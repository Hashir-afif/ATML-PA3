"""Task 2 training: three-stage curriculum for the SigLIP-Qwen VLM (4,500 updates, 72,000 presentations).

    python -m task2_vlm.train --config configs/task2.yaml [--variant default|pool49] [--smoke]

Stage 1 (2,250 updates): connector only, two passes over the 18,000 caption examples (a fresh seeded
permutation per pass). Stage 2 (1,500): connector + newly added LoRA (r=16, alpha=32, q/k/v/o); each draw is
presence (0.5), counting (0.25) or caption replay (0.25). Stage 3 (750): natural VQAv2 (0.7), presence /
counting / caption replay (0.1 each). Each draw contributes two examples (a complete pair, or two captions);
8 draws = one effective batch of 16 (microbatch 8 = 4 draws, accumulation 2). Draws use one seeded sampler
whose state is checkpointed; no batch or stage quotas are enforced and pools may be resampled.

Per stage: AdamW (wd 0.01), linear warmup over 5% of the stage, cosine decay to 10% of the stage's peak LRs; a
fresh optimizer is created at each stage boundary (implementation choice: the parameter set and peak LRs
change). FP16 autocast, dynamic loss scaling, frozen FP16 Qwen, FP32 connector/LoRA/optimizer states;
gradients are unscaled before clipping (norm 1.0) and skipped steps are logged.

Logs: per-update loss, LRs, grad norm, mixture counts; dev metrics at updates 0, every 500, and 2,250 / 3,750 /
4,500; resumable checkpoints every 250 updates (connector, LoRA, optimizer, scheduler, scaler, sampler, stage
progress, RNG); connector/LoRA weights after each stage (stage_k/).
"""
from __future__ import annotations

import argparse
import math
import shutil
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from common.lora import adapter_state_dict, load_adapter_state_dict
from common.utils import (append_jsonl, capture_rng_state, environment_info, load_config, load_json, load_jsonl, log,
                          peak_memory, reset_peak_memory, restore_rng_state, save_json, save_jsonl, set_seed)
from task1_finetune.optim import param_groups
from task2_vlm.evaluate import dev_metrics
from task2_vlm.examples import caption_train_examples, counting_pair, presence_pair, vqa_pair
from task2_vlm.features import FeatureStore
from task2_vlm.model import build_vlm, calibrate_connector, collate, variant_settings


def stage_bounds(cfg) -> list[tuple[int, int]]:
    out, s = [], 0
    for st in cfg["train"]["stages"]:
        out.append((s, s + st["updates"]))
        s += st["updates"]
    return out


def stage_of(cfg, update: int) -> int:
    for k, (a, b) in enumerate(stage_bounds(cfg)):
        if a <= update < b:
            return k
    raise ValueError(update)


def lr_lambda(n_updates: int, warmup_frac: float, min_frac: float):
    warm = max(1, int(round(warmup_frac * n_updates)))

    def f(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        prog = min(1.0, (step - warm) / max(1, n_updates - warm))
        return min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * prog))
    return f


def make_optimizer(cfg, vlm, stage: int):
    st, tc = cfg["train"]["stages"][stage], cfg["train"]
    groups = []
    for g in param_groups(vlm.connector.named_parameters(), tc["weight_decay"]):
        groups.append(dict(g, lr=st["connector_lr"]))
    if st["lora_lr"] is not None:
        lora = [(n, p) for n, p in vlm.qwen.named_parameters() if p.requires_grad]
        for g in param_groups(lora, tc["weight_decay"]):
            groups.append(dict(g, lr=st["lora_lr"]))
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8, foreach=False)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(st["updates"], tc["warmup_frac"], tc["min_lr_frac"]))
    return opt, sched


class Sampler:
    """Stage-1 caption stream (two seeded permutations) and the seeded mixture sampler for Stages 2-3."""

    def __init__(self, cfg, man, pools):
        self.captions = caption_train_examples(man)
        self.cats = pools["categories"]
        self.presence, self.counting = pools["presence"], pools["counting"]
        self.vqa = man["vqa_train"]
        seed = cfg["seed"]
        n_pass = math.ceil(cfg["train"]["stages"][0]["updates"] * 16 / len(self.captions))
        self.caption_order = np.concatenate([np.random.default_rng([seed, 100, k]).permutation(len(self.captions))
                                             for k in range(n_pass)])
        self.rng = np.random.default_rng([seed, 200])

    def stage1(self, update: int, eff: int) -> list[dict]:
        idx = self.caption_order[update * eff:(update + 1) * eff]
        return [self.captions[i] for i in idx]

    def draw(self, mixture: dict) -> tuple[str, list[dict]]:
        comps = list(mixture)
        c = comps[self.rng.choice(len(comps), p=np.array([mixture[k] for k in comps]))]
        if c == "caption":
            i, j = self.rng.integers(len(self.captions), size=2)
            return c, [self.captions[i], self.captions[j]]
        if c == "presence":
            return c, presence_pair(self.presence[self.rng.integers(len(self.presence))], self.cats)
        if c == "counting":
            return c, counting_pair(self.counting[self.rng.integers(len(self.counting))], self.cats)
        return c, vqa_pair(self.vqa[self.rng.integers(len(self.vqa))])


def text_token_ids(vlm, man) -> torch.Tensor:
    tok = vlm.builder.tok
    texts = [c for e in man["captions_train"] for c in e["captions"]] + \
            [p["question"] for p in man["vqa_train"]] + [it["target"] for p in man["vqa_train"] for it in p["items"]]
    ids = [i for t in texts for i in tok(t, add_special_tokens=False).input_ids]
    return torch.tensor(ids)


def save_ckpt(vs, name, vlm, opt, sched, scaler, sampler, update, run_state, complete_marker=True):
    d = vs["ckpt"] / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in vlm.connector.state_dict().items()},
              str(d / "connector.safetensors"))
    lora = adapter_state_dict(vlm.qwen)
    if lora:
        save_file(lora, str(d / "lora.safetensors"))
    if opt is not None:
        torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                    "sampler_rng": sampler.rng.bit_generator.state, "rng": capture_rng_state(), "update": update,
                    "run_state": run_state}, d / "training_state.pt")
    save_json({"update": update, "has_lora": bool(lora)}, d / "meta.json")
    if complete_marker:
        (d / "COMPLETE").write_text("ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task2.yaml")
    ap.add_argument("--variant", default="default", choices=["default", "pool49"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-stage-updates", type=int, nargs=3, default=[4, 3, 3])
    args = ap.parse_args()
    cfg = load_config(args.config)
    tc = cfg["train"]
    if args.smoke:
        for st, n in zip(tc["stages"], args.smoke_stage_updates):
            st["updates"] = n
        tc.update(dev_eval_updates=[0, args.smoke_stage_updates[0], sum(args.smoke_stage_updates)],
                  checkpoint_every=5, calibration_images=16)
        cfg["paths"] = {k: v + "_smoke" for k, v in cfg["paths"].items()}
    vs = variant_settings(cfg, args.variant)
    total = sum(st["updates"] for st in tc["stages"])
    eff = tc["microbatch"] * tc["grad_accum"]
    draws_per_mb = tc["microbatch"] // 2
    log_path, dev_path = vs["results"] / "train_log.jsonl", vs["results"] / "dev_metrics.jsonl"
    if (vs["ckpt"] / "stage_3" / "COMPLETE").exists() and (vs["results"] / "train_summary.json").exists():
        log("training already complete")
        return

    set_seed(cfg["seed"])
    man, pools = load_json(cfg["data"]["manifest"]), load_json(cfg["data"]["pools"])
    store = FeatureStore(cfg)
    sampler = Sampler(cfg, man, pools)
    bounds = stage_bounds(cfg)

    ckpts = sorted(p for p in vs["ckpt"].glob("update_*") if (p / "COMPLETE").exists()) if vs["ckpt"].exists() else []
    resume = ckpts[-1] if ckpts else None
    start = 0
    run_state = {"train_seconds": 0.0, "stage_seconds": {}, "skipped_updates": [], "peak_allocated_gb": 0.0,
                 "peak_reserved_gb": 0.0, "calibration": None, "trainable_params_by_stage": {}}
    if resume is not None:
        st = torch.load(resume / "training_state.pt", weights_only=False)
        start, run_state = st["update"], st["run_state"]
    vlm, info = build_vlm(cfg, args.variant, with_lora=start >= bounds[1][0])
    reset_peak_memory()
    if resume is None:
        calib_imgs = [(e["split"], e["image_id"]) for e in man["captions_train"][:tc["calibration_images"]]]
        run_state["calibration"] = calibrate_connector(vlm.connector, vlm.qwen, store.batch(calib_imgs),
                                                       text_token_ids(vlm, man))
        log(f"connector calibration: {run_state['calibration']}")
        for p in (log_path, dev_path):
            if p.exists():
                p.unlink()
    stage = stage_of(cfg, start) if start < total else len(bounds) - 1
    opt, sched = make_optimizer(cfg, vlm, stage)
    scaler = torch.amp.GradScaler("cuda")
    if resume is not None:
        log(f"resuming from {resume} (update {start}, stage {stage + 1})")
        vlm.connector.load_state_dict(load_file(str(resume / "connector.safetensors")))
        if (resume / "lora.safetensors").exists():
            load_adapter_state_dict(vlm.qwen, load_file(str(resume / "lora.safetensors")))
        if bounds[stage][0] != start:  # mid-stage: restore optimizer/scheduler; at a boundary start fresh
            opt.load_state_dict(st["optimizer"])
            sched.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        sampler.rng.bit_generator.state = st["sampler_rng"]
        restore_rng_state(st["rng"])
        for p in (log_path, dev_path):
            if p.exists():
                save_jsonl([r for r in load_jsonl(p) if r["update"] <= start], p)
    vlm.train()
    pad = vlm.builder.tok.pad_token_id

    def record_trainable(k):
        run_state["trainable_params_by_stage"][str(k + 1)] = {
            "connector": sum(p.numel() for p in vlm.connector.parameters() if p.requires_grad),
            "lora": sum(p.numel() for p in vlm.qwen.parameters() if p.requires_grad)}

    def run_dev(update):
        t0 = time.perf_counter()
        m = dev_metrics(vlm, store, man)
        append_jsonl({"update": update, "stage": stage_of(cfg, max(0, update - 1)) + 1, **m,
                      "seconds": time.perf_counter() - t0}, dev_path)
        log(f"dev @ {update}: caption {m['caption_loss']:.3f} | grounding dNLL {m['grounding_dev']['delta_nll']:.3f} "
            f"| vqa dNLL {m['vqa_dev']['delta_nll']:.3f}")

    if start == 0:
        run_dev(0)
    record_trainable(stage)
    seg_t0 = time.perf_counter()
    for update in range(start, total):
        k = stage_of(cfg, update)
        if k != stage:  # stage boundary
            stage = k
            if stage == 1 and not any(p.requires_grad for p in vlm.qwen.parameters()):
                from common.lora import inject_adapters
                lc = cfg["lora"]
                info["lora"] = inject_adapters(vlm.qwen, lc["targets"], lc["r"], lc["alpha"], lc["dropout"],
                                               seed=cfg["seed"])
                vlm.train()
            opt, sched = make_optimizer(cfg, vlm, stage)
            record_trainable(stage)
        st_cfg = tc["stages"][stage]
        t_upd = time.perf_counter()
        counts = {}
        if stage == 0:
            exs = sampler.stage1(update, eff)
            counts["caption"] = eff // 2
        else:
            exs = []
            for _ in range(eff // 2):
                c, pair = sampler.draw(st_cfg["mixture"])
                counts[c] = counts.get(c, 0) + 1
                exs += pair
        loss_total, trunc = 0.0, 0
        for mb in range(tc["grad_accum"]):
            chunk = exs[mb * tc["microbatch"]:(mb + 1) * tc["microbatch"]]
            built = [vlm.builder.build(e, True, e["target"]) for e in chunk]
            trunc += sum(b["question_truncated"] or b["target_truncated"] for b in built)
            batch = {kk: v.cuda() for kk, v in collate(built, pad).items()}
            feats = store.batch([e["image"] for e in chunk]).cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                nll = vlm.per_example_nll(batch, feats)
            loss = nll.sum() / eff
            scaler.scale(loss).backward()
            loss_total += float(loss.detach())
        scaler.unscale_(opt)
        params = [p for g in opt.param_groups for p in g["params"]]
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, tc["max_grad_norm"]))
        scale = scaler.get_scale()
        lrs = [g["lr"] for g in opt.param_groups]
        scaler.step(opt)
        scaler.update()
        skipped = scaler.get_scale() < scale
        if skipped:
            run_state["skipped_updates"].append(update + 1)
        sched.step()
        opt.zero_grad(set_to_none=True)
        append_jsonl({"update": update + 1, "stage": stage + 1, "loss": loss_total, "connector_lr": lrs[0],
                      "lora_lr": lrs[-1] if st_cfg["lora_lr"] is not None else None, "grad_norm_preclip": gnorm,
                      "loss_scale": scale, "skipped": skipped, "mixture_draws": counts, "truncated_examples": trunc,
                      "seconds": time.perf_counter() - t_upd}, log_path)
        if (update + 1) % 50 == 0 or update == start:
            log(f"update {update + 1}/{total} stage {stage + 1} loss {loss_total:.4f} gnorm {gnorm:.2f} "
                f"{time.perf_counter() - t_upd:.2f}s")
        u = update + 1
        is_stage_end = u in [b for _, b in bounds]
        if is_stage_end or u % tc["checkpoint_every"] == 0:
            seg = time.perf_counter() - seg_t0
            run_state["train_seconds"] += seg
            run_state["stage_seconds"][str(stage + 1)] = run_state["stage_seconds"].get(str(stage + 1), 0.0) + seg
            mem = peak_memory()
            run_state["peak_allocated_gb"] = max(run_state["peak_allocated_gb"], mem["peak_allocated_gb"])
            run_state["peak_reserved_gb"] = max(run_state["peak_reserved_gb"], mem["peak_reserved_gb"])
            if is_stage_end:
                save_ckpt(vs, f"stage_{stage + 1}", vlm, None, None, None, None, u, run_state)
            save_ckpt(vs, f"update_{u:04d}", vlm, opt, sched, scaler, sampler, u, run_state)
            seg_t0 = time.perf_counter()
        if u in tc["dev_eval_updates"]:
            t_dev = time.perf_counter()
            run_dev(u)
            seg_t0 += time.perf_counter() - t_dev  # dev-eval time is logged separately, not counted as training
    rows = load_jsonl(log_path)
    mix = {}
    for r in rows:
        for c, n in r["mixture_draws"].items():
            mix.setdefault(str(r["stage"]), {}).setdefault(c, 0)
            mix[str(r["stage"])][c] += n
    summary = {"variant": args.variant, "seed": cfg["seed"], "updates": total, "examples": total * eff,
               "effective_batch": eff, "stages": tc["stages"], "lora": cfg["lora"], "n_visual_tokens": vs["n_visual"],
               "train_seconds": run_state["train_seconds"], "stage_seconds": run_state["stage_seconds"],
               "peak_gpu_allocated_gb": run_state["peak_allocated_gb"], "peak_gpu_reserved_gb": run_state["peak_reserved_gb"],
               "trainable_params_by_stage": run_state["trainable_params_by_stage"],
               "skipped_updates": run_state["skipped_updates"], "calibration": run_state["calibration"],
               "mixture_draws_by_stage": mix, "truncated_examples_total": sum(r["truncated_examples"] for r in rows),
               "environment": environment_info(), "note": "train_seconds excludes development evaluation time"}
    save_json(summary, vs["results"] / "train_summary.json")
    log(f"done: {summary}")


if __name__ == "__main__":
    main()
