"""Task 1 training: one epoch (1,250 updates) of SFT for a single method.

    python -m task1_finetune.train --config configs/task1.yaml --method full
    methods: full | l2sp | lora4 | lora16 | oplora16

Every run starts from the original Qwen3-0.6B checkpoint with seed 6304 and the same examples, order and
tokenization. The loss for each optimizer update is the summed response-token NLL over its 32 examples
divided by the number of valid response tokens in that effective batch. FP16 autocast with dynamic
gradient scaling; gradients are unscaled before clipping (global norm 1.0); skipped (non-finite) updates are
logged. For L2-SP, lambda * (theta - theta0) is added to the unscaled gradients before clipping, which is
exactly the gradient of SFT + lambda * Omega.

Recovery checkpoints after updates 312/625/937/1,250 hold weights (or adapters), optimizer, scheduler,
gradient-scaler and RNG state plus the data position; an interrupted run resumes automatically from the
latest complete checkpoint. ``--smoke`` runs a few tiny updates into separate directories.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from common.lora import adapter_state_dict, cache_effective_factors, load_adapter_state_dict
from common.utils import (append_jsonl, capture_rng_state, dir_size_bytes, environment_info, load_config,
                          load_json, load_jsonl, log, peak_memory, rel, reset_peak_memory, restore_rng_state, save_json,
                          save_jsonl, set_seed)
from task1_finetune.data import collate, get_train_data, load_tokenizer, n_response_tokens
from task1_finetune.modeling import build_model, sft_loss_sum
from task1_finetune.optim import L2SPReference, OffloadedAdamW, param_groups


DEADLINE_EXIT = 3  # exit code: paused at the session deadline with a recovery checkpoint (not a failure)


def apply_smoke(cfg: dict) -> dict:
    t = cfg["train"]
    t.update(grad_accum=2, total_updates=4, warmup_updates=1, checkpoint_updates=[2, 4], midpoint_update=2)
    cfg["paths"] = {k: v.replace("task1", "task1_smoke", 1) for k, v in cfg["paths"].items()}
    return cfg


def cosine_lambda(warmup: int, total: int):
    def f(step: int) -> float:  # identical to transformers.get_cosine_schedule_with_warmup
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return f


def ckpt_dir(cfg: dict, method: str, update: int) -> Path:
    return rel(cfg["paths"]["checkpoints"]) / method / f"update_{update:04d}"


def latest_complete(cfg: dict, method: str) -> Path | None:
    root = rel(cfg["paths"]["checkpoints"]) / method
    done = sorted(p for p in root.glob("update_*") if (p / "COMPLETE").exists()) if root.exists() else []
    return done[-1] if done else None


def save_checkpoint(cfg, method, kind, model, tok, opt, sched, scaler, update, run_state, l2sp_ref):
    d = ckpt_dir(cfg, method, update)
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    t0 = time.perf_counter()
    if kind == "full":
        model.save_pretrained(d / "model", safe_serialization=True)
        tok.save_pretrained(d / "model")
        weights_bytes = dir_size_bytes(d / "model")
    else:
        save_file(adapter_state_dict(model), str(d / "adapter_raw.safetensors"))
        save_file(adapter_state_dict(model, effective=True), str(d / "adapter_effective.safetensors"))
        weights_bytes = (d / "adapter_effective.safetensors").stat().st_size
    torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                "rng": capture_rng_state(), "update": update, "examples_seen": update * cfg["_eff_batch"],
                "run_state": run_state}, d / "training_state.pt")
    extra = {}
    if l2sp_ref is not None:
        extra["displacement_l2"] = l2sp_ref.displacement(model)
    meta = {"method": method, "update": update, "examples_seen": update * cfg["_eff_batch"],
            "weights_bytes": weights_bytes, "recovery_state_bytes": dir_size_bytes(d), "save_seconds":
            time.perf_counter() - t0, **extra}
    save_json(meta, d / "meta.json")
    (d / "COMPLETE").write_text("ok")
    meta["recovery_state_bytes"] = dir_size_bytes(d)
    save_json(meta, d / "meta.json")
    return meta


def prune_recovery(cfg: dict, method: str, latest: int, keep_weights: set[int]) -> None:
    """Delete superseded recovery state to save disk: optimizer/scheduler/RNG state of every checkpoint older than
    ``latest``, and full-model weights of older checkpoints except the evaluated ones (midpoint, final). Adapter
    files are tiny and kept. Only enabled where disk is limited (``train.prune_recovery``)."""
    for p in (rel(cfg["paths"]["checkpoints"]) / method).glob("update_*"):
        u = int(p.name.split("_")[1])
        if u >= latest or not (p / "COMPLETE").exists():
            continue
        (p / "training_state.pt").unlink(missing_ok=True)
        if u not in keep_weights and (p / "model").exists():
            shutil.rmtree(p / "model")
        meta = load_json(p / "meta.json")
        if not meta.get("pruned"):
            meta["pruned"] = True
            save_json(meta, p / "meta.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1.yaml")
    ap.add_argument("--method", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.smoke:
        cfg = apply_smoke(cfg)
    mcfg = cfg["methods"][args.method]
    kind, tcfg = mcfg["kind"], cfg["train"]
    eff = tcfg["microbatch"] * tcfg["grad_accum"]
    cfg["_eff_batch"] = eff
    total = tcfg["total_updates"]
    results = rel(cfg["paths"]["results"])
    log_path = results / "train_logs" / f"{args.method}.jsonl"
    summary_path = results / "train_summary" / f"{args.method}.json"
    final_dir = ckpt_dir(cfg, args.method, total)
    if (final_dir / "COMPLETE").exists() and summary_path.exists():
        log(f"{args.method}: final checkpoint and summary already exist; nothing to do")
        return

    set_seed(cfg["seed"])
    data = get_train_data(cfg)
    assert len(data["input_ids"]) >= total * eff, "not enough training examples for the budget"
    tok = load_tokenizer(cfg["model_id"])
    device = "cuda"
    reset_peak_memory()
    model, info = build_model(cfg, mcfg, device)
    model.train()
    log(f"{args.method}: trainable params {info['trainable_params']:,} / {info['total_params_unique']:,}")

    groups = param_groups(model.named_parameters(), tcfg["weight_decay"])
    if kind == "full" and cfg.get("offload_optimizer_states", True):
        opt = OffloadedAdamW(groups, lr=mcfg["lr"], betas=tcfg["betas"], eps=tcfg["eps"])
    else:
        opt = torch.optim.AdamW(groups, lr=mcfg["lr"], betas=tuple(tcfg["betas"]), eps=tcfg["eps"], foreach=False)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_lambda(tcfg["warmup_updates"], total))
    scaler = torch.amp.GradScaler("cuda")
    lam = float(mcfg.get("l2sp_lambda", 0.0))
    l2sp_ref = L2SPReference(model) if kind == "full" else None  # used for displacement in both full runs
    trainable = [p for p in model.parameters() if p.requires_grad]

    start, run_state = 0, {"train_seconds": 0.0, "skipped_updates": [], "segments": [],
                           "peak_allocated_gb": 0.0, "peak_reserved_gb": 0.0}
    resume = None if args.no_resume else latest_complete(cfg, args.method)
    if resume is not None:
        log(f"resuming from {resume}")
        st = torch.load(resume / "training_state.pt", weights_only=False)
        if kind == "full":
            for f in sorted((resume / "model").glob("*.safetensors")):
                missing_unexpected = model.load_state_dict(load_file(str(f)), strict=False)
                assert not missing_unexpected.unexpected_keys, missing_unexpected.unexpected_keys
        else:
            load_adapter_state_dict(model, load_file(str(resume / "adapter_raw.safetensors")))
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        restore_rng_state(st["rng"])
        start, run_state = st["update"], st["run_state"]
        rows = [r for r in load_jsonl(log_path) if r["update"] <= start] if log_path.exists() else []
        save_jsonl(rows, log_path)
    elif log_path.exists():
        log_path.unlink()

    # Optional wall-clock budget (e.g. a 12 h Kaggle session): stop at the next update boundary after this epoch
    # time, save a full recovery checkpoint and exit with DEADLINE_EXIT so the orchestrator can resume later.
    deadline = float(os.environ.get("PA3_DEADLINE_EPOCH", "inf"))
    seg_t0 = time.perf_counter()
    seg_start = start
    pad = tok.pad_token_id
    chunk = tcfg["loss_chunk_tokens"]
    for update in range(start, total):
        t_upd = time.perf_counter()
        idx = list(range(update * eff, (update + 1) * eff))
        n_valid = n_response_tokens(data, idx)
        sft = 0.0
        for mb in range(tcfg["grad_accum"]):
            mb_idx = idx[mb * tcfg["microbatch"]:(mb + 1) * tcfg["microbatch"]]
            batch = {k: v.to(device) for k, v in collate(data, mb_idx, pad).items()}
            if kind == "oplora":
                cache_effective_factors(model, True)
            # cache_enabled=False: do not keep FP16 copies of all FP32 master weights alive for the whole
            # forward pass (saves ~1.2 GB for Full FT; casts are recomputed instead, same numerics).
            with torch.autocast("cuda", dtype=torch.float16, cache_enabled=False):
                loss_sum, _ = sft_loss_sum(model, batch["input_ids"], batch["attention_mask"], batch["labels"], chunk)
            loss = loss_sum / n_valid
            scaler.scale(loss).backward()
            if kind == "oplora":
                cache_effective_factors(model, False)
            sft += float(loss.detach())
        scaler.unscale_(opt)
        omega = None
        if lam > 0:
            omega = l2sp_ref.penalty_and_grad(model, lam, add_grad=True)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, tcfg["max_grad_norm"]))
        scale_before = scaler.get_scale()
        lr_used = opt.param_groups[0]["lr"]
        scaler.step(opt)
        scaler.update()
        skipped = scaler.get_scale() < scale_before
        if skipped:
            run_state["skipped_updates"].append(update + 1)
            log(f"update {update + 1}: non-finite gradients, optimizer step skipped (scale {scale_before})")
        sched.step()
        opt.zero_grad(set_to_none=True)
        row = {"update": update + 1, "examples_seen": (update + 1) * eff, "sft_loss": sft, "lr": lr_used,
               "grad_norm_preclip": grad_norm, "loss_scale": scale_before, "skipped": skipped,
               "n_valid_tokens": n_valid, "seconds": time.perf_counter() - t_upd}
        if lam > 0:
            row.update(omega=omega, lambda_omega=lam * omega, total_loss=sft + lam * omega)
        append_jsonl(row, log_path)
        if (update + 1) % 10 == 0 or update == start:
            log(f"{args.method} update {update + 1}/{total} loss {sft:.4f} lr {lr_used:.2e} "
                f"gnorm {grad_norm:.3f} {row['seconds']:.1f}s/upd")
        scheduled = (update + 1) in tcfg["checkpoint_updates"]
        out_of_time = time.time() > deadline and update + 1 < total
        if scheduled or out_of_time:
            seg_time = time.perf_counter() - seg_t0
            mem = peak_memory()
            st_copy = dict(run_state)
            st_copy["train_seconds"] = run_state["train_seconds"] + seg_time
            st_copy["peak_allocated_gb"] = max(run_state["peak_allocated_gb"], mem["peak_allocated_gb"])
            st_copy["peak_reserved_gb"] = max(run_state["peak_reserved_gb"], mem["peak_reserved_gb"])
            st_copy["segments"] = run_state["segments"] + [{"from": seg_start, "to": update + 1, "seconds": seg_time}]
            meta = save_checkpoint(cfg, args.method, kind, model, tok, opt, sched, scaler, update + 1, st_copy, l2sp_ref)
            run_state, seg_t0, seg_start = st_copy, time.perf_counter(), update + 1
            log(f"saved checkpoint at update {update + 1}: {meta}")
            if tcfg.get("prune_recovery", False):
                prune_recovery(cfg, args.method, update + 1, keep_weights={tcfg["midpoint_update"], total})
            if out_of_time:
                log(f"{args.method}: session deadline reached after update {update + 1}; recovery checkpoint saved, "
                    f"exiting with code {DEADLINE_EXIT} (rerun to resume)")
                sys.exit(DEADLINE_EXIT)

    rows = load_jsonl(log_path)
    final_meta = {int(p.name.split("_")[1]): load_json(p / "meta.json") for p in
                  sorted((rel(cfg["paths"]["checkpoints"]) / args.method).glob("update_*")) if (p / "COMPLETE").exists()}
    summary = {
        "method": args.method, "method_config": mcfg, "seed": cfg["seed"], "model_id": cfg["model_id"],
        "train_config": tcfg, "adapter_config": cfg["adapter"] if kind != "full" else None,
        "train_ids_file": cfg["data"]["train_ids_file"], "effective_batch": eff, "updates": total,
        "examples_seen": total * eff, "trainable_params": info["trainable_params"],
        "total_params_unique": info["total_params_unique"], "train_seconds": run_state["train_seconds"],
        "train_segments": run_state["segments"], "peak_gpu_allocated_gb": run_state["peak_allocated_gb"],
        "peak_gpu_reserved_gb": run_state["peak_reserved_gb"], "skipped_updates": run_state["skipped_updates"],
        "final_sft_loss": rows[-1]["sft_loss"] if rows else None,
        "checkpoints": final_meta, "checkpoint_weights_bytes_final": final_meta.get(total, {}).get("weights_bytes"),
        "precision": "FP32 master weights + FP16 autocast" if kind == "full" else "frozen FP16 backbone + FP32 adapters, FP16 autocast",
        "optimizer": type(opt).__name__, "environment": environment_info(),
        "config_overrides": cfg.get("_overrides"), "method_overhead": {},
    }
    if kind != "full":
        summary["method_overhead"].update(adapter=info["adapter"])
    if isinstance(opt, OffloadedAdamW):
        summary["method_overhead"]["optimizer_state_offload"] = {
            "note": "AdamW moments in CPU RAM, streamed per tensor (8 GB GPU constraint); time of last segment",
            "cpu_state_bytes": opt.state_bytes(), "transfer_and_update_seconds_last_segment": opt.transfer_seconds}
    if l2sp_ref is not None:
        summary["method_overhead"]["reference_theta0"] = {
            "storage_dtype": l2sp_ref.storage_dtype, "storage_bytes": l2sp_ref.storage_bytes,
            "n_params": l2sp_ref.n_params, "penalty_seconds_last_segment": l2sp_ref.seconds,
            "transfer_bytes_last_segment": l2sp_ref.transfer_bytes_total,
            "used_in_objective": lam > 0, "lambda": lam}
        summary["displacement_l2"] = {u: m.get("displacement_l2") for u, m in final_meta.items()}
    save_json(summary, summary_path)
    log(f"done: {summary_path}")


if __name__ == "__main__":
    main()
