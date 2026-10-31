"""Task 2 evaluation: development NLL tracking and the final four-condition grounding evaluation.

    python -m task2_vlm.evaluate --config configs/task2.yaml [--variant default|pool49]

Development (called from training): caption reference loss on 96 dev captions; reference-answer NLL on the
grounding and VQAv2 dev suites under correct, complementary-partner and no-image inputs; image-swap penalty
Delta_NLL = NLL(y | q, I_partner) - NLL(y | q, I_correct). Answer NLL excludes the end-of-turn token and is
averaged within each example before averaging examples.

Final (completed Stage 3 model only; never used for checkpoint selection): both final-test suites, greedy
decoding, thinking disabled, 16-new-token cap, batch size 8, under four image conditions with the question and
original references fixed: correct, complementary swap, fixed mismatched image, no image. Grounding: normalized
exact match (presence / counting separately). VQAv2: local soft accuracy against all ten answers (yes/no,
number, other separately). Per suite and condition: pair accuracy, identical-answer rate, empty-answer rate,
partner-following rate, mean reference NLL and generation time per answer; empty outputs stay in denominators.
95% CIs from 4,000 bootstrap resamples of complete pairs, with the same sampled pair IDs for every condition.
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np
import torch

from common.utils import load_config, load_json, log, peak_memory, reset_peak_memory, save_json, save_jsonl
from task2_vlm.examples import caption_dev_examples, eval_items
from task2_vlm.features import FeatureStore
from task2_vlm.model import collate, variant_settings
from task2_vlm.vqa_norm import normalize_answer, soft_accuracy

COND_IMAGE = {"correct": "image", "swap": "partner_image", "mismatch": "mismatch_image", "no_image": None}


def _image_for(item: dict, cond: str):
    key = COND_IMAGE[cond]
    return None if key is None else item[key]


@torch.no_grad()
def nll_values(vlm, store: FeatureStore, exs: list[dict], images: list, include_eot: bool, bs: int = 16) -> np.ndarray:
    out = []
    pad = vlm.builder.tok.pad_token_id
    for s in range(0, len(exs), bs):
        chunk, imgs = exs[s:s + bs], images[s:s + bs]
        with_img = imgs[0] is not None
        built = [vlm.builder.build(e, with_img, e["target"], include_eot=include_eot) for e in chunk]
        batch = {k: v.cuda() for k, v in collate(built, pad).items()}
        feats = store.batch(imgs).cuda() if with_img else None
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(vlm.per_example_nll(batch, feats).float().cpu().numpy())
    return np.concatenate(out)


def dev_metrics(vlm, store: FeatureStore, man: dict) -> dict:
    was_training = vlm.training
    vlm.eval()
    caps = caption_dev_examples(man)
    res = {"caption_loss": float(nll_values(vlm, store, caps, [c["image"] for c in caps], True).mean())}
    for suite in ("grounding_dev", "vqa_dev"):
        items = eval_items(man, suite)
        vals = {c: nll_values(vlm, store, items, [_image_for(it, c) for it in items], False)
                for c in ("correct", "swap", "no_image")}
        qt = np.array([it["qtype"] for it in items])
        entry = {f"nll_{c}": float(v.mean()) for c, v in vals.items()}
        entry["delta_nll"] = float((vals["swap"] - vals["correct"]).mean())
        entry["by_qtype"] = {t: {**{f"nll_{c}": float(v[qt == t].mean()) for c, v in vals.items()},
                                 "delta_nll": float((vals["swap"] - vals["correct"])[qt == t].mean())}
                             for t in sorted(set(qt))}
        res[suite] = entry
    vlm.train(was_training)
    return res


@torch.no_grad()
def generate_condition(vlm, store, items, cond, cfg) -> tuple[list[str], float]:
    ecfg = cfg["eval"]
    pad = vlm.builder.tok.pad_token_id
    preds, seconds = [], 0.0
    for s in range(0, len(items), ecfg["batch_size"]):
        chunk = items[s:s + ecfg["batch_size"]]
        imgs = [_image_for(it, cond) for it in chunk]
        with_img = imgs[0] is not None
        built = [vlm.builder.build(it, with_img, None) for it in chunk]
        batch = {k: v.cuda() for k, v in collate(built, pad, left_pad=True).items()}
        feats = store.batch(imgs).cuda() if with_img else None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            gen = vlm.generate(batch, feats, ecfg["max_new_tokens"])
        torch.cuda.synchronize()
        seconds += time.perf_counter() - t0
        preds += [vlm.builder.tok.decode(g, skip_special_tokens=True).strip() for g in gen]
    return preds, seconds / len(items)


def item_scores(items, preds) -> np.ndarray:
    return np.array([soft_accuracy(p, it["refs"]) if it["kind"] == "vqa" else
                     float(normalize_answer(p) == it["canonical"]) for it, p in zip(items, preds)])


def pair_view(items, values) -> tuple[list[str], np.ndarray]:
    by = defaultdict(dict)
    for it, v in zip(items, values):
        by[it["pair_id"]][it["index"]] = v
    ids = sorted(by)
    return ids, np.array([[by[i][0], by[i][1]] for i in ids])


def summarize_condition(items, preds, nll, seconds) -> dict:
    norm = [normalize_answer(p) for p in preds]
    acc = item_scores(items, preds)
    exact = np.array([float(n == it["canonical"]) for it, n in zip(items, norm)])
    follows = np.array([float(n == it["partner_canonical"]) for it, n in zip(items, norm)])
    _, pair_exact = pair_view(items, exact)
    _, pair_norm = pair_view(items, norm)
    qt = np.array([it["qtype"] for it in items])
    return {"accuracy": float(acc.mean()), "accuracy_by_qtype": {t: float(acc[qt == t].mean()) for t in sorted(set(qt))},
            "exact_match_canonical": float(exact.mean()),
            "pair_accuracy": float((pair_exact.sum(1) == 2).mean()),
            "identical_answer_rate": float((pair_norm[:, 0] == pair_norm[:, 1]).mean()),
            "empty_answer_rate": float(np.mean([n == "" for n in norm])),
            "follows_partner_rate": float(follows.mean()),
            "mean_reference_nll": float(nll.mean()),
            "mean_reference_nll_by_qtype": {t: float(nll[qt == t].mean()) for t in sorted(set(qt))},
            "generation_seconds_per_answer": seconds, "n_questions": len(items)}


def bootstrap(items, per_cond_acc: dict, per_cond_nll: dict, n_boot: int, seed: int) -> dict:
    """Pair-level bootstrap; one set of resampled pair IDs shared by every condition."""
    rng = np.random.default_rng(seed)
    pair_ids, _ = pair_view(items, per_cond_acc["correct"])
    n = len(pair_ids)
    idx = rng.integers(0, n, size=(n_boot, n))
    pa = {c: pair_view(items, v)[1].mean(1) for c, v in per_cond_acc.items()}  # pair-mean accuracy
    pn = {c: pair_view(items, v)[1].mean(1) for c, v in per_cond_nll.items()}

    def ci(x):
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    out = {"n_pairs": n, "n_boot": n_boot, "accuracy_ci95": {c: ci(v[idx].mean(1)) for c, v in pa.items()},
           "correct_minus_condition_accuracy": {}, "condition_minus_correct_nll": {}}
    for c in pa:
        if c == "correct":
            continue
        d = pa["correct"] - pa[c]
        out["correct_minus_condition_accuracy"][c] = {"estimate": float(d.mean()), "ci95": ci(d[idx].mean(1))}
        dn = pn[c] - pn["correct"]
        out["condition_minus_correct_nll"][c] = {"estimate": float(dn.mean()), "ci95": ci(dn[idx].mean(1))}
    return out


def final_eval(cfg: dict, variant: str) -> dict:
    from safetensors.torch import load_file

    from common.lora import load_adapter_state_dict
    from task2_vlm.model import build_vlm

    vs = variant_settings(cfg, variant)
    final = vs["ckpt"] / "stage_3"
    assert (final / "COMPLETE").exists(), f"missing completed Stage 3 model at {final}"
    vlm, _ = build_vlm(cfg, variant, with_lora=True)
    vlm.connector.load_state_dict(load_file(str(final / "connector.safetensors")))
    load_adapter_state_dict(vlm.qwen, load_file(str(final / "lora.safetensors")))
    vlm.eval()
    man = load_json(cfg["data"]["manifest"])
    store = FeatureStore(cfg)
    reset_peak_memory()
    results = {"variant": variant, "checkpoint": str(final), "settings": cfg["eval"], "suites": {}}
    rows = []
    for suite in ("grounding_test", "vqa_test"):
        items = eval_items(man, suite)
        accs, nlls, summ = {}, {}, {}
        for cond in cfg["eval"]["conditions"]:
            preds, sec = generate_condition(vlm, store, items, cond, cfg)
            nll = nll_values(vlm, store, items, [_image_for(it, cond) for it in items], False)
            accs[cond], nlls[cond] = item_scores(items, preds), nll
            summ[cond] = summarize_condition(items, preds, nll, sec)
            for it, p, a, l in zip(items, preds, accs[cond], nll):
                rows.append({"suite": suite, "condition": cond, "pair_id": it["pair_id"], "index": it["index"],
                             "qtype": it["qtype"], "question": it["question"], "image": it["image"],
                             "shown_image": _image_for(it, cond), "references": it["refs"],
                             "canonical": it["canonical"], "partner_canonical": it["partner_canonical"],
                             "prediction": p, "prediction_normalized": normalize_answer(p), "score": float(a),
                             "reference_nll": float(l)})
            log(f"{suite} {cond}: acc {summ[cond]['accuracy']:.3f} pair {summ[cond]['pair_accuracy']:.3f} "
                f"nll {summ[cond]['mean_reference_nll']:.3f}")
        results["suites"][suite] = {"conditions": summ,
                                    "bootstrap": bootstrap(items, accs, nlls, cfg["eval"]["n_bootstrap"], cfg["seed"])}
    results["memory"] = peak_memory()
    save_jsonl(rows, vs["results"] / "final_predictions.jsonl")
    save_json(results, vs["results"] / "final_eval.json")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task2.yaml")
    ap.add_argument("--variant", default="default")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.smoke:
        cfg["paths"] = {k: v + "_smoke" for k, v in cfg["paths"].items()}
    if (variant_settings(cfg, args.variant)["results"] / "final_eval.json").exists():
        log("final_eval.json exists; skipping")
        return
    final_eval(cfg, args.variant)


if __name__ == "__main__":
    main()
