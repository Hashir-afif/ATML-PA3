"""Merge trained adapters into a fresh backbone copy and verify merged vs unmerged logits.

    python -m task1_finetune.merge --config configs/task1.yaml --method lora16

The exported effective factors (projected factors B~, A~ for OPLoRA) are merged in FP32 against the exact
original weights, W = W0 + (alpha / r) B A, and cast to FP16 for evaluation. The check compares logits of
the unmerged model (FP16 backbone + FP32 adapters, eval mode) and the merged FP16 model on a small batch of
training prompts and also verifies the OPLoRA constraints numerically (U_k^T dW = 0 and dW V_k = 0).
"""
from __future__ import annotations

import argparse
import time

import torch
from safetensors.torch import load_file

from common.lora import adapter_modules, load_adapter_state_dict, merge_effective_factors
from common.utils import dir_size_bytes, load_config, log, rel, save_json
from task1_finetune.data import collate, get_train_data, load_tokenizer
from task1_finetune.modeling import build_model, load_causal_lm, original_weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1.yaml")
    ap.add_argument("--method", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-check", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.smoke:
        from task1_finetune.train import apply_smoke
        cfg = apply_smoke(cfg)
    mcfg = cfg["methods"][args.method]
    assert mcfg["kind"] in ("lora", "oplora"), "only adapter methods are merged"
    total = cfg["train"]["total_updates"]
    ck = rel(cfg["paths"]["checkpoints"]) / args.method / f"update_{total:04d}"
    assert (ck / "COMPLETE").exists(), f"missing final checkpoint {ck}"
    out = rel(cfg["paths"]["merged"]) / args.method
    res = rel(cfg["paths"]["results"])
    checked = (res / "merge_checks" / f"{args.method}.json").exists()
    consumers_done = (res / "eval" / args.method / "aggregate.json").exists() and \
        (res / "qualitative" / f"{args.method}.jsonl").exists()
    if checked and ((out / "config.json").exists() or consumers_done):
        log(f"{args.method}: merged model exists or its evaluation and qualitative outputs are complete; skipping")
        return
    tok = load_tokenizer(cfg["model_id"])

    factors = load_file(str(ck / "adapter_effective.safetensors"))
    names = sorted({k.rsplit(".", 1)[0] for k in factors})
    scaling = mcfg["alpha"] / mcfg["r"]
    t0 = time.perf_counter()
    ref = original_weights(cfg["model_id"], names)
    merged = load_causal_lm(cfg["model_id"], torch.float16, "cuda")
    n = merge_effective_factors(merged, factors, scaling, reference_weights=ref)
    merge_seconds = time.perf_counter() - t0
    merged.eval()

    # Unmerged reference model with the raw trained factors.
    unmerged, _ = build_model(cfg, mcfg, "cuda")
    load_adapter_state_dict(unmerged, load_file(str(ck / "adapter_raw.safetensors")))
    unmerged.eval()
    factor_err, constraint = 0.0, {"max_rel_UkT_dW": None, "max_rel_dW_Vk": None}
    rel_u, rel_v = [], []
    for name, m in adapter_modules(unmerged).items():
        with torch.no_grad():
            A, B = m.effective_factors()
            factor_err = max(factor_err, float((A.cpu() - factors[f"{name}.A"]).abs().max()),
                             float((B.cpu() - factors[f"{name}.B"]).abs().max()))
            if mcfg["kind"] == "oplora":
                dW = m.delta_weight()
                nrm = dW.norm().clamp_min(1e-30)
                rel_u.append(float((m.U_k.t() @ dW).norm() / nrm))
                rel_v.append(float((dW @ m.V_k).norm() / nrm))
    if rel_u:
        constraint = {"max_rel_UkT_dW": max(rel_u), "max_rel_dW_Vk": max(rel_v)}

    data = get_train_data(cfg)
    batch = {k: v.cuda() for k, v in collate(data, list(range(args.n_check)), tok.pad_token_id).items()}
    with torch.no_grad():
        lm = merged(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()
        lu = unmerged(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()
    mask = batch["attention_mask"].bool()
    diff = (lm - lu).abs()[mask]
    check = {"method": args.method, "n_modules_merged": n, "scaling": scaling, "merge_seconds": merge_seconds,
             "n_check_sequences": args.n_check, "n_check_tokens": int(mask.sum()),
             "logits_max_abs_diff": float(diff.max()), "logits_mean_abs_diff": float(diff.mean()),
             "argmax_agreement": float((lm.argmax(-1) == lu.argmax(-1))[mask].float().mean()),
             "exported_vs_recomputed_factor_max_abs_diff": factor_err, "oplora_constraints": constraint,
             "dtype_merged": "float16", "note": "merged FP16 W0 + dW vs unmerged FP16 W0 x + FP32-adapter path"}
    del unmerged
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    check["merged_dir_bytes"] = dir_size_bytes(out)
    save_json(check, rel(cfg["paths"]["results"]) / "merge_checks" / f"{args.method}.json")
    log(f"merged {args.method}: {check}")


if __name__ == "__main__":
    main()
