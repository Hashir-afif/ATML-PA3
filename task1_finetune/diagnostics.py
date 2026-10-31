"""Weight-space and activation diagnostics for down_proj in layers 0, 13 and 27.

    python -m task1_finetune.diagnostics --config configs/task1.yaml

For each of the six trained checkpoints (five finals + Full FT midpoint) and each layer:
  rho_k = ||U_k U_k^T dW||_F^2 / ||dW||_F^2   (U_k: leading k=128 left singular vectors of W0)
  eta   = ||dW||_F / ||W0||_F
  R(z)  = ||dW z||_2 / (||W0 z||_2 + 1e-8)   for cached starting-model down_proj inputs z at the last prompt
          token of each probe prompt (32 math GSM8K, 32 non-math HellaSwag).
dW = W_trained - W0 for Full FT variants and (alpha/r) B A (projected factors for OPLoRA) for adapters.
Everything is computed in FP32 from the exact original weights. These are local measurements of one layer's
weight change; they do not capture changes elsewhere in the network and do not establish causes.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from safetensors.torch import load_file

from common.lora import singular_bases
from common.utils import load_config, load_json, log, rel, save_json
from task1_finetune.data import load_tokenizer
from task1_finetune.evaluate import condition_path
from task1_finetune.modeling import checkpoint_tensor, load_causal_lm, original_weights

CONDITIONS = ["full_mid", "full", "l2sp", "lora4", "lora16", "oplora16"]


@torch.no_grad()
def cache_inputs(cfg, layers, probe) -> dict:
    """Run the starting model in FP32 and cache down_proj inputs at the last prompt token (batch size 1)."""
    tok = load_tokenizer(cfg["model_id"])
    model = load_causal_lm(cfg["model_id"], torch.float32, "cuda").eval()
    store: dict[int, list] = {l: [] for l in layers}
    hooks = [model.model.layers[l].mlp.down_proj.register_forward_hook(
        lambda m, inp, out, l=l: store[l].append(inp[0][0, -1].float().cpu())) for l in layers]
    items = probe["gsm8k"] + probe["hellaswag"]
    for it in items:
        ids = tok(it["prompt"], return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        model(input_ids=ids)
    for h in hooks:
        h.remove()
    del model
    torch.cuda.empty_cache()
    return {"z": {l: torch.stack(store[l]) for l in layers}, "groups": [it["group"] for it in items],
            "ids": [f"{'gsm8k' if it['group'] == 'math' else 'hellaswag'}:{it['doc_id']}" for it in items]}


def delta_weight(cfg, cond, name, W0) -> torch.Tensor:
    if cond in ("full_mid", "full", "l2sp"):
        return checkpoint_tensor(condition_path(cfg, cond), f"{name}.weight") - W0
    m = cfg["methods"][cond]
    ck = rel(cfg["paths"]["checkpoints"]) / cond / f"update_{cfg['train']['total_updates']:04d}"
    f = load_file(str(ck / "adapter_effective.safetensors"))
    return (m["alpha"] / m["r"]) * (f[f"{name}.B"].float() @ f[f"{name}.A"].float())


def summarize(x: np.ndarray) -> dict:
    return {"mean": float(x.mean()), "median": float(np.median(x)), "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "min": float(x.min()), "max": float(x.max()), "n": int(len(x))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--conditions", nargs="*", default=CONDITIONS)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.smoke:
        from task1_finetune.train import apply_smoke
        cfg = apply_smoke(cfg)
    dc = cfg["diagnostics"]
    layers, k = dc["layers"], dc["protect_k"]
    out = rel(cfg["paths"]["results"]) / "diagnostics"
    probe = load_json(cfg["data"]["probe_set_file"])
    act_path = out / "base_down_proj_inputs.pt"
    if act_path.exists():
        cache = torch.load(act_path, weights_only=False)
    else:
        cache = cache_inputs(cfg, layers, probe)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(cache, act_path)
    groups = np.array(cache["groups"])
    names = [f"model.layers.{l}.mlp.{dc['module']}" for l in layers]
    W0s = original_weights(cfg["model_id"], names)
    results = {"layers": layers, "protect_k": k, "module": dc["module"], "probe_ids": cache["ids"],
               "precision": "float32", "conditions": {}}
    per_prompt = []
    for cond in args.conditions:
        try:
            cres = {}
            for l, name in zip(layers, names):
                W0 = W0s[name].cuda()
                U_k, V_k = singular_bases(W0, k)
                dW = delta_weight(cfg, cond, name, W0s[name]).cuda()
                dn = dW.norm()
                rho = float((U_k.t() @ dW).norm() ** 2 / dn.clamp_min(1e-30) ** 2)
                eta = float(dn / W0.norm())
                Z = cache["z"][l].cuda()  # (n_probe, d_in)
                R = ((Z @ dW.t()).norm(dim=1) / ((Z @ W0.t()).norm(dim=1) + 1e-8)).cpu().numpy()
                cres[str(l)] = {"rho_k": rho, "eta": eta, "dW_fro": float(dn), "W0_fro": float(W0.norm()),
                                "right_subspace_ratio": float((dW @ V_k).norm() ** 2 / dn.clamp_min(1e-30) ** 2),
                                "R_math": summarize(R[groups == "math"]), "R_non_math": summarize(R[groups == "non_math"]),
                                "R_math_over_non_math_mean": float(R[groups == "math"].mean() / R[groups == "non_math"].mean())}
                per_prompt += [{"condition": cond, "layer": l, "probe_id": pid, "group": g, "R": float(r)}
                               for pid, g, r in zip(cache["ids"], groups, R)]
            results["conditions"][cond] = cres
            log(f"{cond}: " + "; ".join(f"L{l} rho={v['rho_k']:.4f} eta={v['eta']:.2e} "
                                        f"Rm={v['R_math']['mean']:.3e} Rn={v['R_non_math']['mean']:.3e}"
                                        for l, v in cres.items()))
        except (FileNotFoundError, KeyError, AssertionError) as e:
            results["conditions"][cond] = None
            results.setdefault("notes", {})[cond] = f"not computed: {type(e).__name__}: {e}"
            log(f"{cond}: skipped ({e})")
    save_json(results, out / "diagnostics.json")
    import pandas as pd

    pd.DataFrame(per_prompt).to_csv(out / "activation_ratios_per_prompt.csv", index=False)
    log(f"saved {out / 'diagnostics.json'}")


if __name__ == "__main__":
    main()
