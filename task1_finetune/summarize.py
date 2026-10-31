"""Collect every Task 1 result into one JSON, tables and figures.

    python -m task1_finetune.summarize --config configs/task1.yaml

Outputs (results/task1/): task1_results.json, benchmark_table.{csv,md}, resources_table.{csv,md},
l2sp_comparison.json, diagnostics_table.csv and figures/ (loss curves, L2-SP penalty, benchmark deltas,
diagnostics). Missing inputs are recorded as null with a note, never dropped silently.
Paired-bootstrap 95% CIs (1,000 resamples of examples, seed 6304) are given for each change from baseline.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import plotting
from common.utils import load_config, load_json, load_jsonl, log, rel, save_json

PRIMARY = {"gsm8k_cot": ("exact_match,flexible-extract", "exact_match"),
           "hellaswag": ("acc_norm,none", "acc_norm"), "boolq": ("acc,none", "acc"),
           "arc_easy": ("acc_norm,none", "acc_norm")}
RAW = {"gsm8k_cot": ("exact_match,strict-match", None), "hellaswag": ("acc,none", "acc"),
       "arc_easy": ("acc,none", "acc")}
METHODS = ["full", "l2sp", "lora4", "lora16", "oplora16"]


def per_example(res_dir, task) -> dict | None:
    p = res_dir / f"samples_{task}.jsonl"
    if not p.exists():
        return None
    key = "exact_match_flexible_extract" if task == "gsm8k_cot" else PRIMARY[task][1]
    return {r["doc_id"]: float(r[key]) for r in load_jsonl(p)}


def paired_ci(a: dict, b: dict, seed: int, n_boot: int = 1000) -> tuple[float, float]:
    ids = sorted(set(a) & set(b))
    d = np.array([b[i] - a[i] for i in ids])
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    return float(np.percentile(boots, 2.5) * 100), float(np.percentile(boots, 97.5) * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.smoke:
        from task1_finetune.train import apply_smoke
        cfg = apply_smoke(cfg)
    res = rel(cfg["paths"]["results"])
    figs = res / "figures"
    plotting.setup()
    conds = cfg["eval"]["conditions"]
    out = {"seed": cfg["seed"], "conditions": conds, "notes": {}}

    # ---- benchmarks ----
    evals, samples = {}, {}
    for c in conds:
        p = res / "eval" / c / "aggregate.json"
        if p.exists():
            evals[c] = load_json(p)
            samples[c] = {t: per_example(res / "eval" / c, t) for t in PRIMARY}
        else:
            out["notes"][f"eval_{c}"] = "Status: NOT RUN (aggregate.json missing)"
    rows = []
    for c in conds:
        row = {"condition": c}
        for t, (k, _) in PRIMARY.items():
            v = evals.get(c, {}).get("tasks", {}).get(t, {}).get("metrics", {}).get(k)
            row[f"{t}_primary"] = None if v is None else 100 * v
            rk = RAW.get(t, (None,))[0]
            if rk:
                rv = evals.get(c, {}).get("tasks", {}).get(t, {}).get("metrics", {}).get(rk)
                row[f"{t}_{'strict' if t == 'gsm8k_cot' else 'acc_raw'}"] = None if rv is None else 100 * rv
            base = evals.get("baseline", {}).get("tasks", {}).get(t, {}).get("metrics", {}).get(k)
            row[f"{t}_delta_pp"] = None if v is None or base is None else 100 * (v - base)
            if c != "baseline" and c in samples and "baseline" in samples and samples[c][t] and samples["baseline"][t]:
                lo, hi = paired_ci(samples["baseline"][t], samples[c][t], cfg["seed"])
                row[f"{t}_delta_ci95"] = [lo, hi]
        g = evals.get(c, {}).get("tasks", {}).get("gsm8k_cot", {})
        row.update(gsm8k_truncation_rate=g.get("truncation_rate"),
                   gsm8k_flex_unextractable_rate=g.get("flexible_unextractable_rate"),
                   gsm8k_mean_response_tokens=g.get("mean_response_tokens"))
        rows.append(row)
    bench = pd.DataFrame(rows)
    bench.to_csv(res / "benchmark_table.csv", index=False)
    (res / "benchmark_table.md").write_text(bench.drop(columns=[c for c in bench if c.endswith("ci95")]).to_markdown(
        index=False, floatfmt=".2f"), encoding="utf-8")
    out["benchmarks"] = rows

    if evals:
        fig, axes = plotting.plt.subplots(1, 4, figsize=(13, 3.0), sharey=False)
        for ax, t in zip(axes, PRIMARY):
            cs = [c for c in conds if c != "baseline" and c in evals]
            vals = [bench.set_index("condition").loc[c, f"{t}_delta_pp"] for c in cs]
            ax.bar(range(len(cs)), [0 if v is None else v for v in vals], color=[plotting.color(c) for c in cs],
                   width=0.7, edgecolor="white", linewidth=1)
            ax.axhline(0, color=plotting.NEUTRAL, lw=0.8)
            ax.set_xticks(range(len(cs)), [plotting.LABELS[c] for c in cs], rotation=45, ha="right")
            ax.set_title({"gsm8k_cot": "GSM8K (flex EM)", "hellaswag": "HellaSwag (acc_norm)",
                          "boolq": "BoolQ (acc)", "arc_easy": "ARC-Easy (acc_norm)"}[t])
            ax.set_ylabel("Change from baseline (pp)")
        plotting.save(fig, figs / "benchmark_deltas")

    # ---- training logs ----
    logs, summaries = {}, {}
    for m in METHODS:
        lp, sp = res / "train_logs" / f"{m}.jsonl", res / "train_summary" / f"{m}.json"
        if lp.exists():
            logs[m] = load_jsonl(lp)
        if sp.exists():
            summaries[m] = load_json(sp)
        else:
            out["notes"][f"train_{m}"] = "Status: NOT RUN or incomplete (train_summary missing)"
    out["training_trajectories"] = logs
    out["training_summaries"] = summaries
    if logs:
        fig, ax = plotting.plt.subplots(figsize=(6.4, 3.6))
        for i, m in enumerate(METHODS):
            if m not in logs:
                continue
            df = pd.DataFrame(logs[m])
            ax.plot(df["update"], df.sft_loss, color=plotting.color(m), alpha=0.18, lw=0.8)
            ax.plot(df["update"], df.sft_loss.rolling(25, min_periods=1).mean(), color=plotting.color(m),
                    label=plotting.LABELS[m])
        ax.set_xlabel("Optimizer update")
        ax.set_ylabel("SFT loss (nats/token)")
        ax.set_title("Training SFT loss (faint: per update; solid: 25-update mean)")
        ax.legend()
        plotting.save(fig, figs / "sft_loss_curves")
    if "l2sp" in logs:
        df = pd.DataFrame(logs["l2sp"])
        fig, axes = plotting.plt.subplots(1, 2, figsize=(10, 3.2))
        axes[0].plot(df["update"], df.omega, color=plotting.color("l2sp"))
        axes[0].set_xlabel("Optimizer update")
        axes[0].set_ylabel(r"$\Omega(\theta,\theta_0)$")
        axes[0].set_title("L2-SP penalty Omega")
        axes[1].plot(df["update"], df.sft_loss.rolling(25, min_periods=1).mean(), color=plotting.color("l2sp"),
                     label="SFT loss")
        axes[1].plot(df["update"], df.total_loss.rolling(25, min_periods=1).mean(), color=plotting.NEUTRAL, ls="--",
                     label=r"total = SFT + $\lambda\Omega$")
        axes[1].set_xlabel("Optimizer update")
        axes[1].set_ylabel("Loss (25-update mean)")
        axes[1].legend()
        plotting.save(fig, figs / "l2sp_penalty")

    # ---- L2-SP comparison and resources ----
    comp = {}
    for m in ("full", "l2sp"):
        s = summaries.get(m)
        if s:
            comp[m] = {"displacement_l2": s.get("displacement_l2"), "final_sft_loss": s.get("final_sft_loss")}
    if "l2sp" in logs:
        last = logs["l2sp"][-1]
        comp.setdefault("l2sp", {}).update(final_omega=last.get("omega"), final_lambda_omega=last.get("lambda_omega"),
                            final_total_loss=last.get("total_loss"), lambda_value=cfg["methods"]["l2sp"]["l2sp_lambda"])
    save_json(comp, res / "l2sp_comparison.json")
    out["l2sp_comparison"] = comp
    rrows = []
    for m in METHODS:
        s = summaries.get(m)
        if not s:
            rrows.append({"method": m, "status": "NOT RUN"})
            continue
        ck = s.get("checkpoints", {})
        last = ck.get(str(cfg["train"]["total_updates"])) or ck.get(cfg["train"]["total_updates"]) or {}
        ov = s.get("method_overhead", {})
        rrows.append({"method": m, "trainable_params": s["trainable_params"], "train_hours": s["train_seconds"] / 3600,
                      "peak_gpu_allocated_gb": s["peak_gpu_allocated_gb"], "peak_gpu_reserved_gb": s["peak_gpu_reserved_gb"],
                      "checkpoint_weights_mb": (last.get("weights_bytes") or 0) / 2**20,
                      "recovery_state_mb": (last.get("recovery_state_bytes") or 0) / 2**20,
                      "skipped_updates": len(s.get("skipped_updates", [])),
                      "svd_seconds": ov.get("adapter", {}).get("svd_seconds"),
                      "basis_mb": (ov.get("adapter", {}).get("basis_bytes") or 0) / 2**20 or None,
                      "theta0_reference_mb": (ov.get("reference_theta0", {}).get("storage_bytes") or 0) / 2**20 or None,
                      "l2sp_penalty_seconds": ov.get("reference_theta0", {}).get("penalty_seconds_last_segment")
                      if m == "l2sp" else None})
    rdf = pd.DataFrame(rrows)
    rdf.to_csv(res / "resources_table.csv", index=False)
    (res / "resources_table.md").write_text(rdf.to_markdown(index=False, floatfmt=".3g"), encoding="utf-8")
    out["resources"] = rrows
    merge_checks = {m: load_json(res / "merge_checks" / f"{m}.json") for m in METHODS
                    if (res / "merge_checks" / f"{m}.json").exists()}
    out["merge_checks"] = merge_checks

    # ---- diagnostics ----
    dp = res / "diagnostics" / "diagnostics.json"
    if dp.exists():
        diag = load_json(dp)
        out["diagnostics"] = diag
        drows = []
        for c, v in diag["conditions"].items():
            if not v:
                continue
            for l, d in v.items():
                drows.append({"condition": c, "layer": int(l), "rho_k": d["rho_k"], "eta": d["eta"],
                              "R_math_mean": d["R_math"]["mean"], "R_non_math_mean": d["R_non_math"]["mean"],
                              "R_math_over_non_math": d["R_math_over_non_math_mean"]})
        ddf = pd.DataFrame(drows)
        ddf.to_csv(res / "diagnostics_table.csv", index=False)
        if len(ddf):
            cs = list(dict.fromkeys(ddf.condition))
            layers = sorted(ddf.layer.unique())
            fig, axes = plotting.plt.subplots(1, 4, figsize=(14, 3.2))
            w = 0.8 / len(cs)
            for ax, (col, title, logy) in zip(axes, [("rho_k", r"Subspace interference $\rho_k$", False),
                                                    ("eta", r"Relative update $\eta$", True),
                                                    ("R_math_mean", r"Mean $R(z)$, math probes", True),
                                                    ("R_non_math_mean", r"Mean $R(z)$, non-math probes", True)]):
                for i, c in enumerate(cs):
                    sub = ddf[ddf.condition == c].set_index("layer").reindex(layers)
                    ax.bar(np.arange(len(layers)) + (i - (len(cs) - 1) / 2) * w, sub[col], width=w * 0.92,
                           color=plotting.color(c), label=plotting.LABELS[c])
                ax.set_xticks(range(len(layers)), [f"layer {l}" for l in layers])
                ax.set_title(title)
                if logy:
                    ax.set_yscale("log")
            axes[0].legend(fontsize=7.5)
            plotting.save(fig, figs / "diagnostics")
    else:
        out["notes"]["diagnostics"] = "Status: NOT RUN"
    qp = res / "qualitative" / "responses_grouped.json"
    out["qualitative"] = {"responses_grouped": str(qp.relative_to(rel("."))) if qp.exists() else None,
                          "annotation_subset_csv": "results/task1/qualitative/annotation_subset.csv"}
    save_json(out, res / "task1_results.json")
    log(f"saved {res / 'task1_results.json'}; notes: {out['notes']}")
    if len(bench):
        print(bench[["condition"] + [c for c in bench if c.endswith("_primary") or c.endswith("_delta_pp")]].to_string())


if __name__ == "__main__":
    main()
