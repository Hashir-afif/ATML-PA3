"""Task 3 Step 6: figures and tables from the saved offline, adaptive and live results.

    python -m task3_ttc.summarize --config configs/task3.yaml

Example-selection rule for qualitative cases (fixed, outcome-independent ordering): within the held-out split
order, the first question where self-consistency and RWC disagree at N = 32, the first live question whose
revision changed correctness, and the first incorrect early stop of each selected adaptive controller.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import plotting
from common.utils import load_config, load_json, log, rel, save_json

LABEL = {"first": "FirstAns", "sc": "Self-consistency", "best_logp": "Best mean log-prob", "best_selfcert": "Best self-certainty",
         "rwc_selfcert": "RWC (self-certainty)", "shortest": "Shortest", "best_vision": "Best vision", "rwc_vision": "RWC (vision)",
         "best_visconf": "Best VisConf", "rwc_visconf": "RWC (VisConf)", "best_visconf_token": "Best tokenwise VisConf",
         "rwc_visconf_token": "RWC (tokenwise VisConf)"}
ORDER = list(LABEL)


# Color encodes the quality signal (fixed per signal in every panel); panels separate the aggregation rule.
SIGNAL_COLOR = {"sc": plotting.PALETTE[0], "logp": plotting.PALETTE[3], "selfcert": plotting.PALETTE[1],
                "vision": plotting.PALETTE[2], "visconf": plotting.PALETTE[6], "visconf_token": plotting.PALETTE[7],
                "shortest": plotting.PALETTE[4]}


def _signal(m: str) -> str:
    return m.split("_", 1)[1] if m.startswith(("best_", "rwc_")) else m


def accuracy_curves(off: dict, ds: str, figs) -> None:
    d = off["datasets"][ds]["eval"]
    budgets = sorted(int(n) for n in d["coverage"])
    oracle = [d["coverage"][str(n)]["oracle"] for n in budgets]
    panels = {"Highest-score selection": [m for m in d["methods"] if m.startswith("best_") or m == "shortest"],
              "Consensus (SC and RWC)": ["sc"] + [m for m in d["methods"] if m.startswith("rwc_")]}
    fig, axes = plotting.plt.subplots(2, 2, figsize=(11, 7), sharey="row")
    for col, (title, ms) in enumerate(panels.items()):
        ax_n, ax_c = axes[0, col], axes[1, col]
        ax_n.plot(budgets, oracle, color="#2b2b2a", ls="--", marker="o", label="Oracle coverage")
        for ax, xs in ((ax_n, budgets), (ax_c, [d["methods"]["first"][str(n)]["mean_cost"] for n in budgets])):
            ax.plot(xs, [d["methods"]["first"][str(n)]["accuracy"] for n in budgets], color=plotting.NEUTRAL, ls=":",
                    marker="s", label="FirstAns")
        for m in sorted(ms, key=ORDER.index):
            acc = [d["methods"][m][str(n)]["accuracy"] for n in budgets]
            cost = [d["methods"][m][str(n)]["mean_cost"] for n in budgets]
            ls = "--" if m.endswith("_token") else "-"
            ax_n.plot(budgets, acc, marker="o", ls=ls, color=SIGNAL_COLOR[_signal(m)], label=LABEL[m])
            ax_c.plot(cost, acc, marker="o", ls=ls, color=SIGNAL_COLOR[_signal(m)], label=LABEL[m])
        ax_n.set_xscale("log", base=2)
        ax_n.set_xticks(budgets, budgets)
        ax_n.set_xlabel("Candidate budget N")
        ax_n.set_title(f"{ds}: {title}")
        ax_n.legend(fontsize=7)
        ax_c.set_xscale("log")
        ax_c.set_xlabel("Mean generated-token cost per question")
        ax_c.set_title(f"{ds}: {title} vs token cost")
    axes[0, 0].set_ylabel("Held-out accuracy / coverage")
    axes[1, 0].set_ylabel("Held-out accuracy")
    fig.tight_layout()
    plotting.save(fig, figs / f"{ds}_accuracy_curves")


def method_table(off: dict, ds: str) -> pd.DataFrame:
    d = off["datasets"][ds]["eval"]["methods"]
    rows = []
    for m in ORDER:
        if m not in d:
            continue
        for n, e in d[m].items():
            rows.append({"method": m, "N": int(n), "accuracy": e["accuracy"], "ci95": e["accuracy_ci95"],
                         "minus_sc": e.get("minus_sc", {}).get("estimate"), "minus_sc_ci95": e.get("minus_sc", {}).get("ci95"),
                         "mean_cost": e["mean_cost"], "oracle": e["oracle_coverage"]})
    return pd.DataFrame(rows)


def decomposition_table(off: dict, ds: str) -> pd.DataFrame:
    d = off["datasets"][ds]["eval"]["failure_decomposition"]
    return pd.DataFrame([{"method": m, "N": int(n), **v} for m, x in d.items() for n, v in x.items()])


def adaptive_plot(ad: dict, ds: str, figs) -> None:
    d = ad["datasets"][ds]["eval"]
    fig, ax = plotting.plt.subplots(figsize=(5.6, 3.6))
    ns = sorted(int(n) for n in d["fixed_sc"])
    ax.plot([d["fixed_sc"][str(n)]["mean_cost"] for n in ns], [d["fixed_sc"][str(n)]["accuracy"] for n in ns],
            marker="o", color=plotting.PALETTE[0], label="Fixed-N self-consistency")
    for n in ns:
        ax.annotate(f"N={n}", (d["fixed_sc"][str(n)]["mean_cost"], d["fixed_sc"][str(n)]["accuracy"]), fontsize=7,
                    textcoords="offset points", xytext=(3, -9))
    for i, fam in enumerate(("agreement", "quality")):
        st = d[fam]["setting"]
        ax.scatter(d[fam]["mean_cost"], d[fam]["accuracy"], s=60, marker="D", color=plotting.PALETTE[1 + i], zorder=3,
                   label=f"Adaptive {fam} (tau={st['tau']}, gamma={st['gamma']})")
    ax.set_xscale("log")
    ax.set_xlabel("Mean generated-token cost per question")
    ax.set_ylabel("Held-out accuracy")
    ax.set_title(f"{ds}: adaptive vs fixed budgets")
    ax.legend(fontsize=7)
    plotting.save(fig, figs / f"{ds}_adaptive")


def trajectories_plot(off: dict, figs) -> None:
    tr = off.get("illustrative_trajectories") or []
    if not tr:
        return
    fig, axes = plotting.plt.subplots(len(tr), 1, figsize=(8, 2.0 * len(tr)), sharex=False)
    for ax, t in zip(np.atleast_1d(axes), tr):
        c, v = pd.Series(t["c_t"]).rolling(25, min_periods=1).mean(), pd.Series(t["v_t"]).rolling(25, min_periods=1).mean()
        ax.plot(c, color=plotting.PALETTE[0], label="calibrated confidence c_t")
        ax.plot(v / max(v.max(), 1e-8), color=plotting.PALETTE[1], label="visual engagement v_t (scaled to max 1)")
        ax.set_title(f"MathVista {t['sample_id']} rollout {t['rollout_id']} ({t['which']}, "
                     f"{'correct' if t['correct'] else 'incorrect'}; answer {t['answer']}, gold {t['gold']})", fontsize=8)
        ax.set_ylabel("25-token mean")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    np.atleast_1d(axes)[-1].set_xlabel("Generated token index")
    plotting.save(fig, figs / "mathvista_trajectories")


def qualitative_cases(cfg, splits) -> dict:
    out = {}
    res = rel(cfg["paths"]["results"])
    for ds in cfg["data"]["datasets"]:
        p = res / "offline" / f"{ds}_selections.parquet"
        if not p.exists():
            continue
        s = pd.read_parquet(p)
        s = s[(s.N == 32) & (s.order == 0)]
        score = "rwc_visconf" if ds == "mathvista" else "rwc_selfcert"
        piv = s.pivot_table(index="sample_id", columns="method", values=["selected_answer", "correct"], aggfunc="first")
        for qid in splits[ds]["eval"]:
            if qid in piv.index and piv.loc[qid, ("selected_answer", "sc")] != piv.loc[qid, ("selected_answer", score)]:
                out[f"{ds}_selection_disagreement"] = {"sample_id": qid, **{
                    m: {"answer": piv.loc[qid, ("selected_answer", m)], "correct": bool(piv.loc[qid, ("correct", m)])}
                    for m in ("first", "sc", score)}}
                break
        for fam in ("agreement", "quality"):
            dp = res / "adaptive" / f"{ds}_{fam}_eval_decisions.parquet"
            if dp.exists():
                d = pd.read_parquet(dp)
                bad = d[(d.n < 32) & (~d.sc_correct)]
                if len(bad):
                    order = {q: i for i, q in enumerate(splits[ds]["eval"])}
                    r = bad.assign(k=bad.sample_id.map(order)).sort_values(["k", "order"]).iloc[0]
                    out[f"{ds}_{fam}_incorrect_early_stop"] = {k: (v.item() if hasattr(v, "item") else v)
                                                              for k, v in r.drop("k").to_dict().items()}
    lp = res / "live" / "live_results.json"
    if lp.exists():
        live = load_json(lp)
        for q in live["per_question"]:
            if q["initial_correct"] != q["revised_correct"]:
                out["live_revision_change"] = q
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = rel(cfg["paths"]["results"])
    figs = res / "figures"
    plotting.setup()
    splits = load_json(cfg["data"]["splits_file"])
    out = {"notes": {}}
    op = res / "offline" / "offline_results.json"
    if op.exists():
        off = load_json(op)
        out["offline"] = off
        for ds in off["datasets"]:
            accuracy_curves(off, ds, figs)
            t = method_table(off, ds)
            t.to_csv(res / f"{ds}_method_table.csv", index=False)
            (res / f"{ds}_method_table_N32.md").write_text(
                t[t.N == 32].drop(columns=["ci95", "minus_sc_ci95"]).to_markdown(index=False, floatfmt=".3f"), encoding="utf-8")
            decomposition_table(off, ds).to_csv(res / f"{ds}_failure_decomposition.csv", index=False)
            sig = off["datasets"][ds]["eval"]["signals_N32"]["signals"]
            pd.DataFrame([{"signal": s, **{k: v["mean"] for k, v in d.items()}} for s, d in sig.items()]).to_csv(
                res / f"{ds}_signal_associations.csv", index=False)
        trajectories_plot(off, figs)
    else:
        out["notes"]["offline"] = "Status: NOT RUN"
    ap_ = res / "adaptive" / "adaptive_results.json"
    if ap_.exists():
        ad = load_json(ap_)
        out["adaptive"] = ad
        for ds in ad["datasets"]:
            adaptive_plot(ad, ds, figs)
    else:
        out["notes"]["adaptive"] = "Status: NOT RUN"
    lp = res / "live" / "live_results.json"
    if lp.exists():
        live = load_json(lp)
        out["live"] = {k: v for k, v in live.items() if k != "per_question"}
        pd.DataFrame([{"strategy": s, **v} for s, v in live["strategies"].items()]).to_csv(res / "live_table.csv", index=False)
    else:
        out["notes"]["live"] = "Status: NOT RUN"
    out["qualitative_cases"] = qualitative_cases(cfg, splits)
    ds_ = res / "data_summary.json"
    out["data_summary"] = load_json(ds_) if ds_.exists() else None
    save_json(out, res / "task3_results.json")
    log(f"saved {res / 'task3_results.json'}; notes {out['notes']}")


if __name__ == "__main__":
    main()
