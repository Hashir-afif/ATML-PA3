"""Task 3 Step 5: adaptive sampling (offline simulation of stopping between completed rollouts).

    python -m task3_ttc.adaptive --config configs/task3.yaml

Candidates are revealed in the saved orders; stopping is checked after n in {4, 8, 16}, otherwise all 32 are
used. Both controllers return the current self-consistency answer.
  agreement(tau):           stop if a unique, non-NO_ANSWER leading answer has >= tau * n votes.
  quality_aware(tau,gamma): the agreement condition AND the SC winner equals the RWC winner AND the RWC vote margin
                            (top minus runner-up weight, over total weight; runner-up 0 if only one group) >= gamma.
                            Quality score: self-certainty (MATH) or VisConf (MathVista), with VisConf calibration
                            and ranks recomputed from the revealed candidates only.
Setting selection on each dataset's development split only: among tau in {0.5,...,1.0} (x gamma in
{0,0.1,0.25,0.5}) plus the always-32 fallback, choose the lowest mean token cost whose development accuracy is
no more than 1 pp below fixed-32 self-consistency (mean over the ten orders); ties -> higher accuracy, then
smaller tau, then smaller gamma. The chosen settings are frozen and saved before held-out evaluation.
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

from common.utils import load_config, load_json, log, rel, save_json
from task3_ttc.data import load_cache
from task3_ttc.offline import rwc, rwc_margin, visconf, vote


def checkpoint_stats(cfg: dict, ds: str, orders: dict) -> pd.DataFrame:
    """Per (question, order, n): SC leader stats, RWC winner/margin, answer correctness and cost."""
    cache = load_cache(cfg, ds)
    rows = []
    for qid in cache["qids"]:
        q = cache["questions"][qid]
        for o, perm in enumerate(orders[ds][qid]):
            perm = np.array(perm)
            for n in cfg["adaptive"]["checkpoints"] + [32]:
                cand = perm[:n]
                g = q["groups"][cand]
                vals, counts = np.unique(g, return_counts=True)
                top = counts.max()
                unique_leader = (counts == top).sum() == 1
                leader = int(vals[np.argmax(counts)]) if unique_leader else None
                sc_g, _ = vote(g, np.ones(n))
                score = visconf(q, cand)[1] if ds == "mathvista" else q["K"][cand]
                rwc_g, _ = rwc(g, score)
                rows.append({"sample_id": qid, "order": o, "n": n, "leader_unique": bool(unique_leader),
                             "leader_non_na": leader is not None and leader >= 0, "leader_frac": top / n,
                             "sc_group": int(sc_g), "rwc_group": int(rwc_g), "rwc_margin": rwc_margin(g, score),
                             "sc_correct": bool(sc_g >= 0 and q["group_correct"][sc_g]),
                             "cost": int(q["tokens"][cand].sum()), "pool_frac_correct": float(q["correct"].mean())})
    return pd.DataFrame(rows)


def simulate(stats: pd.DataFrame, family: str, tau: float | None, gamma: float | None, checkpoints) -> pd.DataFrame:
    """Per (question, order): stopping n, returned SC answer correctness and cost."""
    s = stats.set_index(["sample_id", "order", "n"])
    final = s.xs(32, level="n")
    stop_n = pd.Series(32, index=final.index)
    if tau is not None:
        decided = pd.Series(False, index=final.index)
        for n in checkpoints:
            x = s.xs(n, level="n")
            ok = x["leader_unique"] & x["leader_non_na"] & (x["leader_frac"] >= tau - 1e-12)
            if family == "quality":
                ok &= (x["sc_group"] == x["rwc_group"]) & (x["rwc_margin"] >= gamma - 1e-12)
            new = ok & ~decided
            stop_n[new] = n
            decided |= ok
    picked = s.reset_index().merge(stop_n.rename("n").reset_index(), on=["sample_id", "order", "n"])
    picked["final32_correct"] = final.loc[list(zip(picked.sample_id, picked.order)), "sc_correct"].to_numpy()
    return picked


def policy_metrics(p: pd.DataFrame) -> dict:
    qacc = p.groupby("sample_id")["sc_correct"].mean()
    early = p[p.n < 32]
    wrong_early = early[~early.sc_correct]
    return {"accuracy": float(qacc.mean()), "mean_cost": float(p["cost"].mean()), "mean_rollouts": float(p["n"].mean()),
            "stop_distribution": {int(k): float(v) for k, v in p["n"].value_counts(normalize=True).sort_index().items()},
            "early_stop_rate": float((p.n < 32).mean()),
            "incorrect_early_stops": int(len(wrong_early)),
            "incorrect_early_stops_recoverable_by_fixed32": int(wrong_early["final32_correct"].sum()),
            "early_stops_correct_rate": float(early["sc_correct"].mean()) if len(early) else None,
            "n_question_orders": int(len(p))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ac = cfg["adaptive"]
    splits, orders = load_json(cfg["data"]["splits_file"]), load_json(cfg["data"]["orders_file"])
    res_dir = rel(cfg["paths"]["results"]) / "adaptive"
    res_dir.mkdir(parents=True, exist_ok=True)
    sel_path = res_dir / "selected_settings.json"
    out = {"config": ac, "datasets": {}}
    selected = load_json(sel_path) if sel_path.exists() else {}
    for ds in cfg["data"]["datasets"]:
        sp = res_dir / f"{ds}_checkpoint_stats.parquet"
        stats = pd.read_parquet(sp) if sp.exists() else checkpoint_stats(cfg, ds, orders)
        stats.to_parquet(sp)
        dev = stats[stats.sample_id.isin(splits[ds]["dev"])]
        ev = stats[stats.sample_id.isin(splits[ds]["eval"])]
        grid = {"agreement": [(t, None) for t in ac["tau"]],
                "quality": list(itertools.product(ac["tau"], ac["gamma"]))}
        dsres = {"dev_grid": {}, "eval": {}}
        if ds not in selected:  # selection uses development data only, then is frozen on disk
            base = policy_metrics(simulate(dev, "agreement", None, None, ac["checkpoints"]))
            selected[ds] = {"fixed32_dev_accuracy": base["accuracy"]}
            for fam, settings in grid.items():
                cands = [{"tau": None, "gamma": None, **base}]
                for tau, gamma in settings:
                    m = policy_metrics(simulate(dev, fam, tau, gamma, ac["checkpoints"]))
                    cands.append({"tau": tau, "gamma": gamma, **m})
                dsres["dev_grid"][fam] = cands
                feas = [c for c in cands if c["accuracy"] >= base["accuracy"] - ac["tolerance_pp"] / 100 - 1e-12]
                best = sorted(feas, key=lambda c: (c["mean_cost"], -c["accuracy"],
                                                   99 if c["tau"] is None else c["tau"],
                                                   99 if c["gamma"] is None else c["gamma"]))[0]
                selected[ds][fam] = {"tau": best["tau"], "gamma": best["gamma"], "dev_accuracy": best["accuracy"],
                                     "dev_mean_cost": best["mean_cost"]}
            save_json(selected, sel_path)
            save_json(dsres["dev_grid"], res_dir / f"{ds}_dev_grid.json")
        # held-out evaluation with frozen settings
        for fam in ("agreement", "quality"):
            st = selected[ds][fam]
            p = simulate(ev, fam, st["tau"], st["gamma"], ac["checkpoints"])
            p.to_parquet(res_dir / f"{ds}_{fam}_eval_decisions.parquet")
            dsres["eval"][fam] = {"setting": st, **policy_metrics(p)}
            agree = p.merge(ev[ev.n == 4][["sample_id", "order", "leader_frac"]].rename(columns={"leader_frac": "agree4"}),
                            on=["sample_id", "order"])
            dsres["eval"][fam]["stop_by_pool_correct_fraction"] = {
                str(b): float(g["n"].mean()) for b, g in agree.groupby(pd.cut(agree.pool_frac_correct,
                                                                              [-0.01, 0.25, 0.5, 0.75, 1.0]), observed=True)}
            dsres["eval"][fam]["mean_rollouts_by_agreement_at_4"] = {
                str(k): float(g["n"].mean()) for k, g in agree.groupby("agree4")}
        dsres["eval"]["fixed_sc"] = {}
        sel_off = pd.read_parquet(rel(cfg["paths"]["results"]) / "offline" / f"{ds}_selections.parquet")
        sc = sel_off[(sel_off.method == "sc") & (sel_off.split == "eval")]
        for N, g in sc.groupby("N"):
            dsres["eval"]["fixed_sc"][int(N)] = {"accuracy": float(g.groupby("sample_id")["correct"].mean().mean()),
                                                 "mean_cost": float(g["cost"].mean())}
        dsres["selected"] = selected[ds]
        out["datasets"][ds] = dsres
        log(f"{ds}: selected {selected[ds]} | eval " +
            ", ".join(f"{f}: acc {dsres['eval'][f]['accuracy']:.3f} cost {dsres['eval'][f]['mean_cost']:.0f} "
                      f"n {dsres['eval'][f]['mean_rollouts']:.1f}" for f in ("agreement", "quality")))
    save_json(out, res_dir / "adaptive_results.json")


if __name__ == "__main__":
    main()
