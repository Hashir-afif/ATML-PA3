"""Task 3 Steps 1-3: repeated sampling, quality scoring / aggregation, and what the signals capture.

    python -m task3_ttc.offline --config configs/task3.yaml

For each dataset, question, fixed order (10) and budget N in {1,2,4,8,16,32} the candidates are the first N
rollouts of that order. Methods (selection uses only the revealed candidates; gold answers are used only to
grade):
  first             FirstAns (consumes one candidate)
  sc                self-consistency, one vote per rollout
  best_logp         highest mean log-probability L_i
  best_selfcert     highest self-certainty K_i
  rwc_selfcert      rank-weighted consensus with K_i
  shortest          shortest rollout (length-only control, Step 3)
  MathVista only: best_vision / rwc_vision (V_i), best_visconf / rwc_visconf (Z_i = V_i C_i) and the tokenwise
  fusion best_visconf_token / rwc_visconf_token (Z_i^token = mean_t c_it v_it), with c_it calibrated on the
  current candidate set's pooled token KL values (5th / 95th linear-interpolation percentiles).
Ties: highest score or vote -> earliest supporting rollout in the current order; RWC ranks use average ranks.
NO_ANSWER is a normal group in voting and costs and is always graded incorrect. At N = 1 every rule returns the
single candidate. Outcomes are averaged over orders within a question, then over questions. Costs are the
summed retained token counts of the consumed candidates. 95% CIs: 1,000 paired question-level bootstrap
resamples (after averaging over orders).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

from common.utils import load_config, load_json, log, rel, save_json
from task3_ttc.answers import NO_ANSWER
from task3_ttc.data import load_cache

BASE_METHODS = ["first", "sc", "best_logp", "best_selfcert", "rwc_selfcert", "shortest"]
VLM_METHODS = ["best_vision", "rwc_vision", "best_visconf", "rwc_visconf", "best_visconf_token", "rwc_visconf_token"]
REQUIRED = {"math": ["first", "sc", "best_logp", "best_selfcert", "rwc_selfcert"],
            "mathvista": ["first", "sc", "best_logp", "best_selfcert", "rwc_selfcert", "best_vision", "rwc_vision",
                          "best_visconf", "rwc_visconf"]}


def vote(groups: np.ndarray, weights: np.ndarray) -> tuple[int, int]:
    """Weighted vote over groups (in current order). Returns (group, position of earliest supporting rollout)."""
    W, first = {}, {}
    for pos, (g, w) in enumerate(zip(groups.tolist(), weights.tolist())):
        W[g] = W.get(g, 0.0) + w
        first.setdefault(g, pos)
    best = max(W.values())
    winners = [g for g in W if abs(W[g] - best) <= 1e-9]
    g = min(winners, key=lambda x: first[x])
    return g, first[g]


def rwc_weights(scores: np.ndarray) -> np.ndarray:
    n = len(scores)
    if n == 1:
        return np.ones(1)
    r = rankdata(-scores, method="average")  # rank 1 = best
    return (n - r) / (n - 1)


def rwc(groups: np.ndarray, scores: np.ndarray) -> tuple[int, int]:
    return vote(groups, rwc_weights(scores))


def rwc_margin(groups: np.ndarray, scores: np.ndarray) -> float:
    w = rwc_weights(scores)
    W = {}
    for g, x in zip(groups.tolist(), w.tolist()):
        W[g] = W.get(g, 0.0) + x
    vals = sorted(W.values(), reverse=True)
    tot = sum(vals)
    return 0.0 if tot <= 0 else (vals[0] - (vals[1] if len(vals) > 1 else 0.0)) / tot


def visconf(q: dict, cand: np.ndarray, qlo: float = 0.05, qhi: float = 0.95) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """C_i, Z_i = V_i C_i and Z_i^token = mean_t c_it v_it, calibrated on the current candidate set only."""
    pooled = np.concatenate([q["kl_tok"][i] for i in cand])
    lo, hi = np.percentile(pooled, [100 * qlo, 100 * qhi])  # linear interpolation (NumPy default)
    den = max(hi - lo, 1e-8)
    C, Zt = np.empty(len(cand)), np.empty(len(cand))
    for j, i in enumerate(cand):
        c = np.clip((q["kl_tok"][i] - lo) / den, 0.0, 1.0)
        C[j] = c.mean()
        Zt[j] = (c * q["v_tok"][i]).mean()
    return C, q["V"][cand] * C, Zt


def select_all(q: dict, cand: np.ndarray, methods: list[str]) -> dict[str, tuple[int, int]]:
    """method -> (selected group, selected rollout id)."""
    g = q["groups"][cand]
    out = {}
    pick = lambda pos: (int(g[pos]), int(cand[pos]))  # noqa: E731
    for m in methods:
        if m == "first":
            out[m] = pick(0)
        elif m == "sc":
            grp, pos = vote(g, np.ones(len(cand)))
            out[m] = (int(grp), int(cand[pos]))
        elif m == "shortest":
            out[m] = pick(int(np.argmin(q["tokens"][cand])))
    scores = {"logp": q["L"][cand], "selfcert": q["K"][cand]}
    if "V" in q and any(m.endswith(("vision", "visconf", "visconf_token")) for m in methods):
        _, Z, Zt = visconf(q, cand)
        scores.update(vision=q["V"][cand], visconf=Z, visconf_token=Zt)
    for m in methods:
        if m.startswith("best_"):
            out[m] = pick(int(np.argmax(scores[m[5:]])))  # argmax returns the earliest maximum
        elif m.startswith("rwc_"):
            grp, pos = rwc(g, scores[m[4:]])
            out[m] = (int(grp), int(cand[pos]))
    return out


def is_correct(q: dict, grp: int) -> bool:
    return grp >= 0 and bool(q["group_correct"][grp])


def run_dataset(cfg: dict, ds: str, splits: dict, orders: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache = load_cache(cfg, ds)
    methods = BASE_METHODS + (VLM_METHODS if ds == "mathvista" else [])
    budgets = cfg["offline"]["budgets"]
    split_of = {q: "dev" for q in splits[ds]["dev"]} | {q: "eval" for q in splits[ds]["eval"]}
    sel_rows, cov_rows = [], []
    for qi, qid in enumerate(cache["qids"]):
        q = cache["questions"][qid]
        for o, perm in enumerate(orders[ds][qid]):
            perm = np.array(perm)
            for N in budgets:
                cand = perm[:N]
                cov_rows.append({"sample_id": qid, "split": split_of[qid], "order": o, "N": N,
                                 "oracle": bool(q["correct"][cand].any()),
                                 "distinct_groups": len(set(q["groups"][cand].tolist())),
                                 "distinct_groups_excl_na": len(set(q["groups"][cand].tolist()) - {-1}),
                                 "frac_correct": float(q["correct"][cand].mean())})
                cost_all = int(q["tokens"][cand].sum())
                for m, (grp, rid) in select_all(q, cand, methods).items():
                    sel_rows.append({"sample_id": qid, "split": split_of[qid], "order": o, "N": N, "method": m,
                                     "selected_group": grp, "selected_rollout": rid,
                                     "selected_answer": NO_ANSWER if grp < 0 else q["group_repr"][grp],
                                     "correct": is_correct(q, grp),
                                     "cost": int(q["tokens"][cand[0]]) if m == "first" else cost_all})
        if qi % 200 == 0:
            log(f"{ds}: {qi}/{len(cache['qids'])} questions")
    return pd.DataFrame(sel_rows), pd.DataFrame(cov_rows)


def bootstrap_summary(sel: pd.DataFrame, cov: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Question-level accuracies (mean over orders); CIs for accuracy and paired differences vs sc / first."""
    acc = sel.groupby(["method", "N", "sample_id"])["correct"].mean().unstack("sample_id")
    cost = sel.groupby(["method", "N"])["cost"].mean()
    qids = acc.columns
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(qids), size=(n_boot, len(qids)))
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]  # noqa: E731
    oracle = cov.groupby(["N", "sample_id"])["oracle"].mean().unstack("sample_id")[qids]
    out = {}
    for (m, N), row in acc.iterrows():
        v = row.to_numpy()
        entry = {"accuracy": float(v.mean()), "accuracy_ci95": ci(v[idx].mean(1)), "mean_cost": float(cost[(m, N)]),
                 "oracle_coverage": float(oracle.loc[N].mean())}
        for ref in ("sc", "first"):
            if m != ref and (ref, N) in acc.index:
                d = v - acc.loc[(ref, N)].to_numpy()
                entry[f"minus_{ref}"] = {"estimate": float(d.mean()), "ci95": ci(d[idx].mean(1))}
        out.setdefault(m, {})[int(N)] = entry
    return out


def failure_decomposition(sel: pd.DataFrame, cov: pd.DataFrame) -> dict:
    df = sel.merge(cov[["sample_id", "order", "N", "oracle"]], on=["sample_id", "order", "N"])
    df["sampling_failure"] = ~df["correct"] & ~df["oracle"]
    df["selection_failure"] = ~df["correct"] & df["oracle"]
    g = df.groupby(["method", "N", "sample_id"])[["sampling_failure", "selection_failure", "correct"]].mean()
    g = g.groupby(["method", "N"]).mean()
    return {m: {int(N): r.to_dict() for (mm, N), r in g.iterrows() if mm == m} for m in g.index.get_level_values(0).unique()}


def signal_analysis(cfg: dict, ds: str, qids: list[str]) -> dict:
    """N = 32 within-question associations between length, scores and correctness (+ confidence-vision for VLM)."""
    cache = load_cache(cfg, ds)
    names = ["L", "K"] + (["V", "Z", "Zt"] if ds == "mathvista" else [])
    res = {s: {"spearman_length": [], "spearman_correct": [], "top_minus_bottom_quartile": [],
               "top_minus_bottom_quartile_mixed": []} for s in names + ["neg_length"]}
    cv_rollout, cov_identity = [], 0.0
    for qid in qids:
        q = cache["questions"][qid]
        cand = np.arange(32)
        sc = {"L": q["L"], "K": q["K"], "neg_length": -q["tokens"].astype(float)}
        if ds == "mathvista":
            C, Z, Zt = visconf(q, cand)
            sc.update(V=q["V"], Z=Z, Zt=Zt)
            pooled = np.concatenate(q["kl_tok"])
            lo, hi = np.percentile(pooled, [5, 95])
            for i in cand:
                c = np.clip((q["kl_tok"][i] - lo) / max(hi - lo, 1e-8), 0, 1)
                v = q["v_tok"][i]
                if len(c) > 2 and c.std() > 0 and v.std() > 0:
                    cv_rollout.append(pearsonr(c, v)[0])
                cov = np.mean(c * v) - np.mean(c) * np.mean(v)
                cov_identity = max(cov_identity, abs((Zt[i] - Z[i]) - cov))
        corr = q["correct"].astype(float)
        mixed = 0 < corr.mean() < 1
        for s, x in sc.items():
            if np.std(x) > 0:
                if s != "neg_length":
                    res[s]["spearman_length"].append(spearmanr(x, q["tokens"])[0])
                if mixed:
                    res[s]["spearman_correct"].append(spearmanr(x, corr)[0])
            order = np.lexsort((cand, -x))  # descending score, ties by rollout id
            d = corr[order[:8]].mean() - corr[order[-8:]].mean()
            res[s]["top_minus_bottom_quartile"].append(d)
            if mixed:
                res[s]["top_minus_bottom_quartile_mixed"].append(d)
    summ = {s: {k: {"mean": float(np.nanmean(v)) if v else None, "n_questions": len(v)} for k, v in d.items()}
            for s, d in res.items()}
    out = {"signals": summ, "notes": "neg_length = shorter is better (length-only control)"}
    if ds == "mathvista":
        a = np.array(cv_rollout)
        out["confidence_vision_within_rollout_pearson"] = {
            "mean": float(np.nanmean(a)), "median": float(np.nanmedian(a)), "frac_positive": float(np.mean(a > 0)),
            "n_rollouts": int(len(a))}
        out["tokenwise_minus_trajectory_equals_cov_max_abs_error"] = float(cov_identity)
    return out


def illustrative(cfg: dict, splits: dict) -> list[dict]:
    """Fixed rule: first n MathVista eval questions (split order) with both correct and incorrect rollouts;
    for each, the highest-Z and lowest-Z rollout at N = 32, with their token-level c_t and v_t series."""
    cache = load_cache(cfg, "mathvista")
    out = []
    for qid in splits["mathvista"]["eval"]:
        q = cache["questions"][qid]
        if not (0 < q["correct"].mean() < 1):
            continue
        cand = np.arange(32)
        _, Z, _ = visconf(q, cand)
        pooled = np.concatenate(q["kl_tok"])
        lo, hi = np.percentile(pooled, [5, 95])
        for tag, i in (("highest_Z", int(np.argmax(Z))), ("lowest_Z", int(np.argmin(Z)))):
            c = np.clip((q["kl_tok"][i] - lo) / max(hi - lo, 1e-8), 0, 1)
            out.append({"sample_id": qid, "rollout_id": i, "which": tag, "correct": bool(q["correct"][i]),
                        "Z": float(Z[i]), "answer": NO_ANSWER if q["groups"][i] < 0 else q["group_repr"][q["groups"][i]],
                        "gold": q["gold"], "c_t": c.tolist(), "v_t": q["v_tok"][i].tolist()})
        if len(out) >= 2 * cfg["offline"]["n_illustrative_questions"]:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    splits, orders = load_json(cfg["data"]["splits_file"]), load_json(cfg["data"]["orders_file"])
    res_dir = rel(cfg["paths"]["results"]) / "offline"
    res_dir.mkdir(parents=True, exist_ok=True)
    out = {"budgets": cfg["offline"]["budgets"], "required_methods": REQUIRED, "datasets": {}}
    for ds in cfg["data"]["datasets"]:
        sp = res_dir / f"{ds}_selections.parquet"
        cp = res_dir / f"{ds}_coverage.parquet"
        if sp.exists() and cp.exists():
            sel, cov = pd.read_parquet(sp), pd.read_parquet(cp)
        else:
            sel, cov = run_dataset(cfg, ds, splits, orders)
            sel.to_parquet(sp)
            cov.to_parquet(cp)
        d = {}
        for split in ("eval", "dev"):
            s, c = sel[sel.split == split], cov[cov.split == split]
            d[split] = {
                "methods": bootstrap_summary(s, c, cfg["offline"]["n_bootstrap"], cfg["seed"]),
                "coverage": {int(N): {"oracle": float(g.groupby("sample_id")["oracle"].mean().mean()),
                                      "distinct_groups": float(g["distinct_groups"].mean()),
                                      "distinct_groups_excl_na": float(g["distinct_groups_excl_na"].mean())}
                             for N, g in c.groupby("N")},
                "failure_decomposition": failure_decomposition(s, c),
                "signals_N32": signal_analysis(cfg, ds, splits[ds][split]),
            }
        out["datasets"][ds] = d
        m = d["eval"]["methods"]
        log(f"{ds} eval @N=32: " + ", ".join(f"{k} {v[32]['accuracy']:.3f}" for k, v in m.items()) +
            f" | oracle {d['eval']['coverage'][32]['oracle']:.3f}")
    out["illustrative_trajectories"] = illustrative(cfg, splits)
    save_json(out, res_dir / "offline_results.json")
    log(f"saved {res_dir / 'offline_results.json'}")


if __name__ == "__main__":
    main()
