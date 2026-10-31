"""Load and join the supplied rollouts, verify alignment, fix splits/orders and build the answer/score cache.

    python -m task3_ttc.data --config configs/task3.yaml

* Join generations / probability_metrics (/ attention_metrics) on (dataset, sample_id, rollout_id); verify that
  every per-token list has length ``n_generated_text_tokens``.
* Splits (fixed file): per dataset, sort the 1,000 question IDs lexicographically, permute with
  ``numpy.random.default_rng(6304)``; first 200 = development, remaining 800 = held-out evaluation; the first 50
  MATH evaluation questions in this permutation form the live-generation subset.
* Orders (fixed file): per dataset, ``default_rng([6304, 1])``; for each question in sorted-ID order, ten
  permutations of the 32 rollout IDs. Budget-N candidates are nested prefixes of each order.
* Per rollout: answer group (NO_ANSWER = -1), group correctness, retained token count T_i, mean log-probability
  L_i, self-certainty K_i (mean token KL(U || p)), and for MathVista the visual engagement V_i =
  mean_t A_img / (A_img + A_prompt + A_gen) plus the token-level KL and engagement arrays needed for VisConf.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle

import numpy as np
import pandas as pd

from common.utils import load_config, load_json, log, rel, save_fixed_file, save_json
from task3_ttc.answers import NO_ANSWER, group_math_exact, group_math_question, group_mathvista_question

KEYS = ["dataset", "sample_id", "rollout_id"]


def load_joined(cfg: dict, ds: str) -> pd.DataFrame:
    d = rel(cfg["data"]["raw_dir"]) / cfg["data"]["datasets"][ds]["folder"]
    g = pd.read_parquet(d / "generations.parquet")
    p = pd.read_parquet(d / "probability_metrics.parquet")
    df = g.merge(p, on=KEYS, how="inner", validate="one_to_one")
    if cfg["data"]["datasets"][ds]["kind"] == "vlm":
        a = pd.read_parquet(d / "attention_metrics.parquet").rename(
            columns={"n_generated_text_tokens": "attn_prev_counts"})
        df = df.merge(a, on=KEYS, how="inner", validate="one_to_one")
    assert len(df) == len(g) == 32000, (len(df), len(g))
    return df.sort_values(["sample_id", "rollout_id"]).reset_index(drop=True)


def alignment_report(df: pd.DataFrame, vlm: bool) -> dict:
    n = df["n_generated_text_tokens"].to_numpy()
    rep = {"rows": len(df), "token_ids_len_mismatch": int((df["generated_token_ids"].map(len) != n).sum()),
           "logp_len_mismatch": int((df["logp"].map(len) != n).sum()),
           "kl_len_mismatch": int((df["kl_u_p"].map(len) != n).sum()),
           "zero_length_rollouts": int((n == 0).sum()), "capped_rollouts": int(df["hit_max_new_tokens"].sum()),
           "null_extractions": int(df["extracted_answer"].isna().sum()), "retained_tokens_total": int(n.sum()),
           "rollouts_per_question": sorted(set(df.groupby("sample_id").size().tolist()))}
    if vlm:
        for c in ("img_attn_total", "prompt_text_attn_total", "generated_text_attn_total", "attn_prev_counts"):
            rep[f"{c}_len_mismatch"] = int((df[c].map(len) != n).sum())
    return rep


def make_splits(cfg: dict) -> dict:
    out = {"seed": cfg["seed"], "rule": "sorted IDs permuted by numpy default_rng(seed); first n_dev = dev"}
    for ds in cfg["data"]["datasets"]:
        g = pd.read_parquet(rel(cfg["data"]["raw_dir"]) / cfg["data"]["datasets"][ds]["folder"] / "generations.parquet",
                            columns=["sample_id"])
        ids = sorted(g["sample_id"].unique())
        perm = [ids[i] for i in np.random.default_rng(cfg["seed"]).permutation(len(ids))]
        out[ds] = {"permuted": perm, "dev": perm[:cfg["data"]["n_dev"]], "eval": perm[cfg["data"]["n_dev"]:]}
    out["math"]["live"] = out["math"]["eval"][:cfg["live"]["n_questions"]] if "live" in cfg else None
    return out


def make_orders(cfg: dict, splits: dict) -> dict:
    out = {"seed": [cfg["seed"], 1], "rule": "per dataset default_rng([seed, 1]); sorted question IDs; 10 perms each"}
    for ds in cfg["data"]["datasets"]:
        rng = np.random.default_rng([cfg["seed"], 1])
        out[ds] = {q: [rng.permutation(cfg["data"]["n_rollouts"]).tolist() for _ in range(cfg["data"]["n_orders"])]
                   for q in sorted(splits[ds]["permuted"])}
    return out


def run_math_grouping(jobs: dict, timeout: float, workers: int = 6) -> dict:
    results, pending = {}, list(jobs)
    while pending:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(workers)
        asyncs = [(q, pool.apply_async(group_math_question, (jobs[q],))) for q in pending]
        pending = []
        for i, (q, a) in enumerate(asyncs):
            try:
                results[q] = a.get(timeout=timeout)
            except mp.TimeoutError:
                log(f"math-verify timeout on {q}; exact-string fallback")
                results[q] = group_math_exact(jobs[q])
                pending = [qq for qq, _ in asyncs[i + 1:]]
                break
            except Exception as e:  # never silently drop a question
                log(f"grouping error on {q}: {e}; exact-string fallback")
                results[q] = group_math_exact(jobs[q])
        pool.terminate()
        pool.join()
    return results


def _answers(sub: pd.DataFrame) -> list:
    return [x if isinstance(x, str) else None for x in sub["extracted_answer"]]


def build_cache(cfg: dict, ds: str) -> dict:
    vlm = cfg["data"]["datasets"][ds]["kind"] == "vlm"
    df = load_joined(cfg, ds)
    rep = alignment_report(df, vlm)
    log(f"{ds}: {rep}")
    meta = load_json(cfg["data"]["mathvista_metadata"]) if ds == "mathvista" else None
    qids = sorted(df["sample_id"].unique())
    by_q = {q: sub for q, sub in df.groupby("sample_id")}
    if ds == "math":
        jobs = {q: (_answers(by_q[q]),
                    by_q[q]["gold_answer"].iloc[0], cfg["offline"]["math_verify_max_len"]) for q in qids}
        grouped = run_math_grouping(jobs, cfg["offline"]["grouping_timeout_s"])
    else:
        grouped = {q: group_mathvista_question(_answers(by_q[q]),
                                               by_q[q]["gold_answer"].iloc[0], meta[q]) for q in qids}
    cache = {"dataset": ds, "qids": qids, "alignment": rep, "questions": {}}
    rows = []
    for q in qids:
        sub, gr = by_q[q], grouped[q]
        assert sub["rollout_id"].tolist() == list(range(32))
        groups = np.array(gr["groups"])
        gc = np.array(gr["group_correct"], dtype=bool)
        correct = np.array([g >= 0 and gc[g] for g in groups], dtype=bool)
        entry = {"groups": groups, "group_correct": gc, "group_repr": gr["group_repr"], "correct": correct,
                 "tokens": sub["n_generated_text_tokens"].to_numpy(), "capped": sub["hit_max_new_tokens"].to_numpy(),
                 "L": np.array([np.mean(x) if len(x) else np.nan for x in sub["logp"]]),
                 "K": np.array([np.mean(x) if len(x) else np.nan for x in sub["kl_u_p"]]),
                 "gold": sub["gold_answer"].iloc[0], "fallback": gr["fallback"]}
        if vlm:
            v_tok = []
            for im, pr, ge in zip(sub["img_attn_total"], sub["prompt_text_attn_total"], sub["generated_text_attn_total"]):
                im, den = np.asarray(im, float), np.asarray(im, float) + np.asarray(pr, float) + np.asarray(ge, float)
                v_tok.append(np.divide(im, den, out=np.zeros_like(im), where=den > 0))
            entry["v_tok"] = v_tok
            entry["kl_tok"] = [np.asarray(x, float) for x in sub["kl_u_p"]]
            entry["V"] = np.array([x.mean() if len(x) else np.nan for x in v_tok])
        cache["questions"][q] = entry
        for r, (ex, g) in enumerate(zip(sub["extracted_answer"], groups)):
            rows.append({"sample_id": q, "rollout_id": r, "extracted_answer": ex if isinstance(ex, str) else None,
                         "group_id": int(g), "group_repr": NO_ANSWER if g < 0 else gr["group_repr"][g],
                         "correct": bool(correct[r]), "tokens": int(entry["tokens"][r])})
    out = rel(cfg["data"]["answer_cache"])
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(rel(cfg["paths"]["results"]) / f"answer_groups_{ds}.csv", index=False)
    with open(out / f"{ds}.pkl", "wb") as f:
        pickle.dump(cache, f)
    return cache


def load_cache(cfg: dict, ds: str) -> dict:
    p = rel(cfg["data"]["answer_cache"]) / f"{ds}.pkl"
    if not p.exists():
        return build_cache(cfg, ds)
    with open(p, "rb") as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    sp, op = rel(cfg["data"]["splits_file"]), rel(cfg["data"]["orders_file"])
    if not sp.exists():
        save_fixed_file(make_splits(cfg), sp)
    if not op.exists():
        save_fixed_file(make_orders(cfg, load_json(sp)), op)
    import math_verify

    summary = {"math_verify_version": getattr(math_verify, "__version__", None)}
    for ds in cfg["data"]["datasets"]:
        c = load_cache(cfg, ds)
        qs = c["questions"].values()
        summary[ds] = {"alignment": c["alignment"], "n_questions": len(c["qids"]),
                       "rollout_accuracy": float(np.mean([q["correct"].mean() for q in qs])),
                       "mean_groups_per_question": float(np.mean([len(q["group_repr"]) for q in qs])),
                       "questions_with_grouping_fallback": int(sum(q["fallback"] for q in qs))}
    save_json(summary, rel(cfg["paths"]["results"]) / "data_summary.json")
    log(f"task3 data ready: {summary}")


if __name__ == "__main__":
    main()
