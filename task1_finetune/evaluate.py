"""Zero-shot lm-evaluation-harness evaluation of one Task 1 condition.

    python -m task1_finetune.evaluate --config configs/task1.yaml --condition baseline
    conditions: baseline | full_mid | full | l2sp | lora4 | lora16 | oplora16

GSM8K: ``gsm8k_cot``, num_fewshot=0, non-thinking chat template, greedy decoding (task default), stock stop
sequences and answer filters, 1,024-new-token cap. Primary metric: exact match after flexible extraction;
unextractable answers count as incorrect (handled by the harness filter). HellaSwag / BoolQ / ARC-Easy:
stock completion prompts without a chat template, candidate likelihood. All in FP16, dropout disabled
(eval mode). Per-example outputs, extracted GSM8K answers and truncation flags are saved.

Truncation flag: the harness strips stop sequences and does not expose the generated token count, so a
response is flagged as truncated when its re-tokenized length is within 4 tokens of the cap.
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

from common.utils import (environment_info, load_config, log, peak_memory, rel, reset_peak_memory, save_json,
                          save_jsonl)

TRUNC_SLACK = 4


def condition_path(cfg: dict, cond: str) -> str:
    ck = rel(cfg["paths"]["checkpoints"])
    total, mid = cfg["train"]["total_updates"], cfg["train"]["midpoint_update"]
    if cond == "baseline":
        return cfg["model_id"]
    if cond == "full_mid":
        return str(ck / "full" / f"update_{mid:04d}" / "model")
    if cond in ("full", "l2sp"):
        return str(ck / cond / f"update_{total:04d}" / "model")
    if cond in cfg["methods"]:
        return str(rel(cfg["paths"]["merged"]) / cond)
    raise ValueError(cond)


def _gsm8k_rows(samples: list[dict], tok, cap: int) -> list[dict]:
    by_doc: dict[int, dict] = defaultdict(dict)
    for s in samples:
        r = by_doc[s["doc_id"]]
        resp = s["resps"][0][0] if isinstance(s["resps"][0], list) else s["resps"][0]
        n_tok = len(tok(resp, add_special_tokens=False)["input_ids"])
        a = s.get("arguments")
        first = next(iter(a.values())) if isinstance(a, dict) and a else (a[0] if a else None)
        prompt = first.get("arg_0") if isinstance(first, dict) else (first[0] if first else None)
        r.update(doc_id=s["doc_id"], question=s["doc"]["question"], target=s["target"], prompt=prompt, response=resp,
                 n_response_tokens=n_tok, truncated=n_tok >= cap - TRUNC_SLACK)
        filt = s["filter"].replace("-", "_")
        r[f"extracted_{filt}"] = s["filtered_resps"][0] if isinstance(s["filtered_resps"], list) else s["filtered_resps"]
        r[f"exact_match_{filt}"] = s.get("exact_match")
    return [by_doc[k] for k in sorted(by_doc)]


def _likelihood_rows(samples: list[dict]) -> list[dict]:
    rows = []
    for s in samples:
        rows.append({"doc_id": s["doc_id"], "target": s["target"],
                     "loglikelihoods": [float(r[0][0]) if isinstance(r[0], (list, tuple)) else float(r[0])
                                        for r in s["resps"]],
                     **{k: s[k] for k in ("acc", "acc_norm") if k in s}})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1.yaml")
    ap.add_argument("--condition", required=True)
    ap.add_argument("--limit", type=int, default=None, help="smoke test only: examples per task")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.smoke:
        from task1_finetune.train import apply_smoke
        cfg = apply_smoke(cfg)
    ecfg = cfg["eval"]
    out = rel(cfg["paths"]["results"]) / "eval" / args.condition
    if (out / "aggregate.json").exists() and args.limit is None:
        log(f"{args.condition}: aggregate.json exists; skipping")
        return
    import lm_eval
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoTokenizer

    path = condition_path(cfg, args.condition)
    assert path == cfg["model_id"] or Path(path).exists(), f"missing model for {args.condition}: {path}"
    tok = AutoTokenizer.from_pretrained(path)
    seeds = dict(random_seed=cfg["seed"], numpy_random_seed=cfg["seed"], torch_random_seed=cfg["seed"],
                 fewshot_random_seed=cfg["seed"])
    agg = {"condition": args.condition, "model_path": path, "seed": cfg["seed"], "limit": args.limit,
           "lm_eval_version": lm_eval.__version__, "environment": environment_info(), "settings": ecfg,
           "tasks": {}}
    reset_peak_memory()

    g = ecfg["gsm8k"]
    lm = HFLM(pretrained=path, dtype=ecfg["dtype"], batch_size=g["batch_size"], device="cuda",
              enable_thinking=False)
    lm.model.eval()
    t0 = time.perf_counter()
    res = simple_evaluate(model=lm, tasks=[g["task"]], num_fewshot=g["num_fewshot"],
                          apply_chat_template=g["apply_chat_template"],
                          gen_kwargs={"max_gen_toks": g["max_gen_toks"]}, log_samples=True, limit=args.limit,
                          bootstrap_iters=1000, **seeds)
    gsm_seconds = time.perf_counter() - t0
    rows = _gsm8k_rows(res["samples"][g["task"]], tok, g["max_gen_toks"])
    save_jsonl(rows, out / f"samples_{g['task']}.jsonl")
    agg["tasks"][g["task"]] = {"metrics": res["results"][g["task"]], "n": len(rows), "seconds": gsm_seconds,
                               "truncation_rate": sum(r["truncated"] for r in rows) / max(1, len(rows)),
                               "flexible_unextractable_rate": sum(r.get("extracted_flexible_extract") == "[invalid]"
                                                                  for r in rows) / max(1, len(rows)),
                               "strict_unextractable_rate": sum(r.get("extracted_strict_match") == "[invalid]"
                                                                for r in rows) / max(1, len(rows)),
                               "mean_response_tokens": sum(r["n_response_tokens"] for r in rows) / max(1, len(rows)),
                               "config": res["configs"][g["task"]]}

    lcfg = ecfg["likelihood"]
    lm.batch_size_per_gpu = lcfg["batch_size"]
    t0 = time.perf_counter()
    res = simple_evaluate(model=lm, tasks=lcfg["tasks"], num_fewshot=lcfg["num_fewshot"],
                          apply_chat_template=lcfg["apply_chat_template"], log_samples=True, limit=args.limit,
                          bootstrap_iters=1000, **seeds)
    lik_seconds = time.perf_counter() - t0
    for task in lcfg["tasks"]:
        rows = _likelihood_rows(res["samples"][task])
        save_jsonl(rows, out / f"samples_{task}.jsonl")
        agg["tasks"][task] = {"metrics": res["results"][task], "n": len(rows), "config": res["configs"][task]}
    agg["likelihood_seconds_total"] = lik_seconds
    agg["memory"] = peak_memory()
    save_json(agg, out / ("aggregate.json" if args.limit is None else f"aggregate_limit{args.limit}.json"))
    log(f"{args.condition}: " + ", ".join(f"{t}: {v['metrics']}" for t, v in agg["tasks"].items())[:2000])


if __name__ == "__main__":
    main()
