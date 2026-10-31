"""Task 3 Step 4: parallel sampling vs a single longer response vs sequential refinement (live generation).

    python -m task3_ttc.live --config configs/task3.yaml [--limit N]

Model: Qwen/Qwen2.5-3B-Instruct at the revision pinned in the supplied README, FP16, no quantization, batch size 1,
temperature 0.7 / top-p 0.8 / top-k 20 / repetition penalty 1.0, native chat template with the system message
"You are a helpful assistant." and the problem followed by the supplied step-by-step instruction. The 50 problems
are the first 50 MATH held-out questions in the fixed permutation; all strategies share a 3,072-token output
allowance per question:
  single      one response capped at 3,072 new tokens
  parallel    three independent responses capped at 1,024 each; self-consistency over their extracted answers
  sequential  parallel response #1 reused as the initial answer, then one critique-and-revision response (cap
              2,048) after the fixed review instruction; the revised answer is returned
Each call is independently seeded (SHA-256 of "6304|sample_id|strategy|k"), timed end-to-end on the GPU (model
loading excluded) and saved immediately with its prompt, response, seed, input/output token counts and truncation
flag, so an interrupted run resumes. The shared initial call is charged to both parallel and sequential costs.
Gold answers are used only for grading after generation.
"""
from __future__ import annotations

import argparse
import hashlib
import time

import numpy as np
import torch

from common.utils import (append_jsonl, environment_info, load_config, load_json, load_jsonl, log, peak_memory, rel,
                          reset_peak_memory, save_json)
from task3_ttc.answers import NO_ANSWER, extract_boxed, group_math_question


def call_seed(sample_id: str, strategy: str, k: int, base: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{base}|{sample_id}|{strategy}|{k}".encode()).digest()[:8], "big") % (2**63)


class Generator:
    def __init__(self, cfg: dict):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        lc = cfg["live"]
        self.lc = lc
        self.tok = AutoTokenizer.from_pretrained(lc["model_id"], revision=lc["revision"])
        t0 = time.perf_counter()
        self.model = AutoModelForCausalLM.from_pretrained(lc["model_id"], revision=lc["revision"], torch_dtype=torch.float16,
                                                          attn_implementation="sdpa").cuda().eval()
        self.load_seconds = time.perf_counter() - t0
        self.eos = list({self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.eos_token_id})

    @torch.no_grad()
    def __call__(self, messages: list[dict], max_new: int, seed: int) -> dict:
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = self.tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new, do_sample=True,
                                  temperature=self.lc["temperature"], top_p=self.lc["top_p"], top_k=self.lc["top_k"],
                                  repetition_penalty=self.lc["repetition_penalty"], eos_token_id=self.eos,
                                  pad_token_id=self.tok.pad_token_id or self.eos[0])
        torch.cuda.synchronize()
        sec = time.perf_counter() - t0
        new = out[0, ids.shape[1]:].tolist()
        stopped = len(new) > 0 and new[-1] in self.eos
        n_out = len(new) - (1 if stopped else 0)  # retained tokens exclude the stop token
        text = self.tok.decode(new[:n_out], skip_special_tokens=True)
        return {"prompt": prompt, "response": text, "input_tokens": int(ids.shape[1]), "output_tokens": n_out,
                "generated_including_stop": len(new), "truncated": not stopped and len(new) >= max_new,
                "seconds": sec, "extracted": extract_boxed(text), "seed": seed, "max_new_tokens": max_new}


def grade_set(answers: list, gold: str) -> dict:
    return group_math_question((answers, gold, 300))


def summarize(cfg, calls: list[dict], problems: dict) -> dict:
    by = {}
    for c in calls:
        by.setdefault(c["sample_id"], {})[(c["strategy"], c["k"])] = c
    per_q, agg = [], {s: {"correct": [], "in": [], "out": [], "sec": [], "trunc": [], "extract": []}
                      for s in ("single", "parallel", "sequential")}
    trans = {"wrong_to_correct": 0, "correct_to_wrong": 0, "correct_to_correct": 0, "wrong_to_wrong": 0}
    for qid, cs in by.items():
        gold = problems[qid]["gold"]
        need = [("single", 0)] + [("parallel", k) for k in range(cfg["live"]["parallel_n"])] + [("revision", 0)]
        if not all(n in cs for n in need):
            continue
        single, par, rev = cs[("single", 0)], [cs[("parallel", k)] for k in range(cfg["live"]["parallel_n"])], cs[("revision", 0)]
        g_single = grade_set([single["extracted"]], gold)
        single_ok = g_single["groups"][0] >= 0 and g_single["group_correct"][g_single["groups"][0]]
        gp = grade_set([p["extracted"] for p in par], gold)
        from task3_ttc.offline import vote
        sc_g, _ = vote(np.array(gp["groups"]), np.ones(len(par)))
        par_ok = sc_g >= 0 and gp["group_correct"][sc_g]
        init_ok = gp["groups"][0] >= 0 and gp["group_correct"][gp["groups"][0]]
        g_rev = grade_set([rev["extracted"]], gold)
        rev_ok = g_rev["groups"][0] >= 0 and g_rev["group_correct"][g_rev["groups"][0]]
        trans[f"{'correct' if init_ok else 'wrong'}_to_{'correct' if rev_ok else 'wrong'}"] += 1
        rows = {"single": ([single], single_ok), "parallel": (par, par_ok), "sequential": ([par[0], rev], rev_ok)}
        rec = {"sample_id": qid, "gold": gold}
        for s, (calls_s, ok) in rows.items():
            a = agg[s]
            a["correct"].append(ok)
            a["in"].append(sum(c["input_tokens"] for c in calls_s))
            a["out"].append(sum(c["output_tokens"] for c in calls_s))
            a["sec"].append(sum(c["seconds"] for c in calls_s))
            a["trunc"].append(np.mean([c["truncated"] for c in calls_s]))
            final = calls_s[-1] if s != "parallel" else None
            a["extract"].append(final["extracted"] is not None if final else np.mean([c["extracted"] is not None for c in calls_s]))
            rec[s] = {"correct": bool(ok), "answer": (NO_ANSWER if sc_g < 0 else gp["group_repr"][sc_g]) if s == "parallel"
                      else (calls_s[-1]["extracted"] or NO_ANSWER), "input_tokens": a["in"][-1],
                      "output_tokens": a["out"][-1], "seconds": a["sec"][-1]}
        rec["initial_correct"], rec["revised_correct"] = bool(init_ok), bool(rev_ok)
        per_q.append(rec)
    res = {"n_questions": len(per_q), "transitions_initial_to_revised": trans, "per_question": per_q, "strategies": {}}
    for s, a in agg.items():
        if not a["correct"]:
            continue
        res["strategies"][s] = {"accuracy": float(np.mean(a["correct"])), "mean_input_tokens": float(np.mean(a["in"])),
                                "mean_output_tokens": float(np.mean(a["out"])), "mean_seconds": float(np.mean(a["sec"])),
                                "truncation_rate": float(np.mean(a["trunc"])), "extraction_rate": float(np.mean(a["extract"]))}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3.yaml")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N questions only")
    args = ap.parse_args()
    cfg = load_config(args.config)
    lc = cfg["live"]
    splits = load_json(cfg["data"]["splits_file"])
    qids = splits["math"]["live"][:args.limit] if args.limit else splits["math"]["live"]
    import pandas as pd

    g = pd.read_parquet(rel(cfg["data"]["raw_dir"]) / "qwen25-math" / "generations.parquet",
                        columns=["sample_id", "rollout_id", "question_text", "gold_answer"])
    g = g[g.rollout_id == 0].set_index("sample_id")
    problems = {q: {"question": g.loc[q, "question_text"], "gold": g.loc[q, "gold_answer"]} for q in qids}
    out_dir = rel(cfg["paths"]["results"]) / "live"
    calls_path = out_dir / ("calls.jsonl" if not args.limit else f"calls_limit{args.limit}.jsonl")
    done = {(c["sample_id"], c["strategy"], c["k"]) for c in load_jsonl(calls_path)} if calls_path.exists() else set()
    gen = Generator(cfg)
    reset_peak_memory()
    t_all = time.perf_counter()
    for qi, qid in enumerate(qids):
        user = problems[qid]["question"] + lc["suffix"]
        base = [{"role": "system", "content": lc["system"]}, {"role": "user", "content": user}]
        plan = [("single", 0, lc["single_max_new"])] + [("parallel", k, lc["parallel_max_new"]) for k in range(lc["parallel_n"])]
        for strategy, k, cap in plan:
            if (qid, strategy, k) in done:
                continue
            r = gen(base, cap, call_seed(qid, strategy, k, cfg["seed"]))
            append_jsonl({"sample_id": qid, "strategy": strategy, "k": k, **r}, calls_path)
        if (qid, "revision", 0) not in done:
            init = next(c for c in load_jsonl(calls_path) if c["sample_id"] == qid and c["strategy"] == "parallel" and c["k"] == 0)
            msgs = base + [{"role": "assistant", "content": init["response"]},
                           {"role": "user", "content": lc["review_instruction"]}]
            r = gen(msgs, lc["revision_max_new"], call_seed(qid, "revision", 0, cfg["seed"]))
            append_jsonl({"sample_id": qid, "strategy": "revision", "k": 0, **r}, calls_path)
        log(f"live {qi + 1}/{len(qids)} done ({time.perf_counter() - t_all:.0f}s)")
    res = summarize(cfg, load_jsonl(calls_path), problems)
    res.update(model_id=lc["model_id"], revision=lc["revision"], settings=lc, memory=peak_memory(),
               model_load_seconds=gen.load_seconds, environment=environment_info(), question_ids=qids)
    save_json(res, out_dir / ("live_results.json" if not args.limit else f"live_results_limit{args.limit}.json"))
    log(f"live summary: {res['strategies']} transitions {res['transitions_initial_to_revised']}")


if __name__ == "__main__":
    main()
