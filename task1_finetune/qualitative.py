"""Qualitative generations: 24 fixed prompts x 7 conditions = 168 greedy responses (512-token cap).

    python -m task1_finetune.qualitative --config configs/task1.yaml --condition baseline   # one condition
    python -m task1_finetune.qualitative --config configs/task1.yaml --export               # group + annotation sheet

Prompts come unchanged from ``first_plot_questions.yaml`` (8 questions in plain, JSON-system-prompt and template
formats). Thinking is disabled. The annotation subset is the first and last underlying question in the file's
plain-question order, in all three formats and all seven conditions (42 responses). The exported CSV holds
automatic helper columns (truncation, length, JSON validity, template-key coverage, repeated 4-gram rate,
math-marker counts) and blank columns for the manual annotation, which is the student's own judgement.
"""
from __future__ import annotations

import argparse
import json
import re
import time

import torch
import yaml

from common.utils import load_config, load_jsonl, log, peak_memory, rel, reset_peak_memory, save_json, save_jsonl
from task1_finetune.data import prompt_text
from task1_finetune.evaluate import condition_path


def load_prompts(path) -> list[dict]:
    with open(rel(path), "r", encoding="utf-8") as f:
        entries = yaml.safe_load(f)
    assert len(entries) == 24, len(entries)
    plain = [e["id"] for e in entries if not e["id"].endswith(("_json", "_template"))]
    out = []
    for e in entries:
        fmt = "json" if e["id"].endswith("_json") else "template" if e["id"].endswith("_template") else "plain"
        base = e["id"].removesuffix("_json").removesuffix("_template")
        assert len(e["paraphrases"]) == 1
        out.append({"id": e["id"], "base_question": base, "question_index": plain.index(base), "format": fmt,
                    "system": e.get("system"), "user": e["paraphrases"][0]})
    return out


@torch.no_grad()
def generate(cfg, cond) -> list[dict]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = condition_path(cfg, cond)
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16).cuda().eval()
    prompts = load_prompts(cfg["data"]["qualitative_file"])
    qc = cfg["qualitative"]
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    rows = []
    reset_peak_memory()
    t0 = time.perf_counter()
    for i in range(0, len(prompts), qc["batch_size"]):
        chunk = prompts[i:i + qc["batch_size"]]
        texts = [prompt_text(tok, p["user"], p["system"]) for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=qc["max_new_tokens"], do_sample=False, temperature=None,
                             top_p=None, top_k=None, eos_token_id=eos, pad_token_id=tok.pad_token_id)
        for p, text, g in zip(chunk, texts, gen[:, enc.input_ids.shape[1]:]):
            ids = g.tolist()
            ended = any(t in eos for t in ids)
            n_new = next((j for j, t in enumerate(ids) if t in eos), len(ids))
            rows.append({**p, "condition": cond, "model_path": path, "prompt_text": text,
                         "response": tok.decode(ids[:n_new], skip_special_tokens=True), "n_new_tokens": n_new,
                         "truncated": not ended})
    log(f"{cond}: {len(rows)} responses in {time.perf_counter() - t0:.1f}s, {peak_memory()}")
    return rows


def repeated_4gram_rate(text: str) -> float:
    w = text.split()
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 0.0 if not grams else 1 - len(set(grams)) / len(grams)


def auto_features(r: dict) -> dict:
    resp = r["response"]
    f = {"auto_truncated": r["truncated"], "auto_n_new_tokens": r["n_new_tokens"],
         "auto_repeated_4gram_rate": round(repeated_4gram_rate(resp), 4),
         "auto_math_markers": len(re.findall(r"\\boxed|\\frac|\$|=|\bstep \d|\btherefore\b", resp, flags=re.I)),
         "auto_json_valid": None, "auto_template_keys_present": None}
    if r["format"] == "json":
        s = resp.strip()
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
        try:
            json.loads(s)
            f["auto_json_valid"] = True
        except Exception:
            f["auto_json_valid"] = False
    if r["format"] == "template":
        keys = re.findall(r"^(\w+) = #", r["user"], flags=re.M)
        f["auto_template_keys_present"] = f"{sum(bool(re.search(rf'^\s*{k}\s*=', resp, flags=re.M)) for k in keys)}/{len(keys)}"
    return f


def export(cfg) -> None:
    import pandas as pd

    qdir = rel(cfg["paths"]["results"]) / "qualitative"
    conds = cfg["eval"]["conditions"]
    rows, missing = [], []
    for c in conds:
        p = qdir / f"{c}.jsonl"
        if p.exists():
            rows += load_jsonl(p)
        else:
            missing.append(c)
    grouped: dict = {}
    for r in rows:
        grouped.setdefault(r["base_question"], {}).setdefault(r["format"], {})[r["condition"]] = {
            "response": r["response"], "n_new_tokens": r["n_new_tokens"], "truncated": r["truncated"]}
    save_json({"conditions": conds, "missing_conditions": missing, "n_responses": len(rows), "by_question": grouped},
              qdir / "responses_grouped.json")
    n_q = max(r["question_index"] for r in rows) if rows else 7
    subset = [r for r in rows if r["question_index"] in (0, n_q)]
    order = {c: i for i, c in enumerate(conds)}
    fmt_order = {"plain": 0, "json": 1, "template": 2}
    subset.sort(key=lambda r: (r["question_index"], fmt_order[r["format"]], order[r["condition"]]))
    manual = ["format_compliance", "coherence_relevance", "repetition", "unnecessary_math_framing",
              "inappropriate_content", "notes"]
    table = [{"question_id": r["id"], "question_index": r["question_index"], "format": r["format"],
              "condition": r["condition"], **auto_features(r), **{m: "" for m in manual},
              "response": r["response"]} for r in subset]
    pd.DataFrame(table).to_csv(qdir / "annotation_subset.csv", index=False, encoding="utf-8")
    log(f"exported {len(rows)} responses ({len(subset)} in annotation subset); missing: {missing}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1.yaml")
    ap.add_argument("--condition")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.smoke:
        from task1_finetune.train import apply_smoke
        cfg = apply_smoke(cfg)
    if args.export:
        export(cfg)
        return
    out = rel(cfg["paths"]["results"]) / "qualitative" / f"{args.condition}.jsonl"
    if out.exists():
        log(f"{out} exists; skipping")
        return
    save_jsonl(generate(cfg, args.condition), out)


if __name__ == "__main__":
    main()
