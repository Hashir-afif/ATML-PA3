"""Task 1 training data: the first 40,000 eligible MetaMathQA examples in source order.

Eligibility (manual): non-empty question and response; formatted length <= 2,048 tokens; neither ``query``
nor ``original_question`` matches a GSM8K test question after text normalization. Retained examples are
never truncated or shuffled.

Formatting: the user turn is rendered with Qwen3's chat template and ``enable_thinking=False`` plus the
generation prompt (which ends with the empty ``<think>\\n\\n</think>\\n\\n`` block). The supplied response
and the end-of-turn token ``<|im_end|>`` follow. Prompt and response are tokenized separately so the loss
mask is exact: loss is computed only on response tokens and ``<|im_end|>``.

Run ``python -m task1_finetune.data --config configs/task1.yaml`` to build the fixed ID file and the
tokenized cache.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import unicodedata

import torch

from common.utils import load_config, load_json, log, rel, save_fixed_file


def normalize_text(s: str) -> str:
    """Normalization used for the GSM8K overlap filter: NFKC, lowercase, alphanumerics only, single spaces."""
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def end_of_turn_id(tok) -> int:
    i = tok.convert_tokens_to_ids("<|im_end|>")
    assert isinstance(i, int) and i != tok.unk_token_id, "tokenizer lacks <|im_end|>"
    return i


def prompt_text(tok, question: str, system: str | None = None) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": question}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def gsm8k_test_questions(cfg: dict) -> set[str]:
    from datasets import load_dataset

    ds = load_dataset(cfg["data"]["gsm8k_id"], cfg["data"]["gsm8k_config"], split="test")
    return {normalize_text(q) for q in ds["question"]}


def build_train_ids(cfg: dict) -> dict:
    """Scan MetaMathQA in source order and record the first N eligible indices (fixed file)."""
    from datasets import load_dataset

    dcfg = cfg["data"]
    tok = load_tokenizer(cfg["model_id"])
    eot = end_of_turn_id(tok)
    gsm = gsm8k_test_questions(cfg)
    ds = load_dataset(dcfg["dataset_id"], split=dcfg["split"])
    n_target, max_len = dcfg["n_examples"], dcfg["max_length"]
    kept, excluded = [], {"empty": 0, "too_long": 0, "gsm8k_overlap": 0}
    chunk = 2048
    for start in range(0, len(ds), chunk):
        rows = ds[start:start + chunk]
        cand = []
        for j, (q, r, oq) in enumerate(zip(rows["query"], rows["response"], rows["original_question"])):
            if not q or not q.strip() or not r or not r.strip():
                excluded["empty"] += 1
                continue
            if normalize_text(q) in gsm or (oq and normalize_text(oq) in gsm):
                excluded["gsm8k_overlap"] += 1
                continue
            cand.append((start + j, q, r))
        if cand:
            p_ids = tok([prompt_text(tok, q) for _, q, _ in cand], add_special_tokens=False)["input_ids"]
            r_ids = tok([r for _, _, r in cand], add_special_tokens=False)["input_ids"]
            for (idx, _, _), p, r in zip(cand, p_ids, r_ids):
                if len(p) + len(r) + 1 > max_len:
                    excluded["too_long"] += 1
                    continue
                kept.append(idx)
                if len(kept) == n_target:
                    break
        if len(kept) == n_target:
            last_scanned = kept[-1]
            break
        log(f"scanned {start + chunk} rows, kept {len(kept)}")
    else:
        raise RuntimeError(f"Only {len(kept)} eligible examples found")
    digest = hashlib.sha256(",".join(map(str, kept)).encode()).hexdigest()
    return {"dataset_id": dcfg["dataset_id"], "split": dcfg["split"], "n_examples": len(kept),
            "source_indices": kept, "last_source_index_scanned": last_scanned,
            "excluded_before_last_kept": excluded, "max_length": max_len, "normalization": "NFKC, lowercase, "
            "non-alphanumerics to spaces, collapsed whitespace", "sha256_of_indices": digest,
            "eot_token_id": eot, "note": "Source order, no shuffling or truncation."}


def build_tokenized(cfg: dict, ids: dict) -> dict:
    from datasets import load_dataset

    tok = load_tokenizer(cfg["model_id"])
    eot = end_of_turn_id(tok)
    ds = load_dataset(cfg["data"]["dataset_id"], split=cfg["data"]["split"])
    sub = ds.select(ids["source_indices"])
    p_ids = tok([prompt_text(tok, q) for q in sub["query"]], add_special_tokens=False)["input_ids"]
    r_ids = tok(list(sub["response"]), add_special_tokens=False)["input_ids"]
    input_ids, n_prompt = [], []
    for p, r in zip(p_ids, r_ids):
        seq = p + r + [eot]
        assert len(seq) <= cfg["data"]["max_length"]
        input_ids.append(torch.tensor(seq, dtype=torch.int32))
        n_prompt.append(len(p))
    return {"input_ids": input_ids, "n_prompt": torch.tensor(n_prompt, dtype=torch.int32),
            "sha256_of_indices": ids["sha256_of_indices"], "model_id": cfg["model_id"]}


def get_train_data(cfg: dict) -> dict:
    """Load (building if needed) the tokenized training set; always validated against the fixed ID file."""
    ids_path = rel(cfg["data"]["train_ids_file"])
    if not ids_path.exists():
        log("building fixed training-ID file")
        save_fixed_file(build_train_ids(cfg), ids_path)
    ids = load_json(ids_path)
    cache = rel(cfg["data"]["tokenized_cache"])
    if cache.exists():
        data = torch.load(cache, weights_only=False)
        if data.get("sha256_of_indices") == ids["sha256_of_indices"]:
            return data
        log("tokenized cache does not match the fixed ID file; rebuilding cache")
    data = build_tokenized(cfg, ids)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache)
    return data


def collate(data: dict, indices: list[int], pad_id: int) -> dict:
    """Right-padded microbatch; labels are -100 on prompt tokens and padding."""
    seqs = [data["input_ids"][i].long() for i in indices]
    T = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    labels = torch.full((len(seqs), T), -100, dtype=torch.long)
    attn = torch.zeros((len(seqs), T), dtype=torch.long)
    for b, (s, i) in enumerate(zip(seqs, indices)):
        n = len(s)
        input_ids[b, :n] = s
        attn[b, :n] = 1
        npr = int(data["n_prompt"][i])
        labels[b, npr:n] = s[npr:]
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


def n_response_tokens(data: dict, indices: list[int]) -> int:
    return int(sum(len(data["input_ids"][i]) - int(data["n_prompt"][i]) for i in indices))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    data = get_train_data(cfg)
    lens = [len(x) for x in data["input_ids"]]
    resp = [len(x) - int(n) for x, n in zip(data["input_ids"], data["n_prompt"])]
    ids = load_json(cfg["data"]["train_ids_file"])
    log(f"{len(lens)} examples | mean length {sum(lens) / len(lens):.1f} | max {max(lens)} | "
        f"total tokens {sum(lens)} | response tokens {sum(resp)} | exclusions {ids['excluded_before_last_kept']}")


if __name__ == "__main__":
    main()
