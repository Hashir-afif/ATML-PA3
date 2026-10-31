"""Build the fixed activation-probe set (32 GSM8K test questions + 32 HellaSwag validation contexts).

    python -m task1_finetune.probe_set --config configs/task1.yaml

Sampled with ``numpy.random.default_rng(6304)`` before any model result is inspected; IDs and the exact
prompt strings in the benchmark formats are saved once and reused for every model:
  * GSM8K: harness ``gsm8k_cot`` zero-shot text ("Q: ...\\nA:") inside the non-thinking chat template with the
    generation prompt (the format used for GSM8K evaluation).
  * HellaSwag: the harness context string (no chat template), before any candidate continuation.
"""
from __future__ import annotations

import argparse

import numpy as np

from common.utils import load_config, log, rel, save_fixed_file
from task1_finetune.data import load_tokenizer, prompt_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    path = rel(cfg["data"]["probe_set_file"])
    if path.exists():
        log(f"{path} exists (fixed file); not regenerating")
        return
    from lm_eval.tasks import TaskManager, get_task_dict

    tasks = get_task_dict(["gsm8k_cot", "hellaswag"], TaskManager())
    gsm, hs = tasks["gsm8k_cot"], tasks["hellaswag"]
    gsm_docs, hs_docs = list(gsm.test_docs()), list(hs.validation_docs())
    assert len(gsm_docs) == 1319 and len(hs_docs) == 10042, (len(gsm_docs), len(hs_docs))
    rng = np.random.default_rng(cfg["seed"])
    dc = cfg["diagnostics"]
    gi = sorted(rng.choice(len(gsm_docs), dc["n_probe_gsm8k"], replace=False).tolist())
    hi = sorted(rng.choice(len(hs_docs), dc["n_probe_hellaswag"], replace=False).tolist())
    tok = load_tokenizer(cfg["model_id"])
    probe = {
        "seed": cfg["seed"], "sampling": "numpy default_rng(seed).choice(n, k, replace=False), GSM8K then HellaSwag",
        "gsm8k": [{"doc_id": i, "question": gsm_docs[i]["question"],
                   "prompt": prompt_text(tok, gsm.doc_to_text(gsm_docs[i])), "group": "math"} for i in gi],
        "hellaswag": [{"doc_id": i, "prompt": hs.doc_to_text(hs_docs[i]), "group": "non_math"} for i in hi],
    }
    save_fixed_file(probe, path)
    log(f"saved probe set to {path}: GSM8K {gi[:5]}..., HellaSwag {hi[:5]}...")


if __name__ == "__main__":
    main()
