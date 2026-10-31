"""Post-hoc leakage / isolation checks across all tasks (run once all data files exist).

    python -m scripts.leakage_check

Task 1: exact overlap between normalized MetaMathQA training queries / original questions and GSM8K test
questions, plus a looser prefix check (first 60 normalized characters). Task 2: image overlap between training
images (including grounding pools) and every evaluation suite, and between evaluation suites. Task 3: dev / eval
question disjointness and live-subset membership in the held-out split. Results: results/leakage_check.json.
"""
from __future__ import annotations

from itertools import combinations

from common.utils import load_config, load_json, log, rel, save_json


def task1(cfg) -> dict:
    from datasets import load_dataset

    from task1_finetune.data import normalize_text

    ids = load_json(cfg["data"]["train_ids_file"])["source_indices"]
    ds = load_dataset(cfg["data"]["dataset_id"], split=cfg["data"]["split"]).select(ids)
    gsm = [normalize_text(q) for q in load_dataset(cfg["data"]["gsm8k_id"], cfg["data"]["gsm8k_config"], split="test")["question"]]
    gset, gpre = set(gsm), {q[:60] for q in gsm}
    exact = prefix = 0
    for q, oq in zip(ds["query"], ds["original_question"]):
        nq, no = normalize_text(q), normalize_text(oq or "")
        exact += nq in gset or no in gset
        prefix += nq[:60] in gpre or no[:60] in gpre
    probe = load_json(cfg["data"]["probe_set_file"]) if rel(cfg["data"]["probe_set_file"]).exists() else None
    return {"n_train": len(ids), "exact_overlap_with_gsm8k_test": exact, "prefix60_overlap_with_gsm8k_test": prefix,
            "probe_set_present": probe is not None,
            "note": "GSM8K test is evaluation-only; HellaSwag/BoolQ/ARC are not used in training."}


def task2(cfg) -> dict:
    man, pools = load_json(cfg["data"]["manifest"]), load_json(cfg["data"]["pools"])
    train = {("train2014", i) for i in man["images"]["train2014"]}
    train |= {(it["split"], it["image_id"]) for p in man["vqa_train"] for it in p["items"]}
    train |= {("train2014", e["image_id"]) for e in man["captions_train"]}
    train |= {("train2014", x) for e in pools["presence"] for x in e[1:]}
    train |= {("train2014", e[1]) for e in pools["counting"]} | {("train2014", e[3]) for e in pools["counting"]}
    suites = {k: {(it["split"], it["image_id"]) for p in man[k] for it in p["items"]}
              for k in ("vqa_dev", "vqa_test", "grounding_dev", "grounding_test")}
    suites["captions_dev"] = {(e["split"], e["image_id"]) for e in man["captions_dev"]}
    for k in man["mismatch"]:
        suites[k] |= {("val2014", v) for v in man["mismatch"][k].values()}
    return {"train_vs_eval_overlap": {k: len(train & v) for k, v in suites.items()},
            "eval_pairwise_overlap": {f"{a}|{b}": len(suites[a] & suites[b]) for a, b in combinations(suites, 2)},
            "vqa_question_id_overlap_train_eval": len({it["question_id"] for p in man["vqa_train"] for it in p["items"]} &
                                                      {it["question_id"] for k in ("vqa_dev", "vqa_test") for p in man[k]
                                                       for it in p["items"]})}


def task3(cfg) -> dict:
    sp = load_json(cfg["data"]["splits_file"])
    out = {}
    for ds in cfg["data"]["datasets"]:
        out[ds] = {"dev_eval_overlap": len(set(sp[ds]["dev"]) & set(sp[ds]["eval"])), "n_dev": len(sp[ds]["dev"]),
                   "n_eval": len(sp[ds]["eval"])}
    out["math"]["live_subset_in_eval"] = set(sp["math"]["live"]) <= set(sp["math"]["eval"])
    return out


def main():
    res = {}
    for name, fn, cfgp in (("task1", task1, "configs/task1.yaml"), ("task2", task2, "configs/task2.yaml"),
                           ("task3", task3, "configs/task3.yaml")):
        try:
            res[name] = fn(load_config(cfgp))
        except FileNotFoundError as e:
            res[name] = {"status": "NOT RUN", "reason": str(e)}
    save_json(res, rel("results/leakage_check.json"))
    log(f"leakage check: {res}")


if __name__ == "__main__":
    main()
