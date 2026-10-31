"""Turn the fixed manifest / pools into training examples and evaluation question items."""
from __future__ import annotations

from task2_vlm.data_prep import counting_question, presence_question


def caption_train_examples(man: dict) -> list[dict]:
    return [{"kind": "caption", "target": c, "image": (e["split"], e["image_id"])}
            for e in man["captions_train"] for c in e["captions"]]


def caption_dev_examples(man: dict) -> list[dict]:
    return [{"kind": "caption", "target": e["caption"], "image": (e["split"], e["image_id"]), "id": e["ann_id"]}
            for e in man["captions_dev"]]


def presence_pair(entry: list, cats: dict) -> list[dict]:
    cat, pos, neg = entry
    q = presence_question(cats[str(cat)])
    return [{"kind": "presence", "question": q, "target": "yes", "image": ("train2014", pos)},
            {"kind": "presence", "question": q, "target": "no", "image": ("train2014", neg)}]


def counting_pair(entry: list, cats: dict) -> list[dict]:
    cat, a, ka, b, kb = entry
    q = counting_question(cats[str(cat)])
    return [{"kind": "counting", "question": q, "target": str(ka), "image": ("train2014", a)},
            {"kind": "counting", "question": q, "target": str(kb), "image": ("train2014", b)}]


def vqa_pair(p: dict) -> list[dict]:
    return [{"kind": "vqa", "question": p["question"], "target": it["target"], "image": (it["split"], it["image_id"])}
            for it in p["items"]]


def eval_items(man: dict, suite: str) -> list[dict]:
    """One item per question with its correct, partner (complementary) and fixed mismatched image."""
    items = []
    is_vqa = suite.startswith("vqa")
    for p in man[suite]:
        for j, it in enumerate(p["items"]):
            other = p["items"][1 - j]
            items.append({
                "suite": suite, "pair_id": p["pair_id"], "index": j,
                "kind": "vqa" if is_vqa else p["kind"],
                "qtype": p["answer_type"] if is_vqa else p["kind"],
                "question": p["question"],
                "target": it["target"] if is_vqa else it["answer"],
                "canonical": it["canonical"], "partner_canonical": other["canonical"],
                "refs": it["answers"] if is_vqa else [it["answer"]],
                "image": (it["split"], it["image_id"]), "partner_image": (other["split"], other["image_id"]),
                "mismatch_image": (it["split"], man["mismatch"][suite][f"{p['pair_id']}:{j}"]),
            })
    return items
