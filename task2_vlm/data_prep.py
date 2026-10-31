"""Task 2 data selection: COCO captions, annotation-derived grounding pairs, VQAv2 complementary pairs.

    python -m task2_vlm.data_prep --config configs/task2.yaml            # select IDs (fixed files) + download images
    python -m task2_vlm.data_prep --config configs/task2.yaml --inspect  # label-inspection contact sheets

Raw inputs (data/task2/raw, downloaded by run_overnight.sh): COCO 2014 caption + instance annotations, VQAv2
questions/annotations and complementary pairs for train2014 and val2014. Only the selected images are fetched
from images.cocodataset.org.

Selection (seed 6304; one independent NumPy generator per component, ``default_rng([6304, k])``):
  * Training (train2014): 6,000 caption images x 3 captions (first three by annotation id); 640/256/640
    yes-no/number/other VQAv2 complementary pairs; reusable presence / counting pair pools.
  * Dev and final test (val2014): VQAv2 pairs 12/8/12 and 36/24/36; grounding 24 and 64 pairs (half presence,
    half counting); 96 dev caption images with one caption each. Every evaluation image is used once, so all
    evaluation suites are image-disjoint from one another; train2014 vs val2014 keeps them disjoint from
    training.
  * VQAv2 pairs: identical question text, same answer type, different images, normalized consensus answers
    that differ, and >= 8/10 annotator agreement on each image. Training target = an original answer string
    of the consensus group; all ten answers are kept for scoring.
  * Grounding labels from instance annotations (bounding-box areas): presence positive = category annotated
    with a non-crowd instance covering >= 1% of the image; negative = no annotation of that category at all.
    Counting = 1-4 non-crowd instances, no crowd annotation of the category, every instance >= 0.5% of the
    image. Pairs share the question and differ in answer.
  * Mismatched-image mapping, fixed before any generation: for each evaluation question, an image from
    another pair of the same suite (never either image of its own pair).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

from common.utils import load_config, load_json, log, rel, save_fixed_file, save_json
from task2_vlm.vqa_norm import consensus, consensus_target, normalize_answer

IRREGULAR_PLURALS = {"person": "people", "mouse": "mice", "sheep": "sheep", "skis": "skis", "scissors": "scissors",
                     "knife": "knives", "broccoli": "broccoli", "skateboard": "skateboards"}


def plural(name: str) -> str:
    words = name.split()
    last = words[-1]
    if last in IRREGULAR_PLURALS:
        p = IRREGULAR_PLURALS[last]
    elif last.endswith(("s", "x", "ch", "sh")):
        p = last + "es"
    else:
        p = last + "s"
    return " ".join(words[:-1] + [p])


def article(name: str) -> str:
    return ("an " if name[0] in "aeiou" else "a ") + name


def presence_question(name: str) -> str:
    return f"Does this image contain {article(name)}?"


def counting_question(name: str) -> str:
    return f"How many {plural(name)} can you see?"


def coco_filename(split: str, image_id: int) -> str:
    return f"COCO_{split}_{image_id:012d}.jpg"


def image_path(cfg: dict, split: str, image_id: int) -> Path:
    return rel(cfg["data"]["image_dir"]) / split / coco_filename(split, image_id)


# ----------------------------------------------------------------------------------------------- loading raw
def load_raw(cfg: dict, split: str) -> dict:
    raw = rel(cfg["data"]["raw_dir"])
    tag = "train2014" if split == "train2014" else "val2014"
    with open(raw / "annotations" / f"captions_{tag}.json", encoding="utf-8") as f:
        caps = json.load(f)
    with open(raw / "annotations" / f"instances_{tag}.json", encoding="utf-8") as f:
        inst = json.load(f)
    with open(raw / f"v2_OpenEnded_mscoco_{tag}_questions.json", encoding="utf-8") as f:
        qs = {q["question_id"]: q for q in json.load(f)["questions"]}
    with open(raw / f"v2_mscoco_{tag}_annotations.json", encoding="utf-8") as f:
        anns = {a["question_id"]: a for a in json.load(f)["annotations"]}
    with open(raw / f"v2_mscoco_{tag}_complementary_pairs.json", encoding="utf-8") as f:
        pairs = json.load(f)
    captions = defaultdict(list)
    for c in sorted(caps["annotations"], key=lambda c: c["id"]):
        captions[c["image_id"]].append((c["id"], c["caption"].strip()))
    sizes = {im["id"]: (im["width"], im["height"]) for im in inst["images"]}
    objects = defaultdict(lambda: defaultdict(list))  # image -> category -> [(bbox_area_frac, iscrowd)]
    for a in inst["annotations"]:
        w, h = sizes[a["image_id"]]
        objects[a["image_id"]][a["category_id"]].append((a["bbox"][2] * a["bbox"][3] / (w * h), a["iscrowd"]))
    cats = {c["id"]: c["name"] for c in inst["categories"]}
    return {"split": split, "captions": captions, "objects": objects, "sizes": sizes, "cats": cats,
            "questions": qs, "annotations": anns, "pairs": pairs}


# ---------------------------------------------------------------------------------------------------- labels
def presence_label(objs: dict, cat: int, min_area: float) -> str | None:
    inst = objs.get(cat, [])
    if not inst:
        return "no"
    if any(c == 0 and a >= min_area for a, c in inst):
        return "yes"
    return None  # annotated but only crowd / tiny instances: ambiguous, excluded


def count_label(objs: dict, cat: int, lo: int, hi: int, min_area: float) -> int | None:
    inst = objs.get(cat, [])
    if not inst or any(c == 1 for _, c in inst) or any(a < min_area for a, _ in inst):
        return None
    return len(inst) if lo <= len(inst) <= hi else None


# ------------------------------------------------------------------------------------------------- VQA pairs
def vqa_candidates(raw: dict, min_agree: int) -> dict[str, list[dict]]:
    out = defaultdict(list)
    qs, anns = raw["questions"], raw["annotations"]
    for q1, q2 in sorted(tuple(p) for p in raw["pairs"]):
        if q1 not in qs or q2 not in qs or q1 not in anns or q2 not in anns:
            continue
        a, b = qs[q1], qs[q2]
        if a["question"].strip().lower() != b["question"].strip().lower() or a["image_id"] == b["image_id"]:
            continue
        aa, ab = anns[q1], anns[q2]
        if aa["answer_type"] != ab["answer_type"]:
            continue
        ra = [x["answer"] for x in aa["answers"]]
        rb = [x["answer"] for x in ab["answers"]]
        ca, ka = consensus(ra)
        cb, kb = consensus(rb)
        if ka < min_agree or kb < min_agree or ca == cb or not ca or not cb:
            continue
        items = []
        for qid, q, refs, c, k in ((q1, a, ra, ca, ka), (q2, b, rb, cb, kb)):
            items.append({"question_id": qid, "image_id": q["image_id"], "split": raw["split"], "answers": refs,
                          "canonical": c, "target": consensus_target(refs), "agreement": k})
        out[aa["answer_type"]].append({"answer_type": aa["answer_type"], "question": a["question"].strip(),
                                       "items": items})
    return out


def select_vqa(cands: dict, counts: dict, rng, used_images: set | None, prefix: str) -> list[dict]:
    chosen = []
    for atype in ("yes/no", "number", "other"):
        pool = cands[atype]
        n = 0
        for i in rng.permutation(len(pool)):
            p = pool[i]
            imgs = {it["image_id"] for it in p["items"]}
            if used_images is not None and imgs & used_images:
                continue
            chosen.append(dict(p, pair_id=f"{prefix}-vqa-{len(chosen):04d}"))
            if used_images is not None:
                used_images |= imgs
            n += 1
            if n == counts[atype]:
                break
        if n < counts[atype]:
            raise RuntimeError(f"only {n} {atype} pairs available for {prefix}")
    return chosen


# ------------------------------------------------------------------------------------------- grounding pairs
def eligibility(raw: dict, images: list[int], cfg: dict) -> tuple[dict, dict]:
    d = cfg["data"]
    lo, hi = d["count_range"]
    pos, neg = defaultdict(list), defaultdict(list)
    counts = defaultdict(lambda: defaultdict(list))
    for img in images:
        objs = raw["objects"].get(img, {})
        for cat in raw["cats"]:
            lab = presence_label(objs, cat, d["presence_min_area_frac"])
            if lab == "yes":
                pos[cat].append(img)
            elif lab == "no":
                neg[cat].append(img)
            c = count_label(objs, cat, lo, hi, d["count_min_area_frac"])
            if c is not None:
                counts[cat][c].append(img)
    return {"pos": pos, "neg": neg}, counts


def sample_presence(rng, pres, cats, used: set | None):
    eligible = sorted(c for c in cats if pres["pos"][c] and pres["neg"][c])
    for _ in range(10000):
        cat = eligible[rng.integers(len(eligible))]
        p = pres["pos"][cat][rng.integers(len(pres["pos"][cat]))]
        n = pres["neg"][cat][rng.integers(len(pres["neg"][cat]))]
        if used is None or (p not in used and n not in used and p != n):
            return cat, p, n
    raise RuntimeError("could not sample a presence pair")


def sample_counting(rng, counts, used: set | None):
    eligible = sorted(c for c in counts if sum(1 for v in counts[c].values() if v) >= 2)
    for _ in range(10000):
        cat = eligible[rng.integers(len(eligible))]
        vals = sorted(k for k, v in counts[cat].items() if v)
        ka, kb = rng.choice(vals, 2, replace=False)
        a = counts[cat][int(ka)][rng.integers(len(counts[cat][int(ka)]))]
        b = counts[cat][int(kb)][rng.integers(len(counts[cat][int(kb)]))]
        if used is None or (a not in used and b not in used and a != b):
            return cat, int(a), int(ka), int(b), int(kb)
    raise RuntimeError("could not sample a counting pair")


def grounding_eval_pairs(raw, rng, n_pres, n_count, used: set, cfg, prefix) -> list[dict]:
    images = sorted(img for img in raw["sizes"] if img not in used)
    pres, counts = eligibility(raw, images, cfg)
    out = []
    for _ in range(n_pres):
        cat, p, n = sample_presence(rng, pres, raw["cats"], used)
        used |= {p, n}
        name = raw["cats"][cat]
        out.append({"pair_id": f"{prefix}-grd-{len(out):04d}", "kind": "presence", "category_id": cat,
                    "category": name, "question": presence_question(name),
                    "items": [{"image_id": p, "split": raw["split"], "answer": "yes"},
                              {"image_id": n, "split": raw["split"], "answer": "no"}]})
    for _ in range(n_count):
        cat, a, ka, b, kb = sample_counting(rng, counts, used)
        used |= {a, b}
        name = raw["cats"][cat]
        out.append({"pair_id": f"{prefix}-grd-{len(out):04d}", "kind": "counting", "category_id": cat,
                    "category": name, "question": counting_question(name),
                    "items": [{"image_id": a, "split": raw["split"], "answer": str(ka)},
                              {"image_id": b, "split": raw["split"], "answer": str(kb)}]})
    for p in out:
        for it in p["items"]:
            it["canonical"] = normalize_answer(it["answer"])
    return out


def mismatch_mapping(pairs: list[dict], rng) -> dict[str, int]:
    """question key (pair_id:index) -> image id from a different pair in the same suite."""
    mapping = {}
    for i, p in enumerate(pairs):
        for j in range(2):
            k = rng.integers(len(pairs) - 1)
            k = k if k < i else k + 1
            mapping[f"{p['pair_id']}:{j}"] = int(pairs[k]["items"][int(rng.integers(2))]["image_id"])
    return mapping


# -------------------------------------------------------------------------------------------------- download
def download_images(cfg: dict, wanted: list[tuple[str, int]]) -> list[str]:
    def fetch(item):
        split, img = item
        p = image_path(cfg, split, img)
        if p.exists() and p.stat().st_size > 0:
            return None
        p.parent.mkdir(parents=True, exist_ok=True)
        url = f"http://images.cocodataset.org/{split}/{coco_filename(split, img)}"
        for _ in range(4):
            try:
                r = requests.get(url, timeout=60)
                if r.status_code == 200:
                    tmp = p.with_suffix(".part")
                    tmp.write_bytes(r.content)
                    tmp.replace(p)
                    return None
            except requests.RequestException:
                pass
        return url

    with ThreadPoolExecutor(cfg["data"]["download_workers"]) as ex:
        failed = [u for u in ex.map(fetch, wanted) if u]
    return failed


# ------------------------------------------------------------------------------------------------------ main
def build_manifest(cfg: dict) -> tuple[dict, dict]:
    d, seed = cfg["data"], cfg["seed"]
    rng = lambda k: np.random.default_rng([seed, k])  # noqa: E731
    log("loading train2014 annotations")
    tr = load_raw(cfg, "train2014")
    vqa_train = select_vqa(vqa_candidates(tr, d["vqa_min_agreement"]), d["vqa_counts"]["train"], rng(1), None, "train")
    cap_imgs = sorted(i for i, c in tr["captions"].items() if len(c) >= d["captions_per_train_image"])
    cap_sel = sorted(rng(2).choice(cap_imgs, d["n_caption_train_images"], replace=False).tolist())
    captions_train = [{"image_id": i, "split": "train2014",
                       "ann_ids": [a for a, _ in tr["captions"][i][:d["captions_per_train_image"]]],
                       "captions": [c for _, c in tr["captions"][i][:d["captions_per_train_image"]]]} for i in cap_sel]
    train_images = sorted(set(cap_sel) | {it["image_id"] for p in vqa_train for it in p["items"]})
    log(f"building grounding pools over {len(train_images)} training images")
    pres, counts = eligibility(tr, train_images, cfg)
    r9 = rng(9)
    pools = {"presence": [], "counting": [], "categories": {str(k): v for k, v in tr["cats"].items()},
             "images": train_images, "split": "train2014"}
    for _ in range(d["grounding_pool_size"]["presence"]):
        cat, p, n = sample_presence(r9, pres, tr["cats"], None)
        pools["presence"].append([int(cat), int(p), int(n)])
    for _ in range(d["grounding_pool_size"]["counting"]):
        pools["counting"].append([int(x) for x in sample_counting(r9, counts, None)])
    del tr

    log("loading val2014 annotations")
    va = load_raw(cfg, "val2014")
    used: set = set()
    vcands = vqa_candidates(va, d["vqa_min_agreement"])
    vqa_test = select_vqa(vcands, d["vqa_counts"]["test"], rng(3), used, "test")
    vqa_dev = select_vqa(vcands, d["vqa_counts"]["dev"], rng(4), used, "dev")
    g = d["grounding_pairs"]
    grd_test = grounding_eval_pairs(va, rng(5), g["test"]["presence"], g["test"]["counting"], used, cfg, "test")
    grd_dev = grounding_eval_pairs(va, rng(6), g["dev"]["presence"], g["dev"]["counting"], used, cfg, "dev")
    cap_dev_pool = sorted(i for i in va["captions"] if i not in used)
    cap_dev = sorted(rng(7).choice(cap_dev_pool, d["n_caption_dev_images"], replace=False).tolist())
    used |= set(cap_dev)
    captions_dev = [{"image_id": i, "split": "val2014", "ann_id": va["captions"][i][0][0],
                     "caption": va["captions"][i][0][1]} for i in cap_dev]
    r8 = rng(8)
    suites = {"vqa_dev": vqa_dev, "vqa_test": vqa_test, "grounding_dev": grd_dev, "grounding_test": grd_test}
    manifest = {
        "seed": seed, "generator": "numpy default_rng([seed, k]) per component", "rules": {
            k: d[k] for k in ("vqa_min_agreement", "presence_min_area_frac", "count_min_area_frac", "count_range")},
        "captions_train": captions_train, "captions_dev": captions_dev,
        "vqa_train": vqa_train, **suites,
        "mismatch": {name: mismatch_mapping(pairs, r8) for name, pairs in suites.items()},
        "images": {"train2014": train_images,
                   "val2014": sorted(used)},
    }
    return manifest, pools


def contact_sheets(cfg: dict) -> None:
    """Render evaluation grounding pairs with their labels so labels can be checked by eye before training."""
    from PIL import Image, ImageDraw

    man = load_json(cfg["data"]["manifest"])
    out = rel(cfg["paths"]["results"]) / "label_inspection"
    out.mkdir(parents=True, exist_ok=True)
    for suite in ("grounding_dev", "grounding_test"):
        cells = [(p["question"], it["answer"], image_path(cfg, it["split"], it["image_id"]), p["pair_id"])
                 for p in man[suite] for it in p["items"]]
        per_page, W, H = 24, 260, 300
        for page in range(0, len(cells), per_page):
            sheet = Image.new("RGB", (6 * W, 4 * H), "white")
            draw = ImageDraw.Draw(sheet)
            for k, (q, a, path, pid) in enumerate(cells[page:page + per_page]):
                im = Image.open(path).convert("RGB")
                im.thumbnail((W - 10, H - 60))
                x, y = (k % 6) * W, (k // 6) * H
                sheet.paste(im, (x + 5, y + 5))
                draw.text((x + 5, y + H - 52), f"{pid}\n{q[:42]}\nlabel: {a}", fill="black")
            sheet.save(out / f"{suite}_p{page // per_page + 1}.png")
    log(f"contact sheets in {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task2.yaml")
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.inspect:
        contact_sheets(cfg)
        return
    mpath, ppath = rel(cfg["data"]["manifest"]), rel(cfg["data"]["pools"])
    if not mpath.exists():
        manifest, pools = build_manifest(cfg)
        save_fixed_file(pools, ppath)
        save_fixed_file(manifest, mpath)
    man = load_json(mpath)
    wanted = [("train2014", i) for i in man["images"]["train2014"]] + [("val2014", i) for i in man["images"]["val2014"]]
    log(f"downloading up to {len(wanted)} images")
    failed = download_images(cfg, wanted)
    stats = {"n_train_images": len(man["images"]["train2014"]), "n_eval_images": len(man["images"]["val2014"]),
             "n_caption_train_examples": sum(len(c["captions"]) for c in man["captions_train"]),
             "vqa_pairs": {k: len(man[k]) for k in ("vqa_train", "vqa_dev", "vqa_test")},
             "grounding_pairs": {k: len(man[k]) for k in ("grounding_dev", "grounding_test")},
             "failed_downloads": failed}
    save_json(stats, rel(cfg["paths"]["results"]) / "data_stats.json")
    if failed:
        raise RuntimeError(f"{len(failed)} image downloads failed, e.g. {failed[:3]}; rerun to retry")
    log(f"data ready: {stats}")


if __name__ == "__main__":
    main()
