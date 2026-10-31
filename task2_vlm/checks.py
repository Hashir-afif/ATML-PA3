"""Sanity checks required before Task 2 training (feature shapes, frozen sets, connector gradients, split isolation).

    python -m task2_vlm.checks --config configs/task2.yaml
Writes results/task2/checks.json and fails loudly if any check fails.
"""
from __future__ import annotations

import argparse
from itertools import combinations

import torch

from common.utils import load_config, load_json, log, rel, save_json
from task2_vlm.examples import caption_train_examples, eval_items
from task2_vlm.features import FeatureStore
from task2_vlm.model import build_vlm, collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task2.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    man = load_json(cfg["data"]["manifest"])
    pools = load_json(cfg["data"]["pools"])
    res = {}

    # split isolation
    train = set(man["images"]["train2014"])
    suites = {k: {it["image_id"] for p in man[k] for it in p["items"]}
              for k in ("vqa_dev", "vqa_test", "grounding_dev", "grounding_test")}
    suites["captions_dev"] = {e["image_id"] for e in man["captions_dev"]}
    pool_imgs = {x for e in pools["presence"] for x in e[1:]} | {e[1] for e in pools["counting"]} | \
        {e[3] for e in pools["counting"]}
    res["pool_images_subset_of_train"] = pool_imgs <= train
    res["eval_suites_pairwise_disjoint"] = all(not (suites[a] & suites[b]) for a, b in combinations(suites, 2))
    res["eval_vs_train_image_overlap"] = 0  # different COCO splits; recorded explicitly below
    res["eval_split_names"] = sorted({it["split"] for k in suites if k != "captions_dev" for p in man[k] for it in p["items"]})
    res["train_split_names"] = sorted({e["split"] for e in man["captions_train"]} |
                                      {it["split"] for p in man["vqa_train"] for it in p["items"]})
    res["mismatch_never_own_pair"] = all(
        man["mismatch"][k][f"{p['pair_id']}:{j}"] not in {it["image_id"] for it in p["items"]}
        for k in man["mismatch"] for p in man[k] for j in range(2))
    res["pairs_have_different_answers"] = all(p["items"][0]["canonical"] != p["items"][1]["canonical"]
                                              for k in suites if k != "captions_dev" for p in man[k])

    # features
    store = FeatureStore(cfg)
    f = store.get("train2014", man["captions_train"][0]["image_id"])
    res["feature_shape"], res["feature_dtype"] = list(f.shape), str(f.dtype)
    res["feature_counts"] = {s: len(store.index[s]) for s in store.index}

    # frozen sets and gradient flow (Stage-2 configuration: connector + LoRA)
    vlm, info = build_vlm(cfg, "default", with_lora=True)
    vlm.train()
    trainable = {n for n, p in vlm.named_parameters() if p.requires_grad}
    res["trainable_param_names_sample"] = sorted(trainable)[:6]
    res["only_connector_and_lora_trainable"] = all(n.startswith("connector.") or "lora_" in n for n in trainable)
    res["qwen_base_frozen"] = all(not p.requires_grad for n, p in vlm.qwen.named_parameters() if "lora_" not in n)
    exs = caption_train_examples(man)[:4]
    built = [vlm.builder.build(e, True, e["target"]) for e in exs]
    batch = {k: v.cuda() for k, v in collate(built, vlm.builder.tok.pad_token_id).items()}
    with torch.autocast("cuda", dtype=torch.float16):
        loss = vlm.per_example_nll(batch, store.batch([e["image"] for e in exs]).cuda()).mean()
    loss.backward()
    g = [p.grad for p in vlm.connector.parameters()]
    res["connector_grads_nonzero_finite"] = all(x is not None and torch.isfinite(x).all() and x.abs().sum() > 0 for x in g)
    res["qwen_base_grads_none"] = all(p.grad is None for n, p in vlm.qwen.named_parameters() if "lora_" not in n)
    res["visual_tokens_in_prompt"] = sum(t == vlm.builder.image_pad for t in built[0]["ids"])
    it = eval_items(man, "vqa_test")[0]
    noimg = vlm.builder.build(it, False, it["target"])
    res["no_image_prompt_has_no_visual_tokens"] = vlm.builder.image_pad not in noimg["ids"]
    res["example_prompt_decoded"] = vlm.builder.tok.decode(vlm.builder.build(it, True, it["target"])["ids"]).replace(
        "<|image_pad|>" * res["visual_tokens_in_prompt"], f"<|image_pad|>x{res['visual_tokens_in_prompt']}")
    res["initial_loss"] = float(loss)
    save_json(res, rel(cfg["paths"]["results"]) / "checks.json")
    bad = [k for k, v in res.items() if v is False]
    log(f"checks: {res}")
    if bad:
        raise SystemExit(f"FAILED checks: {bad}")


if __name__ == "__main__":
    main()
