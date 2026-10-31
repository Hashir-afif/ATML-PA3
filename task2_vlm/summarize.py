"""Collect Task 2 results: learning curves, final tables, qualitative pairs, resources.

    python -m task2_vlm.summarize --config configs/task2.yaml [--variant default|pool49]

Qualitative selection rule (fixed in advance, independent of outcomes): from each final-test suite, the first two
pairs of every question type in manifest order. Each panel shows both images of the pair with the shared
question, the original references and the predictions under all four image conditions.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import plotting
from common.utils import load_config, load_json, load_jsonl, log, rel, save_json
from task2_vlm.data_prep import image_path
from task2_vlm.model import variant_settings


def learning_curves(cfg, vs, figs):
    tl, dl = vs["results"] / "train_log.jsonl", vs["results"] / "dev_metrics.jsonl"
    if not tl.exists() or not dl.exists():
        return None
    tr, dev = pd.DataFrame(load_jsonl(tl)), load_jsonl(dl)
    bounds = np.cumsum([s["updates"] for s in cfg["train"]["stages"]])[:-1]
    du = [d["update"] for d in dev]
    fig, axes = plotting.plt.subplots(1, 4, figsize=(15, 3.2))
    axes[0].plot(tr["update"], tr["loss"], color=plotting.PALETTE[0], alpha=0.2, lw=0.7)
    axes[0].plot(tr["update"], tr["loss"].rolling(50, min_periods=1).mean(), color=plotting.PALETTE[0])
    axes[0].set_title("Training loss (50-update mean)")
    axes[0].set_ylabel("Example-averaged NLL")
    axes[1].plot(du, [d["caption_loss"] for d in dev], marker="o", color=plotting.PALETTE[0])
    axes[1].set_title("Dev caption reference loss")
    axes[1].set_ylabel("NLL (nats/token)")
    for i, (suite, name) in enumerate([("grounding_dev", "Grounding"), ("vqa_dev", "VQAv2")]):
        for c, lab in (("correct", "correct"), ("swap", "complementary"), ("no_image", "no image")):
            axes[2].plot(du, [d[suite][f"nll_{c}"] for d in dev], marker="o" if i == 0 else "s",
                         ls="-" if i == 0 else "--", color=plotting.color({"swap": "swap"}.get(c, c)),
                         label=f"{name}: {lab}")
        axes[3].plot(du, [d[suite]["delta_nll"] for d in dev], marker="o", color=plotting.PALETTE[i], label=name)
    axes[2].set_title("Dev reference-answer NLL")
    axes[2].set_ylabel("NLL (nats/token)")
    axes[2].legend(fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    axes[3].axhline(0, color=plotting.NEUTRAL, lw=0.8)
    axes[3].set_title(r"Image-swap penalty $\Delta_{NLL}$")
    axes[3].set_ylabel("NLL(partner) - NLL(correct)")
    axes[3].legend()
    for ax in axes:
        ax.set_xlabel("Main-run update")
        for b in bounds:
            ax.axvline(b, color="#b5b4ae", ls=":", lw=1)
    plotting.save(fig, figs / "learning_curves")
    return {"dev": dev, "train_loss_by_update": tr[["update", "stage", "loss"]].to_dict("list")}


def final_tables(vs) -> tuple[dict | None, pd.DataFrame | None]:
    p = vs["results"] / "final_eval.json"
    if not p.exists():
        return None, None
    res = load_json(p)
    rows = []
    for suite, s in res["suites"].items():
        bs = s["bootstrap"]
        for cond, m in s["conditions"].items():
            row = {"suite": suite, "condition": cond, "accuracy": m["accuracy"],
                   "accuracy_ci95": bs["accuracy_ci95"][cond], "pair_accuracy": m["pair_accuracy"],
                   "identical_answer_rate": m["identical_answer_rate"], "empty_answer_rate": m["empty_answer_rate"],
                   "follows_partner_rate": m["follows_partner_rate"], "mean_reference_nll": m["mean_reference_nll"],
                   "gen_ms_per_answer": 1000 * m["generation_seconds_per_answer"],
                   **{f"acc_{t}": v for t, v in m["accuracy_by_qtype"].items()}}
            if cond != "correct":
                row["correct_minus_cond_acc"] = bs["correct_minus_condition_accuracy"][cond]["estimate"]
                row["correct_minus_cond_acc_ci95"] = bs["correct_minus_condition_accuracy"][cond]["ci95"]
                row["cond_minus_correct_nll"] = bs["condition_minus_correct_nll"][cond]["estimate"]
                row["cond_minus_correct_nll_ci95"] = bs["condition_minus_correct_nll"][cond]["ci95"]
            rows.append(row)
    return res, pd.DataFrame(rows)


def qualitative(cfg, vs, figs) -> list[dict]:
    p = vs["results"] / "final_predictions.jsonl"
    if not p.exists():
        return []
    from PIL import Image

    rows = load_jsonl(p)
    man = load_json(cfg["data"]["manifest"])
    pred = {(r["suite"], r["pair_id"], r["index"], r["condition"]): r["prediction"] for r in rows}
    chosen = []
    for suite in ("grounding_test", "vqa_test"):
        seen = {}
        for pair in man[suite]:
            t = pair.get("answer_type", pair.get("kind"))
            if seen.get(t, 0) < 2:
                seen[t] = seen.get(t, 0) + 1
                chosen.append((suite, pair))
    out = []
    for page in range(0, len(chosen), 5):
        sel = chosen[page:page + 5]
        fig, axes = plotting.plt.subplots(len(sel), 2, figsize=(9, 3.8 * len(sel)))
        axes = np.atleast_2d(axes)
        for r, (suite, pair) in enumerate(sel):
            for j, it in enumerate(pair["items"]):
                ax = axes[r, j]
                ax.imshow(Image.open(image_path(cfg, it["split"], it["image_id"])).convert("RGB"))
                ax.axis("off")
                refs = it.get("answers") or [it["answer"]]
                ps = {c: pred.get((suite, pair["pair_id"], j, c), "?") for c in cfg["eval"]["conditions"]}
                ax.set_title(f"{pair['question'][:60]}\nref: {it['canonical']} | correct: {ps['correct']} | "
                             f"swap: {ps['swap']}\nmismatch: {ps['mismatch']} | no image: {ps['no_image']}",
                             fontsize=7.5, loc="left")
                out.append({"suite": suite, "pair_id": pair["pair_id"], "index": j, "question": pair["question"],
                            "canonical": it["canonical"], "references": refs, "predictions": ps})
        plotting.save(fig, figs / f"qualitative_pairs_p{page // 5 + 1}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task2.yaml")
    ap.add_argument("--variant", default="default")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.smoke:
        cfg["paths"] = {k: v + "_smoke" for k, v in cfg["paths"].items()}
    vs = variant_settings(cfg, args.variant)
    figs = vs["results"] / "figures"
    plotting.setup()
    out = {"variant": args.variant, "notes": {}}
    out["learning"] = learning_curves(cfg, vs, figs) or out["notes"].setdefault("learning", "Status: NOT RUN")
    res, table = final_tables(vs)
    if table is not None:
        table.to_csv(vs["results"] / "final_table.csv", index=False)
        (vs["results"] / "final_table.md").write_text(
            table.drop(columns=[c for c in table if c.endswith("ci95")]).to_markdown(index=False, floatfmt=".3f"),
            encoding="utf-8")
        out["final"] = res
    else:
        out["notes"]["final"] = "Status: NOT RUN"
    out["qualitative"] = qualitative(cfg, vs, figs)
    for name in ("train_summary.json",):
        p = vs["results"] / name
        out[name.removesuffix(".json")] = load_json(p) if p.exists() else None
    fc = rel(cfg["paths"]["results"]) / "feature_caching.json"
    out["feature_caching"] = load_json(fc) if fc.exists() else None
    ts = out["train_summary"]
    if ts:
        res_rows = [{"stage": k, "trainable_connector": v["connector"], "trainable_lora": v["lora"],
                     "train_hours": ts["stage_seconds"].get(k, 0) / 3600} for k, v in ts["trainable_params_by_stage"].items()]
        pd.DataFrame(res_rows).to_csv(vs["results"] / "resources_table.csv", index=False)
        out["resources"] = {"by_stage": res_rows, "peak_gpu_allocated_gb": ts["peak_gpu_allocated_gb"],
                            "feature_caching": out["feature_caching"]}
    ds = rel(cfg["paths"]["results"]) / "data_stats.json"
    out["data_stats"] = load_json(ds) if ds.exists() else None
    save_json(out, vs["results"] / "task2_results.json")
    log(f"saved {vs['results'] / 'task2_results.json'}; notes {out['notes']}")
    if table is not None:
        print(table[["suite", "condition", "accuracy", "pair_accuracy", "mean_reference_nll"]].to_string())


if __name__ == "__main__":
    main()
