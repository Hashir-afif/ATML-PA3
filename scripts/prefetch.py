"""Download every public model and dataset the pipeline needs into the Hugging Face cache, so the long run can
proceed with HF_HUB_OFFLINE=1.

    python -m scripts.prefetch                  # everything (Tasks 1-3; used by run_overnight.sh's setup phase)
    python -m scripts.prefetch --only task1      # only what Task 1 needs (used by the Kaggle notebook, which does
                                                  # not run Tasks 2-3 locally and should not pay for their assets)
"""
from __future__ import annotations

import argparse
import zipfile

import requests

from common.utils import load_config, log, rel

MODEL_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "merges.txt", "vocab.json"]
INPUT_URLS = {
    "task1": {"data/task1/first_plot_questions.yaml":
        "https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/main/evaluation/first_plot_questions.yaml"},
    "task3": {"data/task3/mathvista_testmini.json": "https://raw.githubusercontent.com/lupantech/MathVista/main/data/testmini.json"},
}
TASK3_DRIVE = "https://drive.google.com/drive/folders/1utX-Gfi9vvaoi1mi0pzNpDGL1o1fAh23"  # ATML-PA3-Data
TASK2_ZIPS = ["http://images.cocodataset.org/annotations/annotations_trainval2014.zip"] + [
    f"https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/{n}.zip" for n in (
        "v2_Questions_Train_mscoco", "v2_Questions_Val_mscoco", "v2_Annotations_Train_mscoco",
        "v2_Annotations_Val_mscoco", "v2_Complementary_Pairs_Train_mscoco", "v2_Complementary_Pairs_Val_mscoco")]


def _download_small_inputs(task: str) -> None:
    """Small course-specified external files, fetched only when missing (never committed to the repository)."""
    for path, url in INPUT_URLS.get(task, {}).items():
        p = rel(path)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            p.write_bytes(r.content)
            log(f"downloaded {path}")


def prefetch_task1() -> None:
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from lm_eval.tasks import TaskManager, get_task_dict

    t1 = load_config("configs/task1.yaml")
    _download_small_inputs("task1")
    snapshot_download(t1["model_id"], allow_patterns=MODEL_PATTERNS)
    load_dataset(t1["data"]["dataset_id"], split="train")
    load_dataset(t1["data"]["gsm8k_id"], t1["data"]["gsm8k_config"], split="test")
    # lm-eval task datasets, downloaded through the harness so the exact dataset builders are cached
    td = get_task_dict([t1["eval"]["gsm8k"]["task"]] + t1["eval"]["likelihood"]["tasks"], TaskManager())
    for name, task in td.items():
        n = len(list(task.test_docs() if task.has_test_docs() else task.validation_docs()))
        log(f"{name}: {n} evaluation docs cached")


def prefetch_task2() -> None:
    from huggingface_hub import snapshot_download

    t2 = load_config("configs/task2.yaml")
    raw2 = rel(t2["data"]["raw_dir"])
    raw2.mkdir(parents=True, exist_ok=True)
    for url in TASK2_ZIPS:
        z = raw2 / url.rsplit("/", 1)[1]
        if not z.exists():
            log(f"downloading {url}")
            with requests.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(z, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            with zipfile.ZipFile(z) as zf:
                zf.extractall(raw2)
    snapshot_download(t2["vision_id"], allow_patterns=MODEL_PATTERNS)


def prefetch_task3() -> None:
    from huggingface_hub import snapshot_download

    t3 = load_config("configs/task3.yaml")
    _download_small_inputs("task3")
    raw3 = rel(t3["data"]["raw_dir"])
    if not (raw3 / "README.MD").exists():
        import gdown

        gdown.download_folder(TASK3_DRIVE, output=str(raw3), quiet=False)
    snapshot_download(t3["live"]["model_id"], revision=t3["live"]["revision"], allow_patterns=MODEL_PATTERNS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["task1", "task2", "task3", "all"], default="all")
    args = ap.parse_args()
    fns = {"task1": prefetch_task1, "task2": prefetch_task2, "task3": prefetch_task3}
    for task in (fns if args.only == "all" else [args.only]):
        fns[task]()
    log(f"prefetch complete ({args.only})")


if __name__ == "__main__":
    main()
