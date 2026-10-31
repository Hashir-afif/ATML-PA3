"""Download every public model and dataset the pipeline needs into the Hugging Face cache, so the long run can
proceed with HF_HUB_OFFLINE=1.

    python -m scripts.prefetch
"""
from __future__ import annotations

import zipfile

import requests

from common.utils import load_config, log, rel

INPUT_URLS = {
    "data/task1/first_plot_questions.yaml":
        "https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/main/evaluation/first_plot_questions.yaml",
    "data/task3/mathvista_testmini.json": "https://raw.githubusercontent.com/lupantech/MathVista/main/data/testmini.json",
}
TASK3_DRIVE = "https://drive.google.com/drive/folders/1utX-Gfi9vvaoi1mi0pzNpDGL1o1fAh23"  # ATML-PA3-Data
TASK2_ZIPS = ["http://images.cocodataset.org/annotations/annotations_trainval2014.zip"] + [
    f"https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/{n}.zip" for n in (
        "v2_Questions_Train_mscoco", "v2_Questions_Val_mscoco", "v2_Annotations_Train_mscoco",
        "v2_Annotations_Val_mscoco", "v2_Complementary_Pairs_Train_mscoco", "v2_Complementary_Pairs_Val_mscoco")]


def download_inputs(t2: dict, t3: dict) -> None:
    """Course-specified external inputs, fetched only when missing (never committed to the repository)."""
    for path, url in INPUT_URLS.items():
        p = rel(path)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            p.write_bytes(r.content)
            log(f"downloaded {path}")
    raw3 = rel(t3["data"]["raw_dir"])
    if not (raw3 / "README.MD").exists():
        import gdown

        gdown.download_folder(TASK3_DRIVE, output=str(raw3), quiet=False)
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


def main():
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    t1, t2, t3 = (load_config(f"configs/task{i}.yaml") for i in (1, 2, 3))
    download_inputs(t2, t3)
    pats = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "merges.txt", "vocab.json"]
    snapshot_download(t1["model_id"], allow_patterns=pats)
    snapshot_download(t2["vision_id"], allow_patterns=pats)
    snapshot_download(t3["live"]["model_id"], revision=t3["live"]["revision"], allow_patterns=pats)
    load_dataset(t1["data"]["dataset_id"], split="train")
    load_dataset(t1["data"]["gsm8k_id"], t1["data"]["gsm8k_config"], split="test")
    # lm-eval task datasets (downloaded through the harness so the exact dataset builders are cached)
    from lm_eval.tasks import TaskManager, get_task_dict

    td = get_task_dict(["gsm8k_cot", "hellaswag", "boolq", "arc_easy"], TaskManager())
    for name, task in td.items():
        n = len(list(task.test_docs() if task.has_test_docs() else task.validation_docs()))
        log(f"{name}: {n} evaluation docs cached")
    log("prefetch complete")


if __name__ == "__main__":
    main()
