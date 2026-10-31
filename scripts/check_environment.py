"""Environment and input check (run before the overnight pipeline).

    python -m scripts.check_environment
Verifies CUDA, GPU memory, library versions, free disk space and the presence of required input files.
"""
from __future__ import annotations

import shutil
import sys

from common.utils import environment_info, log, rel, save_json

REQUIRED = ["configs/task1.yaml", "configs/task2.yaml", "configs/task3.yaml", "data/task1/first_plot_questions.yaml",
            "data/task3/mathvista_testmini.json", "data/task3/raw/README.MD",
            "data/task3/raw/qwen25-math/generations.parquet", "data/task3/raw/qwen25-math/probability_metrics.parquet",
            "data/task3/raw/qwen25vl-mathvista/generations.parquet",
            "data/task3/raw/qwen25vl-mathvista/probability_metrics.parquet",
            "data/task3/raw/qwen25vl-mathvista/attention_metrics.parquet"]


def main():
    import torch

    info = environment_info()
    info["cuda_available"] = torch.cuda.is_available()
    info["gpu_total_gb"] = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else None
    info["disk_free_gb"] = shutil.disk_usage(rel(".")).free / 2**30
    info["missing_inputs"] = [p for p in REQUIRED if not rel(p).exists()]
    for mod in ("lm_eval", "math_verify", "datasets", "accelerate"):
        try:
            m = __import__(mod)
            info[f"{mod}_version"] = getattr(m, "__version__", "unknown")
        except ImportError:
            info[f"{mod}_version"] = None
    save_json(info, rel("results/environment.json"))
    log(f"environment: {info}")
    problems = []
    if not info["cuda_available"]:
        problems.append("CUDA not available")
    if info["missing_inputs"]:
        problems.append(f"missing inputs {info['missing_inputs']}")
    if info["disk_free_gb"] < 120:
        problems.append(f"only {info['disk_free_gb']:.0f} GB free; Task 1 recovery checkpoints need ~70 GB")
    if problems:
        sys.exit("; ".join(problems))


if __name__ == "__main__":
    main()
