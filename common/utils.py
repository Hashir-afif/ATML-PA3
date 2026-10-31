"""Shared helpers: config loading, seeding, JSON I/O, timing and GPU-memory bookkeeping.

Everything here is task-agnostic. Paths are always relative to the repository root, which is the
directory containing this package's parent; scripts are run from the root with ``python -m``.
"""
from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent


def rel(path: str | Path) -> Path:
    """Resolve a repository-relative path (absolute paths are returned unchanged)."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_config(path: str | Path) -> dict:
    """Load a YAML config, applying optional overrides from ``PA3_OVERRIDES_<STEM>`` (e.g. PA3_OVERRIDES_TASK1).

    The variable holds a JSON object of dotted keys -> values, e.g.
    ``{"offload_optimizer_states": false, "paths.checkpoints": "/kaggle/tmp/checkpoints/task1"}``. It is meant
    for platform-specific settings (storage paths, memory strategy), never for experimental hyperparameters;
    applied overrides are kept under ``cfg["_overrides"]`` so run summaries record them.
    """
    with open(rel(path), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    raw = os.environ.get(f"PA3_OVERRIDES_{Path(path).stem.upper()}")
    if raw:
        overrides = json.loads(raw)
        for dotted, value in overrides.items():
            node = cfg
            *parents, last = dotted.split(".")
            for k in parents:
                node = node.setdefault(k, {})
            node[last] = value
        cfg["_overrides"] = overrides
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set):
        return sorted(o)
    try:
        import torch

        if isinstance(o, torch.Tensor):
            return o.tolist()
    except ImportError:
        pass
    return str(o)


def _sanitize(o: Any):
    """Replace NaN/inf by None so files stay valid JSON (a null always carries a note elsewhere)."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    return o


def save_json(obj: Any, path: str | Path, indent: int = 2) -> Path:
    p = rel(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_sanitize(json.loads(json.dumps(obj, default=_json_default))), f, indent=indent,
                  ensure_ascii=False)
    os.replace(tmp, p)
    return p


def load_json(path: str | Path) -> Any:
    with open(rel(path), "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(rows: list[dict], path: str | Path) -> Path:
    p = rel(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=_json_default, ensure_ascii=False) + "\n")
    return p


def load_jsonl(path: str | Path) -> list[dict]:
    with open(rel(path), "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(row: dict, path: str | Path) -> None:
    p = rel(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=_json_default, ensure_ascii=False) + "\n")


def save_fixed_file(obj: Any, path: str | Path) -> Path:
    """Write a fixed split/ID file exactly once. If it exists it is never regenerated."""
    p = rel(path)
    if p.exists():
        raise FileExistsError(f"{p} is a fixed file and already exists; refusing to overwrite.")
    return save_json(obj, p)


def dir_size_bytes(path: str | Path) -> int:
    p = rel(path)
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def environment_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": sys.platform, "git_commit": git_commit()}
    try:
        import torch
        import transformers

        info.update(torch=torch.__version__, transformers=transformers.__version__,
                    cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except ImportError:
        pass
    return info


class Timer:
    def __init__(self):
        self.start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.start


@contextmanager
def cuda_timer(store: dict, key: str):
    """Accumulate synchronized wall-clock seconds under ``store[key]``."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time.perf_counter()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    store[key] = store.get(key, 0.0) + time.perf_counter() - t


def reset_peak_memory() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {"peak_allocated_gb": None, "peak_reserved_gb": None, "note": "CUDA unavailable"}
    return {"peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30}


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def capture_rng_state() -> dict:
    import torch

    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
