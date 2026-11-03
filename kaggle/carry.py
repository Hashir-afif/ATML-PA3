"""Carry Task 1 state between Kaggle sessions (run from the repository root inside the notebook).

    python kaggle/carry.py import    # copy the previous session's output (attached as an input) into place
    python kaggle/carry.py export    # copy what the next session needs into /kaggle/working/carry
    python kaggle/carry.py status    # list finished / missing Task 1 outputs

Kaggle keeps only /kaggle/working (~20 GB) when a version is saved; checkpoints are written to fast scratch space
(PA3_CKPT_ROOT, default /kaggle/tmp/checkpoints/task1) during a session. The export keeps per method and checkpoint:
meta.json, COMPLETE and adapter factors; full-model weights only for evaluated checkpoints (Full FT midpoint and
finals) or the latest checkpoint of an unfinished run; optimizer/RNG state only for the latest checkpoint of an
unfinished run (so it resumes exactly). results/task1 lives in the cloned repository under /kaggle/working and is
saved with the output automatically. Merged adapter models are not carried; they are cheap to rebuild.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORK = Path(os.environ.get("PA3_WORK_DIR", "/kaggle/working"))
INPUT = Path(os.environ.get("PA3_INPUT_DIR", "/kaggle/input"))
CKPT = Path(os.environ.get("PA3_CKPT_ROOT", "/kaggle/tmp/checkpoints/task1"))
CARRY = WORK / "carry"
TOTAL, MIDPOINT = 1250, 625
METHODS = ["full", "l2sp", "lora4", "lora16", "oplora16"]
CONDITIONS = ["baseline", "full_mid", "full", "l2sp", "lora4", "lora16", "oplora16"]


def size_gb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 2**30 if p.exists() else 0.0


def find_previous() -> Path | None:
    """Newest carry manifest among attached inputs (outputs of earlier versions of this notebook)."""
    best, best_t = None, -1.0
    for m in INPUT.glob("**/carry/manifest.json") if INPUT.exists() else []:
        try:
            t = json.loads(m.read_text())["saved_at"]
        except Exception:
            continue
        if t > best_t:
            best, best_t = m.parent.parent, t
    return best


def copy_tree_additive(src: Path, dst: Path) -> tuple[int, int]:
    """Copy every file under src to the matching path under dst, skipping any file dst already has.

    Used for results/task1: the freshly cloned repo may already carry finished, git-committed results (as
    happened here -- a locally run, pushed baseline evaluation), and a previous session's own copy of
    results/task1 can be from a crashed, partial attempt. A blind `copytree(dirs_exist_ok=True)` would let
    that partial attempt silently overwrite good, already-committed files with mixed-provenance data. Returns
    (files copied, files skipped because dst already had them).
    """
    copied = skipped = 0
    for f in src.rglob("*"):
        if f.is_dir():
            continue
        rel = f.relative_to(src)
        target = dst / rel
        if target.exists():
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
        copied += 1
    return copied, skipped


def do_import() -> None:
    prev = find_previous()
    if prev is None:
        print("no previous session output attached; starting fresh")
        return
    print(f"importing previous session from {prev}")
    res = next(iter(prev.glob("**/results/task1")), None)
    if res is not None:
        copied, skipped = copy_tree_additive(res, REPO / "results" / "task1")
        print(f"  results/task1 <- {res} ({copied} files copied, {skipped} already present and kept)")
    # Checkpoints are intentionally overwritten: each update directory is self-contained (named by update
    # number) and the latest copy of an in-progress update's state is always the one to resume from.
    src = prev / "carry" / "checkpoints" / "task1"
    if src.exists():
        CKPT.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, CKPT, dirs_exist_ok=True)
        print(f"  checkpoints ({size_gb(src):.1f} GB) -> {CKPT}")


def do_export() -> None:
    dst_root = CARRY / "checkpoints" / "task1"
    if dst_root.exists():
        shutil.rmtree(dst_root)
    for m in METHODS:
        mdir = CKPT / m
        cks = sorted((p for p in mdir.glob("update_*") if (p / "COMPLETE").exists()), key=lambda p: p.name) \
            if mdir.exists() else []
        if not cks:
            continue
        finished = (mdir / f"update_{TOTAL:04d}" / "COMPLETE").exists()
        latest = cks[-1]
        for p in cks:
            u = int(p.name.split("_")[1])
            d = dst_root / m / p.name
            d.mkdir(parents=True, exist_ok=True)
            for f in ["meta.json", "COMPLETE", "adapter_raw.safetensors", "adapter_effective.safetensors"]:
                if (p / f).exists():
                    shutil.copy2(p / f, d / f)
            keep_weights = u == TOTAL or (m == "full" and u == MIDPOINT) or (not finished and p == latest)
            if keep_weights and (p / "model").exists():
                shutil.copytree(p / "model", d / "model", dirs_exist_ok=True)
            if not finished and p == latest and (p / "training_state.pt").exists():
                shutil.copy2(p / "training_state.pt", d / "training_state.pt")
    CARRY.mkdir(parents=True, exist_ok=True)
    gb = size_gb(CARRY)
    (CARRY / "manifest.json").write_text(json.dumps({"saved_at": time.time(), "carry_gb": gb}, indent=2))
    print(f"carried {gb:.2f} GB to {CARRY}; repository dir {size_gb(REPO):.2f} GB")
    if gb + size_gb(REPO) > 18.5:
        print("WARNING: output is close to Kaggle's ~20 GB /kaggle/working limit")


def do_status() -> bool:
    res = REPO / "results" / "task1"
    missing = [f"train {m}" for m in METHODS if not (res / "train_summary" / f"{m}.json").exists()]
    missing += [f"eval {c}" for c in CONDITIONS if not (res / "eval" / c / "aggregate.json").exists()]
    missing += [f"qualitative {c}" for c in CONDITIONS if not (res / "qualitative" / f"{c}.jsonl").exists()]
    dpath = res / "diagnostics" / "diagnostics.json"
    diag = json.loads(dpath.read_text()).get("conditions", {}) if dpath.exists() else {}
    missing += [f"diagnostics {c}" for c in CONDITIONS[1:] if not diag.get(c)]
    if missing:
        print("TASK 1 NOT FINISHED -- remaining: " + ", ".join(missing))
        print("Attach this version's output as an input and run the notebook again to continue.")
        return False
    print("TASK 1 ALL DONE -- download results_task1.zip from the output tab")
    return True


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"import": do_import, "export": do_export, "status": do_status}[cmd]()
