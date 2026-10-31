"""Shared matplotlib style for report figures (PNG + PDF).

Categorical colors are assigned per entity in a fixed order (never by rank) from a CVD-validated palette, so
the same method keeps the same color in every figure.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
NEUTRAL = "#52514e"

ENTITY_COLORS = {
    # Task 1 conditions
    "baseline": NEUTRAL, "full_mid": "#8fb8ea", "full": PALETTE[0], "l2sp": PALETTE[1], "lora4": PALETTE[2],
    "lora16": PALETTE[3], "oplora16": PALETTE[6],
    # Task 2 image conditions
    "correct": PALETTE[0], "swap": PALETTE[1], "mismatch": PALETTE[2], "no_image": PALETTE[3],
}

LABELS = {"baseline": "Baseline", "full_mid": "Full FT (mid)", "full": "Full FT", "l2sp": "Full FT + L2-SP",
          "lora4": "LoRA-4", "lora16": "LoRA-16", "oplora16": "OPLoRA-16", "correct": "Correct",
          "swap": "Complementary swap", "mismatch": "Mismatched", "no_image": "No image"}


def color(name: str, i: int = 0) -> str:
    return ENTITY_COLORS.get(name, PALETTE[i % len(PALETTE)])


def setup() -> None:
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8.5,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "lines.linewidth": 1.6, "lines.markersize": 5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
        "grid.linewidth": 0.6, "axes.edgecolor": "#8a8984", "axes.axisbelow": True, "figure.dpi": 120,
        "savefig.dpi": 200, "savefig.bbox": "tight", "legend.frameon": False,
    })


def save(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
