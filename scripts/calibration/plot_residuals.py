"""Build the residual-evolution plot from saved cycle artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
ART_DIR = _ROOT / "calibration_artifacts"
PLOTS_DIR = _ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def load_cycles() -> list[dict]:
    cycles = []
    for p in sorted(ART_DIR.glob("cycle_*.json")):
        cycles.append(json.loads(p.read_text()))
    return cycles


def main() -> None:
    cycles = load_cycles()
    if not cycles:
        print("No cycle artifacts found.")
        return

    nums = [c["cycle"] for c in cycles]
    r1 = [c["final_residuals"]["T1_arb_5bp_share"] for c in cycles]
    r2 = [c["final_residuals"]["T2_retail_5bp_share"] for c in cycles]
    r3 = [c["final_residuals"]["T3_markout_bps"] for c in cycles]
    losses = [c["scipy_result"]["fun"] for c in cycles]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.axhspan(-0.02, 0.02, color="#27ae60", alpha=0.10, label="±2% target")
    ax.axhline(0, color="#666", lw=0.5)
    ax.plot(nums, r1, "o-", color="#e74c3c", lw=2, markersize=8, label="T1 arb_5bp_share")
    ax.plot(nums, r2, "s-", color="#2980b9", lw=2, markersize=8, label="T2 retail_5bp_share")
    ax.plot(nums, r3, "^-", color="#27ae60", lw=2, markersize=8, label="T3 markout_bps")
    ax.set_xlabel("Cycle", fontsize=11)
    ax.set_ylabel("Relative residual", fontsize=11)
    ax.set_title("Residual Evolution (Calibration Seeds)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, frameon=False, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:+.0f}%"))

    ax = axes[1]
    ax.plot(nums, losses, "o-", color="#8e44ad", lw=2, markersize=8)
    ax.set_xlabel("Cycle", fontsize=11)
    ax.set_ylabel("Loss (sum of squared relative residuals)", fontsize=11)
    ax.set_title("Loss Evolution", fontsize=12, fontweight="bold")
    ax.set_yscale("log")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    out_path = PLOTS_DIR / "calibration_residuals.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
