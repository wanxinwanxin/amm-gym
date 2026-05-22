"""Find the spikes in the simulated per-trade markout distribution."""
from __future__ import annotations

import sys
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKTREE))

import numpy as np  # noqa: E402


def histogram_modes(values: np.ndarray, bin_width: float, lo: float, hi: float, top_n: int = 8) -> None:
    edges = np.arange(lo, hi + bin_width, bin_width)
    counts, edges = np.histogram(values, bins=edges)
    # find the top_n bins by count
    order = np.argsort(counts)[::-1]
    total = counts.sum()
    print(f"\nHistogram modes ({bin_width} bps bins, range [{lo}, {hi}]):")
    print(f"  bins covering {total/values.size*100:.1f}% of {values.size} samples")
    print(f"  {'bps_range':>16}  {'count':>8}  {'pct':>6}")
    for idx in order[:top_n]:
        bps_lo, bps_hi = edges[idx], edges[idx + 1]
        c = counts[idx]
        print(f"  [{bps_lo:+7.3f},{bps_hi:+7.3f})  {c:8d}  {c/values.size*100:5.2f}%")


def main() -> None:
    markouts = np.load(WORKTREE / "reports" / "spike_markouts.npy")
    print(f"Loaded {markouts.size:,} per-trade markouts (bps).")

    # check zoom: §5 plot likely uses a -25..+25 or similar range
    in_zoom = markouts[(markouts >= -25) & (markouts <= 25)]
    print(f"In [-25, 25]: {in_zoom.size:,} samples ({in_zoom.size/markouts.size*100:.1f}%).")

    # high-resolution histogram across the bulk
    histogram_modes(in_zoom, 0.5, -25, 25, top_n=12)
    histogram_modes(in_zoom, 0.1, -25, 25, top_n=12)
    histogram_modes(in_zoom, 0.05, -25, 25, top_n=12)

    # Bonus: dig into near-+5 and near-0 specifically
    print("\nFine-bin counts near +5 bps (0.01 bps bins, [4.9, 5.1]):")
    edges = np.arange(4.9, 5.10 + 0.01, 0.01)
    counts, edges = np.histogram(markouts, bins=edges)
    for c, e0, e1 in zip(counts, edges[:-1], edges[1:]):
        if c > 0:
            print(f"  [{e0:.4f}, {e1:.4f})  {c} samples ({c/markouts.size*100:.2f}%)")

    print("\nFine-bin counts near 0 bps (0.01 bps bins, [-0.5, 0.5]):")
    edges = np.arange(-0.5, 0.5 + 0.01, 0.01)
    counts, edges = np.histogram(markouts, bins=edges)
    for c, e0, e1 in zip(counts, edges[:-1], edges[1:]):
        if c > 0:
            print(f"  [{e0:.4f}, {e1:.4f})  {c} samples ({c/markouts.size*100:.2f}%)")

    print("\nFine-bin counts near +10 bps (0.01 bps bins, [9.5, 10.5]):")
    edges = np.arange(9.5, 10.5 + 0.01, 0.01)
    counts, edges = np.histogram(markouts, bins=edges)
    for c, e0, e1 in zip(counts, edges[:-1], edges[1:]):
        if c > 0:
            print(f"  [{e0:.4f}, {e1:.4f})  {c} samples ({c/markouts.size*100:.2f}%)")


if __name__ == "__main__":
    main()
