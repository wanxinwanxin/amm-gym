"""Phase 1 diagnostic: plot the observed price-impact cloud for the non-5bp
portion of router-routed WETH/USDC transactions.

Input:  analysis/weth_usdc_90d/non5bp_impact_sample_7d.csv
Output: plots/impact_curve_observed.png

The plot is the central deliverable for Phase 1 of the impact-curve calibration:
its shape (V2-like / not-V2-like) drives whether Plan A (USD-weighted squared)
or Plan B (Huber) is more defensible in Phase 2.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_PATH = REPO_ROOT / "analysis" / "weth_usdc_90d" / "non5bp_impact_sample_7d.csv"
OUT_PATH = REPO_ROOT / "plots" / "impact_curve_observed.png"


def load_sample() -> pd.DataFrame:
    df = pd.read_csv(SAMPLE_PATH)
    # Drop rare mixed-direction txs per spec.
    df = df[df["n_distinct_sides"] == 1].copy()
    # Drop dust (numerical artifacts).
    df = df[df["size_usd"] > 1.0].copy()
    # Optional: clip pathological spread outliers from plotting (keep them in fit).
    return df.reset_index(drop=True)


def decile_centerline(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """USD-weighted median spread per log-size decile (centerline overlay)."""
    log_sizes = np.log10(df["size_usd"].values)
    edges = np.linspace(log_sizes.min(), log_sizes.max(), 21)  # 20 bins
    centers = []
    median_spreads = []
    for i in range(len(edges) - 1):
        mask = (log_sizes >= edges[i]) & (log_sizes < edges[i + 1])
        if mask.sum() < 10:
            continue
        sub = df.loc[mask]
        # USD-weighted median
        sorted_sub = sub.sort_values("observed_spread_bps")
        w = sorted_sub["size_usd"].values
        cw = np.cumsum(w)
        half = cw[-1] / 2.0
        idx = int(np.searchsorted(cw, half))
        idx = min(idx, len(sorted_sub) - 1)
        median_spreads.append(sorted_sub.iloc[idx]["observed_spread_bps"])
        centers.append(10 ** ((edges[i] + edges[i + 1]) / 2))
    return np.asarray(centers), np.asarray(median_spreads)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    df = load_sample()
    print(f"loaded {len(df):,} txs")
    print(f"  size_usd: min={df['size_usd'].min():.2f}, max={df['size_usd'].max():,.0f}, median={df['size_usd'].median():,.0f}")
    print(f"  observed_spread_bps: min={df['observed_spread_bps'].min():.1f}, max={df['observed_spread_bps'].max():.1f}")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Hex-bin the cloud. Clip y-range for readability (the fit will see full range).
    y_clip_lo, y_clip_hi = -50, 100
    mask = (df["observed_spread_bps"] >= y_clip_lo) & (df["observed_spread_bps"] <= y_clip_hi)
    plot_df = df[mask]

    hb = ax.hexbin(
        plot_df["size_usd"], plot_df["observed_spread_bps"],
        xscale="log", yscale="linear",
        gridsize=60, mincnt=1, cmap="viridis", bins="log",
    )
    cb = fig.colorbar(hb, ax=ax, label="log10(count)")

    # Decile centerline (depth-weighted median)
    centers, medians = decile_centerline(df)
    ax.plot(centers, medians, color="white", lw=3.5, alpha=0.95, zorder=5)
    ax.plot(centers, medians, color="#e74c3c", lw=2.0, alpha=0.95, zorder=6,
            label="USD-weighted median (per log-size decile)")

    ax.axhline(0, color="#555", lw=0.8, ls="--", alpha=0.7)

    ax.set_xlim(50, 1e6)  # x-axis explicit per spec
    ax.set_ylim(y_clip_lo, y_clip_hi)
    ax.set_xlabel("Tx size (USD, log scale)", fontsize=11)
    ax.set_ylabel("Observed price-impact spread (bps, LP-positive sign)", fontsize=11)

    n_total = len(df)
    n_clipped = (mask.sum())
    pct_neg = (df["observed_spread_bps"] < 0).mean() * 100
    ax.set_title(
        f"Phase 1 — Observed price-impact cloud, non-5bp portion of router-routed WETH/USDC txs\n"
        f"7-day window 2026-05-14..2026-05-20 · n_txs={n_total:,} ({n_clipped:,} shown after y-clip) · "
        f"{pct_neg:.1f}% negative-spread points",
        fontsize=10.5, fontweight="bold",
    )
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.grid(True, which="both", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
