"""Plot non-5bp router-routed price impact under two reference choices:
  (a) pool_mid_pre  — V3 pre-trade marginal price (V2's natural coordinate)
  (b) fair_price    — Binance benchmark mid at block

For the apples-to-apples 24k-tx overlap subset, both references are plotted
side by side: binned-medians (USD-volume-weighted) and a scatter cloud.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parent.parent.parent
JOIN_CSV = REPO / "analysis/weth_usdc_90d/non5bp_impact_sample_pool_vs_fair_7d.csv"
POOL_CSV = REPO / "analysis/weth_usdc_90d/non5bp_impact_sample_v3_pool_mid_7d.csv"
FAIR_CSV = REPO / "analysis/weth_usdc_90d/non5bp_impact_sample_7d.csv"
OUT_DIR = REPO / "plots"
OUT_DIR.mkdir(exist_ok=True)


def load_joined() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sizes = []
    spread_pool = []
    spread_fair = []
    with JOIN_CSV.open() as fh:
        r = csv.DictReader(fh)
        for row in r:
            try:
                s = float(row["size_usd_v3"])
                sp = float(row["spread_pool_bps"])
                sf = float(row["spread_fair_bps"])
            except (TypeError, ValueError):
                continue
            if not (s > 0 and math.isfinite(sp) and math.isfinite(sf)):
                continue
            sizes.append(s)
            spread_pool.append(sp)
            spread_fair.append(sf)
    return np.array(sizes), np.array(spread_pool), np.array(spread_fair)


def usd_binned_stats(
    sizes: np.ndarray,
    spread: np.ndarray,
    bin_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For each bin, compute USD-volume-weighted median, USD-weighted mean, p25/p75."""
    bin_idx = np.digitize(sizes, bin_edges) - 1
    median = np.full(len(bin_edges) - 1, np.nan)
    mean = np.full(len(bin_edges) - 1, np.nan)
    p25 = np.full(len(bin_edges) - 1, np.nan)
    p75 = np.full(len(bin_edges) - 1, np.nan)
    for i in range(len(bin_edges) - 1):
        m = bin_idx == i
        if m.sum() < 5:
            continue
        sp = spread[m]
        w = sizes[m]
        # USD-weighted percentiles: sort sp, sum w, find weighted quantile
        order = np.argsort(sp)
        sp_sorted = sp[order]
        w_sorted = w[order]
        cw = np.cumsum(w_sorted) / w_sorted.sum()
        median[i] = sp_sorted[np.searchsorted(cw, 0.5)]
        p25[i] = sp_sorted[np.searchsorted(cw, 0.25)]
        p75[i] = sp_sorted[np.searchsorted(cw, 0.75)]
        mean[i] = (w_sorted * sp_sorted).sum() / w_sorted.sum()
    return median, mean, p25, p75


def plot_side_by_side() -> None:
    sizes, sp_pool, sp_fair = load_joined()
    print(f"loaded {len(sizes)} joined rows  (size>0, finite spreads)")
    print(f"  size USD: median={np.median(sizes):,.0f}  p25={np.percentile(sizes,25):,.0f}  p75={np.percentile(sizes,75):,.0f}  max={sizes.max():,.0f}")
    print(f"  spread vs pool_mid_pre: mean={sp_pool.mean():.2f} bps  median={np.median(sp_pool):.2f} bps")
    print(f"  spread vs fair_price:   mean={sp_fair.mean():.2f} bps  median={np.median(sp_fair):.2f} bps")
    print(f"  USD-w mean(pool): {((sp_pool*sizes).sum()/sizes.sum()):.3f} bps")
    print(f"  USD-w mean(fair): {((sp_fair*sizes).sum()/sizes.sum()):.3f} bps")

    # Log-spaced bin edges; clip
    lo = max(10.0, sizes.min())
    hi = sizes.max() * 1.01
    bin_edges = np.logspace(math.log10(lo), math.log10(hi), 25)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    med_p, mean_p, p25_p, p75_p = usd_binned_stats(sizes, sp_pool, bin_edges)
    med_f, mean_f, p25_f, p75_f = usd_binned_stats(sizes, sp_fair, bin_edges)

    # === Plot 1: USD-weighted binned medians + IQR band, both refs ===
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xscale("log")
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)

    valid_p = ~np.isnan(med_p)
    valid_f = ~np.isnan(med_f)
    ax.plot(centers[valid_p], med_p[valid_p], "o-", color="C0", label="vs pool_mid_pre (V3, new)", lw=2)
    ax.fill_between(centers[valid_p], p25_p[valid_p], p75_p[valid_p], color="C0", alpha=0.15, label="IQR (pool)")
    ax.plot(centers[valid_f], med_f[valid_f], "s-", color="C3", label="vs fair_price (Binance, old)", lw=2)
    ax.fill_between(centers[valid_f], p25_f[valid_f], p75_f[valid_f], color="C3", alpha=0.15, label="IQR (fair)")

    ax.set_xlabel("Tx size (USD, log scale)")
    ax.set_ylabel("Observed spread (bps, signed)")
    ax.set_title(
        f"Non-5bp router-routed price impact: pool_mid_pre vs fair reference\n"
        f"7d window  |  V3 only  |  n={len(sizes):,} txs overlap  |  USD-volume-weighted bin medians + IQR"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left")
    plt.tight_layout()
    out = OUT_DIR / "impact_curve_pool_vs_fair.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")

    # === Plot 2: scatter (subsampled) ===
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True, sharey=True)
    rng = np.random.default_rng(0)
    n = len(sizes)
    keep = min(n, 6000)
    idx = rng.choice(n, keep, replace=False)

    for ax_i, sp, ref_name, color in [
        (axes[0], sp_pool, "pool_mid_pre", "C0"),
        (axes[1], sp_fair, "fair_price", "C3"),
    ]:
        ax_i.scatter(sizes[idx], sp[idx], s=4, alpha=0.18, color=color)
        ax_i.set_xscale("log")
        ax_i.axhline(0, color="black", lw=0.7, alpha=0.5)
        med = med_p if ref_name == "pool_mid_pre" else med_f
        valid = ~np.isnan(med)
        ax_i.plot(centers[valid], med[valid], "o-", color="black", lw=2, label="USD-w median")
        ax_i.set_xlabel("Tx size (USD)")
        ax_i.set_title(f"reference: {ref_name}")
        ax_i.legend(loc="upper left")
        ax_i.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("Observed spread (bps)")
    # Clip y range for readability — long tails can dominate
    p_lo, p_hi = np.percentile(np.concatenate([sp_pool, sp_fair]), [0.5, 99.5])
    for ax_i in axes:
        ax_i.set_ylim(p_lo - 2, p_hi + 2)
    fig2.suptitle(f"Per-tx observed spread vs size — pool vs fair reference ({len(sizes):,} txs)", y=1.02)
    plt.tight_layout()
    out2 = OUT_DIR / "impact_curve_pool_vs_fair_scatter.png"
    fig2.savefig(out2, dpi=130, bbox_inches="tight")
    print(f"wrote {out2}")

    # === Plot 3: per-tx diff (fair - pool) vs size — directly shows reference effect ===
    diff = sp_fair - sp_pool  # how much higher (more positive) fair-ref spread is
    fig3, ax3 = plt.subplots(figsize=(11, 5))
    ax3.set_xscale("log")
    ax3.axhline(0, color="black", lw=0.7)
    # binned median of diff
    med_d, _, _, _ = usd_binned_stats(sizes, diff, bin_edges)
    valid_d = ~np.isnan(med_d)
    keep = min(len(diff), 6000)
    idx = rng.choice(len(diff), keep, replace=False)
    ax3.scatter(sizes[idx], diff[idx], s=4, alpha=0.18, color="C4")
    ax3.plot(centers[valid_d], med_d[valid_d], "o-", color="black", lw=2, label="USD-w median (fair − pool)")
    ax3.set_xlabel("Tx size (USD)")
    ax3.set_ylabel("Spread_fair − Spread_pool  (bps)")
    ax3.set_title("Reference-choice effect on observed spread per tx")
    ax3.set_ylim(-30, 30)
    ax3.grid(True, which="both", alpha=0.3)
    ax3.legend()
    plt.tight_layout()
    out3 = OUT_DIR / "impact_curve_reference_diff.png"
    fig3.savefig(out3, dpi=130)
    print(f"wrote {out3}")

    # === Summary stats per bin ===
    print()
    print(f"{'bin_lo':>12} {'bin_hi':>12} {'n_tx':>6} {'med_pool':>10} {'med_fair':>10} {'fair-pool':>10}")
    for i in range(len(bin_edges) - 1):
        m = (sizes >= bin_edges[i]) & (sizes < bin_edges[i + 1])
        if m.sum() < 5:
            continue
        print(
            f"{bin_edges[i]:>12,.0f} {bin_edges[i+1]:>12,.0f} {m.sum():>6} "
            f"{med_p[i]:>10.2f} {med_f[i]:>10.2f} {med_f[i] - med_p[i]:>10.2f}"
        )


if __name__ == "__main__":
    plot_side_by_side()
