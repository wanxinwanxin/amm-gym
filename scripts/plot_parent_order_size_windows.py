"""Visualize router-parent USD-size distributions across 6m/1y/2y windows.

Reads analysis/weth_usdc_90d/router_parent_order_size_windows.csv and writes a
multi-panel PNG to analysis/weth_usdc_90d/plots/router_parent_order_sizes.png.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "analysis" / "weth_usdc_90d" / "router_parent_order_size_windows.csv"
OUT_PATH = ROOT / "analysis" / "weth_usdc_90d" / "plots" / "router_parent_order_sizes.png"


def load_curves() -> dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, int]]:
    """Return {(window, mode, side): (pct_array, size_usd_array, parent_count)}."""

    buckets: dict[tuple[str, str, str], list[tuple[float, float]]] = defaultdict(list)
    counts: dict[tuple[str, str, str], int] = {}
    with CSV_PATH.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (row["window_name"], row["mode"], row["side_group"])
            buckets[key].append((float(row["pct"]), float(row["size_usd"])))
            counts[key] = int(row["parent_count"])
    out: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, int]] = {}
    for key, points in buckets.items():
        points.sort()
        pct = np.array([p[0] for p in points], dtype=float)
        size = np.array([p[1] for p in points], dtype=float)
        out[key] = (pct, size, counts[key])
    return out


def _format_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def main() -> None:
    curves = load_curves()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    window_colors = {"6m": "#1f77b4", "1y": "#2ca02c", "2y": "#d62728"}

    # Panel 1: CDF (log-x), strict 'all', three windows
    ax = axes[0]
    for window in ("6m", "1y", "2y"):
        pct, size, n = curves[(window, "strict", "all")]
        positive = size > 0
        ax.plot(
            size[positive],
            pct[positive] / 100.0,
            label=f"{window} (n={_format_count(n)})",
            color=window_colors[window],
            linewidth=1.6,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Parent order size (USD, log scale)")
    ax.set_ylabel("CDF")
    ax.set_title("Router-parent CDF — strict, all sides")
    ax.set_xlim(1.0, 1e7)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    # Panel 2: Buy vs Sell, 6m strict, log-x CDF
    ax = axes[1]
    side_colors = {"buy_eth": "#ff7f0e", "sell_eth": "#9467bd", "all": "#7f7f7f"}
    for side in ("all", "buy_eth", "sell_eth"):
        pct, size, n = curves[("6m", "strict", side)]
        positive = size > 0
        ax.plot(
            size[positive],
            pct[positive] / 100.0,
            label=f"{side} (n={_format_count(n)})",
            color=side_colors[side],
            linewidth=1.6,
            linestyle="-" if side == "all" else "--",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Parent order size (USD, log scale)")
    ax.set_ylabel("CDF")
    ax.set_title("6m strict — buy vs sell")
    ax.set_xlim(1.0, 1e7)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    # Panel 3: Survival function (1 - CDF) on log-log axes — heavy-tail view
    ax = axes[2]
    for window in ("6m", "1y", "2y"):
        pct, size, n = curves[(window, "strict", "all")]
        survival = 1.0 - pct / 100.0
        keep = (size > 0) & (survival > 0)
        ax.plot(
            size[keep],
            survival[keep],
            label=f"{window}",
            color=window_colors[window],
            linewidth=1.6,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Parent order size (USD, log scale)")
    ax.set_ylabel("Survival 1 − CDF (log scale)")
    ax.set_title("Tail survival — strict, all sides")
    ax.set_xlim(10.0, 1e7)
    ax.set_ylim(1e-6, 1.0)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)

    fig.suptitle(
        "WETH/USDC router-parent order size distribution (USD, dex_trades aggregated to tx hash)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(OUT_PATH, dpi=140)

    # Quick numeric summary printed to stdout for the caller.
    print(f"Wrote {OUT_PATH.relative_to(ROOT)}")
    for window in ("6m", "1y", "2y"):
        pct, size, n = curves[(window, "strict", "all")]
        idx = {p: int(np.searchsorted(pct, p)) for p in (10.0, 50.0, 90.0, 99.0, 99.9)}
        print(
            f"  {window} strict/all  n={_format_count(n):>6}  "
            f"p10=${size[idx[10.0]]:>10,.0f}  "
            f"p50=${size[idx[50.0]]:>10,.0f}  "
            f"p90=${size[idx[90.0]]:>10,.0f}  "
            f"p99=${size[idx[99.0]]:>10,.0f}  "
            f"p99.9=${size[idx[99.9]]:>10,.0f}"
        )


if __name__ == "__main__":
    main()
