"""Roll up the per-day markout CSV into the requested windows ending on the
latest available date, and emit:

  1. reports/markout_windows.csv -- one row per window with USD-weighted bps,
     simple per-swap bps, n_swaps, total_usd, lp_pnl, and date span.
  2. reports/markout_by_window.png -- bar plot of the two markout series side
     by side with a horizontal reference line at +3.637 bps.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WINDOWS_DAYS = [7, 30, 90, 180, 360, 730]
QUOTED_TARGET_BPS = 3.637  # from analysis/weth_usdc_90d/markout_5bp_pool_summary.csv


def rollup(df: pd.DataFrame, end_date: pd.Timestamp, days: int) -> dict:
    start_date = end_date - pd.Timedelta(days=days - 1)
    sub = df[(df["block_date"] >= start_date) & (df["block_date"] <= end_date)]
    if sub.empty:
        return {
            "window_days": days,
            "start_date": None,
            "end_date": end_date.date().isoformat(),
            "actual_days_present": 0,
            "n_swaps": 0,
            "total_usd": 0.0,
            "usd_weighted_next_bps": float("nan"),
            "simple_avg_next_bps": float("nan"),
            "usd_weighted_15s_bps": float("nan"),
            "simple_avg_15s_bps": float("nan"),
            "lp_pnl_next_dollar": 0.0,
            "lp_pnl_15s_dollar": 0.0,
        }
    n_swaps = int(sub["n_swaps"].sum())
    total_usd = float(sub["total_usd"].sum())
    lp_pnl_next = float(sub["lp_pnl_next_dollar"].sum())
    lp_pnl_15s = float(sub["lp_pnl_15s_dollar"].sum())
    # USD-weighted = SUM(markout_dollar) / SUM(usd_amount) * 1e4
    usd_w_next_bps = lp_pnl_next / total_usd * 1e4 if total_usd > 0 else float("nan")
    usd_w_15s_bps = lp_pnl_15s / total_usd * 1e4 if total_usd > 0 else float("nan")
    # Simple per-swap average across days, weighted by swap count to mimic
    # AVG(markout_next) over all swaps in the window.
    simple_next = (
        float((sub["simple_avg_next_bps"] * sub["n_swaps"]).sum() / sub["n_swaps"].sum())
        if sub["n_swaps"].sum() > 0
        else float("nan")
    )
    simple_15s = (
        float((sub["simple_avg_15s_bps"] * sub["n_swaps"]).sum() / sub["n_swaps"].sum())
        if sub["n_swaps"].sum() > 0
        else float("nan")
    )
    return {
        "window_days": days,
        "start_date": start_date.date().isoformat(),
        "end_date": end_date.date().isoformat(),
        "actual_days_present": int(len(sub)),
        "n_swaps": n_swaps,
        "total_usd": total_usd,
        "usd_weighted_next_bps": usd_w_next_bps,
        "simple_avg_next_bps": simple_next,
        "usd_weighted_15s_bps": usd_w_15s_bps,
        "simple_avg_15s_bps": simple_15s,
        "lp_pnl_next_dollar": lp_pnl_next,
        "lp_pnl_15s_dollar": lp_pnl_15s,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="reports/markout_daily_5bp_pool.csv")
    ap.add_argument("--out-csv", default="reports/markout_windows.csv")
    ap.add_argument("--out-png", default="reports/markout_by_window.png")
    ap.add_argument(
        "--end-date",
        default=None,
        help="Override end date (ISO). Default = max date in input.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    df["block_date"] = pd.to_datetime(df["block_date"])
    end_date = (
        pd.to_datetime(args.end_date)
        if args.end_date is not None
        else df["block_date"].max()
    )

    rows = []
    for d in WINDOWS_DAYS:
        rows.append(rollup(df, end_date, d))

    # Include an "all-available" max window for context
    full_days = (end_date - df["block_date"].min()).days + 1
    rows.append(rollup(df, end_date, full_days))
    rows[-1]["window_days"] = f"max={full_days}"

    out_df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(out_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    plot_rows = out_df.iloc[:-1]  # drop "max" label for the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(plot_rows))
    width = 0.35
    ax.bar(
        x - width / 2,
        plot_rows["usd_weighted_next_bps"],
        width,
        label="USD-volume-weighted",
        color="#c0392b",
    )
    ax.bar(
        x + width / 2,
        plot_rows["simple_avg_next_bps"],
        width,
        label="Simple per-swap avg",
        color="#2980b9",
    )
    ax.axhline(
        QUOTED_TARGET_BPS,
        color="grey",
        linestyle="--",
        linewidth=1.2,
        label=f"Repo target {QUOTED_TARGET_BPS} bps (single-day May-7 simple avg)",
    )
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}d" for d in plot_rows["window_days"]])
    ax.set_ylabel("LP markout (bps, next-block / next-swap basis)")
    ax.set_xlabel("Trailing window ending " + end_date.date().isoformat())
    ax.set_title("Uniswap V3 5bp WETH/USDC pool: LP markout by horizon")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, linestyle=":")
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    print(f"\nWrote: {args.out_csv}\nWrote: {args.out_png}")


if __name__ == "__main__":
    main()
