"""Render the figures used by `presentation/calibration_explainer.html`.

This script pulls the calibration artifacts from the `calibration-rework`
branch via `git show` so we don't have to vendor the JSON / CSV onto the
`html-cleanup` branch. Outputs land in `plots/`:

  1. `calibration_residuals_evolution.png` — residual evolution + loss across
     the 8 calibration cycles.
  2. `calibration_residuals_final.png` — bar chart of the final 6 residuals
     (3 metrics × 2 seed sets) against the ±2% target band.
  3. `calibration_pool_flow_splits.png` — empirical T1/T2 sources: USD-volume
     splits of arb vs retail flow across 5bp and other-fee pools.
  4. `calibration_markout_windows.png` — the +3.637 vs -1.05 retarget figure:
     USD-weighted markout vs window length, overlaid with the prior single-day
     reference line.

Run via:

    .venv/bin/python scripts/plot_calibration_explainer.py
"""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PLOTS = REPO / "plots"
PLOTS.mkdir(exist_ok=True)

# Palette pulled from `presentation/helpers.py` STYLE / RETAIL_STYLE so the
# explainer figures match the existing presentation deck.
COLOR_OBSERVED = "#2c3e50"   # on-chain / target
COLOR_CALIB = "#27ae60"      # calibration seeds (realistic green)
COLOR_HOLDOUT = "#e74c3c"    # held-out seeds (challenge red)
COLOR_LOSS = "#8e44ad"
COLOR_SIMPLE = "#7f8c8d"

CYCLES = [
    "cycle_01",
    "cycle_02",
    "cycle_03",
    "cycle_04",
    "cycle_05",
    "cycle_06",
    "cycle_07_de_fast",
    "cycle_08",
]


def git_show(path: str) -> bytes:
    """Read a blob from the `calibration-rework` branch without checking out."""
    return subprocess.check_output(
        ["git", "show", f"calibration-rework:{path}"], cwd=REPO,
    )


def load_cycle(stem: str) -> dict:
    return json.loads(git_show(f"calibration_artifacts/{stem}.json"))


def load_final() -> dict:
    return json.loads(git_show("calibration_final.json"))


def load_pool_flow_splits() -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(git_show("calibration_artifacts/pool_flow_splits.csv")))


def load_markout_windows() -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(git_show("reports/markout_windows.csv")))
    df = df[~df["window_days"].astype(str).str.startswith("max=")].copy()
    df["window_days"] = df["window_days"].astype(int)
    return df.sort_values("window_days").reset_index(drop=True)


def _apply_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=10)


def _residuals(cycle: dict) -> dict[str, float]:
    for key in ("final_residuals", "calib_residuals"):
        if key in cycle:
            return cycle[key]
    raise KeyError(f"no residuals dict in cycle {cycle.get('cycle')}")


def plot_residual_evolution() -> Path:
    """Plot residual + loss evolution for cycles 3-8 (post-T3-retarget).

    Cycles 1-2 used the prior single-day `+3.637 bps` T3 target which was later
    corrected to `-1.05 bps` (the 7d USD-weighted figure). They are excluded so
    the residuals are comparable across cycles. Cycles 6-7 are diagnostic
    excursions (coarse coordinate descent, fast differential evolution) that
    confirm no alternative basin beats Cycle 3's joint-NM minimum.
    """
    keep = ["cycle_03", "cycle_04", "cycle_05", "cycle_06", "cycle_07_de_fast", "cycle_08"]
    labels = ["3\nNM joint", "4\nbasin\nconfirm", "5\nweighted\nNM", "6\ncoord\nscan", "7\nDE\nfast", "8\nfinal\nNM"]
    cycles = [load_cycle(s) for s in keep]

    nums = list(range(len(cycles)))
    r1 = [_residuals(c)["T1_arb_5bp_share"] for c in cycles]
    r2 = [_residuals(c)["T2_retail_5bp_share"] for c in cycles]
    r3 = [_residuals(c)["T3_markout_bps"] for c in cycles]
    losses = [a * a + b * b + cc * cc for a, b, cc in zip(r1, r2, r3)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    ax = axes[0]
    ax.axhspan(-0.02, 0.02, color=COLOR_CALIB, alpha=0.12, label="±2% target band")
    ax.axhline(0, color="#666", lw=0.5)
    ax.plot(nums, r1, "o-", color=COLOR_HOLDOUT, lw=2, markersize=8, label="T1 arb 5bp share")
    ax.plot(nums, r2, "s-", color="#2980b9", lw=2, markersize=8, label="T2 retail 5bp share")
    ax.plot(nums, r3, "^-", color=COLOR_CALIB, lw=2, markersize=8, label="T3 markout (bps)")
    ax.set_xticks(nums)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Cycle", fontsize=11)
    ax.set_ylabel("Relative residual", fontsize=11)
    ax.set_title("Residual evolution — calibration seeds (post-T3-retarget)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:+.0f}%"))
    _apply_style(ax)

    ax = axes[1]
    ax.plot(nums, losses, "o-", color=COLOR_LOSS, lw=2, markersize=8)
    ax.set_xticks(nums)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Cycle", fontsize=11)
    ax.set_ylabel("Sum of squared relative residuals", fontsize=11)
    ax.set_title("Loss evolution (log scale)", fontsize=12, fontweight="bold")
    ax.set_yscale("log")
    _apply_style(ax)

    fig.tight_layout()
    out = PLOTS / "calibration_residuals_evolution.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")
    return out


def plot_final_residuals() -> Path:
    final = load_final()
    cal = final["calibration_residuals"]
    hol = final["holdout_residuals"]

    labels = ["T1\narb 5bp share", "T2\nretail 5bp share", "T3\nmarkout (bps)"]
    cal_vals = [
        cal["T1_arb_5bp_share"],
        cal["T2_retail_5bp_share"],
        cal["T3_markout_bps"],
    ]
    hol_vals = [
        hol["T1_arb_5bp_share"],
        hol["T2_retail_5bp_share"],
        hol["T3_markout_bps"],
    ]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.axhspan(-0.02, 0.02, color=COLOR_CALIB, alpha=0.12, label="±2% DONE target")
    ax.axhline(0, color="#666", lw=0.6)

    bars_c = ax.bar(
        x - width / 2, cal_vals, width,
        color=COLOR_CALIB, edgecolor="white",
        label="Calibration seeds (5)",
    )
    bars_h = ax.bar(
        x + width / 2, hol_vals, width,
        color=COLOR_HOLDOUT, edgecolor="white",
        label="Held-out seeds (5)",
    )

    for bars in (bars_c, bars_h):
        for b in bars:
            h = b.get_height()
            ax.annotate(
                f"{h*100:+.1f}%",
                (b.get_x() + b.get_width() / 2, h),
                xytext=(0, 4 if h >= 0 else -12),
                textcoords="offset points",
                ha="center", fontsize=9, color="#222",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Relative residual (sim − target) / target", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:+.0f}%"))
    ax.set_title(
        "Final calibration residuals — PARTIAL (within ±12%, above ±2% DONE)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10, frameon=False, loc="best")
    _apply_style(ax)
    ax.set_ylim(-0.15, 0.15)

    fig.tight_layout()
    out = PLOTS / "calibration_residuals_final.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")
    return out


def plot_pool_flow_splits() -> Path:
    df = load_pool_flow_splits()

    pivot = df.pivot(index="flow_kind", columns="pool_bucket", values="usd_volume").fillna(0.0)
    pivot = pivot[["5bp", "other"]]
    totals = pivot.sum(axis=1)
    share = pivot.div(totals, axis=0).loc[["arb", "retail"]]

    x = np.arange(len(share))
    width = 0.55

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(
        x, share["5bp"], width,
        color=COLOR_CALIB, edgecolor="white", label="5bp pool",
    )
    ax.bar(
        x, share["other"], width, bottom=share["5bp"],
        color=COLOR_OBSERVED, alpha=0.85, edgecolor="white", label="Other pools",
    )

    for i, kind in enumerate(share.index):
        s5 = float(share.loc[kind, "5bp"])
        ax.text(x[i], s5 / 2, f"{s5:.3f}",
                ha="center", va="center", color="white", fontsize=12, fontweight="bold")
        ax.text(x[i], s5 + (1 - s5) / 2, f"{1 - s5:.3f}",
                ha="center", va="center", color="white", fontsize=10)

    xlabels = [f"{k}\n(${totals.loc[k]/1e6:,.0f}M)" for k in share.index]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=11)
    ax.set_ylabel("USD-volume share", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_title(
        "Empirical T1 / T2 — USD-volume splits by flow kind",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), ncol=2)
    _apply_style(ax)

    fig.tight_layout()
    out = PLOTS / "calibration_pool_flow_splits.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")
    return out


def plot_markout_windows() -> Path:
    df = load_markout_windows()
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(
        x, df["usd_weighted_next_bps"], 0.6,
        color=COLOR_CALIB, edgecolor="white",
        label="USD-weighted markout (bps)",
    )

    for i, v in enumerate(df["usd_weighted_next_bps"].values):
        ax.annotate(
            f"{v:+.2f}",
            (x[i], v),
            xytext=(0, 4 if v >= 0 else -12),
            textcoords="offset points",
            ha="center", fontsize=9, color="#1c603c",
        )

    ax.axhline(
        3.637, color=COLOR_SIMPLE, lw=1.5, ls="--",
        label="Prior figure: +3.637 bps (single-day simple avg)",
    )
    ax.axhline(
        -1.05, color=COLOR_HOLDOUT, lw=1.5, ls=":",
        label="Calibration target T3: -1.05 bps (7d USD-weighted)",
    )
    ax.axhline(0, color="#555", lw=0.7, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}d" for d in df["window_days"]], fontsize=10)
    ax.set_xlabel("Window length ending 2026-05-19", fontsize=11)
    ax.set_ylabel("LP markout (bps)", fontsize=11)
    ax.set_title(
        "USD-weighted next-block markout vs window length — 5bp WETH/USDC",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), ncol=3)
    ax.set_ylim(-2.5, 4.5)
    _apply_style(ax)

    fig.tight_layout()
    out = PLOTS / "calibration_markout_windows.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")
    return out


def main() -> None:
    plot_residual_evolution()
    plot_final_residuals()
    plot_pool_flow_splits()
    plot_markout_windows()


if __name__ == "__main__":
    main()
