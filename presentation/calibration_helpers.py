"""Helpers for the three-target calibration presentation notebook.

Loaders + plotters specific to `presentation/calibration_three_targets.ipynb`.
Style/palette is shared with `helpers.py` (same `_apply_style`, same observed /
realistic / challenge colors).

Sections covered:
- Section 2: load + plot pool-flow splits (T1, T2 empirical sources).
- Section 3: load + plot the markout size-bucket decomposition + window-length
  decomposition. This is where the +3.637 -> -1.05 retarget is justified.
- Section 4: load calibration_final.json + cycle_*.json artifacts, render the
  residual table, the sim-vs-target bar chart, and the residual evolution
  plot across cycles.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from presentation.helpers import STYLE, _apply_style

# ---------------------------------------------------------------------------
# Paths (relative to this file)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = REPO_ROOT / "calibration_artifacts"
REPORTS_DIR = REPO_ROOT / "reports"
ANALYSIS_DIR = REPO_ROOT / "analysis" / "weth_usdc_90d"
CALIBRATION_FINAL = REPO_ROOT / "calibration_final.json"

# ---------------------------------------------------------------------------
# Local palette: derived from helpers.STYLE so the look stays consistent.
# ---------------------------------------------------------------------------
CALIB_STYLE = {
    "target": {"color": STYLE["observed"]["color"], "label": "Target (on-chain)"},
    "calibration": {"color": STYLE["realistic"]["color"], "label": "Sim — calibration seeds"},
    "holdout": {"color": STYLE["challenge"]["color"], "label": "Sim — held-out seeds"},
    "simple": {"color": "#7f8c8d", "label": "Simple per-swap average"},
    "weighted": {"color": STYLE["realistic"]["color"], "label": "USD-volume-weighted"},
}

# The targets, kept in one place for use across cells.
TARGETS = {
    "T1_arb_5bp_share": 0.337330,
    "T2_retail_5bp_share": 0.782049,
    "T3_markout_bps": -1.05,
}

TARGET_LABELS = {
    "T1_arb_5bp_share": "T1 · arb 5bp share",
    "T2_retail_5bp_share": "T2 · retail 5bp share",
    "T3_markout_bps": "T3 · markout (bps)",
}


# ===================================================================
# Section 2: empirical pool-flow splits (T1, T2 sources)
# ===================================================================

def load_pool_flow_splits() -> pd.DataFrame:
    """Load the BigQuery-derived arb/retail × 5bp/other USD-volume splits.

    Schema: flow_kind, pool_bucket, swap_count, usd_volume.
    """
    return pd.read_csv(ARTIFACTS_DIR / "pool_flow_splits.csv")


def plot_pool_flow_splits(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Stacked bar chart: USD-volume share by pool bucket for each flow type.

    T1 is the arb row's 5bp share; T2 is the retail row's 5bp share.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))

    pivot = df.pivot(index="flow_kind", columns="pool_bucket", values="usd_volume").fillna(0.0)
    pivot = pivot[["5bp", "other"]]  # column order
    totals = pivot.sum(axis=1)
    share = pivot.div(totals, axis=0)
    share = share.loc[["arb", "retail"]]  # row order

    x = np.arange(len(share))
    width = 0.6

    bars_5bp = ax.bar(
        x, share["5bp"], width,
        color=STYLE["realistic"]["color"], label="5bp pool", edgecolor="white",
    )
    bars_other = ax.bar(
        x, share["other"], width, bottom=share["5bp"],
        color=STYLE["observed"]["color"], alpha=0.85, label="Other pools", edgecolor="white",
    )

    for i, kind in enumerate(share.index):
        sh = float(share.loc[kind, "5bp"])
        ax.text(
            x[i], sh / 2, f"{sh:.3f}",
            ha="center", va="center", color="white", fontsize=11, fontweight="bold",
        )
        ax.text(
            x[i], sh + (1 - sh) / 2, f"{1 - sh:.3f}",
            ha="center", va="center", color="white", fontsize=10,
        )

    labels = [f"{k}\n(${totals.loc[k]/1e6:,.0f}M)" for k in share.index]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("USD-volume share", fontsize=11)
    ax.set_ylim(0, 1.02)
    ax.set_title(
        "Empirical T1 / T2 sources — USD volume splits\n"
        "T1 = arb's 5bp share · T2 = retail's 5bp share",
        fontsize=12, fontweight="bold",
    )
    _apply_style(ax)
    return ax


# ===================================================================
# Section 3: the +3.637 -> -1.05 retarget (the centerpiece)
# ===================================================================

def load_size_bucket_markout() -> pd.DataFrame:
    """Per-size-bucket markout decomposition (30d window).

    Schema: size_bucket, n_swaps, total_usd, simple_avg_bps, usd_weighted_bps,
    lp_pnl_usd.
    """
    df = pd.read_csv(REPORTS_DIR / "markout_by_size_bucket_30d.csv")
    return df.sort_values("size_bucket").reset_index(drop=True)


SIZE_BUCKET_LABELS = {
    "0_lt_100": "< $100",
    "1_100_1k": "$100 – $1k",
    "2_1k_10k": "$1k – $10k",
    "3_10k_100k": "$10k – $100k",
    "4_100k_1M": "$100k – $1M",
    "5_gte_1M": ">= $1M",
}


def plot_size_bucket_markout(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Grouped bar chart: simple per-swap avg vs USD-weighted markout by size bucket.

    The whale flip (sub-$10k buckets positive; $100k+ buckets negative on a
    USD-weighted basis) is the visual story.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4.5))

    df = df.copy()
    labels = [SIZE_BUCKET_LABELS.get(b, b) for b in df["size_bucket"]]
    x = np.arange(len(df))
    width = 0.38

    simple = df["simple_avg_bps"].values
    weighted = df["usd_weighted_bps"].values

    ax.bar(
        x - width / 2, simple, width,
        color=CALIB_STYLE["simple"]["color"], label="Simple per-swap avg (bps)",
        edgecolor="white",
    )
    ax.bar(
        x + width / 2, weighted, width,
        color=CALIB_STYLE["weighted"]["color"], label="USD-weighted (bps)",
        edgecolor="white",
    )

    for i, v in enumerate(simple):
        ax.annotate(
            f"{v:+.2f}",
            (x[i] - width / 2, v),
            xytext=(0, 4 if v >= 0 else -10),
            textcoords="offset points",
            ha="center", fontsize=8, color="#555",
        )
    for i, v in enumerate(weighted):
        ax.annotate(
            f"{v:+.2f}",
            (x[i] + width / 2, v),
            xytext=(0, 4 if v >= 0 else -10),
            textcoords="offset points",
            ha="center", fontsize=8, color="#1c603c",
        )

    ax.axhline(0, color="#555", lw=0.7, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15)
    ax.set_xlabel("Parent-order USD bucket", fontsize=11)
    ax.set_ylabel("LP markout (bps)", fontsize=11)
    ax.set_title(
        "Size-bucket markout decomposition (30d, 5bp pool)\n"
        "Whales flip the sign once we weight by USD volume",
        fontsize=12, fontweight="bold",
    )
    _apply_style(ax)
    return ax


def load_markout_windows() -> pd.DataFrame:
    """Per-window USD-weighted markout (7d, 30d, 90d, 180d, 360d, 730d).

    Drops the `max=504` aggregate row to keep the bar chart clean.
    """
    df = pd.read_csv(REPORTS_DIR / "markout_windows.csv")
    df = df[~df["window_days"].astype(str).str.startswith("max=")].copy()
    df["window_days"] = df["window_days"].astype(int)
    return df.sort_values("window_days").reset_index(drop=True)


def plot_markout_windows(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Bar chart of USD-weighted markout vs window length.

    Overlays the misleading +3.637 reference line so the discrepancy is visible.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))

    x = np.arange(len(df))
    bars = ax.bar(
        x, df["usd_weighted_next_bps"], 0.6,
        color=CALIB_STYLE["weighted"]["color"], edgecolor="white",
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
        3.637, color=CALIB_STYLE["simple"]["color"], lw=1.5, ls="--",
        label="Prior figure: +3.637 bps (single-day simple avg, May 7 2026)",
    )
    ax.axhline(
        -1.05, color=STYLE["challenge"]["color"], lw=1.5, ls=":",
        label="Calibration target T3: -1.05 bps (7d USD-weighted)",
    )
    ax.axhline(0, color="#555", lw=0.7, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}d" for d in df["window_days"]], fontsize=10)
    ax.set_xlabel("Window length (days, ending 2026-05-19)", fontsize=11)
    ax.set_ylabel("LP markout (bps)", fontsize=11)
    ax.set_title(
        "USD-weighted next-block markout vs window length (5bp pool)",
        fontsize=12, fontweight="bold",
    )
    _apply_style(ax)
    return ax


# ===================================================================
# Section 4: calibration results
# ===================================================================

def load_final() -> dict:
    """Load calibration_final.json."""
    return json.loads(CALIBRATION_FINAL.read_text())


def final_params_table(final: dict) -> pd.DataFrame:
    """Tidy display table of the final parameters."""
    p = final["params"]
    rows = [
        ("submission_depth_y", f"{p['submission_depth_y']:,.0f}", "USDC (5bp pool virtual reserve)"),
        ("normalizer_fee",     f"{p['normalizer_fee']:.6f}",      f"≈ {p['normalizer_fee']*1e4:.1f} bps"),
        ("normalizer_depth_y", f"{p['normalizer_depth_y']:,.0f}", "USDC (aggregated non-5bp pools)"),
        ("submission_fee (fixed)", f"{0.0005:.4f}", "5 bps — pool's posted fee"),
    ]
    return pd.DataFrame(rows, columns=["parameter", "value", "notes"])


def residuals_dataframe(final: dict) -> pd.DataFrame:
    """Long-form residuals table: 3 metrics × 2 seed sets.

    Columns: target_id, label, target, sim_calibration, residual_calib_pct,
    sim_holdout, residual_holdout_pct.
    """
    rows = []
    cal_m = final["calibration_metrics"]
    cal_r = final["calibration_residuals"]
    hol_m = final["holdout_metrics"]
    hol_r = final["holdout_residuals"]

    name_map = {
        "T1_arb_5bp_share": "arb_5bp_share",
        "T2_retail_5bp_share": "retail_5bp_share",
        "T3_markout_bps": "markout_bps",
    }
    for tid, mname in name_map.items():
        rows.append({
            "target_id": tid,
            "label": TARGET_LABELS[tid],
            "target": final["targets"][tid.split("_", 1)[0]],
            "sim_calibration": cal_m[mname],
            "residual_calib_pct": cal_r[tid] * 100,
            "sim_holdout": hol_m[mname],
            "residual_holdout_pct": hol_r[tid] * 100,
        })
    return pd.DataFrame(rows)


def plot_sim_vs_target(final: dict, ax: plt.Axes | None = None) -> plt.Axes:
    """Per-target grouped bar chart: target + calibration + holdout, with residual annotations.

    Each metric is rendered on its own subplot so the y-axes don't conflict
    (T1/T2 are shares ~0–1, T3 is a small negative bps figure).
    """
    df = residuals_dataframe(final)
    n = len(df)

    if ax is None:
        fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2))
    else:
        # Allow caller to pass a single axis; we still build subplots.
        fig = ax.figure
        axes = fig.subplots(1, n)

    for ax_i, (_, row) in zip(axes, df.iterrows()):
        labels = ["target", "calibration", "held-out"]
        values = [row["target"], row["sim_calibration"], row["sim_holdout"]]
        colors = [
            CALIB_STYLE["target"]["color"],
            CALIB_STYLE["calibration"]["color"],
            CALIB_STYLE["holdout"]["color"],
        ]
        bars = ax_i.bar(labels, values, color=colors, width=0.6, edgecolor="white")

        # Residual annotations above each non-target bar.
        ax_i.annotate(
            "—",
            (0, values[0]),
            xytext=(0, 4 if values[0] >= 0 else -12),
            textcoords="offset points",
            ha="center", fontsize=9, color="#555",
        )
        for j, (resid, v) in enumerate(zip(
            [row["residual_calib_pct"], row["residual_holdout_pct"]],
            values[1:],
        )):
            ax_i.annotate(
                f"{resid:+.2f}%",
                (j + 1, v),
                xytext=(0, 4 if v >= 0 else -12),
                textcoords="offset points",
                ha="center", fontsize=9,
                color="#1c603c" if abs(resid) <= 15 else STYLE["challenge"]["color"],
            )

        ax_i.axhline(0, color="#555", lw=0.5, alpha=0.5)
        ax_i.set_title(row["label"], fontsize=11, fontweight="bold")
        ax_i.tick_params(labelsize=9)
        ax_i.spines["top"].set_visible(False)
        ax_i.spines["right"].set_visible(False)

    fig.suptitle(
        "Calibrated simulator vs target (5 calibration seeds, 5 held-out seeds)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    return axes


def load_cycle_history() -> pd.DataFrame:
    """Walk all `cycle_*.json` artifacts and pull final per-cycle residuals.

    Returns long-form: cycle, seed_set in {calibration, holdout}, target_id,
    residual_pct. Cycles without holdout records (cycle_01) only contribute
    calibration rows.
    """
    cycle_files = sorted(ARTIFACTS_DIR.glob("cycle_*.json"))
    rows = []
    for p in cycle_files:
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        # Skip the auxiliary DE-fast cycle (no proper residuals payload).
        if "de_fast" in p.stem:
            continue
        # The schemas differ across cycles; harmonise.
        calib = d.get("calib_residuals") or d.get("final_residuals")
        hold = d.get("holdout_residuals")
        # Cycle number — fall back to the filename if not stored.
        try:
            cyc = int(d.get("cycle", "".join(c for c in p.stem if c.isdigit())))
        except Exception:
            cyc = int("".join(c for c in p.stem if c.isdigit()) or 0)
        if calib:
            for tid, val in calib.items():
                rows.append({
                    "cycle": cyc, "seed_set": "calibration",
                    "target_id": tid, "residual_pct": abs(val) * 100,
                })
        if hold:
            for tid, val in hold.items():
                rows.append({
                    "cycle": cyc, "seed_set": "holdout",
                    "target_id": tid, "residual_pct": abs(val) * 100,
                })
    return pd.DataFrame(rows).sort_values(["cycle", "seed_set", "target_id"]).reset_index(drop=True)


def plot_cycle_residuals(history: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Line plot: |residual %| vs cycle, one line per (target × seed-set).

    Calibration = solid; held-out = dashed. Log-scale y so that the order-of-
    magnitude drop from cycle 6 (DE detour) back to cycle 8 is visible.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    target_color = {
        "T1_arb_5bp_share": "#1f77b4",
        "T2_retail_5bp_share": "#ff7f0e",
        "T3_markout_bps": "#2ca02c",
    }

    for tid, color in target_color.items():
        for seed_set, ls, marker in [("calibration", "-", "o"), ("holdout", "--", "s")]:
            sub = history[
                (history["target_id"] == tid) & (history["seed_set"] == seed_set)
            ].sort_values("cycle")
            if sub.empty:
                continue
            ax.plot(
                sub["cycle"], sub["residual_pct"].clip(lower=1e-3),
                ls=ls, marker=marker, color=color, markersize=6,
                label=f"{TARGET_LABELS[tid]} · {seed_set}",
                alpha=0.85,
            )

    ax.axhspan(0, 2, color="#27ae60", alpha=0.08)
    ax.axhline(2, color="#27ae60", lw=1.2, ls=":", label="2% tolerance (DONE threshold)")

    ax.set_xlabel("Cycle", fontsize=11)
    ax.set_ylabel("|Residual| (%)", fontsize=11)
    ax.set_yscale("log")
    ax.set_title(
        "Residual evolution across calibration cycles\n"
        "Cycle 6 was a DE detour; cycles 1–5 used the +3.637 prior; cycles 3,5,8 use the -1.05 retarget",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper right")
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


# ===================================================================
# Section 5: per-trade markout distribution overlay (validation)
# ===================================================================

def run_calibrated_markout(
    final: dict,
    seeds: tuple[int, ...] | None = None,
    n_steps: int = 5_000,
) -> dict:
    """Run the calibrated realistic sim and collect per-trade submission markouts.

    Thin wrapper around `presentation.helpers.run_calibrated_sim` that just
    threads in the final params. Returns the same dict that
    `plot_markout_comparison` / `plot_markout_qq` expect.
    """
    from presentation.helpers import run_calibrated_sim  # local import: avoids cycles

    if seeds is None:
        seeds = tuple(final.get("seeds_holdout") or final.get("seeds_calibration") or (42, 43, 44, 45, 46))
    p = final["params"]
    return run_calibrated_sim(
        normalizer_fee=p["normalizer_fee"],
        normalizer_depth_y=p["normalizer_depth_y"],
        submission_depth_y=p["submission_depth_y"],
        submission_fee=0.0005,
        n_steps=n_steps,
        seeds=seeds,
    )
