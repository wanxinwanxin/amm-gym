"""Helper functions for the realistic simulator presentation notebook."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ANALYSIS_DIR = Path(__file__).resolve().parent.parent / "analysis" / "weth_usdc_90d"

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
STYLE = {
    "challenge": {"color": "#e74c3c", "label": "Simple AMM Challenge (GBM)"},
    "observed": {"color": "#2c3e50", "label": "Observed (Binance 12s)"},
    "realistic": {"color": "#27ae60", "label": "Realistic Simulator"},
}

RETAIL_STYLE = {
    "challenge": {"color": "#e74c3c", "label": "Simple AMM Challenge (Lognormal)"},
    "observed": {"color": "#2c3e50", "label": "Observed (On-chain parent orders)"},
    "realistic": {"color": "#27ae60", "label": "Realistic Simulator"},
}


def _apply_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10, frameon=False)


# ===================================================================
# SECTION 1: Return Distributions
# ===================================================================

def load_binance_quantiles(window: str = "90d") -> pd.DataFrame:
    """Load pre-computed Binance 12-second return quantiles."""
    path = ANALYSIS_DIR / "binance_return_quantiles_multi_window.csv"
    df = pd.read_csv(path)
    return df[df["window_name"] == window][["pct", "ret_bps"]].reset_index(drop=True)


def generate_challenge_return_quantiles(
    sigma: float = 0.000945, n_samples: int = 1_000_000, seed: int = 42,
) -> pd.DataFrame:
    """Generate GBM return quantiles at the same percentile grid as Binance."""
    rng = np.random.default_rng(seed)
    log_returns = sigma * rng.standard_normal(n_samples)
    pct_grid = np.arange(0, 100.1, 0.1)
    quantiles_bps = np.percentile(log_returns, pct_grid) * 10_000
    return pd.DataFrame({"pct": pct_grid, "ret_bps": quantiles_bps})


def generate_realistic_return_quantiles(
    n_steps: int = 1_000_000, seed: int = 42,
) -> pd.DataFrame:
    """Generate regime-switching return quantiles."""
    from arena_eval.exact_simple_amm.dynamics import RegimeSwitchingReturnProcess

    proc = RegimeSwitchingReturnProcess(
        initial_price=100.0,
        invcdf_path=ANALYSIS_DIR / "regimes_invcdf.csv",
        trans_matrix_path=ANALYSIS_DIR / "regimes_transition_matrix.csv",
        start_regime=3,
        seed=seed,
    )
    log_returns = []
    prev = proc.current_price
    for _ in range(n_steps):
        new_price = proc.step()
        log_returns.append(math.log(new_price / prev))
        prev = new_price

    pct_grid = np.arange(0, 100.1, 0.1)
    quantiles_bps = np.percentile(log_returns, pct_grid) * 10_000
    return pd.DataFrame({"pct": pct_grid, "ret_bps": quantiles_bps})


def plot_return_hist_overlay(
    distributions: list[tuple[pd.DataFrame, str]],
    *,
    title: str = "",
    trim_pct: float = 0.5,
    bins: int = 80,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Overlay semi-transparent histograms for return distributions.

    Uses quantile data as pseudo-samples — works because the quantile
    grid is uniformly spaced in probability.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    for df, style_key in distributions:
        mask = (df["pct"] >= trim_pct) & (df["pct"] <= 100 - trim_pct)
        vals = df.loc[mask, "ret_bps"].values
        s = STYLE[style_key]
        ax.hist(
            vals, bins=bins, density=True,
            color=s["color"], alpha=0.45, label=s["label"],
            edgecolor="none",
        )

    ax.set_xlabel("12-second log-return (bps)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    _apply_style(ax)
    return ax


def plot_return_qq(
    df_x: pd.DataFrame,
    df_y: pd.DataFrame,
    x_style: str,
    y_style: str,
    *,
    title: str = "",
    trim_pct: float = 0.5,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """QQ plot comparing two return distributions (in bps).

    trim_pct: clip the extreme tails (0-trim_pct and 100-trim_pct percentiles)
    to keep the chart readable.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    mask_x = (df_x["pct"] >= trim_pct) & (df_x["pct"] <= 100 - trim_pct)
    mask_y = (df_y["pct"] >= trim_pct) & (df_y["pct"] <= 100 - trim_pct)
    x_vals = df_x.loc[mask_x, "ret_bps"].values
    y_vals = df_y.loc[mask_y, "ret_bps"].values

    # Interpolate to common grid if lengths differ
    n = min(len(x_vals), len(y_vals))
    if len(x_vals) != len(y_vals):
        common = np.linspace(0, 1, n)
        x_vals = np.interp(common, np.linspace(0, 1, len(x_vals)), x_vals)
        y_vals = np.interp(common, np.linspace(0, 1, len(y_vals)), y_vals)

    ax.scatter(x_vals, y_vals, s=4, alpha=0.5, color=STYLE[y_style]["color"])
    lo = min(x_vals.min(), y_vals.min())
    hi = max(x_vals.max(), y_vals.max())
    ax.plot([lo, hi], [lo, hi], ls="--", color="#999", lw=1, label="y = x")
    ax.set_xlabel(f"{STYLE[x_style]['label']} (bps)", fontsize=10)
    ax.set_ylabel(f"{STYLE[y_style]['label']} (bps)", fontsize=10)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    _apply_style(ax)
    return ax


def plot_return_cdf_overlay(
    distributions: list[tuple[pd.DataFrame, str]],
    *,
    title: str = "",
    trim_pct: float = 0.5,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Overlay CDF curves for multiple return distributions.

    distributions: list of (dataframe, style_key) tuples.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    for df, style_key in distributions:
        mask = (df["pct"] >= trim_pct) & (df["pct"] <= 100 - trim_pct)
        s = STYLE[style_key]
        ax.plot(
            df.loc[mask, "ret_bps"],
            df.loc[mask, "pct"] / 100,
            color=s["color"],
            label=s["label"],
            lw=2,
            alpha=0.85,
        )

    ax.set_xlabel("12-second log-return (bps)", fontsize=11)
    ax.set_ylabel("CDF", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    _apply_style(ax)
    return ax


def plot_return_robustness(
    windows: list[str] = ("90d", "180d", "360d"),
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Show Binance return CDFs across different lookback windows."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    colors = {"90d": "#2c3e50", "180d": "#2980b9", "360d": "#8e44ad"}
    for w in windows:
        df = load_binance_quantiles(w)
        mask = (df["pct"] >= 0.5) & (df["pct"] <= 99.5)
        ax.plot(
            df.loc[mask, "ret_bps"], df.loc[mask, "pct"] / 100,
            color=colors.get(w, "#333"), label=f"Binance {w}", lw=2, alpha=0.85,
        )

    ax.set_xlabel("12-second log-return (bps)", fontsize=11)
    ax.set_ylabel("CDF", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title("Binance Return Distributions — Robustness Across Windows", fontsize=12, fontweight="bold")
    _apply_style(ax)
    return ax


# ===================================================================
# SECTION 2: Retail Order Size Distributions
# ===================================================================

def load_parent_order_quantiles(
    window: str = "6m", mode: str = "strict", side: str = "all",
) -> pd.DataFrame:
    """Load pre-computed parent-order USD size quantiles from BigQuery data."""
    path = ANALYSIS_DIR / "router_parent_order_size_windows.csv"
    df = pd.read_csv(path)
    sel = df[
        (df["window_name"] == window)
        & (df["mode"] == mode)
        & (df["side_group"] == side)
    ][["pct", "size_usd", "parent_count"]].reset_index(drop=True)
    return sel


def generate_challenge_retail_quantiles(
    mean_size: float = 20.0,
    size_sigma: float = 1.2,
    n_samples: int = 1_000_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate lognormal retail order size quantiles (challenge mode)."""
    rng = np.random.default_rng(seed)
    ln_mu = math.log(mean_size) - 0.5 * size_sigma ** 2
    sizes = rng.lognormal(ln_mu, size_sigma, size=n_samples)
    pct_grid = np.arange(0, 100.1, 0.1)
    quantiles = np.percentile(sizes, pct_grid)
    return pd.DataFrame({"pct": pct_grid, "size_usd": quantiles})


def generate_realistic_retail_quantiles(
    n_steps: int = 1_000_000, seed: int = 42,
) -> pd.DataFrame:
    """Generate empirical USD-size retail order sizes from the realistic simulator."""
    from arena_eval.exact_simple_amm.dynamics import EmpiricalUSDSizeRetailTrader

    trader = EmpiricalUSDSizeRetailTrader(
        arrival_rate=857_035 / 1_303_200,  # cleaned 6m parent-order rate
        usd_quantiles_path=ANALYSIS_DIR / "parent_order_usd_quantiles.csv",
        buy_prob=0.4413,
        seed=seed,
    )
    sizes = []
    for _ in range(n_steps):
        orders = trader.generate_orders(fair_price=100.0)
        for o in orders:
            sizes.append(o.size)

    pct_grid = np.arange(0, 100.1, 0.1)
    quantiles = np.percentile(sizes, pct_grid)
    return pd.DataFrame({"pct": pct_grid, "size_usd": quantiles})


def get_arrival_rate_comparison() -> pd.DataFrame:
    """Return a summary table comparing arrival rates across models."""
    # Observed: from parent order data across all ETH/USDC pools
    parent_df = pd.read_csv(ANALYSIS_DIR / "router_parent_order_size_windows.csv")
    rows = []
    for w in ["6m", "1y", "2y"]:
        sel = parent_df[
            (parent_df["window_name"] == w)
            & (parent_df["mode"] == "strict")
            & (parent_df["side_group"] == "all")
        ].iloc[0]
        days = sel["horizon_days"]
        blocks = days * 24 * 3600 / 12
        rate = sel["parent_count"] / blocks
        rows.append({"source": f"Observed ({w})", "rate_per_block": rate, "orders": int(sel["parent_count"]), "blocks": int(blocks)})
    rows.append({"source": "Challenge Model", "rate_per_block": 0.8, "orders": None, "blocks": None})
    rows.append({"source": "Realistic Simulator (single-pool)", "rate_per_block": 186_085 / 645_123, "orders": None, "blocks": None})
    rows.append({"source": "Realistic Simulator (cross-pool)", "rate_per_block": 857_035 / 1_303_200, "orders": None, "blocks": None})
    return pd.DataFrame(rows)


def plot_arrival_rate_comparison(ax: plt.Axes | None = None) -> plt.Axes:
    """Bar chart comparing arrival rates."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    df = get_arrival_rate_comparison()
    labels = df["source"].values
    rates = df["rate_per_block"].values
    colors = ["#2c3e50", "#34495e", "#7f8c8d",  # observed windows
              "#e74c3c",  # challenge
              "#e67e22",  # single-pool
              "#27ae60"]  # cross-pool

    bars = ax.barh(range(len(labels)), rates, color=colors[:len(labels)])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Arrival rate (parent orders per 12s block)", fontsize=11)
    ax.set_title("Retail Order Arrival Rates", fontsize=12, fontweight="bold")
    for i, (bar, rate) in enumerate(zip(bars, rates)):
        ax.text(rate + 0.02, i, f"{rate:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, max(rates) * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()
    return ax


def plot_retail_hist_overlay(
    distributions: list[tuple[pd.DataFrame, str]],
    *,
    title: str = "",
    log_x: bool = True,
    trim_pct: float = 1.0,
    bins: int = 60,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Overlay semi-transparent histograms for retail order size distributions."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    for df, style_key in distributions:
        mask = (df["pct"] >= trim_pct) & (df["pct"] <= 100 - trim_pct)
        vals = df.loc[mask, "size_usd"].values
        vals = vals[vals > 0]  # drop zeros for log scale
        s = RETAIL_STYLE[style_key]
        if log_x:
            log_vals = np.log10(vals)
            ax.hist(
                log_vals, bins=bins, density=True,
                color=s["color"], alpha=0.45, label=s["label"],
                edgecolor="none",
            )
        else:
            ax.hist(
                vals, bins=bins, density=True,
                color=s["color"], alpha=0.45, label=s["label"],
                edgecolor="none",
            )

    if log_x:
        ax.set_xlabel("Order size (log₁₀ USD)", fontsize=11)
    else:
        ax.set_xlabel("Order size (USD)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    _apply_style(ax)
    return ax


def plot_retail_cdf_overlay(
    distributions: list[tuple[pd.DataFrame, str]],
    *,
    title: str = "",
    log_x: bool = True,
    trim_pct: float = 0.5,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Overlay CDF curves for retail order size distributions."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    for df, style_key in distributions:
        mask = (df["pct"] >= trim_pct) & (df["pct"] <= 100 - trim_pct)
        s = RETAIL_STYLE[style_key]
        ax.plot(
            df.loc[mask, "size_usd"],
            df.loc[mask, "pct"] / 100,
            color=s["color"],
            label=s["label"],
            lw=2,
            alpha=0.85,
        )

    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("Order size (USD)", fontsize=11)
    ax.set_ylabel("CDF", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    _apply_style(ax)
    return ax


def plot_retail_qq(
    df_x: pd.DataFrame,
    df_y: pd.DataFrame,
    x_style: str,
    y_style: str,
    *,
    title: str = "",
    log_scale: bool = True,
    trim_pct: float = 1.0,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """QQ plot comparing two retail order size distributions."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    mask_x = (df_x["pct"] >= trim_pct) & (df_x["pct"] <= 100 - trim_pct)
    mask_y = (df_y["pct"] >= trim_pct) & (df_y["pct"] <= 100 - trim_pct)
    x_vals = df_x.loc[mask_x, "size_usd"].values
    y_vals = df_y.loc[mask_y, "size_usd"].values

    n = min(len(x_vals), len(y_vals))
    if len(x_vals) != len(y_vals):
        common = np.linspace(0, 1, n)
        x_vals = np.interp(common, np.linspace(0, 1, len(x_vals)), x_vals)
        y_vals = np.interp(common, np.linspace(0, 1, len(y_vals)), y_vals)

    ax.scatter(x_vals, y_vals, s=4, alpha=0.5, color=RETAIL_STYLE[y_style]["color"])
    lo = min(x_vals.min(), y_vals.min())
    hi = max(x_vals.max(), y_vals.max())
    ax.plot([lo, hi], [lo, hi], ls="--", color="#999", lw=1, label="y = x")
    ax.set_xlabel(f"{RETAIL_STYLE[x_style]['label']} (USD)", fontsize=10)
    ax.set_ylabel(f"{RETAIL_STYLE[y_style]['label']} (USD)", fontsize=10)
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    _apply_style(ax)
    return ax




# ===================================================================
# SECTION 3: Two-Pool Architecture & Impact-Curve Calibration
# ===================================================================

# Frozen submission-pool parameters: real Uniswap V3 5bp WETH/USDC on-chain
# (sqrt_price_x96 × liquidity_L → virtual depths, May 2026).
REAL_POOL_FEE = 0.0005  # 5 bps
REAL_VIRTUAL_USDC = 212_157_626.44


def load_impact_curve_sample() -> pd.DataFrame:
    """Empirical router-routed non-5bp WETH/USDC impact sample
    (size, spread vs pool_mid_pre). 7d window, V3 only."""
    path = ANALYSIS_DIR / "non5bp_impact_sample_v3_pool_mid_7d.csv"
    return pd.read_csv(path)


def load_impact_curve_fit() -> dict:
    """Plan A and Plan B V2 fits of the normalizer pool."""
    import json
    return json.loads((ANALYSIS_DIR / "impact_curve_fit_pool_mid.json").read_text())


def load_validation() -> dict:
    """Validation payload from validate_pool_mid.py."""
    import json
    return json.loads((ANALYSIS_DIR / "validation_retail_pool_mid.json").read_text())


def load_real_retail_markouts() -> pd.DataFrame:
    """Per-swap retail markouts on the 5bp pool (real on-chain, 6 days)."""
    return pd.read_csv(ANALYSIS_DIR / "markout_5bp_pool_retail.csv")


def load_sim_retail_markouts() -> pd.DataFrame:
    """Per-trade retail markouts on the 5bp submission pool (simulator)."""
    return pd.read_csv(ANALYSIS_DIR / "markout_5bp_pool_sim_retail.csv")


def v2_spread_bps(size_usd: np.ndarray, phi: float, depth_usdc: float) -> np.ndarray:
    """V2 constant-product impact + fee model in spread-bps space.
    spread = (phi + (1-phi) S/D) / (1-phi)  [retail-side, pool-frame]."""
    S = np.asarray(size_usd, dtype=np.float64)
    return ((phi + (1.0 - phi) * S / depth_usdc) / (1.0 - phi)) * 1e4


def plot_impact_curve_fit(ax: plt.Axes | None = None,
                          plan_key: str = "plan_b",
                          n_bins: int = 30) -> plt.Axes:
    """Empirical impact cloud (USD-weighted binned median) overlaid with the
    fitted V2 curve. Spread is referenced to the lagged (pre-trade) fair price —
    the Binance mid one 12s step before each trade."""
    sample = load_impact_curve_sample()
    fit = load_impact_curve_fit()
    plan = fit[plan_key]
    phi = plan["phi"]; depth = plan["depth_usdc"]

    size = sample["size_usd"].to_numpy()
    spread = sample["observed_spread_fair_lag_bps"].to_numpy()

    # Flag sandwich / transient-MEV-excursion victims: trades that filled against a
    # pool whose PRE-TRADE mid was already pushed off fair (a separate frontrun tx
    # earlier in the block moved the pool; a backrun restores it within the block).
    # Signature: pre-trade pool mid >0.5% from the contemporaneous fair, while the
    # fill sat ~on that pushed mid (|pool-spread|<20 bps => a victim, not a
    # self-mover). These ~0.7% of trades are MEV slippage, not the pool's
    # mechanical impact, which is why they sit above the V2 curve. Visual call-out
    # only — the fit (loaded from the JSON) and the binned-median curve are unchanged.
    pool_mid = sample["pool_mid_pre_blended"].to_numpy()
    fair_ct = sample["fair_price_blended"].to_numpy()
    pool_spread = sample["observed_spread_pool_bps"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        pushed = np.abs(pool_mid / fair_ct - 1.0)
    sandwiched = (pushed > 0.005) & (np.abs(pool_spread) < 20.0)

    # USD-weighted log-binned medians (each bin's median spread, weight = size_usd)
    s_clip = np.clip(size, 1.0, np.inf)
    edges = np.logspace(np.log10(max(s_clip.min(), 10)),
                        np.log10(s_clip.max()), n_bins + 1)
    centers = []
    medians = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s_clip >= lo) & (s_clip < hi)
        if m.sum() < 3:
            continue
        centers.append(np.exp(0.5 * (np.log(lo) + np.log(hi))))
        medians.append(np.median(spread[m]))
    centers = np.array(centers); medians = np.array(medians)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(size[~sandwiched], spread[~sandwiched], s=3, color="#bdc3c7", alpha=0.25,
               label="empirical (per-tx)")
    ax.scatter(size[sandwiched], spread[sandwiched], s=12, color="#e74c3c", alpha=0.75,
               edgecolor="none", zorder=5,
               label=f"sandwiched — pool pushed off fair (n={int(sandwiched.sum())})")
    ax.plot(centers, medians, "o-", color="#2c3e50", lw=2, label="empirical (binned median)")

    sizes_fit = np.logspace(2, np.log10(size.max()), 200)
    ax.plot(sizes_fit, v2_spread_bps(sizes_fit, phi, depth),
            "-", color="#27ae60", lw=2.5,
            label=f"V2 fit ({plan_key}): φ={phi*1e4:.2f} bps, D=${depth/1e6:.1f}M")

    ax.set_xscale("log")
    ax.set_xlabel("Trade size (USD)")
    ax.set_ylabel("Spread vs lagged fair (bps)")
    ax.set_title("Non-5bp retail impact curve — empirical vs V2 fit",
                 fontweight="bold", fontsize=12)
    # The lagged-fair cloud has heavy ± tails (12s of price drift per trade);
    # clip the view so the binned-median curve and V2 fit stay legible.
    ax.set_ylim(-10, 60)
    _apply_style(ax)
    return ax


def plot_retail_share_bars(ax: plt.Axes | None = None) -> plt.Axes:
    """Bar chart: retail volume share + retail fee share at 5bp pool, sim vs real."""
    v = load_validation()
    r = v["real_targets"]; s = v["sim_results"]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))

    metrics = ["Volume share", "Fee share"]
    real_vals = [r["retail_volume_share_5bp"], r["retail_fee_share_5bp"]]
    sim_vals  = [s["retail_volume_share_mean"], s["retail_fee_share_mean"]]
    sim_stds  = [s["retail_volume_share_std"],  s["retail_fee_share_std"]]

    x = np.arange(len(metrics))
    w = 0.35
    ax.bar(x - w/2, real_vals, w, color="#2c3e50", label="real (on-chain)")
    ax.bar(x + w/2, sim_vals,  w, color="#27ae60", label="sim",
           yerr=sim_stds, capsize=4, ecolor="#1c603c")
    for i, (rv, sv) in enumerate(zip(real_vals, sim_vals)):
        ax.text(i - w/2, rv + 0.015, f"{rv*100:.1f}%", ha="center", fontsize=9)
        ax.text(i + w/2, sv + 0.015, f"{sv*100:.1f}%", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_title("Retail flow at 5bp pool — sim vs real",
                 fontweight="bold", fontsize=12)
    _apply_style(ax)
    return ax


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    v = values[order]; w = weights[order]
    cum = np.cumsum(w)
    cutoff = q * cum[-1]
    idx = np.minimum(np.searchsorted(cum, cutoff), len(v) - 1)
    return v[idx]


def plot_retail_markout_overlay(ax: plt.Axes | None = None,
                                clip_bps: float | None = None) -> plt.Axes:
    """USD-weighted retail markout_15s density: real vs sim, on the 5bp pool.

    Shows the full RAW range by default (clip_bps=None) — no truncation — so the
    sim's right tail resolves to its true location (a few large trades marking out
    to ~+250 bps) instead of piling into an artificial edge bin. Pass a clip_bps
    to truncate the view.
    """
    real = load_real_retail_markouts()
    sim  = load_sim_retail_markouts()
    rm_v = real["markout_15s_bps"].to_numpy(); rw = real["usd_amount"].to_numpy()
    sm_v = sim["markout_bps"].to_numpy();      sw = sim["usd_amount"].to_numpy()

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))

    if clip_bps is not None:
        rm_v = np.clip(rm_v, -clip_bps, clip_bps)
        sm_v = np.clip(sm_v, -clip_bps, clip_bps)
        lo, hi = -clip_bps, clip_bps
    else:
        lo = float(min(rm_v.min(), sm_v.min()))
        hi = float(max(rm_v.max(), sm_v.max()))
    bins = np.linspace(lo, hi, 120)
    ax.hist(rm_v, bins=bins, weights=rw, density=True,
            alpha=0.55, color="#2c3e50", label="real (USD-weighted)")
    ax.hist(sm_v, bins=bins, weights=sw, density=True,
            alpha=0.55, color="#27ae60", label="sim (USD-weighted)")

    # USD-weighted means (computed on the raw, unclipped values)
    rm = float((real["markout_15s_bps"] * real["usd_amount"]).sum() / real["usd_amount"].sum())
    sm = float((sim["markout_bps"] * sim["usd_amount"]).sum() / sim["usd_amount"].sum())
    ax.axvline(rm, color="#2c3e50", ls="--", lw=1.2, label=f"real μ = {rm:+.2f}")
    ax.axvline(sm, color="#27ae60", ls="--", lw=1.2, label=f"sim μ = {sm:+.2f}")

    ax.set_xlabel("markout_15s (bps, LP-positive) — raw, unclipped")
    ax.set_ylabel("USD-weighted density")
    ax.set_title("Retail markout_15s at 5bp pool — sim vs real (USD-w)",
                 fontweight="bold", fontsize=12)
    ax.set_xlim(lo, hi)
    _apply_style(ax)
    return ax


def plot_retail_markout_qq(ax: plt.Axes | None = None,
                           q_grid: tuple = (1, 5, 10, 25, 50, 75, 90, 95, 99)) -> plt.Axes:
    """USD-weighted QQ plot of retail markout_15s: real vs sim."""
    real = load_real_retail_markouts()
    sim  = load_sim_retail_markouts()

    qs = np.array(q_grid) / 100.0
    real_q = _weighted_quantile(real["markout_15s_bps"].to_numpy(),
                                real["usd_amount"].to_numpy(), qs)
    sim_q  = _weighted_quantile(sim["markout_bps"].to_numpy(),
                                sim["usd_amount"].to_numpy(),  qs)

    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5))

    ax.scatter(real_q, sim_q, s=60, color="#27ae60")
    for q_pct, x, y in zip(q_grid, real_q, sim_q):
        ax.annotate(f"p{q_pct}", (x, y), textcoords="offset points",
                    xytext=(6, -4), fontsize=8, color="#555")
    lo = min(real_q.min(), sim_q.min())
    hi = max(real_q.max(), sim_q.max())
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="#888", lw=1)
    ax.set_xlabel("real markout_15s quantile (bps, USD-w)")
    ax.set_ylabel("sim markout quantile (bps, USD-w)")
    ax.set_title("Markout QQ — sim vs real", fontweight="bold", fontsize=12)
    _apply_style(ax)
    return ax


def get_validation_summary_table() -> pd.DataFrame:
    """Display-friendly table of the three validation metrics: sim vs real."""
    v = load_validation()
    r = v["real_targets"]; s = v["sim_results"]; p = v["calibrated_normalizer"]

    rows = [
        ("Retail volume share @5bp",
         f"{r['retail_volume_share_5bp']*100:.2f}%",
         f"{s['retail_volume_share_mean']*100:.2f}% ± {s['retail_volume_share_std']*100:.2f}pp"),
        ("Retail fee share @5bp",
         f"{r['retail_fee_share_5bp']*100:.2f}%",
         f"{s['retail_fee_share_mean']*100:.2f}% ± {s['retail_fee_share_std']*100:.2f}pp"),
        ("Retail markout_15s — USD-w mean",
         f"{r['markout_15s_usd_w_mean_bps']:+.2f} bps",
         f"{s['markout_usd_w_mean_bps']:+.2f} bps"),
        ("Normalizer φ",          "—", f"{p['phi_bps']:.2f} bps"),
        ("Normalizer depth",      "—", f"${p['depth_usdc_m']:.1f}M"),
        ("Submission pool fee",   "5.00 bps (frozen)", "5.00 bps (frozen)"),
        ("Submission pool depth", "$212.2M (frozen)",  "$212.2M (frozen)"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Real (on-chain)", "Simulator"])
