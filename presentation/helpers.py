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
        arrival_rate=1_294_178 / 1_303_200,  # cross-pool parent order rate
        usd_quantiles_path=ANALYSIS_DIR / "parent_order_usd_quantiles.csv",
        buy_prob=0.4842,
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
    rows.append({"source": "Realistic Simulator (cross-pool)", "rate_per_block": 1_294_178 / 1_303_200, "orders": None, "blocks": None})
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
# SECTION 3: Pool Liquidity & Fee Calibration
# ===================================================================

# Real Uniswap v3 WETH/USDC 0.05% pool parameters (from pool_stats_6m.csv)
REAL_POOL_FEE = 0.0005  # 5 bps
REAL_VIRTUAL_USDC = 212_157_626.44
# On-chain calibration targets (all WETH/USDC pools, all protocols, avg of May 1+7 2026)
REAL_VOLUME_SHARE_5BP = 0.311  # 5bp pool gets ~31% of total WETH/USDC volume
REAL_FEE_SHARE_5BP = 0.155    # 5bp pool earns ~15.5% of total WETH/USDC fees

# On-chain markout distribution on the 5bp WETH/USDC pool, per-swap (no USD-weighting).
# markout_next = LP profit per swap in bps, benchmarked to next-block mid (Binance).
# RETAIL-ONLY = router-routed txs (the same definition the simulator uses to filter
# the sim distribution it produces). Aggregated over the 6 days of the calibration
# window 2026-05-14..2026-05-19 (2026-05-20 has no markouts populated on the pool).
# n_swaps = 8,963.
#
# The full per-swap-percentile curve lives in
# `markout_5bp_pool_percentiles_retail.csv` (1001 quantile points) — this is what
# `plot_markout_comparison` consumes.
OBSERVED_MARKOUT = {
    "avg_bps": 0.734, "std_bps": 3.547,
    "p5": -3.185, "p25": -0.911, "p50": 0.076, "p75": 1.310, "p95": 8.314,
}

# For reference — the all-flow (retail+arb) per-swap distribution on the same pool
# (May 7 2026 snapshot, n=6328). Kept for historical comparison; the sim collects
# retail-only so the calibration overlay should not use this.
OBSERVED_MARKOUT_ALLFLOW = {
    "avg_bps": 3.637, "std_bps": 4.456,
    "p5": -1.94, "p25": 0.25, "p50": 3.05, "p75": 6.93, "p95": 10.44,
}


def _build_sim(
    normalizer_fee: float,
    normalizer_depth_y: float,
    submission_depth_y: float = REAL_VIRTUAL_USDC,
    submission_fee: float = REAL_POOL_FEE,
    n_steps: int = 10_000,
    seed: int = 42,
):
    """Create a configured ExactSimpleAMMSimulator."""
    from arena_eval.exact_simple_amm.config import ExactSimpleAMMConfig
    from arena_eval.exact_simple_amm.simulator import ExactSimpleAMMSimulator
    from arena_eval.exact_simple_amm.strategies import FixedFeeStrategy

    initial_price = 100.0
    norm_y = normalizer_depth_y
    norm_x = norm_y / initial_price
    frac = submission_depth_y / norm_y

    cfg = ExactSimpleAMMConfig(
        n_steps=n_steps,
        initial_price=initial_price,
        initial_x=norm_x,
        initial_y=norm_y,
        submission_liquidity_fraction=frac,
        evaluator_kind="real_data",
        price_process_kind="regime_switching",
        retail_flow_kind="empirical_usd_size",
        retail_arrival_rate=1_294_178 / 1_303_200,
        retail_buy_prob=0.4842,
        regime_invcdf_path=str(ANALYSIS_DIR / "regimes_invcdf.csv"),
        regime_transition_path=str(ANALYSIS_DIR / "regimes_transition_matrix.csv"),
        retail_usd_quantiles_path=str(ANALYSIS_DIR / "parent_order_usd_quantiles.csv"),
    )
    return ExactSimpleAMMSimulator(
        config=cfg,
        submission_strategy=FixedFeeStrategy(
            bid_fee=submission_fee, ask_fee=submission_fee,
        ),
        normalizer_strategy=FixedFeeStrategy(
            bid_fee=normalizer_fee, ask_fee=normalizer_fee,
        ),
        seed=seed,
    )


def measure_sim_shares(
    normalizer_fee: float,
    normalizer_depth_y: float,
    submission_depth_y: float = REAL_VIRTUAL_USDC,
    submission_fee: float = REAL_POOL_FEE,
    n_steps: int = 5_000,
    seeds: tuple[int, ...] = (42, 43),
) -> dict:
    """Run simulation and return volume share and fee share of the submission pool."""
    total_sub_vol = 0.0
    total_norm_vol = 0.0
    total_sub_fees = 0.0
    total_norm_fees = 0.0
    for seed in seeds:
        sim = _build_sim(
            normalizer_fee, normalizer_depth_y,
            submission_depth_y, submission_fee, n_steps, seed,
        )
        sim.run()
        res = sim.result()
        total_sub_vol += res.retail_volume_submission_y
        total_norm_vol += res.retail_volume_normalizer_y
        # Fee earned ≈ volume × fee_rate (submission uses submission_fee, normalizer uses normalizer_fee)
        total_sub_fees += res.retail_volume_submission_y * submission_fee
        total_norm_fees += res.retail_volume_normalizer_y * normalizer_fee
    total_vol = total_sub_vol + total_norm_vol
    total_fees = total_sub_fees + total_norm_fees
    return {
        "vol_share": total_sub_vol / total_vol if total_vol > 0 else 0.5,
        "fee_share": total_sub_fees / total_fees if total_fees > 0 else 0.5,
    }


def calibrate_2d(
    target_vol_share: float = REAL_VOLUME_SHARE_5BP,
    target_fee_share: float = REAL_FEE_SHARE_5BP,
    submission_depth_y: float = REAL_VIRTUAL_USDC,
    submission_fee: float = REAL_POOL_FEE,
    fee_grid: tuple[float, ...] = (0.0003, 0.0005, 0.001, 0.003, 0.005, 0.007, 0.01, 0.012, 0.015, 0.02, 0.03),
    n_steps: int = 5_000,
    seeds: tuple[int, ...] = (42, 43),
    depth_tol: float = 0.005,
    verbose: bool = True,
) -> tuple[dict, list[dict]]:
    """2D calibration: sweep normalizer fee, binary-search depth for volume share,
    then pick the fee where fee share also matches.

    Returns (best_result, full_log).
    """
    full_log = []

    for nf in fee_grid:
        if verbose:
            print(f"\n--- Normalizer fee = {nf * 1e4:.1f} bps ---")
        # Binary search depth for volume share
        lo, hi = 1e7, 50e9
        best_depth = None
        for iteration in range(15):
            mid = math.sqrt(lo * hi)
            shares = measure_sim_shares(
                nf, mid, submission_depth_y, submission_fee, n_steps, seeds,
            )
            entry = {
                "normalizer_fee": nf,
                "normalizer_fee_bps": nf * 1e4,
                "depth_y": mid,
                "depth_M": mid / 1e6,
                "vol_share": shares["vol_share"],
                "fee_share": shares["fee_share"],
                "vol_share_error": shares["vol_share"] - target_vol_share,
                "fee_share_error": shares["fee_share"] - target_fee_share,
                "iteration": iteration,
            }
            full_log.append(entry)
            if verbose:
                print(f"  depth=${mid/1e6:,.1f}M → vol={shares['vol_share']*100:.1f}% fee={shares['fee_share']*100:.1f}%")
            if abs(shares["vol_share"] - target_vol_share) < depth_tol:
                best_depth = mid
                break
            if shares["vol_share"] > target_vol_share:
                lo = mid
            else:
                hi = mid
        if best_depth is None:
            best_depth = mid

    # Find the fee point where fee_share is closest to target
    # Only look at converged points (last iteration per fee)
    converged = {}
    for e in full_log:
        nf = e["normalizer_fee"]
        if nf not in converged or e["iteration"] > converged[nf]["iteration"]:
            converged[nf] = e
    converged_list = sorted(converged.values(), key=lambda x: x["normalizer_fee"])
    best = min(converged_list, key=lambda x: abs(x["fee_share_error"]))

    if verbose:
        print(f"\n=== Best calibration ===")
        print(f"  Normalizer fee:  {best['normalizer_fee_bps']:.1f} bps")
        print(f"  Normalizer depth: ${best['depth_M']:.1f}M")
        print(f"  Vol share: {best['vol_share']*100:.1f}% (target: {target_vol_share*100:.1f}%)")
        print(f"  Fee share: {best['fee_share']*100:.1f}% (target: {target_fee_share*100:.1f}%)")

    return best, full_log


def run_calibrated_sim(
    normalizer_fee: float,
    normalizer_depth_y: float,
    submission_depth_y: float = REAL_VIRTUAL_USDC,
    submission_fee: float = REAL_POOL_FEE,
    n_steps: int = 10_000,
    seeds: tuple[int, ...] = (42, 43, 44),
    collect_markouts: bool = True,
) -> dict:
    """Run full simulation with calibrated parameters.

    Collects per-trade markouts on the submission pool and aggregate metrics.
    Per-trade markout is computed against the **next block's** ``fair_price``
    (the canonical "next-block mid" used by the on-chain ``markout_next_bps``
    column in ``analysis/weth_usdc_90d/markout_5bp_pool_percentiles.csv`` and
    by T3 in this calibration). Using the same-block ``fair_price`` would put
    every tiny retail trade on the deterministic post-arb-spot fee boundary at
    exactly +0 bps or +10 bps, producing two narrow spikes that don't exist in
    the empirical data - see ``reports/markout_spike_investigation.md``.
    """
    per_trade_markouts_bps: list[float] = []
    total_sub_vol = 0.0
    total_norm_vol = 0.0
    total_retail_edge_sub = 0.0
    total_arb_loss_sub = 0.0

    def trade_edge(amount_x: float, amount_y: float, trader_side: str, fair: float) -> float:
        if trader_side == "sell_x":
            # Retail sells X (buys Y), LP buys X
            return amount_x * fair - amount_y
        # Retail buys X (sells Y), LP sells X
        return amount_y - amount_x * fair

    for seed in seeds:
        sim = _build_sim(
            normalizer_fee, normalizer_depth_y,
            submission_depth_y, submission_fee, n_steps, seed,
        )
        # Defer-by-one-block buffer so each retail trade is marked against the
        # next block's fair_price (apples-to-apples with the empirical
        # next-block markout). Trades from the last block of each seed have no
        # next block and are dropped (~0.03% of samples; negligible for the
        # distribution overlay).
        pending: list[tuple[float, float, str]] = []
        while not sim.done:
            step = sim.step_once()
            if collect_markouts:
                # First, finalise any pending retail trades against this block's
                # newly drawn fair_price.
                new_fair = step["fair_price"]
                for amount_x, amount_y, trader_side in pending:
                    edge = trade_edge(amount_x, amount_y, trader_side, new_fair)
                    if amount_y > 0:
                        per_trade_markouts_bps.append(edge / amount_y * 10_000)
                pending.clear()
                # Buffer this block's retail submission trades for the next block.
                for ev in step["trade_events"]:
                    if ev["source"] != "retail" or ev["venue"] != "submission":
                        continue
                    pending.append(
                        (float(ev["amount_x"]), float(ev["amount_y"]), str(ev["trader_side"]))
                    )

        res = sim.result()
        total_sub_vol += res.retail_volume_submission_y
        total_norm_vol += res.retail_volume_normalizer_y
        total_retail_edge_sub += res.retail_edge_submission
        total_arb_loss_sub += res.arb_loss_submission

    total = total_sub_vol + total_norm_vol
    vol_share = total_sub_vol / total if total > 0 else 0.5
    agg_markout = (total_retail_edge_sub / total_sub_vol * 10_000) if total_sub_vol > 0 else 0.0

    markouts = np.array(per_trade_markouts_bps)
    return {
        "markouts_bps": markouts,
        "volume_share_submission": vol_share,
        "aggregate_markout_bps": agg_markout,
        "retail_edge_sub": total_retail_edge_sub,
        "arb_loss_sub": total_arb_loss_sub,
        "normalizer_fee": normalizer_fee,
        "normalizer_depth_y": normalizer_depth_y,
        "submission_depth_y": submission_depth_y,
        "submission_fee": submission_fee,
        "n_seeds": len(seeds),
        "n_steps": n_steps,
    }


def plot_calibration_2d(
    full_log: list[dict],
    target_vol_share: float = REAL_VOLUME_SHARE_5BP,
    target_fee_share: float = REAL_FEE_SHARE_5BP,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot 2D calibration: fee share vs normalizer fee, with converged depth annotations."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    # Extract converged points (last iteration per fee)
    converged = {}
    for e in full_log:
        nf = e["normalizer_fee"]
        if nf not in converged or e["iteration"] > converged[nf]["iteration"]:
            converged[nf] = e
    pts = sorted(converged.values(), key=lambda x: x["normalizer_fee"])

    fees_bps = [p["normalizer_fee_bps"] for p in pts]
    fee_shares = [p["fee_share"] * 100 for p in pts]
    vol_shares = [p["vol_share"] * 100 for p in pts]

    ax.plot(fees_bps, fee_shares, "o-", color="#27ae60", lw=2, markersize=8,
            label="Simulated fee share (vol-share matched)")
    ax.axhline(target_fee_share * 100, color="#e74c3c", ls="--", lw=1.5, alpha=0.7)
    ax.text(fees_bps[-1] * 0.7, target_fee_share * 100 + 1,
            f"Target: {target_fee_share * 100:.1f}%", color="#e74c3c", fontsize=9)

    # Annotate depths
    for p in pts:
        ax.annotate(f"${p['depth_M']:.0f}M\nvol={p['vol_share']*100:.0f}%",
                     (p["normalizer_fee_bps"], p["fee_share"] * 100),
                     textcoords="offset points", xytext=(0, 14),
                     fontsize=7, color="#8b8fa3", ha="center")

    ax.set_xlabel("Normalizer Fee (bps)", fontsize=11)
    ax.set_ylabel("5bp Pool Fee Share (%)", fontsize=11)
    ax.set_title("2D Calibration: Fee Share vs Normalizer Fee\n(each point depth-matched for volume share)",
                 fontsize=12, fontweight="bold")
    ax.set_xscale("log")
    _apply_style(ax)
    return ax


def plot_markout_comparison(
    sim_data: dict,
    observed: dict | None = None,
    trim_pct: float = 1.0,
    bins: int = 80,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Overlaid histograms comparing simulated vs observed markout distributions."""
    if observed is None:
        observed = OBSERVED_MARKOUT
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    markouts = sim_data["markouts_bps"]

    # Load observed percentile curve as pseudo-samples (uniformly spaced in probability).
    # Retail-only on the 5bp pool — same slice as the sim's `markouts_bps`
    # (filtered by source == 'retail', venue == 'submission').
    obs_df = pd.read_csv(ANALYSIS_DIR / "markout_5bp_pool_percentiles_retail.csv")
    obs_vals = obs_df["markout_next_bps"].values

    # Trim both to the same range for readability
    lo = min(np.percentile(markouts, trim_pct), np.percentile(obs_vals, trim_pct))
    hi = max(np.percentile(markouts, 100 - trim_pct), np.percentile(obs_vals, 100 - trim_pct))
    sim_trimmed = markouts[(markouts >= lo) & (markouts <= hi)]
    obs_trimmed = obs_vals[(obs_vals >= lo) & (obs_vals <= hi)]

    ax.hist(obs_trimmed, bins=bins, density=True, color=STYLE["observed"]["color"],
            alpha=0.45, edgecolor="none", label="Observed (on-chain)")
    ax.hist(sim_trimmed, bins=bins, density=True, color=STYLE["realistic"]["color"],
            alpha=0.45, edgecolor="none", label="Simulated (calibrated)")

    # Annotate key stats
    sim_avg = np.mean(markouts)
    sim_med = np.median(markouts)
    txt = (
        f"Simulated:  avg={sim_avg:.2f} bps, median={sim_med:.2f} bps\n"
        f"Observed:   avg={observed['avg_bps']:.2f} bps, median={observed['p50']:.2f} bps"
    )
    ax.text(0.02, 0.97, txt, transform=ax.transAxes, fontsize=8.5,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="#ccc"))

    ax.set_xlabel("LP Markout per Trade (bps)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Markout Distribution: Simulated vs Observed",
                 fontsize=12, fontweight="bold")
    _apply_style(ax)
    return ax


def plot_markout_qq(
    sim_data: dict,
    observed: dict | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """QQ-style comparison: simulated markout percentiles vs observed."""
    if observed is None:
        observed = OBSERVED_MARKOUT
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    pct_keys = ["p5", "p25", "p50", "p75", "p95"]
    pct_vals = [5, 25, 50, 75, 95]
    obs_vals = [observed[k] for k in pct_keys]
    sim_vals = [float(np.percentile(sim_data["markouts_bps"], p)) for p in pct_vals]

    ax.scatter(obs_vals, sim_vals, s=80, color="#27ae60", zorder=5, edgecolors="white")
    for label, ox, sy in zip(pct_keys, obs_vals, sim_vals):
        ax.annotate(label, (ox, sy), textcoords="offset points",
                    xytext=(8, 8), fontsize=9, color="#8b8fa3")

    lo = min(min(obs_vals), min(sim_vals)) - 1
    hi = max(max(obs_vals), max(sim_vals)) + 1
    ax.plot([lo, hi], [lo, hi], ls="--", color="#999", lw=1, label="y = x")

    ax.set_xlabel("Observed Markout (bps)", fontsize=11)
    ax.set_ylabel("Simulated Markout (bps)", fontsize=11)
    ax.set_title("Markout QQ: Simulated vs Observed", fontsize=12, fontweight="bold")
    _apply_style(ax)
    return ax


def get_calibration_summary(sim_data: dict) -> pd.DataFrame:
    """Summary table comparing calibrated simulation outcomes vs on-chain observations."""
    markouts = sim_data["markouts_bps"]
    # Compute simulated fee share
    sub_fees = sim_data.get("retail_edge_sub", 0)
    sub_vol = sim_data["volume_share_submission"]
    norm_vol = 1 - sub_vol
    sim_fee_share = (sub_vol * sim_data["submission_fee"]) / (
        sub_vol * sim_data["submission_fee"] + norm_vol * sim_data["normalizer_fee"]
    ) if sim_data.get("normalizer_fee", 0) > 0 else 0

    rows = {
        "Volume Share (5bp pool)": {
            "Observed": f"{REAL_VOLUME_SHARE_5BP * 100:.1f}%",
            "Simulated": f"{sim_data['volume_share_submission'] * 100:.1f}%",
        },
        "Fee Share (5bp pool)": {
            "Observed": f"{REAL_FEE_SHARE_5BP * 100:.1f}%",
            "Simulated": f"{sim_fee_share * 100:.1f}%",
        },
        "Avg LP Markout (bps)": {
            "Observed": f"{OBSERVED_MARKOUT['avg_bps']:.2f}",
            "Simulated": f"{np.mean(markouts):.2f}",
        },
        "Median LP Markout (bps)": {
            "Observed": f"{OBSERVED_MARKOUT['p50']:.2f}",
            "Simulated": f"{np.median(markouts):.2f}",
        },
        "Markout p5 (bps)": {
            "Observed": f"{OBSERVED_MARKOUT['p5']:.2f}",
            "Simulated": f"{np.percentile(markouts, 5):.2f}",
        },
        "Markout p95 (bps)": {
            "Observed": f"{OBSERVED_MARKOUT['p95']:.2f}",
            "Simulated": f"{np.percentile(markouts, 95):.2f}",
        },
        "Normalizer Depth": {
            "Observed": "—",
            "Simulated": f"${sim_data['normalizer_depth_y'] / 1e6:.1f}M",
        },
        "Normalizer Fee": {
            "Observed": "—",
            "Simulated": f"{sim_data['normalizer_fee'] * 1e4:.0f} bps",
        },
    }
    return pd.DataFrame(rows).T.rename_axis("Metric")
