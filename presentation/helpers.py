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
    """Broad 19-router parent-order USD size quantiles (kept for robustness checks)."""
    path = ANALYSIS_DIR / "router_parent_order_size_windows.csv"
    df = pd.read_csv(path)
    sel = df[
        (df["window_name"] == window)
        & (df["mode"] == mode)
        & (df["side_group"] == side)
    ][["pct", "size_usd", "parent_count"]].reset_index(drop=True)
    return sel


def load_strict_retail_quantiles() -> pd.DataFrame:
    """Strict-retail (Uniswap first-party FE ∪ MetaMask, 30d) parent-order USD size
    quantiles — the distribution the simulator samples retail order sizes from
    (parent_order_usd_quantiles.csv). This is the §2 "observed" series."""
    return pd.read_csv(ANALYSIS_DIR / "parent_order_usd_quantiles.csv")


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
        arrival_rate=98_676 / 216_000,  # strict-retail 30d parent-order rate
        usd_quantiles_path=ANALYSIS_DIR / "parent_order_usd_quantiles.csv",
        buy_prob=0.4627,
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
    """Arrival-rate comparison on the strict-retail cohort (Uniswap first-party FE
    ∪ MetaMask, 30d window 2026-04-21..05-20) — the same cohort used for calibration
    and validation. The realistic simulator matches the observed strict rate by
    construction. (The broad 19-router 6m/1y/2y rates remain in
    router_parent_order_size_windows.csv for robustness, but no longer drive this chart.)"""
    strict_rate = 98_676 / 216_000  # = config.EMPIRICAL_PARENT_ORDER_ARRIVAL_RATE
    rows = [
        {"source": "Observed (strict retail, 30d)", "rate_per_block": strict_rate, "orders": 98_676, "blocks": 216_000},
        {"source": "Challenge Model", "rate_per_block": 0.8, "orders": None, "blocks": None},
        {"source": "Realistic Simulator", "rate_per_block": strict_rate, "orders": None, "blocks": None},
    ]
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
    """§4 normalizer calibration sample: strict-retail impact across ALL non-5bp
    WETH/USDC venues (Uniswap V3+V4+V2, Fluid, Balancer, Curve, …) — the true
    competitor of the held-out 5bp pool. Spread vs lagged fair, from markout_prod
    (no pool-mid column). 30d window."""
    return pd.read_csv(ANALYSIS_DIR / "non5bp_allvenue_impact_30d.csv")


def load_impact_curve_sample_v3() -> pd.DataFrame:
    """V3-only non-5bp sample — kept for the sandwich diagnostic, which needs
    per-pool sqrtPrice mids (only V3 carries them)."""
    return pd.read_csv(ANALYSIS_DIR / "non5bp_impact_sample_v3_pool_mid_7d.csv")


def load_impact_curve_fit_v3() -> dict:
    """V2 fit of the V3-only non-5bp sample (for the sandwich diagnostic chart)."""
    import json
    return json.loads((ANALYSIS_DIR / "impact_curve_fit_v3nonsbp.json").read_text())


def load_impact_curve_fit() -> dict:
    """Plan A and Plan B V2 fits of the normalizer pool."""
    import json
    return json.loads((ANALYSIS_DIR / "impact_curve_fit_pool_mid.json").read_text())


def load_sandwich_victims() -> pd.DataFrame:
    """All-venue WETH/USDC sandwich VICTIM swaps over the 30d window, reconstructed
    from the Allium heuristic sandwich table (uniswap-allium.ethereum.dex_sandwich_trades):
    swaps sitting between a front and back attacker leg on the same pool+block.
    Columns: victim_tx, protocol, pool, usd_amount. The all-venue complement to the
    V3-only pool-mid sandwich diagnostic (Chart 3b)."""
    return pd.read_csv(ANALYSIS_DIR / "sandwich_victims_allvenue_30d.csv")


def sandwich_census():
    """(by_protocol, by_sample_venue) — the all-venue WETH/USDC sandwich-victim
    census from the heuristic table, and how many of our strict-retail calibration
    trades are sandwich victims, broken down by venue."""
    vic = load_sandwich_victims()
    by_protocol = (vic.groupby("protocol")
                   .agg(victims=("victim_tx", "nunique"),
                        usd_m=("usd_amount", lambda s: round(s.sum() / 1e6, 2)))
                   .sort_values("victims", ascending=False).reset_index())
    samp = load_impact_curve_sample()
    f = samp[(samp["n_distinct_sides"] == 1) & (samp["size_usd"] > 1.0)
             & np.isfinite(samp["observed_spread_fair_lag_bps"])].copy()
    f["venue"] = _venue_labels(f["top_pool"], f["top_project"], f["observed_spread_fair_lag_bps"])
    vset = set(vic["victim_tx"].str.lower())
    f["v"] = f["tx_hash"].str.lower().isin(vset)
    rows = []
    for ven, sub in f.groupby("venue"):
        vv = sub[sub["v"]]
        rows.append(dict(venue=ven, n=len(sub), victims=int(sub["v"].sum()),
                         victim_pct=round(100 * sub["v"].mean(), 2),
                         med_spread_victim_bps=(round(float(vv["observed_spread_fair_lag_bps"].median()), 1)
                                                if len(vv) else None)))
    by_sample_venue = pd.DataFrame(rows).sort_values("victims", ascending=False).reset_index(drop=True)
    return by_protocol, by_sample_venue


def plot_sandwich_census(ax: plt.Axes | None = None) -> plt.Axes:
    """Bar chart: % of our strict-retail trades that are sandwich victims, by venue
    (Allium heuristic). Shows the non-V3 sandwiching — concentrated in Uniswap V4 —
    that the V3-only Chart 3b cannot see, and that the fee-tier band (V2/Balancer/
    Curve) is essentially un-sandwiched (so that band is fee, not MEV)."""
    _, by_venue = sandwich_census()
    d = by_venue[by_venue["n"] >= 50].copy()
    short = {"Uniswap V3 1bp — wide-vs-fair tail (~1/3 sandwich, ~2/3 lag)": "V3 1bp wide-vs-fair tail",
             "Uniswap V4 (all tiers, mostly low-fee)": "Uniswap V4",
             "Uniswap V3 (incl. 1bp pool)": "Uniswap V3 (mass)",
             "Uniswap V2 (30bp fee)": "Uniswap V2", "Balancer (~30bp)": "Balancer",
             "Curve (~30bp)": "Curve", "other": "other"}
    d["label"] = d["venue"].map(lambda v: short.get(v, v))
    d = d.sort_values("victim_pct")
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(d["label"], d["victim_pct"], color="#e74c3c", alpha=0.85)
    for b, (_, r) in zip(bars, d.iterrows()):
        ax.text(b.get_width() + 0.6, b.get_y() + b.get_height() / 2,
                f"{int(r['victims'])} of {int(r['n']):,}", va="center", fontsize=8)
    ax.set_xlabel("% of venue's strict-retail trades flagged as sandwich victims")
    ax.set_title("All-venue sandwich census — non-V3 sandwiching is concentrated in Uniswap V4",
                 fontweight="bold", fontsize=11)
    ax.set_xlim(0, max(d["victim_pct"].max() * 1.25, 5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    return ax


def load_allpools_impact_sample() -> pd.DataFrame:
    """Full-market impact sample: strict-retail WETH/USDC trades across ALL venues
    (Uniswap V3+V4+V2, Fluid, Swaap, Pancake, Balancer, Curve, …), incl. the 5bp
    pool. Spread vs lagged fair; built from markout_prod (no pool-mid column)."""
    return pd.read_csv(ANALYSIS_DIR / "allpools_impact_sample_30d.csv")


def load_allpools_impact_fit() -> dict:
    """V2 fit of the FULL-MARKET normalizer — the whole current ETH/USDC market a
    new submission strategy competes against (5bp pool included, all venues)."""
    import json
    return json.loads((ANALYSIS_DIR / "impact_curve_fit_allpools.json").read_text())


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
                          n_bins: int = 30,
                          sample: pd.DataFrame | None = None,
                          fit: dict | None = None,
                          title: str = "Retail impact curve (all non-5bp venues) — empirical vs V2 fit") -> plt.Axes:
    """Empirical impact cloud (USD-weighted binned median) overlaid with the
    fitted V2 curve. Spread is referenced to the lagged (pre-trade) fair price —
    the Binance mid one 12s step before each trade. Defaults to the §4 non-5bp
    sample/fit; pass `sample`/`fit` to plot a different cohort (e.g. the
    full-market all-venue sample). The red sandwich overlay is shown only when the
    sample carries a pool-mid column (the all-venue markout_prod sample does not)."""
    if sample is None:
        sample = load_impact_curve_sample()
    if fit is None:
        fit = load_impact_curve_fit()
    plan = fit[plan_key]
    phi = plan["phi"]; depth = plan["depth_usdc"]

    size = sample["size_usd"].to_numpy()
    spread = sample["observed_spread_fair_lag_bps"].to_numpy()

    # Flag sandwich / transient-MEV-excursion victims (only if we have a pre-trade
    # pool mid): trades that filled against a pool whose PRE-TRADE mid was already
    # pushed off fair — pre-trade pool mid >0.5% from contemporaneous fair, while
    # the fill sat ~on that pushed mid (|pool-spread|<20 bps). MEV slippage, not
    # mechanical impact. Visual call-out only; the fit is unchanged.
    has_pool_mid = {"pool_mid_pre_blended", "fair_price_blended",
                    "observed_spread_pool_bps"}.issubset(sample.columns)
    if has_pool_mid:
        with np.errstate(divide="ignore", invalid="ignore"):
            pushed = np.abs(sample["pool_mid_pre_blended"].to_numpy()
                            / sample["fair_price_blended"].to_numpy() - 1.0)
        sandwiched = (pushed > 0.005) & (np.abs(sample["observed_spread_pool_bps"].to_numpy()) < 20.0)
    else:
        sandwiched = np.zeros(len(size), dtype=bool)

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
    if sandwiched.any():
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
    ax.set_title(title, fontweight="bold", fontsize=12)
    # The lagged-fair cloud has heavy ± tails (12s of price drift per trade);
    # clip the view so the binned-median curve and V2 fit stay legible.
    ax.set_ylim(-10, 60)
    _apply_style(ax)
    return ax


# Well-known WETH/USDC venue addresses, used to split the "uniswap" project label
# into V2/V3/V4 and to call out the prominent pools driving the elevated spread band.
_V4_POOLMANAGER = "0x000000000004444c5dc75cb358380d2e3de08a90"  # Uniswap V4 singleton
_V2_PAIR        = "0xb4e16d0168e52d35cacd2c6185b44281ec28c9dc"  # Uniswap V2 USDC/WETH (30bp)
_V3_1BP_POOL    = "0xe0554a476a092703abdb3ef35c80e0d76d32939f"  # Uniswap V3 1bp (sandwich-heavy)

# venue label -> (color, z, point size, alpha). The Uniswap-V3 mass (incl. the 1bp
# pool, which is the dominant non-5bp venue) is the grey calibration cloud, behind;
# the 30bp fee-tier venues are highlighted on top, plus the 1bp pool's high-spread
# subset (likely sandwich/MEV — 1bp fee can't produce >20 bps spread).
_VENUE_STYLE = {
    "Uniswap V3 (incl. 1bp pool)":          ("#cfd6dc", 1, 4,  0.25),
    "Uniswap V4 (all tiers, mostly low-fee)": ("#9b59b6", 4, 7, 0.30),
    "Uniswap V2 (30bp fee)":                ("#e67e22", 5, 7, 0.30),
    "Balancer (~30bp)":                     ("#2980b9", 6, 22, 0.90),
    "Curve (~30bp)":                        ("#16a085", 6, 22, 0.90),
    "Uniswap V3 1bp — wide-vs-fair tail (~1/3 sandwich, ~2/3 lag)": ("#e74c3c", 7, 14, 0.80),
    "other":                                ("#95a5a6", 3, 10, 0.5),
}
_VENUE_ORDER = list(_VENUE_STYLE.keys())


def _venue_labels(top_pool, top_project, spread) -> np.ndarray:
    """Vectorised venue+version label for each row (uses spread to split the 1bp
    pool's likely-sandwich tail from its normal cheap flow)."""
    tp = np.asarray([str(x).lower() for x in top_pool])
    pj = np.asarray([str(x).lower() for x in top_project])
    sp = np.asarray(spread, dtype=float)
    out = np.full(len(tp), "other", dtype=object)
    out[pj == "uniswap"] = "Uniswap V3 (incl. 1bp pool)"
    out[pj == "balancer"] = "Balancer (~30bp)"
    out[pj == "curve"] = "Curve (~30bp)"
    out[tp == _V2_PAIR] = "Uniswap V2 (30bp fee)"
    out[tp == _V4_POOLMANAGER] = "Uniswap V4 (all tiers, mostly low-fee)"
    out[(tp == _V3_1BP_POOL) & (sp > 20.0)] = "Uniswap V3 1bp — wide-vs-fair tail (~1/3 sandwich, ~2/3 lag)"
    return out


def load_mm_fee_txs() -> set:
    """Tx hashes that paid the MetaMask 87.5bps fee (the MetaMask-router cohort, 30d)."""
    s = pd.read_csv(ANALYSIS_DIR / "mm_fee_txs_30d.csv")["tx_hash"].astype(str).str.lower()
    return set(s)


def _router_labels(tx_hash) -> np.ndarray:
    """Per-order router: MetaMask if the tx paid the MM fee, else Uniswap first-party FE.
    (The §4 strict cohort is uni_fe ∪ mm_fee, disjoint, so not-MM ⇒ Uniswap.)"""
    mm = load_mm_fee_txs()
    return np.where(np.asarray([h.lower() for h in tx_hash]) == np.asarray([h.lower() for h in tx_hash]),
                    np.where([h.lower() in mm for h in tx_hash], "MetaMask", "Uniswap"), "Uniswap")


# router-coloured categories -> (color, z, point size, alpha)
_ROUTER_STYLE = {
    "Uniswap FE": ("#2980b9", 2, 4, 0.18),
    "MetaMask (other venues)": ("#e67e22", 5, 10, 0.55),
    "MetaMask → Curve/Balancer (30bp)": ("#c0392b", 7, 22, 0.95),
}
_ROUTER_ORDER = list(_ROUTER_STYLE.keys())


def plot_impact_curve_by_router(ax: plt.Axes | None = None, plan_key: str = "plan_b") -> plt.Axes:
    """The all-non-5bp-venue impact cloud coloured by ORDER ROUTER (Uniswap first-party
    FE vs MetaMask), with the MetaMask→Curve/Balancer subset (the 30bp-venue routing
    interaction) called out. Each dot is one order (tx); router is a per-order property.
    Binned medians are drawn per router; same V2 fit as Chart 3a."""
    sample = load_impact_curve_sample()
    fit = load_impact_curve_fit()
    phi = fit[plan_key]["phi"]; depth = fit[plan_key]["depth_usdc"]
    s = sample[(sample["n_distinct_sides"] == 1) & (sample["size_usd"] > 1.0)
               & np.isfinite(sample["observed_spread_fair_lag_bps"])].copy()
    router = _router_labels(s["tx_hash"])
    proj = s["top_project"].astype(str).str.lower().to_numpy()
    cat = np.where(router == "Uniswap", "Uniswap FE",
                   np.where(np.isin(proj, ["curve", "balancer"]),
                            "MetaMask → Curve/Balancer (30bp)", "MetaMask (other venues)"))
    s["cat"] = cat
    size = s["size_usd"].to_numpy(); spread = s["observed_spread_fair_lag_bps"].to_numpy()

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))
    for c in _ROUTER_ORDER:
        m = (s["cat"] == c).to_numpy()
        if not m.any():
            continue
        color, z, sz, al = _ROUTER_STYLE[c]
        ax.scatter(size[m], spread[m], s=sz, color=color, alpha=al, edgecolor="none", zorder=z,
                   label=f"{c} (n={int(m.sum()):,})")
    # USD-weighted log-binned medians per router
    edges = np.logspace(0, np.log10(max(size.max(), 10)), 26)
    for rt, col in [("Uniswap", "#1b4f72"), ("MetaMask", "#a04000")]:
        mr = router == rt
        cen, med = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mm2 = mr & (size >= lo) & (size < hi)
            if mm2.sum() >= 8:
                cen.append(np.sqrt(lo * hi)); med.append(np.median(spread[mm2]))
        ax.plot(cen, med, "o-", color=col, lw=2, ms=3, zorder=8, label=f"{rt} binned median")
    sizes_fit = np.logspace(2, np.log10(size.max()), 200)
    ax.plot(sizes_fit, v2_spread_bps(sizes_fit, phi, depth), "-", color="#2c3e50", lw=2, zorder=8,
            label=f"V2 fit: φ={phi*1e4:.2f} bps, D=${depth/1e6:.1f}M")
    ax.set_xscale("log"); ax.set_ylim(-10, 60)
    ax.set_xlabel("Order size (USD)"); ax.set_ylabel("Spread vs lagged fair (bps)")
    ax.set_title("Chart 3a (by router) — Uniswap FE vs MetaMask; MetaMask→Curve/Balancer separated",
                 fontweight="bold", fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, framealpha=0.95, markerscale=1.6, borderaxespad=0.0)
    ax.figure.subplots_adjust(right=0.7)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.grid(alpha=0.25)
    return ax


def plot_impact_curve_by_venue(ax: plt.Axes | None = None,
                               plan_key: str = "plan_b") -> plt.Axes:
    """The all-non-5bp-venue impact cloud (Chart 3a) coloured by venue + version,
    to show what makes up the elevated ~30 bps band above the V3 calibration mass.
    The Uniswap-V3 bulk (incl. the 1bp pool, the dominant non-5bp venue) is grey,
    behind; the 30bp fee-tier venues — Uniswap V2, the Uniswap V4 WETH/USDC pool,
    Balancer and Curve — are highlighted (they sit ~flat at their fee, not on the
    impact curve), plus the 1bp pool's high-spread tail (likely sandwich/MEV, since
    a 1bp fee cannot produce >20 bps). Same data and V2 fit as Chart 3a."""
    sample = load_impact_curve_sample()
    fit = load_impact_curve_fit()
    phi = fit[plan_key]["phi"]; depth = fit[plan_key]["depth_usdc"]

    s = sample[(sample["n_distinct_sides"] == 1) & (sample["size_usd"] > 1.0)
               & np.isfinite(sample["observed_spread_fair_lag_bps"])].copy()
    s["venue"] = _venue_labels(s["top_pool"], s["top_project"],
                               s["observed_spread_fair_lag_bps"])

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    # grey V3 mass first, highlighted venues on top (draw in style-z order)
    for venue in _VENUE_ORDER:
        m = (s["venue"] == venue).to_numpy()
        if not m.any():
            continue
        color, z, size_pt, alpha = _VENUE_STYLE[venue]
        ax.scatter(s.loc[m, "size_usd"], s.loc[m, "observed_spread_fair_lag_bps"],
                   s=size_pt, color=color, alpha=alpha, edgecolor="none", zorder=z,
                   label=f"{venue} (n={int(m.sum()):,})")
        # bold median marker for the highlighted (non-grey, non-other) venues
        if venue not in ("Uniswap V3 (incl. 1bp pool)", "other"):
            ax.scatter([np.median(s.loc[m, "size_usd"])],
                       [np.median(s.loc[m, "observed_spread_fair_lag_bps"])],
                       s=120, color=color, edgecolor="black", lw=1.2, zorder=z + 1, marker="D")

    sizes_fit = np.logspace(2, np.log10(s["size_usd"].max()), 200)
    ax.plot(sizes_fit, v2_spread_bps(sizes_fit, phi, depth), "-", color="#2c3e50", lw=2.0,
            zorder=8, label=f"V2 fit: φ={phi*1e4:.2f} bps, D=${depth/1e6:.1f}M")

    ax.set_xscale("log")
    ax.set_xlabel("Trade size (USD)")
    ax.set_ylabel("Spread vs lagged fair (bps)")
    ax.set_title("Chart 3a (by venue) — the elevated band is the 30bp fee tiers (V2 / Balancer / Curve)\n"
                 "(◆ = per-venue median; they sit flat at their fee, not on the impact curve. V4 is mostly low-fee.)",
                 fontweight="bold", fontsize=10.5)
    ax.set_ylim(-10, 60)
    _apply_style(ax)  # sets spines/ticks AND a default 'best' legend — override it below so ours wins
    # legend OUTSIDE the axes (right strip) so the dense scatter never covers it.
    # MUST come after _apply_style(), whose ax.legend() call would otherwise clobber
    # this placement back to loc='best' (on top of the cloud).
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8,
              framealpha=0.95, markerscale=1.6, borderaxespad=0.0)
    ax.figure.subplots_adjust(right=0.66)
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


# ===================================================================
# SECTION 7: Retail-sample robustness (conservative ↔ aggressive dial)
# ===================================================================
#
# The strict cohort (Uniswap FE ∪ MetaMask) is a subset of true retail, so it is
# the CONSERVATIVE anchor: lower arrival rate, thinner size tail. The broad
# 19-router cohort is the AGGRESSIVE anchor: heavier tail (up to ~$7M), but mixes
# in MM/whale/arb flow. Rather than pick one, we interpolate between them and let
# the user dial the assumption.
_STRICT_ARRIVAL = 98_676 / 216_000      # = config.EMPIRICAL_PARENT_ORDER_ARRIVAL_RATE (strict 30d)
_BROAD_ARRIVAL  = 857_035 / 1_303_200   # broad 19-router 6m
_STRICT_BUY = 0.4627
_BROAD_BUY  = 0.4413


def load_broad_retail_quantiles() -> pd.DataFrame:
    """Broad 19-router cohort (6m) parent-order USD size quantiles — the
    'aggressive' robustness anchor (heavier tail incl. MM/whale flow)."""
    df = pd.read_csv(ANALYSIS_DIR / "router_parent_order_size_windows.csv")
    sel = df[(df["window_name"] == "6m") & (df["mode"] == "strict") & (df["side_group"] == "all")]
    return sel[["pct", "size_usd"]].sort_values("pct").reset_index(drop=True)


def interpolate_retail_inputs(aggressiveness: float):
    """Interpolate retail sim inputs between the strict (conservative, a=0) and
    broad 19-router (aggressive, a=1) anchors. Order sizes are log-interpolated
    per percentile; arrival rate and buy share linearly. a may exceed 1 to
    extrapolate past the broad cohort. Returns (arrival, buy_prob, quantiles_df)."""
    a = float(max(aggressiveness, 0.0))
    strict = load_strict_retail_quantiles().sort_values("pct").reset_index(drop=True)
    broad = load_broad_retail_quantiles()
    pct = strict["pct"].to_numpy()
    ss = strict["size_usd"].to_numpy()
    bs = np.interp(pct, broad["pct"].to_numpy(), broad["size_usd"].to_numpy())
    eps = 1e-9
    size = np.exp((1.0 - a) * np.log(np.maximum(ss, eps)) + a * np.log(np.maximum(bs, eps)))
    arrival = (1.0 - a) * _STRICT_ARRIVAL + a * _BROAD_ARRIVAL
    buy = (1.0 - a) * _STRICT_BUY + a * _BROAD_BUY
    return arrival, buy, pd.DataFrame({"pct": pct, "size_usd": size})


def run_retail_robustness(aggressiveness: float, seeds=(42, 43, 44), n_steps: int = 5000) -> dict:
    """Run the §5 validation sim with retail inputs interpolated at `aggressiveness`
    (0 = strict/conservative, 1 = broad/aggressive). (φ, D) stay at the calibrated
    values and the normalizer is held at fair (as in validate_pool_mid.py). Returns
    the three sim validation metrics."""
    import os
    import tempfile

    from arena_eval.exact_simple_amm.config import ExactSimpleAMMConfig
    from arena_eval.exact_simple_amm.simulator import ExactSimpleAMMSimulator
    from arena_eval.exact_simple_amm.strategies import FixedFeeStrategy

    fit = load_impact_curve_fit()
    phi = fit["plan_b"]["phi"]; depth = fit["plan_b"]["depth_usdc"]
    arrival, buy, q = interpolate_retail_inputs(aggressiveness)

    tf = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
    q.to_csv(tf.name, index=False); tf.close()

    initial_price = 100.0
    norm_y = depth; norm_x = norm_y / initial_price
    frac = REAL_VIRTUAL_USDC / norm_y
    vol_shares, fee_shares, usd, mkt = [], [], [], []
    try:
        for seed in seeds:
            cfg = ExactSimpleAMMConfig(
                n_steps=n_steps, initial_price=initial_price, initial_x=norm_x, initial_y=norm_y,
                submission_liquidity_fraction=frac, evaluator_kind="real_data",
                price_process_kind="regime_switching", retail_flow_kind="empirical_usd_size",
                retail_arrival_rate=arrival, retail_buy_prob=buy,
                regime_invcdf_path=str(ANALYSIS_DIR / "regimes_invcdf.csv"),
                regime_transition_path=str(ANALYSIS_DIR / "regimes_transition_matrix.csv"),
                retail_usd_quantiles_path=tf.name, normalizer_tracks_fair=True,
            )
            sim = ExactSimpleAMMSimulator(
                config=cfg,
                submission_strategy=FixedFeeStrategy(bid_fee=REAL_POOL_FEE, ask_fee=REAL_POOL_FEE),
                normalizer_strategy=FixedFeeStrategy(bid_fee=phi, ask_fee=phi),
                seed=seed,
            )
            pending: list = []
            while not sim.done:
                step = sim.step_once(); nf = step["fair_price"]
                for ax_, ay_, side_, u_ in pending:
                    if ay_ > 0:
                        edge = (ax_ * nf - ay_) if side_ == "sell_x" else (ay_ - ax_ * nf)
                        usd.append(u_); mkt.append(edge / ay_ * 1e4)
                pending = []
                for ev in step["trade_events"]:
                    if ev["source"] != "retail" or ev["venue"] != "submission":
                        continue
                    pending.append((float(ev["amount_x"]), float(ev["amount_y"]),
                                    str(ev["trader_side"]), float(ev["amount_y"])))
            r = sim.result()
            rs = r.retail_volume_submission_y; rn = r.retail_volume_normalizer_y
            vol_shares.append(rs / max(rs + rn, 1e-12))
            fee_shares.append((rs * REAL_POOL_FEE) / max(rs * REAL_POOL_FEE + rn * phi, 1e-12))
    finally:
        os.unlink(tf.name)
    usd = np.asarray(usd); mkt = np.asarray(mkt)
    return {
        "aggressiveness": float(aggressiveness),
        "arrival_rate": arrival,
        "vol_share": float(np.mean(vol_shares)),
        "fee_share": float(np.mean(fee_shares)),
        "markout_bps": float((mkt * usd).sum() / usd.sum()) if usd.sum() > 0 else float("nan"),
    }


def plot_retail_robustness_sweep(ax: plt.Axes | None = None, alphas=None,
                                 seeds=(42, 43, 44), n_steps: int = 5000) -> plt.Axes:
    """Sweep the retail-aggressiveness dial 0→1 and plot how the sim's validation
    metrics move. Shares on the left axis, markout on the right; dotted lines are
    the strict-measured real targets."""
    if alphas is None:
        alphas = np.linspace(0.0, 1.0, 9)
    res = [run_retail_robustness(a, seeds=seeds, n_steps=n_steps) for a in alphas]
    a = np.array([r["aggressiveness"] for r in res])
    vol = np.array([r["vol_share"] for r in res]) * 100
    fee = np.array([r["fee_share"] for r in res]) * 100
    mk = np.array([r["markout_bps"] for r in res])
    real = load_validation()["real_targets"]

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    ax.plot(a, vol, "o-", color="#2c3e50", label="sim vol share @5bp")
    ax.plot(a, fee, "s-", color="#8e44ad", label="sim fee share @5bp")
    ax.axhline(real["retail_volume_share_5bp"] * 100, color="#2c3e50", ls=":", lw=1,
               label=f"real vol {real['retail_volume_share_5bp']*100:.0f}%")
    ax.axhline(real["retail_fee_share_5bp"] * 100, color="#8e44ad", ls=":", lw=1,
               label=f"real fee {real['retail_fee_share_5bp']*100:.0f}%")
    ax.set_xlabel("retail aggressiveness   (0 = strict / conservative,   1 = broad 19-router / aggressive)")
    ax.set_ylabel("share at 5bp (%)")
    ax.set_ylim(0, 100)

    ax2 = ax.twinx()
    ax2.plot(a, mk, "^-", color="#27ae60", label="sim markout_15s")
    ax2.axhline(real["markout_15s_usd_w_mean_bps"], color="#27ae60", ls=":", lw=1,
                label=f"real markout {real['markout_15s_usd_w_mean_bps']:+.0f} bps")
    ax2.set_ylabel("USD-weighted markout_15s (bps)", color="#27ae60")
    ax2.tick_params(axis="y", labelcolor="#27ae60")

    ax.set_title("Retail-sample robustness: validation metrics vs aggressiveness",
                 fontweight="bold", fontsize=12)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, frameon=False, loc="center left")
    ax.spines["top"].set_visible(False)
    return ax


def retail_robustness_slider(seeds=(42, 43, 44), n_steps: int = 5000):
    """Interactive ipywidgets slider over retail aggressiveness (live in JupyterLab).
    Drag to dial the retail-sample assumption; the sim re-runs and prints the
    validation metrics. Static (nbconvert) renders the slider at its default (0)."""
    from ipywidgets import FloatSlider, interact

    real = load_validation()["real_targets"]

    def _show(aggressiveness=0.0):
        r = run_retail_robustness(aggressiveness, seeds=seeds, n_steps=n_steps)
        print(f"aggressiveness = {aggressiveness:.2f}   (0 = strict/conservative, 1 = broad 19-router/aggressive)\n")
        print(f"  retail arrival rate : {r['arrival_rate']:.3f}/block")
        print(f"  vol share @5bp      : sim {r['vol_share']*100:5.1f}%    (real {real['retail_volume_share_5bp']*100:.1f}%)")
        print(f"  fee share @5bp      : sim {r['fee_share']*100:5.1f}%    (real {real['retail_fee_share_5bp']*100:.1f}%)")
        print(f"  markout_15s (USD-w) : sim {r['markout_bps']:+6.1f} bps  (real {real['markout_15s_usd_w_mean_bps']:+.1f} bps)")

    interact(_show, aggressiveness=FloatSlider(
        value=0.0, min=0.0, max=1.0, step=0.1, continuous_update=False,
        description="aggressiveness", readout_format=".1f",
        style={"description_width": "initial"}, layout={"width": "65%"}))


# ===================================================================
# SECTION 9: Guidestar-volatile new-strategy backtest (plots cached results)
# Cache produced offline by scripts/calibration/guidestar_backtest.py.
# ===================================================================
GS_COLORS = {
    "Guidestar (real params, 3.5bp floor)": "#8e44ad",
    "flat 3.5bp (=feeInit, dynamics off)": "#e67e22",
    "flat 5bp (incumbent)": "#2980b9",
}


def _gs_bare(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=9)


def _gs_short(nm: str) -> str:
    return nm.split(" (")[0]


def plot_guidestar_dynamics(scenario: dict | None = None):
    """Drive the actual GuidestarVolatileStrategy (real mainnet volatile params) through
    a scripted price path via its before_swap hook, and plot the resulting fee state:
    the price path, the total buy/sell fee, and the four perm/trans components. One
    step = one block. Visualizes the fee mechanics, not a backtest."""
    from arena_eval.core.types import IncomingSwap
    from arena_eval.exact_simple_amm.guidestar_volatile import GuidestarVolatileStrategy
    events = (scenario or {}).get("events", {6: (0.020, True), 7: (0.009, True),
                                             28: (0.025, False), 29: (0.007, False)})
    n = (scenario or {}).get("n_blocks", 64)
    s = GuidestarVolatileStrategy()                       # real mainnet volatile defaults
    s.after_initialize(1.0, 100.0)
    spot = 100.0
    rec = {k: [] for k in ["price", "buyP", "buyT", "sellP", "sellT", "bid", "ask"]}
    for blk in range(n):
        imp, up = events.get(blk, (0.0, True))
        if blk in events:
            spot *= (1 + imp) if up else (1 - imp)
        bid, ask = s.before_swap(IncomingSwap(is_buy=up, size=None, reserve_x=1.0, reserve_y=spot, block=blk))
        rec["price"].append(spot)
        rec["buyP"].append(s._buyPerm / 100); rec["buyT"].append(s._buyTrans / 100)
        rec["sellP"].append(s._sellPerm / 100); rec["sellT"].append(s._sellTrans / 100)
        rec["bid"].append(bid * 1e4); rec["ask"].append(ask * 1e4)
    B = np.arange(n)
    fig, ax = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True,
                           gridspec_kw={"height_ratios": [1, 1.3, 1.3]})
    ax[0].plot(B, rec["price"], color="#2c3e50", lw=2, marker="o", ms=2); ax[0].set_ylabel("ETH price (fair)")
    ax[0].set_title("Guidestar volatile fee mechanics (real mainnet params, 1 step = 1 block)", fontweight="bold")
    for b, (imp, up) in events.items():
        ax[0].annotate(("buy +" if up else "sell −") + f"{imp*100:.1f}%", (b, rec["price"][b]),
                       textcoords="offset points", xytext=(0, 11 if up else -16), ha="center", fontsize=8,
                       color="#c0392b" if up else "#2980b9")
    ax[1].plot(B, rec["ask"], color="#c0392b", lw=2.3, label="buy fee (ask) = buyPerm+buyTrans")
    ax[1].plot(B, rec["bid"], color="#2980b9", lw=2.3, label="sell fee (bid) = sellPerm+sellTrans")
    ax[1].axhline(2 * 3.5, color="grey", ls=":", label="floor 2·feeInit = 7 bps")
    ax[1].set_ylabel("total fee (bps)"); ax[1].legend(fontsize=8)
    ax[2].plot(B, rec["buyP"], color="#c0392b", lw=2, label="buy permanent")
    ax[2].plot(B, rec["buyT"], color="#e67e22", lw=2, ls="--", label="buy transitory")
    ax[2].plot(B, rec["sellP"], color="#2980b9", lw=2, label="sell permanent")
    ax[2].plot(B, rec["sellT"], color="#16a085", lw=2, ls="--", label="sell transitory")
    ax[2].set_ylabel("component (bps)"); ax[2].set_xlabel("block"); ax[2].legend(fontsize=8, ncol=2)
    for a in ax:
        a.grid(alpha=0.25); a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def load_guidestar_backtest() -> dict:
    """Cached Guidestar new-pool backtest results (analysis/.../guidestar_backtest_cache.json)."""
    import json
    return json.loads((ANALYSIS_DIR / "guidestar_backtest_cache.json").read_text())


def plot_gs_cumulative(cache: dict | None = None):
    """Cumulative 15s-forward LP markout ($) over the run, mean±sd over seeds, at the
    primary depth — total, and split into retail (+) and arb/LVR (−) components."""
    if cache is None:
        cache = load_guidestar_backtest()
    x = np.array(cache["cumulative_steps"])
    order = cache["meta"]["pool_order"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), sharex=True)
    comps = [("tot", "TOTAL LP markout"), ("ret", "Retail markout (+)"), ("arb", "Arb / LVR markout (−)")]
    for nm in order:
        for j, (c, _) in enumerate(comps):
            d = cache["cumulative"][nm][c]
            mean = np.array(d["mean"]); sd = np.array(d["sd"])
            ax[j].plot(x, mean, color=GS_COLORS[nm], lw=2, label=_gs_short(nm))
            ax[j].fill_between(x, mean - sd, mean + sd, color=GS_COLORS[nm], alpha=0.12)
    for j, (_, ttl) in enumerate(comps):
        ax[j].set_title(ttl, fontsize=11, fontweight="bold"); ax[j].set_xlabel("step (12s)")
        ax[j].axhline(0, color="grey", lw=0.7); _gs_bare(ax[j])
    ax[0].set_ylabel("cumulative markout ($)"); ax[0].legend(fontsize=8, loc="upper left")
    d_m = cache["meta"]["primary_depth"] / 1e6
    fig.suptitle(f"§9 — new ${d_m:.1f}M pool vs §8 normalizer: cumulative 15s-forward LP markout "
                 f"(mean±sd, {cache['meta']['seeds']} seeds, real volatile params)", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def plot_gs_histograms(cache: dict | None = None):
    """Per-trade 15s markout (bps) distribution at the primary depth (log + linear)."""
    if cache is None:
        cache = load_guidestar_backtest()
    bins = np.array(cache["histogram"]["bins"])
    ctr = 0.5 * (bins[:-1] + bins[1:])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for nm in cache["meta"]["pool_order"]:
        dens = np.array(cache["histogram"][nm])
        ax[0].plot(ctr, dens, drawstyle="steps-mid", lw=2, color=GS_COLORS[nm], label=_gs_short(nm))
        ax[1].plot(ctr, dens, drawstyle="steps-mid", lw=2, color=GS_COLORS[nm])
    ax[0].set_yscale("log"); ax[0].set_title("per-trade markout (bps) — log density (tails)", fontsize=11, fontweight="bold")
    ax[1].set_title("per-trade markout (bps) — linear (body)", fontsize=11, fontweight="bold")
    for a in ax:
        a.axvline(0, color="grey", lw=0.7); a.set_xlabel("LP markout per trade (bps)"); _gs_bare(a)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def load_guidestar_sensitivity() -> dict:
    """Cached §9 sensitivity + diagnostic results (guidestar_sensitivity_cache.json,
    written by scripts/calibration/guidestar_sensitivity.py)."""
    import json
    return json.loads((ANALYSIS_DIR / "guidestar_sensitivity_cache.json").read_text())


def plot_gs_by_size(cache: dict | None = None, diag: dict | None = None, min_count: int = 30):
    """Markout vs captured trade size. The left panel reports the per-bin RETAIL
    markout as the MEDIAN (solid) with the IQR band, not the mean: the size
    distribution is heavy-tailed, so bins above ~$1k hold few retail trades and the
    per-trade markout has ~tens-of-bps noise (15s fair drift), which makes the bin
    MEAN unstable. The mean is overlaid (dotted) for reference; bins with fewer than
    `min_count` retail trades are drawn faint. Centre/right are sums (count-robust)."""
    if cache is None:
        cache = load_guidestar_backtest()
    if diag is None:
        diag = load_guidestar_sensitivity()
    order = cache["meta"]["pool_order"]
    ctr = np.array(cache["by_size"]["centers"])
    dctr = np.array(diag["by_size_diag"]["centers"])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    def _arr(d, k):
        return np.array([np.nan if v is None else v for v in d[k]], dtype=float)

    for nm in order:
        c = GS_COLORS[nm]
        d = diag["by_size_diag"][nm]
        med, mean = _arr(d, "median_bps"), _arr(d, "mean_bps")
        p25, p75 = _arr(d, "p25_bps"), _arr(d, "p75_bps")
        cnt = np.array(d["count"], dtype=float)
        ok = cnt >= min_count
        med_ok = np.where(ok, med, np.nan)
        ax[0].plot(dctr, med_ok, "o-", color=c, lw=2, ms=4, label=_gs_short(nm))
        ax[0].fill_between(dctr, np.where(ok, p25, np.nan), np.where(ok, p75, np.nan), color=c, alpha=0.10)
        ax[0].plot(dctr, mean, ":", color=c, lw=1, alpha=0.5)            # the unstable mean, for reference
        bs = cache["by_size"][nm]
        # mask empty bins: above ~$31k no trade is captured at this depth, so plotting 0
        # there reads as "profitable" when it is really "no data". End the line instead.
        nonempty = np.array(d["all_count"], dtype=float) > 0
        tot = np.where(nonempty, bs["total_usd"], np.nan)
        vol = np.where(nonempty, bs["volume"], np.nan)
        ax[1].plot(ctr, tot, "o-", color=c, lw=2, ms=4)
        ax[2].plot(ctr, vol, "o-", color=c, lw=2, ms=4)
    ax[0].set_title(f"Retail LP markout (bps) by trade size\nmedian + IQR (solid), mean (dotted); bins ≥{min_count} trades",
                    fontsize=10, fontweight="bold")
    ax[0].set_ylabel("markout (bps)"); ax[0].legend(fontsize=8)
    ax[1].set_title("All trades: total LP markout ($) by trade size", fontsize=10.5, fontweight="bold")
    ax[1].set_ylabel("cumulative markout ($, per sim)")
    ax[2].set_title("Captured volume ($) by trade size", fontsize=10.5, fontweight="bold")
    ax[2].set_ylabel("volume ($, per sim)")
    for a in ax:
        a.set_xscale("log"); a.axhline(0, color="grey", lw=0.7); a.set_xlabel("captured notional per trade ($)"); _gs_bare(a)
    fig.tight_layout()
    return fig


def plot_gs_by_size_diag(diag: dict | None = None, min_count: int = 30):
    """Diagnostic for the two unusual by-size features. Top row — the LEFT-panel
    mean-markout rise: (A) retail markout mean vs median vs leave-one-out mean (drop
    the single most extreme trade); mean≈median≈drop1 ⇒ not a single outlier. (B)
    the per-bin retail COUNT (log) with the `min_count` gate ⇒ the rise sits in the
    sparse tail (52→7→1 trades). Bottom row — the CENTRE-panel total-$ spike: (C)
    total markout($) by size with the arb-only component overlaid ⇒ the spike is
    arb/LVR, not retail. (D) the largest-single-trade and top-5 share of the bin's
    |markout| ⇒ ~5% / ~20% across ~40 arb trades, so the spike is not outlier driven."""
    if diag is None:
        diag = load_guidestar_sensitivity()
    order = diag["meta"]["pool_order"]
    ctr = np.array(diag["by_size_diag"]["centers"])
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    def _arr(d, k):
        return np.array([np.nan if v is None else v for v in d[k]], dtype=float)

    for nm in order:
        c = GS_COLORS[nm]
        d = diag["by_size_diag"][nm]
        ax[0, 0].plot(ctr, _arr(d, "mean_bps"), ":", color=c, lw=1.4, alpha=0.8)
        ax[0, 0].plot(ctr, _arr(d, "median_bps"), "o-", color=c, lw=2, ms=4, label=_gs_short(nm))
        ax[0, 0].plot(ctr, _arr(d, "mean_drop1_bps"), "--", color=c, lw=1.2, alpha=0.8)
        ax[0, 1].plot(ctr, _arr(d, "count"), "o-", color=c, lw=2, ms=4, label=_gs_short(nm))
        nonempty = np.array(d["all_count"], dtype=float) > 0   # mask empty bins (no trades captured at this depth)
        ax[1, 0].plot(ctr, np.where(nonempty, _arr(d, "total_usd"), np.nan), "o-", color=c, lw=2, ms=4, label=_gs_short(nm))
        ax[1, 0].plot(ctr, np.where(nonempty, _arr(d, "arb_usd"), np.nan), "x--", color=c, lw=1.2, ms=4, alpha=0.8)
        ax[1, 1].plot(ctr, 100 * _arr(d, "top1_abs_share"), "o-", color=c, lw=2, ms=4, label=_gs_short(nm))
        ax[1, 1].plot(ctr, 100 * _arr(d, "top5_abs_share"), "s:", color=c, lw=1.2, ms=3, alpha=0.7)
    ax[0, 0].plot([], [], "k:", lw=1.4, label="mean"); ax[0, 0].plot([], [], "k-", lw=2, label="median")
    ax[0, 0].plot([], [], "k--", lw=1.2, label="mean drop-1")
    ax[0, 0].set_title("(A) Retail markout: mean vs median vs leave-one-out", fontsize=10, fontweight="bold")
    ax[0, 0].set_ylabel("markout (bps)"); ax[0, 0].legend(fontsize=7.5, ncol=2); ax[0, 0].axhline(0, color="grey", lw=0.7)
    ax[0, 1].axhline(min_count, color="grey", ls=":", lw=1)
    ax[0, 1].annotate(f"min_count = {min_count}", (ctr[0], min_count * 1.2), fontsize=8, color="grey")
    ax[0, 1].set_yscale("log"); ax[0, 1].set_title("(B) Retail trades per size bin", fontsize=10, fontweight="bold")
    ax[0, 1].set_ylabel("count"); ax[0, 1].legend(fontsize=8)
    ax[1, 0].set_title("(C) Total markout ($): all trades (solid) vs arb-only (×--)", fontsize=10, fontweight="bold")
    ax[1, 0].set_ylabel("markout ($, per sim)"); ax[1, 0].legend(fontsize=8); ax[1, 0].axhline(0, color="grey", lw=0.7)
    ax[1, 1].plot([], [], "k-", lw=2, label="largest trade"); ax[1, 1].plot([], [], "ks:", lw=1.2, label="top-5")
    ax[1, 1].set_title("(D) Share of bin |markout| from biggest trades", fontsize=10, fontweight="bold")
    ax[1, 1].set_ylabel("% of bin Σ|markout|"); ax[1, 1].legend(fontsize=7.5, ncol=2); ax[1, 1].set_ylim(0, 100)
    for a in ax.ravel():
        a.set_xscale("log"); a.set_xlabel("captured notional per trade ($)"); _gs_bare(a)
    fig.suptitle("§9 — by-size diagnostics: the left-panel rise (top) and the centre-panel spike (bottom)",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def plot_gs_arrival_decomp(diag: dict | None = None):
    """Decompose final LP markout into retail(+) and arb/LVR(−) components, plus
    retail volume captured, across the retail-arrival multiplier ($1M pool). Explains
    the breakeven chart: the retail component and captured volume scale up with the
    arrival rate, while the arb/LVR loss is roughly flat in it."""
    if diag is None:
        diag = load_guidestar_sensitivity()
    ad = diag["arrival_decomp"]
    order = diag["meta"]["pool_order"]
    m = np.array(ad["mults"], dtype=float)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), sharex=True)
    for nm in order:
        c = GS_COLORS[nm]
        ax[0].plot(m, ad[nm]["retail"], "o-", color=c, lw=2, ms=4, label=_gs_short(nm))
        ax[1].plot(m, ad[nm]["arb"], "o-", color=c, lw=2, ms=4)
        ax[2].plot(m, ad[nm]["volume"], "o-", color=c, lw=2, ms=4)
    ax[0].set_title("Retail markout (+) vs arrival", fontsize=11, fontweight="bold")
    ax[0].set_ylabel("final retail markout ($)"); ax[0].legend(fontsize=8, loc="upper left")
    ax[1].set_title("Arb / LVR markout (−) vs arrival", fontsize=11, fontweight="bold")
    ax[1].set_ylabel("final arb markout ($)")
    ax[2].set_title("Retail volume captured vs arrival", fontsize=11, fontweight="bold")
    ax[2].set_ylabel("retail volume ($, per sim)")
    for a in ax:
        a.axhline(0, color="grey", lw=0.7); a.set_xlabel("retail arrival multiplier (× base 0.46/block)"); _gs_bare(a)
    fig.suptitle("§9 — markout decomposition vs retail arrival ($1M pool)", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def _gs_crossings(ax, x, y, color):
    """Mark $0 crossings of a markout curve."""
    for i in range(len(x) - 1):
        if (y[i] < 0 <= y[i + 1]) or (y[i] >= 0 > y[i + 1]):
            xb = x[i] + (0 - y[i]) / (y[i + 1] - y[i]) * (x[i + 1] - x[i])
            ax.axvline(xb, color=color, ls=":", lw=1.1)


def plot_gs_sizedist(diag: dict | None = None):
    """Sensitivity of the comparison to the retail SIZE distribution ($1M pool).
    Top row — shift the MEAN (multiply every order size by m: mean & sd move together).
    Bottom row — shift the SPREAD/std (log-spread around the fixed median:
    size' = median·(size/median)^s; the median is pinned). Left column: final total
    LP markout ($) per pool with $0 crossings; right column: retail volume captured."""
    if diag is None:
        diag = load_guidestar_sensitivity()
    sd = diag["sizedist"]
    order = diag["meta"]["pool_order"]
    ms, ss = sd["mean_sweep"], sd["std_sweep"]
    base_mean = diag["sizedist"]["base_mean"]
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    xm = np.array(ms["means"], dtype=float)            # realized mean order size ($)
    xs = np.array(ss["log10std"], dtype=float)         # realized log10 spread
    for nm in order:
        c = GS_COLORS[nm]
        tm = np.array(ms[nm]["total"], dtype=float)
        ts = np.array(ss[nm]["total"], dtype=float)
        ax[0, 0].plot(xm, tm, "o-", color=c, lw=2, ms=4, label=_gs_short(nm)); _gs_crossings(ax[0, 0], xm, tm, c)
        ax[0, 1].plot(xm, ms[nm]["volume"], "o-", color=c, lw=2, ms=4)
        ax[1, 0].plot(xs, ts, "o-", color=c, lw=2, ms=4, label=_gs_short(nm)); _gs_crossings(ax[1, 0], xs, ts, c)
        ax[1, 1].plot(xs, ss[nm]["volume"], "o-", color=c, lw=2, ms=4)
    ax[0, 0].set_xscale("log"); ax[0, 1].set_xscale("log")
    ax[0, 0].axvline(base_mean, color="grey", ls="--", lw=1); ax[0, 1].axvline(base_mean, color="grey", ls="--", lw=1)
    ax[1, 0].axvline(diag["sizedist"]["base_log10std"], color="grey", ls="--", lw=1)
    ax[1, 1].axvline(diag["sizedist"]["base_log10std"], color="grey", ls="--", lw=1)
    ax[0, 0].set_title("MEAN shift — final total LP markout ($)", fontsize=10.5, fontweight="bold")
    ax[0, 0].set_ylabel("final markout ($)"); ax[0, 0].legend(fontsize=8)
    ax[0, 1].set_title("MEAN shift — retail volume captured ($)", fontsize=10.5, fontweight="bold")
    ax[0, 1].set_ylabel("retail volume ($, per sim)")
    for a in (ax[0, 0], ax[0, 1]):
        a.set_xlabel("mean order size ($, log)  —  dashed = calibrated")
    ax[1, 0].set_title("SPREAD shift — final total LP markout ($)", fontsize=10.5, fontweight="bold")
    ax[1, 0].set_ylabel("final markout ($)"); ax[1, 0].legend(fontsize=8)
    ax[1, 1].set_title("SPREAD shift — retail volume captured ($)", fontsize=10.5, fontweight="bold")
    ax[1, 1].set_ylabel("retail volume ($, per sim)")
    for a in (ax[1, 0], ax[1, 1]):
        a.set_xlabel("log10 spread of order size  —  dashed = calibrated")
    for a in ax.ravel():
        a.axhline(0, color="grey", lw=0.7); _gs_bare(a)
    fig.suptitle("§9 — sensitivity to the retail size distribution ($1M pool, 24 seeds)", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def plot_gs_breakeven(cache: dict | None = None):
    """Final cumulative LP markout ($) vs retail-arrival multiplier; $0 crossings marked."""
    if cache is None:
        cache = load_guidestar_backtest()
    be = cache["breakeven"]; mults = np.array(be["mults"])
    fig, ax = plt.subplots(figsize=(9, 5))
    for nm in cache["meta"]["pool_order"]:
        y = np.array(be["finals"][nm])
        ax.plot(mults, y, "o-", color=GS_COLORS[nm], lw=2, label=_gs_short(nm))
        for i in range(len(mults) - 1):
            if (y[i] < 0 <= y[i + 1]) or (y[i] <= 0 < y[i + 1]):
                xb = mults[i] + (0 - y[i]) / (y[i + 1] - y[i]) * (mults[i + 1] - mults[i])
                ax.axvline(xb, color=GS_COLORS[nm], ls=":", lw=1.2)
                ax.annotate(f"$0 at {xb:.1f}×", (xb, 0), color=GS_COLORS[nm], fontsize=8)
                break
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("retail arrival multiplier (× base 0.46/block)"); ax.set_ylabel("final cumulative LP markout ($)")
    ax.legend(fontsize=8); _gs_bare(ax)
    ax.set_title(f"Final cumulative markout vs retail arrival ({be['seeds']} seeds)", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def plot_gs_volume(cache: dict | None = None):
    """Retail volume captured ($, per sim) by the candidate pool, per pool and depth."""
    if cache is None:
        cache = load_guidestar_backtest()
    order = cache["meta"]["pool_order"]; depths = cache["meta"]["depths"]
    xpos = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, d in enumerate(depths):
        vols = [cache["volume"][str(d)][nm] for nm in order]
        ax.bar(xpos + (i - 0.5) * w, vols, w, label=f"${d/1e6:.1f}M depth",
               color=[GS_COLORS[n] for n in order], alpha=0.55 + 0.45 * i, edgecolor="black", lw=0.5)
    ax.set_xticks(xpos); ax.set_xticklabels([_gs_short(n) for n in order], fontsize=9)
    ax.set_ylabel("retail volume captured ($, per sim)"); ax.legend(fontsize=8)
    ax.set_title("Retail volume captured by the candidate pool", fontweight="bold", fontsize=11)
    _gs_bare(ax); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    return fig


# ============================ §11 — regime map + unifying strategy ============================
REGIME_COLORS = {"Guidestar": "#8e44ad", "flat 3.5bp": "#e67e22", "flat 5bp": "#2980b9",
                 "FlowAware": "#27ae60", "SizeAware": "#c0392b", "Unified": "#16a085"}
_STATIC3 = ["Guidestar", "flat 3.5bp", "flat 5bp"]


def load_guidestar_regime() -> dict:
    """Cached joint arrival×size regime sweep (guidestar_regime_cache.json,
    written by scripts/calibration/guidestar_regime.py)."""
    import json
    return json.loads((ANALYSIS_DIR / "guidestar_regime_cache.json").read_text())


def _regime_axes(ax, reg):
    """Label a heatmap's axes with the arrival / mean multipliers."""
    am, mm = reg["meta"]["arr_mults"], reg["meta"]["mean_mults"]
    ax.set_xticks(range(len(am))); ax.set_xticklabels([f"{m:g}×" for m in am], fontsize=8)
    ax.set_yticks(range(len(mm))); ax.set_yticklabels([f"{m:g}×" for m in mm], fontsize=8)
    ax.set_xlabel("retail arrival multiplier"); ax.set_ylabel("order-size mean multiplier")


def plot_gs_intuition_schematic():
    """Conceptual visualization of the §10 sensitivity intuition: the two flow
    channels and which fee policy wins in which (arrival × size) regime."""
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=(9.5, 7))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    # diagonal split: defend (upper-left, big & sparse) vs cheap (lower-right, small & frequent)
    ax.fill([0, 10, 0], [0, 10, 10], color="#8e44ad", alpha=0.10)      # upper-left triangle: defend
    ax.fill([0, 10, 10], [0, 0, 10], color="#e67e22", alpha=0.10)      # lower-right triangle: cheap
    ax.plot([0, 10], [0, 10], color="grey", ls="--", lw=1.3)
    ax.annotate("", xy=(9.4, 0.4), xytext=(0.6, 0.4), arrowprops=dict(arrowstyle="->", color="black", lw=1.4))
    ax.text(5, 0.05, "retail arrival rate  →", ha="center", fontsize=11, fontweight="bold")
    ax.annotate("", xy=(0.4, 9.4), xytext=(0.4, 0.6), arrowprops=dict(arrowstyle="->", color="black", lw=1.4))
    ax.text(0.05, 5, "order size  →", ha="center", va="center", rotation=90, fontsize=11, fontweight="bold")
    ax.text(2.6, 8.0, "DEFEND aggressively\n(Guidestar)", ha="center", fontsize=12, fontweight="bold", color="#8e44ad")
    ax.text(2.6, 7.0, "sparse + large ⇒ toxic flow dominates;\nhigher floor + directional skew\nminimise LVR", ha="center", fontsize=9, color="#5b2c6f")
    ax.text(7.4, 2.0, "go CHEAP\n(low / fast-decaying fee)", ha="center", fontsize=12, fontweight="bold", color="#b9620a")
    ax.text(7.4, 1.05, "frequent + small ⇒ benign flow dominates;\nlow fee wins the largest routed share", ha="center", fontsize=9, color="#7e4708")
    ax.text(5.0, 5.2, "middle ground: high volatility + heavy retail\n→ defend on the vol spike, keep the floor moderate",
            ha="center", fontsize=9.5, fontstyle="italic", color="dimgray",
            bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.8))
    # two flow-channel annotations
    ax.text(5.0, 9.6, "toxic ARB / LVR (−): scales with size × volatility · mitigated by a directional fee",
            ha="center", fontsize=9, color="#8e44ad",
            bbox=dict(boxstyle="round", fc="#f5eef8", ec="#8e44ad"))
    ax.text(5.0, 0.7 + 2.6, "", ha="center")
    ax.text(8.7, 4.6, "benign RETAIL (+spread):\nscales with arrival ×\ncaptured share (↑ at low fee)",
            ha="center", fontsize=9, color="#b9620a",
            bbox=dict(boxstyle="round", fc="#fdf2e9", ec="#e67e22"))
    ax.set_title("§11 — sensitivity intuition: two flow channels, two regimes\n"
                 "(FlowAware moves along the dashed diagonal as it senses the flow)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_gs_regime_map(reg: dict | None = None):
    """Data-grounded regime map over the arrival × size grid ($1M pool). Left — which
    static pool has the best final markout in each cell (the §10 intuition made
    spatial). Right — Guidestar minus the best flat baseline ($), with the $0
    boundary; positive = Guidestar's defense wins, negative = a flat low fee wins."""
    from matplotlib.colors import ListedColormap, TwoSlopeNorm
    if reg is None:
        reg = load_guidestar_regime()
    g = reg["grids"]
    tot = {nm: np.array(g[nm]["total"]) for nm in reg["meta"]["pools"]}
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    # (A) winner among the 3 static pools
    stack = np.dstack([tot[nm] for nm in _STATIC3])          # (mean, arr, 3)
    win = np.argmax(stack, axis=2)
    cmap = ListedColormap([REGIME_COLORS[nm] for nm in _STATIC3])
    ax[0].imshow(win, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=2, alpha=0.85)
    for i in range(win.shape[0]):
        for j in range(win.shape[1]):
            ax[0].text(j, i, _STATIC3[win[i, j]].replace("flat ", "").replace("Guidestar", "GS")[:4],
                       ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")
    _regime_axes(ax[0], reg); ax[0].set_title("(A) Best static pool by regime", fontsize=11, fontweight="bold")
    # (B) Guidestar - best flat
    bestflat = np.maximum(tot["flat 3.5bp"], tot["flat 5bp"])
    margin = tot["Guidestar"] - bestflat
    lim = float(np.abs(margin).max())
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    im = ax[1].imshow(margin, origin="lower", aspect="auto", cmap="RdBu_r", norm=norm)
    cs = ax[1].contour(margin, levels=[0], colors="black", linewidths=1.5)
    for i in range(margin.shape[0]):
        for j in range(margin.shape[1]):
            ax[1].text(j, i, f"{margin[i,j]:.0f}", ha="center", va="center", fontsize=7,
                       color="black" if abs(margin[i, j]) < lim * 0.6 else "white")
    _regime_axes(ax[1], reg)
    ax[1].set_title("(B) Guidestar − best flat ($); >0 ⇒ Guidestar wins", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04, label="markout difference ($)")
    fig.suptitle("§11 — arrival × size regime map ($1M pool, 12 seeds)", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def _adaptive_margin_panel(ax, reg, name, best_static, tot):
    from matplotlib.colors import TwoSlopeNorm
    margin = tot[name] - best_static
    lim = max(float(np.abs(margin).max()), 1.0)
    im = ax.imshow(margin, origin="lower", aspect="auto", cmap="RdBu",
                   norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim))
    for i in range(margin.shape[0]):
        for j in range(margin.shape[1]):
            ax.text(j, i, f"{margin[i,j]:.0f}", ha="center", va="center", fontsize=6.5,
                    color="black" if abs(margin[i, j]) < lim * 0.6 else "white")
    _regime_axes(ax, reg)
    return im


def plot_gs_adaptive(reg: dict | None = None):
    """Do the adaptive policies track the upper envelope of the two static regimes?
    Top row — each adaptive strategy minus the best of the three static pools ($) per
    cell (blue/≥0 ⇒ matches or beats the best static there; red/<0 ⇒ lags). Bottom row
    — arrival and size slices (statics + Unified vs the best-static envelope), and a
    tally of how many of the 30 cells each adaptive policy ties-or-beats. FlowAware
    blends on a block-level flow EWMA; SizeAware conditions on the incoming order size;
    Unified does both (size-conditioned retail + flow-gated arb defense)."""
    if reg is None:
        reg = load_guidestar_regime()
    g = reg["grids"]; am = reg["meta"]["arr_mults"]; mm = reg["meta"]["mean_mults"]
    pools = reg["meta"]["pools"]
    tot = {nm: np.array(g[nm]["total"]) for nm in pools}
    best_static = np.maximum.reduce([tot[nm] for nm in _STATIC3])
    adaptive = [nm for nm in ("FlowAware", "SizeAware", "Unified") if nm in pools]
    fig, ax = plt.subplots(2, 3, figsize=(17.5, 9.5))
    for k, nm in enumerate(adaptive):
        im = _adaptive_margin_panel(ax[0, k], reg, nm, best_static, tot)
        ax[0, k].set_title(f"({chr(65+k)}) {nm} − best static ($)", fontsize=10.5, fontweight="bold")
        fig.colorbar(im, ax=ax[0, k], fraction=0.046, pad=0.04)
    slice_pools = [p for p in (_STATIC3 + ["Unified"]) if p in pools]
    i1, j1 = mm.index(1.0), am.index(1.0)
    for nm in slice_pools:
        z = 3 if nm == "Unified" else 2
        ax[1, 0].plot(am, tot[nm][i1, :], "o-", color=REGIME_COLORS[nm], lw=2, ms=4, label=nm, zorder=z)
        ax[1, 1].plot(mm, tot[nm][:, j1], "o-", color=REGIME_COLORS[nm], lw=2, ms=4, label=nm, zorder=z)
    ax[1, 0].plot(am, best_static[i1, :], "k--", lw=1.1, alpha=0.7, label="best static")
    ax[1, 1].plot(mm, best_static[:, j1], "k--", lw=1.1, alpha=0.7, label="best static")
    ax[1, 0].set_xlabel("retail arrival multiplier (mean ×1)"); ax[1, 0].set_ylabel("final total markout ($)")
    ax[1, 0].set_title("(D) arrival slice", fontsize=10.5, fontweight="bold"); ax[1, 0].legend(fontsize=7.5)
    ax[1, 1].set_xlabel("order-size mean multiplier (arrival ×1)"); ax[1, 1].set_ylabel("final total markout ($)")
    ax[1, 1].set_title("(E) size slice", fontsize=10.5, fontweight="bold"); ax[1, 1].legend(fontsize=7.5)
    for a in (ax[1, 0], ax[1, 1]):
        a.axhline(0, color="grey", lw=0.7); _gs_bare(a)
    # (F) tally: cells (of 30) where each adaptive ties/beats the best static (within $1)
    ncells = best_static.size
    counts = [int(np.sum(tot[nm] - best_static >= -1.0)) for nm in adaptive]
    ax[1, 2].bar(range(len(adaptive)), counts, color=[REGIME_COLORS[nm] for nm in adaptive], edgecolor="black", lw=0.5)
    ax[1, 2].set_xticks(range(len(adaptive))); ax[1, 2].set_xticklabels(adaptive, fontsize=8)
    ax[1, 2].set_ylim(0, ncells); ax[1, 2].set_ylabel(f"cells ties/beats best static (of {ncells})")
    ax[1, 2].set_title("(F) envelope coverage", fontsize=10.5, fontweight="bold"); _gs_bare(ax[1, 2])
    for k, c in enumerate(counts):
        ax[1, 2].text(k, c + 0.4, str(c), ha="center", fontsize=9, fontweight="bold")
    fig.suptitle("§11 — adaptive policies vs the static pools ($1M pool, 12 seeds)", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


# ============================ §12 — Nezlobin depth-sweep backtest ============================
NZ_COLORS = {"Nezlobin (doc spec)": "#c0392b", "Guidestar (real params)": "#8e44ad",
             "flat 9bp (4.5/4.5, dyn off)": "#e67e22", "flat 5bp (incumbent)": "#2980b9"}


def load_nezlobin_backtest() -> dict:
    """Cached doc-aligned Nezlobin depth-sweep backtest
    (analysis/.../nezlobin_backtest_cache.json, from scripts/calibration/nezlobin_backtest.py)."""
    import json
    return json.loads((ANALYSIS_DIR / "nezlobin_backtest_cache.json").read_text())


def _nz_depths(cache):
    return sorted(cache["meta"]["depths"])


def plot_nz_depth_sweep(cache: dict | None = None):
    """Headline: final 15s-forward LP markout ($) vs pool depth ($0.5M–$5M), against
    the §8 full-market normalizer, split into total / retail(+) / arb-LVR(−)."""
    if cache is None:
        cache = load_nezlobin_backtest()
    order = cache["meta"]["pool_order"]
    depths = _nz_depths(cache)
    xs = np.array(depths) / 1e6
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6), sharex=True)
    comps = [("tot", "TOTAL LP markout"), ("ret", "Retail markout (+)"), ("arb", "Arb / LVR markout (−)")]
    for nm in order:
        for j, (c, _) in enumerate(comps):
            ys = [cache["finals"][f"{d:.0f}"][nm][c] for d in depths]
            ax[j].plot(xs, ys, "o-", color=NZ_COLORS[nm], lw=2, ms=5, label=_gs_short(nm))
    for j, (_, ttl) in enumerate(comps):
        ax[j].set_title(ttl, fontsize=11, fontweight="bold")
        ax[j].set_xlabel("pool depth ($M, log)"); ax[j].set_xscale("log")
        ax[j].set_xticks(xs); ax[j].set_xticklabels([f"{x:g}" for x in xs])
        ax[j].axhline(0, color="grey", lw=0.7); _gs_bare(ax[j])
    ax[0].set_ylabel("final cumulative markout ($)"); ax[0].legend(fontsize=8)
    d_m = cache["meta"]["primary_depth"] / 1e6
    fig.suptitle(f"§12 — doc-aligned Nezlobin vs baselines: final 15s LP markout by depth "
                 f"(§8 normalizer, {cache['meta']['seeds']} seeds, bottom-of-block arb)", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def plot_nz_volume(cache: dict | None = None):
    """Retail volume captured ($ per sim) vs pool depth, per pool."""
    if cache is None:
        cache = load_nezlobin_backtest()
    order = cache["meta"]["pool_order"]; depths = _nz_depths(cache)
    xs = np.array(depths) / 1e6
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for nm in order:
        ys = [cache["volume"][f"{d:.0f}"][nm] for d in depths]
        ax.plot(xs, ys, "o-", color=NZ_COLORS[nm], lw=2, ms=5, label=_gs_short(nm))
    ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels([f"{x:g}" for x in xs])
    ax.set_xlabel("pool depth ($M, log)"); ax.set_ylabel("retail volume captured ($, per sim)")
    ax.legend(fontsize=8); _gs_bare(ax)
    ax.set_title("§12 — retail volume captured by depth", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def plot_nz_cumulative(cache: dict | None = None):
    """Cumulative 15s-forward LP markout over the run at the primary depth (mean±sd,
    seeds), total + retail(+)/arb(−) components."""
    if cache is None:
        cache = load_nezlobin_backtest()
    x = np.array(cache["cumulative_steps"]); order = cache["meta"]["pool_order"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), sharex=True)
    comps = [("tot", "TOTAL LP markout"), ("ret", "Retail markout (+)"), ("arb", "Arb / LVR markout (−)")]
    for nm in order:
        for j, (c, _) in enumerate(comps):
            d = cache["cumulative"][nm][c]; mean = np.array(d["mean"]); sd = np.array(d["sd"])
            ax[j].plot(x, mean, color=NZ_COLORS[nm], lw=2, label=_gs_short(nm))
            ax[j].fill_between(x, mean - sd, mean + sd, color=NZ_COLORS[nm], alpha=0.10)
    for j, (_, ttl) in enumerate(comps):
        ax[j].set_title(ttl, fontsize=11, fontweight="bold"); ax[j].set_xlabel("step (12s)")
        ax[j].axhline(0, color="grey", lw=0.7); _gs_bare(ax[j])
    ax[0].set_ylabel("cumulative markout ($)"); ax[0].legend(fontsize=8, loc="upper left")
    d_m = cache["meta"]["primary_depth"] / 1e6
    fig.suptitle(f"§12 — cumulative 15s LP markout at ${d_m:g}M depth (mean±sd, {cache['meta']['seeds']} seeds)",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def plot_nz_histograms(cache: dict | None = None):
    """Per-trade 15s markout (bps) distribution at the primary depth (log + linear)."""
    if cache is None:
        cache = load_nezlobin_backtest()
    bins = np.array(cache["histogram"]["bins"]); ctr = 0.5 * (bins[:-1] + bins[1:])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for nm in cache["meta"]["pool_order"]:
        dens = np.array(cache["histogram"][nm])
        ax[0].plot(ctr, dens, drawstyle="steps-mid", lw=2, color=NZ_COLORS[nm], label=_gs_short(nm))
        ax[1].plot(ctr, dens, drawstyle="steps-mid", lw=2, color=NZ_COLORS[nm])
    ax[0].set_yscale("log"); ax[0].set_title("per-trade markout (bps) — log density (tails)", fontsize=11, fontweight="bold")
    ax[1].set_title("per-trade markout (bps) — linear (body)", fontsize=11, fontweight="bold")
    for a in ax:
        a.axvline(0, color="grey", lw=0.7); a.set_xlabel("LP markout per trade (bps)"); _gs_bare(a)
    ax[0].legend(fontsize=8)
    d_m = cache["meta"]["primary_depth"] / 1e6
    fig.suptitle(f"§12 — per-trade markout distribution at ${d_m:g}M depth", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def plot_nz_by_size(cache: dict | None = None, min_count: int = 30):
    """Markout by captured trade size at the primary depth: retail median-gated mean
    markout (bps), total markout ($) over all trades (empty bins masked), and captured
    volume ($), per pool."""
    if cache is None:
        cache = load_nezlobin_backtest()
    order = cache["meta"]["pool_order"]; ctr = np.array(cache["by_size"]["centers"])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for nm in order:
        bs = cache["by_size"][nm]; c = NZ_COLORS[nm]
        rm = np.array([np.nan if v is None else v for v in bs["retail_mean_bps"]], dtype=float)
        ne = np.array(bs["all_count"], dtype=float) > 0
        ax[0].plot(ctr, rm, "o-", color=c, lw=2, ms=4, label=_gs_short(nm))
        ax[1].plot(ctr, np.where(ne, bs["total_usd"], np.nan), "o-", color=c, lw=2, ms=4)
        ax[2].plot(ctr, np.where(ne, bs["volume"], np.nan), "o-", color=c, lw=2, ms=4)
    ax[0].set_title(f"Retail mean LP markout (bps), bins ≥{min_count}", fontsize=10.5, fontweight="bold")
    ax[0].set_ylabel("markout (bps)"); ax[0].legend(fontsize=8)
    ax[1].set_title("All trades: total LP markout ($)", fontsize=10.5, fontweight="bold")
    ax[1].set_ylabel("markout ($, per sim)")
    ax[2].set_title("Captured volume ($)", fontsize=10.5, fontweight="bold")
    ax[2].set_ylabel("volume ($, per sim)")
    for a in ax:
        a.set_xscale("log"); a.axhline(0, color="grey", lw=0.7); a.set_xlabel("captured notional per trade ($)"); _gs_bare(a)
    d_m = cache["meta"]["primary_depth"] / 1e6
    fig.suptitle(f"§12 — markout by captured trade size at ${d_m:g}M depth", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig
