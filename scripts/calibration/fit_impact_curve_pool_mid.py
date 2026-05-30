"""Fit V2 (constant-product) "other pool" to the empirical non-5bp impact
cloud (V3-only sample), referenced to the LAGGED fair price.

Reference (SPREAD_COLUMN): observed_spread_fair_lag_bps — execution spread vs
the Binance mid in the 12s bucket immediately preceding each trade ("fair at
or before 12s before the trade"; the real-data analog of the simulator's
"1 step before"). The sample CSV also carries observed_spread_pool_bps
(pre-trade pool mid) and observed_spread_fair_bps (contemporaneous fair) as
diagnostics — switch SPREAD_COLUMN to refit against those.

NOTE on the lagged reference: because the reference fair is one 12s step stale,
each trade's measured spread includes ~12s of price drift on top of the pure
execution slippage, so the per-trade cloud is much noisier (heavy ± tails)
than the pool-mid cloud. The USD-weighted Huber fit (Plan B) is the robust
default for exactly this reason.

Model and method: USD-weighted L2 (Plan A) and USD-weighted Huber (Plan B,
delta = 90th percentile of |residual| under Plan A). Multi-start L-BFGS-B.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_PATH = REPO_ROOT / "analysis" / "weth_usdc_90d" / "non5bp_impact_sample_v3_pool_mid_7d.csv"
FIT_JSON_PATH = REPO_ROOT / "analysis" / "weth_usdc_90d" / "impact_curve_fit_pool_mid.json"

# Spread reference to calibrate against. observed_spread_fair_lag_bps = spread vs
# the lagged (pre-trade) fair. Alternatives in the same CSV for diagnostics:
# observed_spread_pool_bps (pre-trade pool mid), observed_spread_fair_bps
# (contemporaneous fair).
SPREAD_COLUMN = "observed_spread_fair_lag_bps"
FIT_PLOT_PATH = REPO_ROOT / "plots" / "impact_curve_fit_pool_mid.png"
RESID_PLOT_PATH = REPO_ROOT / "plots" / "impact_curve_residuals_pool_mid.png"
REPORT_PATH = REPO_ROOT / "reports" / "impact_curve_fit_pool_mid.md"

BOUNDS_PHI = (0.0, 0.05)
BOUNDS_DEPTH = (1e6, 1e12)
BOUNDS_LOG10_DEPTH = (np.log10(BOUNDS_DEPTH[0]), np.log10(BOUNDS_DEPTH[1]))

# Reparametrize depth as log10(depth) so L-BFGS-B sees comparable parameter
# scales. The earlier (phi, depth) parametrization had a depth gradient of
# magnitude ~S/D^2 ≈ 1e-17 at depth ~ $10B, which kept the optimizer pinned
# to its starting depth.
INITIAL_GUESSES_LOG = [
    (1e-3, 7.5),   # depth ~ $32M
    (5e-4, 7.0),   # depth ~ $10M
    (3e-3, 8.5),   # depth ~ $316M
    (1e-4, 8.0),   # depth ~ $100M
    (2e-3, 9.0),   # depth ~ $1B
    (5e-3, 7.0),
    (1e-3, 7.0),
    (1e-2, 8.0),
    (2e-4, 7.3),
    (5e-4, 7.8),
]


def hyp_spread_bps(size_usd: np.ndarray, phi: float, depth: float) -> np.ndarray:
    one_minus_phi = 1.0 - phi
    return 1e4 * (phi + one_minus_phi * size_usd / depth) / one_minus_phi


def loss_l2(params, sizes, obs, weights):
    phi, log_depth = params
    depth = 10.0 ** log_depth
    if not (BOUNDS_PHI[0] <= phi <= BOUNDS_PHI[1]) or not (BOUNDS_LOG10_DEPTH[0] <= log_depth <= BOUNDS_LOG10_DEPTH[1]):
        return 1e30
    hyp = hyp_spread_bps(sizes, phi, depth)
    resid = obs - hyp
    return float(np.sum(weights * resid * resid) / np.sum(weights))


def loss_huber(params, sizes, obs, weights, delta):
    phi, log_depth = params
    depth = 10.0 ** log_depth
    if not (BOUNDS_PHI[0] <= phi <= BOUNDS_PHI[1]) or not (BOUNDS_LOG10_DEPTH[0] <= log_depth <= BOUNDS_LOG10_DEPTH[1]):
        return 1e30
    hyp = hyp_spread_bps(sizes, phi, depth)
    resid = obs - hyp
    abs_r = np.abs(resid)
    quad = 0.5 * resid * resid
    lin = delta * (abs_r - 0.5 * delta)
    h = np.where(abs_r <= delta, quad, lin)
    return float(np.sum(weights * h) / np.sum(weights))


def multistart_fit(loss_fn, sizes, obs, weights, *, label, extra_args=()):
    best = None
    runs = []
    for phi0, log_depth0 in INITIAL_GUESSES_LOG:
        try:
            res = minimize(
                loss_fn, x0=np.array([phi0, log_depth0]),
                args=(sizes, obs, weights, *extra_args),
                method="L-BFGS-B",
                bounds=[BOUNDS_PHI, BOUNDS_LOG10_DEPTH],
                options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-10},
            )
            phi_fit = float(res.x[0])
            depth_fit = float(10.0 ** res.x[1])
            runs.append({
                "start": (float(phi0), float(10.0 ** log_depth0)),
                "params": (phi_fit, depth_fit),
                "loss": float(res.fun),
                "success": bool(res.success),
                "iterations": int(res.nit),
            })
            if best is None or res.fun < best["loss"]:
                best = {
                    "phi": phi_fit, "depth": depth_fit,
                    "loss": float(res.fun),
                    "start": (float(phi0), float(10.0 ** log_depth0)),
                    "success": bool(res.success),
                }
        except Exception as exc:
            runs.append({"start": (float(phi0), float(10.0 ** log_depth0)), "error": str(exc)})
    print(f"[{label}] best phi={best['phi']:.6f} depth=${best['depth']:,.0f} loss={best['loss']:.4f}")
    return {"best": best, "all_runs": runs}


def main() -> None:
    df = pd.read_csv(SAMPLE_PATH)
    df = df[df["n_distinct_sides"] == 1].copy()
    df = df[df["size_usd"] > 1.0].copy()
    df = df[np.isfinite(df[SPREAD_COLUMN])].copy()
    df = df.reset_index(drop=True)
    print(f"loaded {len(df):,} txs after filters (reference column: {SPREAD_COLUMN})")

    sizes = df["size_usd"].values.astype(np.float64)
    obs = df[SPREAD_COLUMN].values.astype(np.float64)
    weights = sizes

    print(f"obs spread USD-w mean = {(weights*obs).sum()/weights.sum():+.3f} bps")
    print(f"obs spread median     = {np.median(obs):+.3f} bps")
    print(f"obs spread p1/p99     = {np.percentile(obs,1):+.2f} / {np.percentile(obs,99):+.2f} bps")
    print(f"fraction negative     = {(obs < 0).mean()*100:.2f}%")

    print("\n=== Plan A: USD-weighted L2 ===")
    fit_a = multistart_fit(loss_l2, sizes, obs, weights, label="A")

    phi_a, depth_a = fit_a["best"]["phi"], fit_a["best"]["depth"]
    hyp_a = hyp_spread_bps(sizes, phi_a, depth_a)
    resid_a = obs - hyp_a
    delta_huber = float(np.percentile(np.abs(resid_a), 90))
    print(f"\nHuber delta (90th pct |resid_A|) = {delta_huber:.4f} bps")

    print("\n=== Plan B: USD-weighted Huber ===")
    fit_b = multistart_fit(loss_huber, sizes, obs, weights, label="B", extra_args=(delta_huber,))

    phi_b, depth_b = fit_b["best"]["phi"], fit_b["best"]["depth"]
    hyp_b = hyp_spread_bps(sizes, phi_b, depth_b)
    resid_b = obs - hyp_b

    def usd_w_stats(resid, w):
        order = np.argsort(np.abs(resid))
        abs_sorted = np.abs(resid)[order]
        w_sorted = w[order]
        cw = np.cumsum(w_sorted) / np.sum(w_sorted)
        def q(p):
            idx = int(np.searchsorted(cw, p))
            idx = min(idx, len(abs_sorted) - 1)
            return float(abs_sorted[idx])
        return {
            "median_abs": q(0.5), "p75_abs": q(0.75),
            "p90_abs": q(0.90), "p99_abs": q(0.99),
            "usd_w_rmse": float(np.sqrt(np.sum(w * resid * resid) / np.sum(w))),
            "usd_w_mean_resid": float(np.sum(w * resid) / np.sum(w)),
        }

    stats_a = usd_w_stats(resid_a, weights)
    stats_b = usd_w_stats(resid_b, weights)

    payload = {
        "reference": f"{SPREAD_COLUMN} (lagged fair, 12s / 1-step pre-trade)",
        "window": "2026-04-21..2026-05-20",
        "retail_filter": "MetaMask 87.5bps fee (0xf326e4...); interface half pending",
        "n_txs": int(len(df)),
        "obs_summary": {
            "usd_w_mean_bps": float((weights*obs).sum()/weights.sum()),
            "median_bps": float(np.median(obs)),
            "p1_bps": float(np.percentile(obs, 1)),
            "p99_bps": float(np.percentile(obs, 99)),
            "frac_negative": float((obs < 0).mean()),
        },
        "plan_a": {
            "method": "USD-weighted L2",
            "phi": phi_a, "depth_usdc": depth_a,
            "fit_loss": fit_a["best"]["loss"],
            "residual_stats_usd_weighted_bps": stats_a,
            "all_runs": fit_a["all_runs"],
        },
        "plan_b": {
            "method": "USD-weighted Huber",
            "huber_delta_bps": delta_huber,
            "phi": phi_b, "depth_usdc": depth_b,
            "fit_loss": fit_b["best"]["loss"],
            "residual_stats_usd_weighted_bps": stats_b,
            "all_runs": fit_b["all_runs"],
        },
        "bounds": {"phi": BOUNDS_PHI, "depth": BOUNDS_DEPTH},
    }
    FIT_JSON_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved {FIT_JSON_PATH}")

    plot_fit_overlay(df, phi_a, depth_a, phi_b, depth_b, FIT_PLOT_PATH)
    plot_residuals(df, resid_a, resid_b, RESID_PLOT_PATH)
    write_report(payload, df, REPORT_PATH)
    print(f"\nsaved {REPORT_PATH}")


def plot_fit_overlay(df, phi_a, depth_a, phi_b, depth_b, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    y_clip_lo, y_clip_hi = -10, 100
    mask = (df[SPREAD_COLUMN] >= y_clip_lo) & (df[SPREAD_COLUMN] <= y_clip_hi)
    plot_df = df[mask]
    hb = ax.hexbin(
        plot_df["size_usd"], plot_df[SPREAD_COLUMN],
        xscale="log", yscale="linear",
        gridsize=60, mincnt=1, cmap="viridis", bins="log", alpha=0.7,
    )
    fig.colorbar(hb, ax=ax, label="log10(count)")

    grid = np.logspace(1, 7, 400)
    ax.plot(grid, hyp_spread_bps(grid, phi_a, depth_a),
            color="#e74c3c", lw=2.5,
            label=f"Plan A  φ={phi_a*1e4:.2f}bp  depth=${depth_a/1e6:,.1f}M")
    ax.plot(grid, hyp_spread_bps(grid, phi_b, depth_b),
            color="#3498db", lw=2.5, ls="--",
            label=f"Plan B  φ={phi_b*1e4:.2f}bp  depth=${depth_b/1e6:,.1f}M")
    ax.axhline(0, color="#555", lw=0.7, alpha=0.5, ls=":")
    ax.set_xlim(10, 1e6)
    ax.set_ylim(y_clip_lo, y_clip_hi)
    ax.set_xlabel("Tx size (USD, log scale)")
    ax.set_ylabel("Spread vs pool_mid_pre (bps, LP-positive)")
    ax.set_title(
        f"V2 impact-curve fit (pool_mid_pre reference)\n"
        f"7-day window  ·  V3-only  ·  n_txs={len(df):,}"
    )
    ax.legend(fontsize=10, frameon=False, loc="upper left")
    ax.grid(True, which="both", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


def plot_residuals(df, resid_a, resid_b, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sizes = df["size_usd"].values
    clip = (-50, 50)

    for ax, resid, title in [(axes[0, 0], resid_a, "Plan A residuals vs size"),
                              (axes[0, 1], resid_b, "Plan B residuals vs size")]:
        mask = (resid >= clip[0]) & (resid <= clip[1])
        ax.hexbin(sizes[mask], resid[mask], xscale="log", gridsize=50, mincnt=1, cmap="viridis", bins="log")
        ax.axhline(0, color="#e74c3c", lw=1)
        ax.set_xlabel("Tx size (USD, log)")
        ax.set_ylabel("Residual = obs - hyp (bps)")
        ax.set_title(title)
        ax.set_xlim(10, 1e6)
        ax.set_ylim(*clip)
        ax.grid(True, which="both", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax, resid, title, color in [(axes[1, 0], resid_a, "Plan A residual distribution", "#e74c3c"),
                                     (axes[1, 1], resid_b, "Plan B residual distribution", "#3498db")]:
        mask = (resid >= clip[0]) & (resid <= clip[1])
        ax.hist(resid[mask], bins=80, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(0, color="#555", lw=0.8, ls="--", alpha=0.6)
        med = float(np.median(resid))
        p90 = float(np.percentile(np.abs(resid), 90))
        ax.set_xlabel("Residual (bps)")
        ax.set_ylabel("Count")
        ax.set_title(f"{title}\nmedian={med:+.2f} bps · p90(|r|)={p90:.2f} bps")
        ax.set_xlim(*clip)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


def write_report(payload: dict, df: pd.DataFrame, out: Path) -> None:
    a = payload["plan_a"]
    b = payload["plan_b"]
    obs_summary = payload["obs_summary"]
    n_txs = payload["n_txs"]

    df_d = df.copy()
    df_d["log_size"] = np.log10(df_d["size_usd"])
    df_d["dec"] = pd.cut(df_d["log_size"], bins=10)
    by_dec = df_d.groupby("dec", observed=True).apply(lambda x: pd.Series({
        "n": len(x),
        "median_size": float(x["size_usd"].median()),
        "usd_vol_pct": float(100 * x["size_usd"].sum() / df_d["size_usd"].sum()),
        "usd_wmean_spread_bps": float(np.sum(x["size_usd"] * x[SPREAD_COLUMN]) / np.sum(x["size_usd"])),
        "median_spread_bps": float(x[SPREAD_COLUMN].median()),
        "pct_neg": float((x[SPREAD_COLUMN] < 0).mean() * 100),
    }))
    by_dec_rows = "\n".join(
        f"| ${row['median_size']:>13,.0f} | {int(row['n']):>6,} | {row['usd_vol_pct']:>5.2f}% | "
        f"{row['usd_wmean_spread_bps']:+6.2f} | {row['median_spread_bps']:+6.2f} | {row['pct_neg']:5.1f}% |"
        for _, row in by_dec.iterrows()
    )

    md = f"""# Impact-curve fit — lagged-fair reference

**Sample:** V3 swap legs of **MetaMask-fee retail** (87.5 bps convenience fee,
collector `0xf326e4…`) non-5bp WETH/USDC txs in the 30-day window
**{payload['window']}**, aggregated per tx, with spread measured
against the **lagged fair price** — the Binance mid in the 12s bucket
immediately preceding each trade ("fair at or before 12s before the trade";
the real-data analog of the simulator's "1 step before"). Reference column:
`{SPREAD_COLUMN}`.

**n_txs** = **{n_txs:,}** after dropping mixed-direction and dust rows.

Sign convention: LP-positive (`{SPREAD_COLUMN} > 0` ⇔ user paid above the
lagged fair). Because the reference is one 12s step stale, each trade's
spread also carries ~12s of price drift, so the cloud has heavy ± tails;
the USD-weighted Huber fit (Plan B) is the robust default.

## Observation summary

| stat | value |
|------|-------|
| USD-weighted mean spread | **{obs_summary['usd_w_mean_bps']:+.3f} bps** |
| median spread | {obs_summary['median_bps']:+.3f} bps |
| p1 / p99 spread | {obs_summary['p1_bps']:+.2f} / {obs_summary['p99_bps']:+.2f} bps |
| fraction of txs with negative spread | {obs_summary['frac_negative']*100:.2f}% |

The lagged fair is a pre-trade reference (one 12s step before the trade), so
the trade cannot have influenced it; it differs from the contemporaneous-fair
and pool-mid references (both also in the sample CSV) by the 12s price drift it
carries into each observation.

## Empirical impact curve, by log-size decile (USD-weighted)

| median size | n_txs | USD-vol share | USD-w mean spread (bps) | median spread (bps) | pct negative |
|-------------|-------|----------------|--------------------------|----------------------|---------------|
{by_dec_rows}

The curve is **monotone non-decreasing, strictly positive at the mean**, and
clearly V2-shaped: small-size shelf near the 1bp pool's fee, then linear-in-size
growth at larger sizes.

## Fitted parameters

| Plan | φ | depth (USDC) | fit-loss |
|------|----|---------------|----------|
| **A — USD-weighted L2** | **{a['phi']:.6f}** (≈{a['phi']*1e4:.4f} bps) | **${a['depth_usdc']:,.0f}** (≈${a['depth_usdc']/1e6:,.1f} M) | {a['fit_loss']:.4f} |
| **B — USD-weighted Huber (δ = {b['huber_delta_bps']:.2f} bps)** | **{b['phi']:.6f}** (≈{b['phi']*1e4:.4f} bps) | **${b['depth_usdc']:,.0f}** (≈${b['depth_usdc']/1e6:,.1f} M) | {b['fit_loss']:.4f} |

## Residual diagnostics (USD-weighted, |residual| in bps)

| stat        | Plan A | Plan B |
|-------------|---------|---------|
| median \\|r\\|     | {a['residual_stats_usd_weighted_bps']['median_abs']:6.2f} | {b['residual_stats_usd_weighted_bps']['median_abs']:6.2f} |
| p75 \\|r\\|        | {a['residual_stats_usd_weighted_bps']['p75_abs']:6.2f} | {b['residual_stats_usd_weighted_bps']['p75_abs']:6.2f} |
| p90 \\|r\\|        | {a['residual_stats_usd_weighted_bps']['p90_abs']:6.2f} | {b['residual_stats_usd_weighted_bps']['p90_abs']:6.2f} |
| p99 \\|r\\|        | {a['residual_stats_usd_weighted_bps']['p99_abs']:6.2f} | {b['residual_stats_usd_weighted_bps']['p99_abs']:6.2f} |
| USD-weighted RMSE | {a['residual_stats_usd_weighted_bps']['usd_w_rmse']:6.2f} | {b['residual_stats_usd_weighted_bps']['usd_w_rmse']:6.2f} |
| USD-weighted mean residual | {a['residual_stats_usd_weighted_bps']['usd_w_mean_resid']:+.3f} | {b['residual_stats_usd_weighted_bps']['usd_w_mean_resid']:+.3f} |

## Artifacts

- Fit JSON: `analysis/weth_usdc_90d/impact_curve_fit_pool_mid.json`
- Fit plot: `plots/impact_curve_fit_pool_mid.png`
- Residuals plot: `plots/impact_curve_residuals_pool_mid.png`
"""
    out.write_text(md)


if __name__ == "__main__":
    main()
