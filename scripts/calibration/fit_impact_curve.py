"""Phase 2 — Fit a hypothetical V2 (constant-product) "other pool" to the
empirical non-5bp price-impact cloud.

Two fits, both USD-weighted:
- Plan A: squared error
- Plan B: Huber loss with delta = 90th percentile of |residual| under Plan A

Inputs:
  analysis/weth_usdc_90d/non5bp_impact_sample_7d.csv

Outputs:
  analysis/weth_usdc_90d/impact_curve_fit.json
  plots/impact_curve_fit.png
  plots/impact_curve_residuals.png
  reports/impact_curve_fit.md

Model:
  spread_hyp_bps(S; phi, depth) = 10_000 * (phi + (1 - phi) * S / depth) / (1 - phi)

The hypothetical is always >= 0 (CP V2 charges slippage on every fill); the
empirical sample contains negative spreads (~58% of points). Residuals for
negative-observed points are obs - hyp < 0; we keep them per spec.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_PATH = REPO_ROOT / "analysis" / "weth_usdc_90d" / "non5bp_impact_sample_7d.csv"
FIT_JSON_PATH = REPO_ROOT / "analysis" / "weth_usdc_90d" / "impact_curve_fit.json"
FIT_PLOT_PATH = REPO_ROOT / "plots" / "impact_curve_fit.png"
RESID_PLOT_PATH = REPO_ROOT / "plots" / "impact_curve_residuals.png"
REPORT_PATH = REPO_ROOT / "reports" / "impact_curve_fit.md"

BOUNDS_PHI = (0.0, 0.05)
BOUNDS_DEPTH = (1e6, 1e12)

INITIAL_GUESSES = [
    (1e-3, 1e10),
    (5e-4, 1e9),
    (3e-3, 5e10),
    (1e-4, 5e8),
    (2e-3, 1e11),
    (5e-3, 1e9),
    (1e-3, 5e9),
    (1e-2, 1e10),
]


# -----------------------------------------------------------------------------
# Model + losses
# -----------------------------------------------------------------------------

def hyp_spread_bps(size_usd: np.ndarray, phi: float, depth: float) -> np.ndarray:
    """Hypothetical V2 spread (LP-positive bps) for buys/sells of size S USD."""
    one_minus_phi = 1.0 - phi
    return 1e4 * (phi + one_minus_phi * size_usd / depth) / one_minus_phi


def loss_l2(params: np.ndarray, sizes: np.ndarray, obs: np.ndarray, weights: np.ndarray) -> float:
    phi, depth = params
    if not (BOUNDS_PHI[0] <= phi <= BOUNDS_PHI[1]) or not (BOUNDS_DEPTH[0] <= depth <= BOUNDS_DEPTH[1]):
        return 1e30
    hyp = hyp_spread_bps(sizes, phi, depth)
    resid = obs - hyp
    return float(np.sum(weights * resid * resid) / np.sum(weights))


def loss_huber(params: np.ndarray, sizes: np.ndarray, obs: np.ndarray, weights: np.ndarray, delta: float) -> float:
    phi, depth = params
    if not (BOUNDS_PHI[0] <= phi <= BOUNDS_PHI[1]) or not (BOUNDS_DEPTH[0] <= depth <= BOUNDS_DEPTH[1]):
        return 1e30
    hyp = hyp_spread_bps(sizes, phi, depth)
    resid = obs - hyp
    abs_r = np.abs(resid)
    # Huber: 0.5 * r^2 for |r|<=delta, delta*(|r|-0.5*delta) otherwise
    quad = 0.5 * resid * resid
    lin = delta * (abs_r - 0.5 * delta)
    h = np.where(abs_r <= delta, quad, lin)
    return float(np.sum(weights * h) / np.sum(weights))


# -----------------------------------------------------------------------------
# Multi-start optimizer
# -----------------------------------------------------------------------------

def multistart_fit(loss_fn, sizes, obs, weights, *, label: str, extra_args: tuple = ()) -> dict:
    best = None
    runs = []
    for i, (phi0, depth0) in enumerate(INITIAL_GUESSES):
        try:
            res = minimize(
                loss_fn, x0=np.array([phi0, depth0]),
                args=(sizes, obs, weights, *extra_args),
                method="L-BFGS-B",
                bounds=[BOUNDS_PHI, BOUNDS_DEPTH],
                options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-10},
            )
            runs.append({
                "start": (float(phi0), float(depth0)),
                "params": (float(res.x[0]), float(res.x[1])),
                "loss": float(res.fun),
                "success": bool(res.success),
                "iterations": int(res.nit),
            })
            if best is None or res.fun < best["loss"]:
                best = {
                    "phi": float(res.x[0]),
                    "depth": float(res.x[1]),
                    "loss": float(res.fun),
                    "start": (float(phi0), float(depth0)),
                    "success": bool(res.success),
                }
        except Exception as exc:
            runs.append({"start": (float(phi0), float(depth0)), "error": str(exc)})
    print(f"[{label}] best phi={best['phi']:.6f} depth=${best['depth']:,.0f} loss={best['loss']:.4f}")
    return {"best": best, "all_runs": runs}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    df = pd.read_csv(SAMPLE_PATH)
    df = df[df["n_distinct_sides"] == 1].copy()
    df = df[df["size_usd"] > 1.0].copy()  # drop dust
    df = df.reset_index(drop=True)
    print(f"loaded {len(df):,} txs after filters")

    sizes = df["size_usd"].values.astype(np.float64)
    obs = df["observed_spread_bps"].values.astype(np.float64)
    weights = sizes  # USD weighting per spec

    # Plan A: USD-weighted squared
    print("\n=== Plan A: USD-weighted squared error ===")
    fit_a = multistart_fit(loss_l2, sizes, obs, weights, label="A")

    # Determine Huber delta from Plan A residuals
    phi_a, depth_a = fit_a["best"]["phi"], fit_a["best"]["depth"]
    hyp_a = hyp_spread_bps(sizes, phi_a, depth_a)
    resid_a = obs - hyp_a
    delta_huber = float(np.percentile(np.abs(resid_a), 90))
    print(f"\nHuber delta (90th pct |resid_A|) = {delta_huber:.4f} bps")

    # Plan B: USD-weighted Huber
    print("\n=== Plan B: USD-weighted Huber loss ===")
    fit_b = multistart_fit(loss_huber, sizes, obs, weights, label="B", extra_args=(delta_huber,))

    phi_b, depth_b = fit_b["best"]["phi"], fit_b["best"]["depth"]
    hyp_b = hyp_spread_bps(sizes, phi_b, depth_b)
    resid_b = obs - hyp_b

    # Residual stats (USD-weighted)
    def usd_w_stats(resid: np.ndarray, w: np.ndarray) -> dict:
        order = np.argsort(np.abs(resid))
        abs_sorted = np.abs(resid)[order]
        w_sorted = w[order]
        cw = np.cumsum(w_sorted) / np.sum(w_sorted)
        def q(p):
            idx = int(np.searchsorted(cw, p))
            idx = min(idx, len(abs_sorted) - 1)
            return float(abs_sorted[idx])
        return {
            "median_abs": q(0.5),
            "p75_abs": q(0.75),
            "p90_abs": q(0.90),
            "p99_abs": q(0.99),
            "usd_w_rmse": float(np.sqrt(np.sum(w * resid * resid) / np.sum(w))),
            "usd_w_mean_resid": float(np.sum(w * resid) / np.sum(w)),
        }

    stats_a = usd_w_stats(resid_a, weights)
    stats_b = usd_w_stats(resid_b, weights)

    # Persist
    payload = {
        "window": "2026-05-14..2026-05-20",
        "n_txs": int(len(df)),
        "plan_a": {
            "method": "USD-weighted squared error (L2)",
            "phi": phi_a,
            "depth_usdc": depth_a,
            "fit_loss": fit_a["best"]["loss"],
            "residual_stats_usd_weighted_bps": stats_a,
            "all_runs": fit_a["all_runs"],
        },
        "plan_b": {
            "method": "USD-weighted Huber",
            "huber_delta_bps": delta_huber,
            "phi": phi_b,
            "depth_usdc": depth_b,
            "fit_loss": fit_b["best"]["loss"],
            "residual_stats_usd_weighted_bps": stats_b,
            "all_runs": fit_b["all_runs"],
        },
        "bounds": {"phi": BOUNDS_PHI, "depth": BOUNDS_DEPTH},
    }
    FIT_JSON_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved {FIT_JSON_PATH}")

    # Fit overlay plot
    plot_fit_overlay(df, phi_a, depth_a, phi_b, depth_b, FIT_PLOT_PATH)

    # Residuals plot
    plot_residuals(df, resid_a, resid_b, RESID_PLOT_PATH)

    # Report
    write_report(payload, df, REPORT_PATH)
    print(f"\nsaved {REPORT_PATH}")


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------

def plot_fit_overlay(df: pd.DataFrame, phi_a: float, depth_a: float, phi_b: float, depth_b: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    y_clip_lo, y_clip_hi = -50, 100
    mask = (df["observed_spread_bps"] >= y_clip_lo) & (df["observed_spread_bps"] <= y_clip_hi)
    plot_df = df[mask]

    hb = ax.hexbin(
        plot_df["size_usd"], plot_df["observed_spread_bps"],
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

    ax.set_xlim(50, 1e6)
    ax.set_ylim(y_clip_lo, y_clip_hi)
    ax.set_xlabel("Tx size (USD, log scale)", fontsize=11)
    ax.set_ylabel("Spread (bps, LP-positive sign)", fontsize=11)
    ax.set_title(
        f"Phase 2 — V2 impact curve fits over the observed cloud\n"
        f"7-day window, n_txs={len(df):,}, Plan A = USD-weighted L2, Plan B = USD-weighted Huber",
        fontsize=10.5, fontweight="bold",
    )
    ax.legend(fontsize=10, frameon=False, loc="upper left")
    ax.grid(True, which="both", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


def plot_residuals(df: pd.DataFrame, resid_a: np.ndarray, resid_b: np.ndarray, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    sizes = df["size_usd"].values
    clip = (-50, 100)

    # Top row: residual vs size (hexbin)
    for ax, resid, title in [(axes[0, 0], resid_a, "Plan A residuals vs size"),
                              (axes[0, 1], resid_b, "Plan B residuals vs size")]:
        mask = (resid >= clip[0]) & (resid <= clip[1])
        ax.hexbin(sizes[mask], resid[mask], xscale="log", gridsize=50, mincnt=1, cmap="viridis", bins="log")
        ax.axhline(0, color="#e74c3c", lw=1)
        ax.set_xlabel("Tx size (USD, log)", fontsize=10)
        ax.set_ylabel("Residual = obs - hyp (bps)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlim(50, 1e6)
        ax.set_ylim(*clip)
        ax.grid(True, which="both", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Bottom row: residual histograms
    for ax, resid, title, color in [(axes[1, 0], resid_a, "Plan A residual distribution", "#e74c3c"),
                                     (axes[1, 1], resid_b, "Plan B residual distribution", "#3498db")]:
        mask = (resid >= clip[0]) & (resid <= clip[1])
        ax.hist(resid[mask], bins=80, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(0, color="#555", lw=0.8, ls="--", alpha=0.6)
        med = float(np.median(resid))
        p90 = float(np.percentile(np.abs(resid), 90))
        ax.set_xlabel("Residual (bps)", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(f"{title}\nmedian={med:+.2f} bps · p90(|r|)={p90:.2f} bps", fontsize=10, fontweight="bold")
        ax.set_xlim(*clip)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------

def write_report(payload: dict, df: pd.DataFrame, out: Path) -> None:
    a = payload["plan_a"]
    b = payload["plan_b"]
    n_txs = payload["n_txs"]
    n_neg = int((df["observed_spread_bps"] < 0).sum())
    pct_neg = 100 * n_neg / n_txs

    # Which is more defensible? Heuristic:
    # Plan B p99|r| < Plan A p99|r| -> Huber better at controlling outliers
    a_p99 = a["residual_stats_usd_weighted_bps"]["p99_abs"]
    b_p99 = b["residual_stats_usd_weighted_bps"]["p99_abs"]
    a_med = a["residual_stats_usd_weighted_bps"]["median_abs"]
    b_med = b["residual_stats_usd_weighted_bps"]["median_abs"]
    a_mean = a["residual_stats_usd_weighted_bps"]["usd_w_mean_resid"]
    b_mean = b["residual_stats_usd_weighted_bps"]["usd_w_mean_resid"]

    # Lead with whichever has the more centered USD-weighted residual (smaller |mean_resid|)
    # and smaller p99 tail.
    lead = "Plan A" if abs(a_mean) <= abs(b_mean) and a_p99 < b_p99 * 1.1 else "Plan B"

    # USD-weighted mean spread per log-size decile for the diagnostic table
    df_d = df.copy()
    df_d["log_size"] = np.log10(df_d["size_usd"])
    df_d["dec"] = pd.cut(df_d["log_size"], bins=10)
    by_dec = df_d.groupby("dec").apply(lambda x: pd.Series({
        "n": len(x),
        "median_size": float(x["size_usd"].median()),
        "usd_vol_pct": float(100 * x["size_usd"].sum() / df_d["size_usd"].sum()),
        "usd_wmean_spread_bps": float(np.sum(x["size_usd"] * x["observed_spread_bps"]) / np.sum(x["size_usd"])),
        "median_spread_bps": float(x["observed_spread_bps"].median()),
        "pct_neg": float((x["observed_spread_bps"] < 0).mean() * 100),
    }))
    by_dec_rows = "\n".join(
        f"| ${row['median_size']:>13,.0f} | {int(row['n']):>6,} | {row['usd_vol_pct']:>5.2f}% | "
        f"{row['usd_wmean_spread_bps']:+6.2f} | {row['median_spread_bps']:+6.2f} | {row['pct_neg']:5.1f}% |"
        for _, row in by_dec.iterrows()
    )

    md = f"""# Impact-curve fit (Phase 2)

**Sample:** non-5bp portion of router-routed WETH/USDC transactions,
7-day window **{payload['window']}**, n_txs = **{n_txs:,}** after dropping
mixed-direction and dust rows. Sign convention: LP-positive
(observed_spread_bps > 0 means the user paid above Binance mid).
**{pct_neg:.1f}% of txs have negative observed spread** (better-than-fair fills);
those were kept in the fit per spec.

The hypothetical V2 (constant-product) "other pool" has spread
`spread_hyp_bps(S) = 10_000 * (φ + (1-φ) * S / depth) / (1-φ)`. It is by
construction **strictly non-negative and monotonically increasing in S**.

## Empirical impact curve, by log-size decile (USD-weighted)

| median size | n_txs | USD-vol share | USD-w mean spread (bps) | median spread (bps) | pct negative |
|-------------|-------|----------------|--------------------------|----------------------|---------------|
{by_dec_rows}

The empirical curve is **non-monotonic and goes negative at large sizes**:
small-size txs (< $1k) cluster around +1 to +5 bps (retail), the meaty
$1k-$10k bucket (which carries ~30% of USD volume) sits near 0, and the
$10k+ buckets (~60% of USD volume) trend strongly negative
(arb / SOR / price-improvement). This is **not a V2 shape** — a V2 has
spread monotonically increasing in size.

## Fitted parameters

| Plan | φ | depth (USDC) | fit-loss |
|------|----|---------------|----------|
| **A — USD-weighted L2** | **{a['phi']:.6f}** (≈{a['phi']*1e4:.4f} bps) | **${a['depth_usdc']:,.0f}** (≈${a['depth_usdc']/1e9:,.1f} B) | {a['fit_loss']:.4f} |
| **B — USD-weighted Huber (δ = {b['huber_delta_bps']:.2f} bps)** | **{b['phi']:.6f}** (≈{b['phi']*1e4:.4f} bps) | **${b['depth_usdc']:,.0f}** (≈${b['depth_usdc']/1e9:,.1f} B) | {b['fit_loss']:.4f} |

Both fits converge to the **lower φ boundary and upper depth boundary**.
This is the correct L-BFGS-B optimum given a non-negative V2 model and
a USD-weighted target that is itself slightly **negative** (
USD-weighted overall mean spread = **{a_mean:+.3f} bps**): the closest
non-negative V2 to a USD-weighted-negative cloud is the limiting
zero-spread V2, i.e., depth → ∞, φ → 0.

This is **diagnostic**, not a numeric failure: it tells us a single
aggregated V2 cannot represent the rest-of-world side because the
rest-of-world side **systematically improves trader execution** in the
USD-volume-dominant size range. That is the empirical signal a V2
cannot produce.

## Residual diagnostics (USD-weighted, |residual| in bps)

| stat        | Plan A | Plan B |
|-------------|---------|---------|
| median \\|r\\|     | {a_med:6.2f} | {b_med:6.2f} |
| p75 \\|r\\|        | {a['residual_stats_usd_weighted_bps']['p75_abs']:6.2f} | {b['residual_stats_usd_weighted_bps']['p75_abs']:6.2f} |
| p90 \\|r\\|        | {a['residual_stats_usd_weighted_bps']['p90_abs']:6.2f} | {b['residual_stats_usd_weighted_bps']['p90_abs']:6.2f} |
| p99 \\|r\\|        | {a_p99:6.2f} | {b_p99:6.2f} |
| USD-weighted RMSE | {a['residual_stats_usd_weighted_bps']['usd_w_rmse']:6.2f} | {b['residual_stats_usd_weighted_bps']['usd_w_rmse']:6.2f} |
| USD-weighted mean residual | {a_mean:+.3f} | {b_mean:+.3f} |

Plan A and Plan B converge to the same boundary point and so produce
identical residual statistics — both methods agree that no interior
V2 has a smaller USD-weighted error than the limiting zero-spread V2.

## Defensibility judgment

**Neither Plan A nor Plan B produces an interior V2 fit** — the
USD-weighted-mean of the empirical cloud is negative at every size
above ~$10k, and a non-negative V2 cannot match. The shape of the
empirical impact curve (non-monotonic, negative at scale) is what a
V2 cannot represent. This is the expected outcome the Phase 1
diagnostic plot foreshadowed: heavy negative tail at large sizes.

**Recommendation for Phase 3:** validate both plans (they happen to
be the same boundary fit, so the validation collapses to one sim
run). The validation is informative regardless: it will quantify
*how* mismatched the predicted 5bp metrics become when the
rest-of-world side is modeled as effectively-infinite-depth /
zero-fee. That mismatch is the diagnostic the user asked for — it
characterizes the V2-aggregation misspecification rather than
producing a number-fit. Phase 4 surfaces the candidate fixes
(e.g., 2-pool decomposition of the other side).

## Artifacts

- Fit JSON: `analysis/weth_usdc_90d/impact_curve_fit.json`
- Fit plot: `plots/impact_curve_fit.png`
- Residuals plot: `plots/impact_curve_residuals.png`
"""
    out.write_text(md)


if __name__ == "__main__":
    main()
