"""Held-out validation of the impact-curve-calibrated normalizer pool.

Framework:
  - Submission pool = 5bp Uniswap V3 WETH/USDC, frozen at on-chain config
    (fee=0.0005, virtual_depth_y=$212.16M). Never calibrated.
  - Normalizer pool = V2 with (phi, depth) fit to the empirical non-5bp
    router-routed impact curve (pool_mid_pre referenced). Loaded from
    impact_curve_fit_pool_mid.json (Plan B / USD-weighted Huber by default).

Validation metrics (all retail-only, all on the submission/5bp pool):
  1. Retail volume share at 5bp = retail_vol_5bp / retail_vol_total
  2. Retail fee share at 5bp    = retail_fees_5bp / retail_fees_total
  3. USD-weighted distribution of markout_15s on retail trades at 5bp
     (sim uses next-block fair_price ≈ 12s ≈ 15s convention).

Real-world targets pulled from:
  analysis/weth_usdc_90d/retail_5bp_share_summary.csv     (vol/fee shares)
  analysis/weth_usdc_90d/markout_5bp_pool_retail.csv      (per-swap, USD-weighted)

Outputs:
  analysis/weth_usdc_90d/validation_retail_pool_mid.json
  analysis/weth_usdc_90d/markout_5bp_pool_sim_retail.csv  (per-trade sim)
  plots/validation_retail_pool_mid.png
  reports/validation_retail_pool_mid.md
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from arena_eval.exact_simple_amm.config import ExactSimpleAMMConfig
from arena_eval.exact_simple_amm.simulator import ExactSimpleAMMSimulator
from arena_eval.exact_simple_amm.strategies import FixedFeeStrategy

ANALYSIS = REPO_ROOT / "analysis" / "weth_usdc_90d"
FIT_JSON           = ANALYSIS / "impact_curve_fit_pool_mid.json"
SHARES_CSV         = ANALYSIS / "retail_5bp_share_summary.csv"
REAL_MARKOUT_CSV   = ANALYSIS / "markout_5bp_pool_retail.csv"
OUT_JSON           = ANALYSIS / "validation_retail_pool_mid.json"
OUT_SIM_MARKOUT    = ANALYSIS / "markout_5bp_pool_sim_retail.csv"
PLOT_PATH          = REPO_ROOT / "plots" / "validation_retail_pool_mid.png"
REPORT_PATH        = REPO_ROOT / "reports" / "validation_retail_pool_mid.md"

# Submission pool (frozen)
REAL_POOL_FEE = 0.0005
REAL_VIRTUAL_USDC = 212_157_626.44

# Default plan: Plan B (USD-w Huber, more robust to the heavy upper tail of
# the impact curve). Override with PLAN env var or --plan_a flag.
DEFAULT_PLAN = "plan_b"

SEEDS = (42, 43, 44, 45, 46)
N_STEPS = 5_000


# ---------------------------------------------------------------------------
# Simulator runner
# ---------------------------------------------------------------------------

def build_sim(*, phi: float, depth_y: float, seed: int, n_steps: int = N_STEPS):
    initial_price = 100.0
    norm_y = depth_y
    norm_x = norm_y / initial_price
    frac = REAL_VIRTUAL_USDC / norm_y
    cfg = ExactSimpleAMMConfig(
        n_steps=n_steps,
        initial_price=initial_price,
        initial_x=norm_x,
        initial_y=norm_y,
        submission_liquidity_fraction=frac,
        evaluator_kind="real_data",
        price_process_kind="regime_switching",
        retail_flow_kind="empirical_usd_size",
        retail_arrival_rate=857_035 / 1_303_200,  # 6m cleaned (was 1_294_178 / 1_303_200; Wintermute removed)
        retail_buy_prob=0.4413,                   # 6m cleaned (was 0.4842)
        regime_invcdf_path=str(ANALYSIS / "regimes_invcdf.csv"),
        regime_transition_path=str(ANALYSIS / "regimes_transition_matrix.csv"),
        retail_usd_quantiles_path=str(ANALYSIS / "parent_order_usd_quantiles.csv"),
    )
    return ExactSimpleAMMSimulator(
        config=cfg,
        submission_strategy=FixedFeeStrategy(bid_fee=REAL_POOL_FEE, ask_fee=REAL_POOL_FEE),
        normalizer_strategy=FixedFeeStrategy(bid_fee=phi, ask_fee=phi),
        seed=seed,
    )


def _trade_edge(amount_x: float, amount_y: float, trader_side: str, fair: float) -> float:
    """LP edge per retail trade marked against `fair` (USDC/X)."""
    if trader_side == "sell_x":
        # Retail sells X; LP buys X.
        return amount_x * fair - amount_y
    return amount_y - amount_x * fair


def run_one_seed(*, phi: float, depth_y: float, seed: int) -> dict:
    """Returns aggregate shares + per-trade (usd, markout_bps) for retail
    trades on the submission pool. Per-trade markout uses next-block
    fair_price (≈12s, close to the empirical 15s convention)."""
    sim = build_sim(phi=phi, depth_y=depth_y, seed=seed)
    per_trade_usd: list[float] = []
    per_trade_mkt: list[float] = []
    pending: list[tuple[float, float, str, float]] = []
    while not sim.done:
        step = sim.step_once()
        new_fair = step["fair_price"]
        for amount_x, amount_y, trader_side, usd in pending:
            edge = _trade_edge(amount_x, amount_y, trader_side, new_fair)
            if amount_y > 0:
                per_trade_usd.append(usd)
                per_trade_mkt.append(edge / amount_y * 1e4)
        pending.clear()
        for ev in step["trade_events"]:
            if ev["source"] != "retail" or ev["venue"] != "submission":
                continue
            ax = float(ev["amount_x"]); ay = float(ev["amount_y"])
            side = str(ev["trader_side"])
            # USD weight = USDC leg of the trade (ay is in y-units = USDC).
            pending.append((ax, ay, side, ay))

    r = sim.result()
    arb_sub   = r.arb_volume_submission_y
    arb_norm  = r.arb_volume_normalizer_y
    ret_sub   = r.retail_volume_submission_y
    ret_norm  = r.retail_volume_normalizer_y

    ret_vol_share = ret_sub / max(ret_sub + ret_norm, 1e-12)
    # Fees: each pool charges its own rate on the volume routed to it.
    ret_fees_sub  = ret_sub  * REAL_POOL_FEE
    ret_fees_norm = ret_norm * phi
    ret_fee_share = ret_fees_sub / max(ret_fees_sub + ret_fees_norm, 1e-12)

    return {
        "seed": seed,
        "retail_volume_sub_y":  ret_sub,
        "retail_volume_norm_y": ret_norm,
        "retail_fees_sub_y":    ret_fees_sub,
        "retail_fees_norm_y":   ret_fees_norm,
        "arb_volume_sub_y":     arb_sub,
        "arb_volume_norm_y":    arb_norm,
        "retail_volume_share":  ret_vol_share,
        "retail_fee_share":     ret_fee_share,
        "per_trade_usd":        per_trade_usd,
        "per_trade_markout_bps": per_trade_mkt,
    }


# ---------------------------------------------------------------------------
# Weighted percentile helper
# ---------------------------------------------------------------------------

def weighted_percentile(values: np.ndarray, weights: np.ndarray, pct: float) -> float:
    order = np.argsort(values)
    v = values[order]; w = weights[order]
    cum = np.cumsum(w)
    cutoff = pct / 100.0 * cum[-1]
    idx = min(int(np.searchsorted(cum, cutoff)), len(v) - 1)
    return float(v[idx])


def quantile_grid(values: np.ndarray, weights: np.ndarray, pcts: list[int]) -> dict[int, float]:
    return {p: weighted_percentile(values, weights, p) for p in pcts}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def load_real_targets() -> dict:
    shares = {}
    with SHARES_CSV.open() as fh:
        for r in csv.reader(fh):
            if len(r) == 2 and r[0] in {"retail_volume_share_5bp", "retail_fee_share_5bp"}:
                shares[r[0]] = float(r[1])

    df = pd.read_csv(REAL_MARKOUT_CSV)
    return {
        "retail_volume_share_5bp": shares["retail_volume_share_5bp"],
        "retail_fee_share_5bp":    shares["retail_fee_share_5bp"],
        "markout_15s_per_swap_arr": df["markout_15s_bps"].to_numpy(),
        "usd_arr":                  df["usd_amount"].to_numpy(),
    }


def main(plan_key: str = DEFAULT_PLAN) -> None:
    fit = json.loads(FIT_JSON.read_text())
    plan = fit[plan_key]
    phi = plan["phi"]
    depth = plan["depth_usdc"]
    print(f"[{plan_key}] phi={phi:.6f}  depth=${depth:,.0f}")

    real = load_real_targets()

    per_seed = [run_one_seed(phi=phi, depth_y=depth, seed=s) for s in SEEDS]

    # Aggregate seed-level summaries
    vol_shares = np.array([s["retail_volume_share"] for s in per_seed])
    fee_shares = np.array([s["retail_fee_share"]    for s in per_seed])

    # Pool all retail trades across seeds for USD-weighted markout distribution
    sim_usd  = np.array(sum((s["per_trade_usd"]         for s in per_seed), []))
    sim_mkt  = np.array(sum((s["per_trade_markout_bps"] for s in per_seed), []))
    print(f"sim retail trades on submission pool (all seeds): {len(sim_usd):,}")

    pct_grid = [1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99]
    sim_usd_w = quantile_grid(sim_mkt, sim_usd, pct_grid)
    real_usd_w = quantile_grid(real["markout_15s_per_swap_arr"], real["usd_arr"], pct_grid)

    sim_usd_w_mean  = float((sim_mkt * sim_usd).sum() / sim_usd.sum())
    real_usd_w_mean = float(
        (real["markout_15s_per_swap_arr"] * real["usd_arr"]).sum()
        / real["usd_arr"].sum()
    )

    payload = {
        "framework": (
            "Calibrate (phi, depth) of the V2 normalizer pool on the empirical "
            "non-5bp impact curve (pool_mid_pre, retail-only); validate on the "
            "5bp submission pool's retail-only metrics, which are never seen at fit time."
        ),
        "submission_frozen": {
            "fee": REAL_POOL_FEE,
            "depth_y": REAL_VIRTUAL_USDC,
            "pool": "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640 (Uniswap V3 5bp WETH/USDC)",
        },
        "calibrated_normalizer": {
            "plan": plan_key,
            "phi": phi,
            "phi_bps": phi * 1e4,
            "depth_usdc": depth,
            "depth_usdc_m": depth / 1e6,
        },
        "real_targets": {
            "retail_volume_share_5bp": real["retail_volume_share_5bp"],
            "retail_fee_share_5bp":    real["retail_fee_share_5bp"],
            "markout_15s_usd_w_mean_bps": real_usd_w_mean,
            "markout_15s_usd_w_quantiles_bps": real_usd_w,
        },
        "sim_results": {
            "retail_volume_share_mean": float(vol_shares.mean()),
            "retail_volume_share_std":  float(vol_shares.std()),
            "retail_fee_share_mean":    float(fee_shares.mean()),
            "retail_fee_share_std":     float(fee_shares.std()),
            "markout_usd_w_mean_bps":   sim_usd_w_mean,
            "markout_usd_w_quantiles_bps": sim_usd_w,
            "per_seed": [
                {k: v for k, v in s.items() if k not in {"per_trade_usd", "per_trade_markout_bps"}}
                for s in per_seed
            ],
        },
        "seeds":  list(SEEDS),
        "n_steps": N_STEPS,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT_JSON}")

    # Save per-trade sim markouts so the notebook can do its own overlay
    OUT_SIM_MARKOUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT_SIM_MARKOUT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["usd_amount", "markout_bps"])
        for u, m in zip(sim_usd, sim_mkt):
            w.writerow([f"{u:.4f}", f"{m:.6f}"])
    print(f"wrote {OUT_SIM_MARKOUT}")

    # Plot
    plot_validation(payload, real, sim_usd, sim_mkt, PLOT_PATH)
    print(f"wrote {PLOT_PATH}")

    # Report
    write_report(payload, REPORT_PATH)
    print(f"wrote {REPORT_PATH}")

    # Console summary
    print()
    print("=== Validation summary ===")
    print(f"  retail vol share  real={real['retail_volume_share_5bp']*100:.2f}%  "
          f"sim={vol_shares.mean()*100:.2f}%  Δ={(vol_shares.mean()-real['retail_volume_share_5bp'])*100:+.2f}pp")
    print(f"  retail fee share  real={real['retail_fee_share_5bp']*100:.2f}%  "
          f"sim={fee_shares.mean()*100:.2f}%  Δ={(fee_shares.mean()-real['retail_fee_share_5bp'])*100:+.2f}pp")
    print(f"  markout USD-w μ   real={real_usd_w_mean:+.2f}bps  "
          f"sim={sim_usd_w_mean:+.2f}bps  Δ={sim_usd_w_mean-real_usd_w_mean:+.2f}bps")


def plot_validation(payload: dict, real: dict,
                    sim_usd: np.ndarray, sim_mkt: np.ndarray,
                    out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    r = payload["real_targets"]; s = payload["sim_results"]

    # (1) retail volume share bar
    axes[0].bar(["real", "sim"],
                [r["retail_volume_share_5bp"], s["retail_volume_share_mean"]],
                color=["#2c3e50", "#27ae60"])
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title("Retail volume share at 5bp pool", fontweight="bold")
    axes[0].set_ylabel("share")
    for i, v in enumerate([r["retail_volume_share_5bp"], s["retail_volume_share_mean"]]):
        axes[0].text(i, v + 0.01, f"{v*100:.1f}%", ha="center")

    # (2) retail fee share bar
    axes[1].bar(["real", "sim"],
                [r["retail_fee_share_5bp"], s["retail_fee_share_mean"]],
                color=["#2c3e50", "#27ae60"])
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("Retail fee share at 5bp pool", fontweight="bold")
    axes[1].set_ylabel("share")
    for i, v in enumerate([r["retail_fee_share_5bp"], s["retail_fee_share_mean"]]):
        axes[1].text(i, v + 0.01, f"{v*100:.1f}%", ha="center")

    # (3) markout_15s USD-weighted distribution overlay (kernel-style or hist)
    real_m = real["markout_15s_per_swap_arr"]
    real_w = real["usd_arr"]
    lo, hi = -25, 25
    bins = np.linspace(lo, hi, 60)
    axes[2].hist(np.clip(real_m, lo, hi), bins=bins, weights=real_w,
                 density=True, alpha=0.55, color="#2c3e50", label="real (USD-w)")
    axes[2].hist(np.clip(sim_mkt, lo, hi), bins=bins, weights=sim_usd,
                 density=True, alpha=0.55, color="#27ae60", label="sim (USD-w)")
    axes[2].axvline(r["markout_15s_usd_w_mean_bps"], color="#2c3e50", ls="--", lw=1)
    axes[2].axvline(s["markout_usd_w_mean_bps"],     color="#27ae60", ls="--", lw=1)
    axes[2].set_title("Retail markout_15s — USD-weighted density", fontweight="bold")
    axes[2].set_xlabel("markout (bps, LP-positive)")
    axes[2].legend(loc="upper right", fontsize=9)

    plan = payload["calibrated_normalizer"]
    fig.suptitle(
        f"Validation — submission=5bp (frozen), normalizer={plan['plan']}: "
        f"φ={plan['phi_bps']:.2f}bps, depth=${plan['depth_usdc_m']:.1f}M",
        fontweight="bold")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)


def write_report(payload: dict, out: Path) -> None:
    r = payload["real_targets"]; s = payload["sim_results"]
    p = payload["calibrated_normalizer"]

    def row(p_): return (
        f"| p{p_} | "
        f"{r['markout_15s_usd_w_quantiles_bps'][p_]:+.2f} | "
        f"{s['markout_usd_w_quantiles_bps'][p_]:+.2f} |"
    )
    pct_rows = "\n".join(row(p_) for p_ in [1, 5, 25, 50, 75, 95, 99])

    md = f"""# Retail-only held-out validation — impact-curve framework

Calibration step (not run here): fit V2 (φ, depth) of the normalizer pool to
the empirical non-5bp router-routed WETH/USDC impact curve (pool_mid_pre
referenced). 5bp pool metrics never enter the fit.

This script runs the simulator with:
- submission = 5bp on-chain frozen (fee=0.0005, virtual_depth_y=$212.16M)
- normalizer = **{p['plan']}**: φ = **{p['phi_bps']:.2f} bps**, depth = **${p['depth_usdc_m']:.1f}M**

and compares the three retail-only validation metrics to on-chain reality.

## Retail flow split

| Metric | Real | Sim (mean ± std) |
|--------|------|-------------------|
| Volume share @5bp | {r['retail_volume_share_5bp']*100:.2f}% | {s['retail_volume_share_mean']*100:.2f}% ± {s['retail_volume_share_std']*100:.2f}pp |
| Fee share @5bp    | {r['retail_fee_share_5bp']*100:.2f}% | {s['retail_fee_share_mean']*100:.2f}% ± {s['retail_fee_share_std']*100:.2f}pp |

## Retail markout_15s on the 5bp pool — USD-weighted

| Metric | Real (USD-w) | Sim (USD-w) |
|--------|--------------|-------------|
| mean   | {r['markout_15s_usd_w_mean_bps']:+.3f} bps | {s['markout_usd_w_mean_bps']:+.3f} bps |
{pct_rows}

Convention: markout_15s is LP-positive (a positive value means LP profited
from the trade after the 15-second look-ahead). Real data uses 15s; the sim
uses the next-block fair_price (≈12s, the closest available proxy).

## Artifacts
- Per-trade sim markouts: `analysis/weth_usdc_90d/markout_5bp_pool_sim_retail.csv`
- Validation plot: `plots/validation_retail_pool_mid.png`
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=DEFAULT_PLAN, choices=("plan_a", "plan_b"))
    args = ap.parse_args()
    main(plan_key=args.plan)
