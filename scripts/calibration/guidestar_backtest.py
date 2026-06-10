"""Backtest a NEW (shallow) dynamic-fee pool vs fixed-fee baselines, against the
§8 full-market normalizer, on the calibrated realistic tape.

Metric: 15s-forward LP markout (executed price vs the fair price 15s later) in $,
summed over every swap (retail + arb) hitting the candidate pool. Paired seeds
(identical fair path + retail arrival stream; only the submission fee policy
differs). Three pools at equal shallow depth:
  - Guidestar volatile (dynamic), floor = feeInit
  - flat feeInit  (the "dynamics off" control)
  - flat 5bp      (the incumbent tier)

Deliverables (plots/guidestar_backtest_*.png): cumulative markout-$ curves
(decomposed retail vs arb/LVR), overlaid per-trade markout histograms, volume.

ILLUSTRATIVE Guidestar params (feeInit etc.) pending the real deployment values.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arena_eval.exact_simple_amm.config import ExactSimpleAMMConfig
from arena_eval.exact_simple_amm.simulator import ExactSimpleAMMSimulator
from arena_eval.exact_simple_amm.strategies import FixedFeeStrategy
from arena_eval.exact_simple_amm.guidestar_volatile import GuidestarVolatileStrategy

A = Path("analysis/weth_usdc_90d")
NORM_PHI, NORM_D = 0.000465, 275.1e6      # §8 full-market normalizer (φ=4.65bps, D=$275M)
INIT_PX, STEP_S, MARKOUT_S = 100.0, 12.0, 15.0
FEE_INIT_BPS = 3.5                         # REAL mainnet Guidestar volatile feeInit (350 = 3.5bps)
SEEDS = tuple(range(40, 56))               # 16 paired seeds
N_STEPS = 5000


def make_cfg(depth_usd: float) -> ExactSimpleAMMConfig:
    return ExactSimpleAMMConfig(
        n_steps=N_STEPS, initial_price=INIT_PX, initial_x=NORM_D / INIT_PX, initial_y=NORM_D,
        submission_liquidity_fraction=depth_usd / NORM_D, evaluator_kind="real_data",
        price_process_kind="regime_switching", retail_flow_kind="empirical_usd_size",
        retail_arrival_rate=98_676 / 216_000, retail_buy_prob=0.4627,
        regime_invcdf_path=str(A / "regimes_invcdf.csv"),
        regime_transition_path=str(A / "regimes_transition_matrix.csv"),
        retail_usd_quantiles_path=str(A / "parent_order_usd_quantiles.csv"),
        normalizer_tracks_fair=True,
    )


def pools() -> dict:
    fi = FEE_INIT_BPS / 1e4
    return {
        "Guidestar (real params, 3.5bp floor)": lambda: GuidestarVolatileStrategy(),  # mainnet volatile defaults
        "flat 3.5bp (=feeInit, dynamics off)": lambda: FixedFeeStrategy(fi, fi),
        "flat 5bp (incumbent)": lambda: FixedFeeStrategy(5e-4, 5e-4),
    }


def run_pool(make_strat, depth_usd: float, seed: int):
    """Run one sim; return (fair_path, submission trade records)."""
    sim = ExactSimpleAMMSimulator(config=make_cfg(depth_usd), submission_strategy=make_strat(),
                                  normalizer_strategy=FixedFeeStrategy(NORM_PHI, NORM_PHI), seed=seed)
    fair, trades = [], []
    while not sim.done:
        out = sim.step_once()
        fair.append(out["fair_price"])
        for ev in out["trade_events"]:
            if ev["venue"] == "submission":
                trades.append((out["timestamp"], ev["source"], ev["trader_side"],
                               ev["amount_x"], ev["amount_y"]))
    return np.asarray(fair), trades


def markout(fair: np.ndarray, trades) -> list:
    """15s-forward LP markout ($) per submission trade: executed price vs fair 15s later."""
    n = len(fair)
    tf_off = MARKOUT_S / STEP_S   # 1.25 steps
    out = []
    for (t, src, side, ax, ay) in trades:
        tf = t + tf_off
        i0 = int(np.floor(tf))
        if i0 + 1 >= n:
            continue
        f15 = fair[i0] + (tf - i0) * (fair[i0 + 1] - fair[i0])
        m = (ay - ax * f15) if side == "buy_x" else (ax * f15 - ay)   # LP P&L at fair_15s
        out.append((t, src, m, ay))   # step, source, markout_$, notional_$ (USDC)
    return out


def aggregate(make_strat, depth_usd: float):
    """Mean-over-seeds cumulative markout curves + pooled per-trade bps + volume."""
    cum_tot, cum_ret, cum_arb = [], [], []
    bps_all, retail_vol = [], []
    for s in SEEDS:
        fair, trades = run_pool(make_strat, depth_usd, s)
        recs = markout(fair, trades)
        step_tot = np.zeros(N_STEPS); step_ret = np.zeros(N_STEPS); step_arb = np.zeros(N_STEPS)
        rv = 0.0
        for (t, src, m, notl) in recs:
            step_tot[t] += m
            if src == "retail":
                step_ret[t] += m; rv += notl
            else:
                step_arb[t] += m
            if notl > 1.0:
                bps_all.append(1e4 * m / notl)
        cum_tot.append(np.cumsum(step_tot)); cum_ret.append(np.cumsum(step_ret)); cum_arb.append(np.cumsum(step_arb))
        retail_vol.append(rv)
    return dict(
        cum_tot=np.array(cum_tot), cum_ret=np.array(cum_ret), cum_arb=np.array(cum_arb),
        bps=np.array(bps_all), retail_vol=float(np.mean(retail_vol)),
    )


def main():
    depths = [1_000_000.0, 500_000.0]
    colors = {"Guidestar (real params, 3.5bp floor)": "#8e44ad",
              "flat 3.5bp (=feeInit, dynamics off)": "#e67e22", "flat 5bp (incumbent)": "#2980b9"}
    results = {d: {nm: aggregate(mk, d) for nm, mk in pools().items()} for d in depths}

    # ---- Figure 1: cumulative markout $ (total / retail / arb-LVR), primary depth $1M ----
    D = 1_000_000.0
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), sharex=True)
    x = np.arange(N_STEPS)
    for nm, r in results[D].items():
        for j, key in enumerate(["cum_tot", "cum_ret", "cum_arb"]):
            mean = r[key].mean(0); sd = r[key].std(0)
            ax[j].plot(x, mean, color=colors[nm], lw=2, label=nm)
            ax[j].fill_between(x, mean - sd, mean + sd, color=colors[nm], alpha=0.12)
    for j, ttl in enumerate(["TOTAL LP markout", "Retail markout (+)", "Arb / LVR markout (−)"]):
        ax[j].set_title(ttl, fontsize=11, fontweight="bold"); ax[j].set_xlabel("step (12s)")
        ax[j].axhline(0, color="grey", lw=0.7); ax[j].grid(alpha=.25)
        ax[j].spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel("cumulative markout ($)"); ax[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"New shallow pool ($1M) vs §8 full-market normalizer — 15s-forward LP markout (mean±sd, {len(SEEDS)} seeds, real mainnet volatile params)",
                 fontweight="bold", fontsize=11)
    plt.tight_layout(); plt.savefig("plots/guidestar_backtest_cumulative.png", dpi=115); plt.close()

    # ---- Figure 2: per-trade markout histograms overlaid ($1M) ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    bins = np.linspace(-150, 150, 81)
    for nm, r in results[D].items():
        b = r["bps"]; b = b[np.isfinite(b)]
        ax[0].hist(b, bins=bins, histtype="step", lw=2, color=colors[nm], label=f"{nm}", density=True)
        ax[1].hist(b, bins=bins, histtype="step", lw=2, color=colors[nm], density=True)
    ax[0].set_yscale("log"); ax[0].set_title("Per-trade markout (bps) — log density (tails)", fontsize=11, fontweight="bold")
    ax[1].set_title("Per-trade markout (bps) — linear (body)", fontsize=11, fontweight="bold")
    for a in ax:
        a.axvline(0, color="grey", lw=0.7); a.set_xlabel("LP markout per trade (bps)"); a.grid(alpha=.25)
        a.spines[["top", "right"]].set_visible(False)
    ax[0].legend(fontsize=8)
    fig.suptitle("Per-trade 15s markout distribution ($1M pool) — does Guidestar trim the loss tail?", fontweight="bold", fontsize=11)
    plt.tight_layout(); plt.savefig("plots/guidestar_backtest_histograms.png", dpi=115); plt.close()

    # ---- Figure 3 + summary: final cumulative markout & retail volume by pool/depth ----
    print(f"\n{'depth':>8} {'pool':32s} {'final cum markout $':>20s} {'  (retail / arb-LVR)':>22s} {'retail vol $':>14s}")
    for d in depths:
        for nm, r in results[d].items():
            ft = r["cum_tot"][:, -1].mean(); fr = r["cum_ret"][:, -1].mean(); fa = r["cum_arb"][:, -1].mean()
            print(f"${d/1e6:>5.1f}M {nm:32s} {ft:>20,.0f} {fr:>10,.0f}/{fa:>10,.0f} {r['retail_vol']:>14,.0f}")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    names = list(pools().keys()); xpos = np.arange(len(names)); w = 0.38
    for i, d in enumerate(depths):
        vols = [results[d][nm]["retail_vol"] for nm in names]
        ax.bar(xpos + (i - 0.5) * w, vols, w, label=f"${d/1e6:.1f}M depth",
               color=["#8e44ad", "#e67e22", "#2980b9"], alpha=0.6 + 0.4 * i, edgecolor="black", lw=0.5)
    ax.set_xticks(xpos); ax.set_xticklabels([n.split(" (")[0] for n in names], fontsize=9)
    ax.set_ylabel("retail volume captured ($, per sim)"); ax.legend(fontsize=8)
    ax.set_title("Retail volume captured by the candidate pool", fontweight="bold", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=.25, axis="y")
    plt.tight_layout(); plt.savefig("plots/guidestar_backtest_volume.png", dpi=115); plt.close()
    print("\nsaved plots/guidestar_backtest_{cumulative,histograms,volume}.png")


if __name__ == "__main__":
    main()
