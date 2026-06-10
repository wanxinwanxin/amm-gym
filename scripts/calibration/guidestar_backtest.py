"""Backtest a NEW (shallow) dynamic-fee pool vs fixed-fee baselines, against the
§8 full-market normalizer, on the calibrated realistic tape.

Metric: 15s-forward LP markout (executed price vs the fair price 15s later) in $,
summed over every swap (retail + arb) hitting the candidate pool. Paired seeds
(identical fair path + retail arrival stream; only the submission fee policy
differs). Three pools at equal shallow depth:
  - Guidestar volatile (dynamic), real mainnet params (feeInit=3.5bp floor)
  - flat feeInit (3.5bp) -- the "dynamics off" control
  - flat 5bp     -- the incumbent tier

Figures (plots/guidestar_backtest_*.png):
  cumulative  : cumulative markout-$ curves, decomposed retail vs arb/LVR
  histograms  : per-trade markout distributions
  by_size     : markout vs trade size -- the selection hypothesis (small +, large -;
                does Guidestar keep profitable large trades and shed toxic ones?)
  breakeven   : scale retail arrival until cumulative markout crosses 0
  volume      : retail volume captured
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
NORM_PHI, NORM_D = 0.000465, 275.1e6
INIT_PX, STEP_S, MARKOUT_S = 100.0, 12.0, 15.0
FEE_INIT_BPS = 3.5
BASE_ARRIVAL = 98_676 / 216_000
SEEDS = tuple(range(40, 56))               # 16 paired seeds
N_STEPS = 5000
COLORS = {"Guidestar (real params, 3.5bp floor)": "#8e44ad",
          "flat 3.5bp (=feeInit, dynamics off)": "#e67e22", "flat 5bp (incumbent)": "#2980b9"}


def make_cfg(depth_usd: float, arrival_mult: float = 1.0) -> ExactSimpleAMMConfig:
    return ExactSimpleAMMConfig(
        n_steps=N_STEPS, initial_price=INIT_PX, initial_x=NORM_D / INIT_PX, initial_y=NORM_D,
        submission_liquidity_fraction=depth_usd / NORM_D, evaluator_kind="real_data",
        price_process_kind="regime_switching", retail_flow_kind="empirical_usd_size",
        retail_arrival_rate=BASE_ARRIVAL * arrival_mult, retail_buy_prob=0.4627,
        regime_invcdf_path=str(A / "regimes_invcdf.csv"),
        regime_transition_path=str(A / "regimes_transition_matrix.csv"),
        retail_usd_quantiles_path=str(A / "parent_order_usd_quantiles.csv"),
        normalizer_tracks_fair=True,
    )


def pools() -> dict:
    fi = FEE_INIT_BPS / 1e4
    return {
        "Guidestar (real params, 3.5bp floor)": lambda: GuidestarVolatileStrategy(),
        "flat 3.5bp (=feeInit, dynamics off)": lambda: FixedFeeStrategy(fi, fi),
        "flat 5bp (incumbent)": lambda: FixedFeeStrategy(5e-4, 5e-4),
    }


def run_pool(make_strat, depth_usd, seed, arrival_mult=1.0):
    sim = ExactSimpleAMMSimulator(config=make_cfg(depth_usd, arrival_mult), submission_strategy=make_strat(),
                                  normalizer_strategy=FixedFeeStrategy(NORM_PHI, NORM_PHI), seed=seed)
    fair, trades = [], []
    while not sim.done:
        out = sim.step_once()
        fair.append(out["fair_price"])
        for ev in out["trade_events"]:
            if ev["venue"] == "submission":
                trades.append((out["timestamp"], ev["source"], ev["trader_side"], ev["amount_x"], ev["amount_y"]))
    return np.asarray(fair), trades


def markout(fair, trades):
    """15s-forward LP markout ($) per submission trade -> (step, source, markout_$, notional_$)."""
    n = len(fair); off = MARKOUT_S / STEP_S
    out = []
    for (t, src, side, ax, ay) in trades:
        tf = t + off; i0 = int(np.floor(tf))
        if i0 + 1 >= n:
            continue
        f15 = fair[i0] + (tf - i0) * (fair[i0 + 1] - fair[i0])
        m = (ay - ax * f15) if side == "buy_x" else (ax * f15 - ay)
        out.append((t, src, m, ay))
    return out


def aggregate(make_strat, depth_usd, arrival_mult=1.0):
    cum_tot, cum_ret, cum_arb, retail_vol = [], [], [], []
    per_trade = []   # (notional, markout_$, source) pooled across seeds
    for s in SEEDS:
        fair, trades = run_pool(make_strat, depth_usd, s, arrival_mult)
        st, sr, sa = (np.zeros(N_STEPS) for _ in range(3)); rv = 0.0
        for (t, src, m, notl) in markout(fair, trades):
            st[t] += m
            (sr if src == "retail" else sa)[t] += m
            if src == "retail":
                rv += notl
            per_trade.append((notl, m, src))
        cum_tot.append(np.cumsum(st)); cum_ret.append(np.cumsum(sr)); cum_arb.append(np.cumsum(sa)); retail_vol.append(rv)
    pt = np.array(per_trade, dtype=object)
    return dict(cum_tot=np.array(cum_tot), cum_ret=np.array(cum_ret), cum_arb=np.array(cum_arb),
                retail_vol=float(np.mean(retail_vol)),
                notl=np.array([p[0] for p in per_trade]), mk=np.array([p[1] for p in per_trade]),
                src=np.array([p[2] for p in per_trade]))


def fig_cumulative(res, depth):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), sharex=True)
    x = np.arange(N_STEPS)
    for nm, r in res.items():
        for j, key in enumerate(["cum_tot", "cum_ret", "cum_arb"]):
            mean, sd = r[key].mean(0), r[key].std(0)
            ax[j].plot(x, mean, color=COLORS[nm], lw=2, label=nm)
            ax[j].fill_between(x, mean - sd, mean + sd, color=COLORS[nm], alpha=0.12)
    for j, ttl in enumerate(["TOTAL LP markout", "Retail markout (+)", "Arb / LVR markout (−)"]):
        ax[j].set_title(ttl, fontsize=11, fontweight="bold"); ax[j].set_xlabel("step (12s)")
        ax[j].axhline(0, color="grey", lw=0.7); ax[j].grid(alpha=.25); ax[j].spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel("cumulative markout ($)"); ax[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"New shallow pool (${depth/1e6:.1f}M) vs §8 normalizer — 15s-forward LP markout (mean±sd, {len(SEEDS)} seeds, real volatile params)",
                 fontweight="bold", fontsize=11)
    plt.tight_layout(); plt.savefig("plots/guidestar_backtest_cumulative.png", dpi=115); plt.close()


def fig_histograms(res):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6)); bins = np.linspace(-150, 150, 81)
    for nm, r in res.items():
        b = 1e4 * r["mk"][r["notl"] > 1.0] / r["notl"][r["notl"] > 1.0]; b = b[np.isfinite(b)]
        ax[0].hist(b, bins=bins, histtype="step", lw=2, color=COLORS[nm], label=nm, density=True)
        ax[1].hist(b, bins=bins, histtype="step", lw=2, color=COLORS[nm], density=True)
    ax[0].set_yscale("log"); ax[0].set_title("per-trade markout (bps) — log density (tails)", fontsize=11, fontweight="bold")
    ax[1].set_title("per-trade markout (bps) — linear (body)", fontsize=11, fontweight="bold")
    for a in ax:
        a.axvline(0, color="grey", lw=0.7); a.set_xlabel("LP markout per trade (bps)"); a.grid(alpha=.25); a.spines[["top", "right"]].set_visible(False)
    ax[0].legend(fontsize=8)
    fig.suptitle("Per-trade 15s markout distribution ($1M pool)", fontweight="bold", fontsize=11)
    plt.tight_layout(); plt.savefig("plots/guidestar_backtest_histograms.png", dpi=115); plt.close()


def fig_by_size(res):
    """Markout vs trade size: the selection hypothesis. Bins by captured notional ($)."""
    edges = np.logspace(0, 6, 19)  # $1 .. $1M
    ctr = np.sqrt(edges[:-1] * edges[1:])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for nm, r in res.items():
        notl, mk, src = r["notl"], r["mk"], r["src"]
        ret = src == "retail"
        # panel 0: RETAIL mean markout bps by size
        # panel 1: ALL total markout $ by size (per sim)
        # panel 2: captured volume $ by size (per sim, all trades)
        mean_bps_ret, tot_usd_all, vol_all = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mr = ret & (notl >= lo) & (notl < hi)
            ma = (notl >= lo) & (notl < hi)
            mean_bps_ret.append(np.mean(1e4 * mk[mr] / notl[mr]) if mr.sum() >= 5 else np.nan)
            tot_usd_all.append(mk[ma].sum() / len(SEEDS))
            vol_all.append(notl[ma].sum() / len(SEEDS))
        ax[0].plot(ctr, mean_bps_ret, "o-", color=COLORS[nm], lw=2, ms=4, label=nm)
        ax[1].plot(ctr, tot_usd_all, "o-", color=COLORS[nm], lw=2, ms=4)
        ax[2].plot(ctr, vol_all, "o-", color=COLORS[nm], lw=2, ms=4)
    ax[0].set_title("Retail: mean LP markout (bps) by trade size", fontsize=10.5, fontweight="bold")
    ax[0].set_ylabel("mean markout (bps)"); ax[0].legend(fontsize=7.5)
    ax[1].set_title("All trades: total LP markout ($) by trade size", fontsize=10.5, fontweight="bold")
    ax[1].set_ylabel("cumulative markout ($, per sim)")
    ax[2].set_title("Captured volume ($) by trade size", fontsize=10.5, fontweight="bold")
    ax[2].set_ylabel("volume ($, per sim)")
    for a in ax:
        a.set_xscale("log"); a.axhline(0, color="grey", lw=0.7); a.set_xlabel("captured notional per trade ($)")
        a.grid(alpha=.25); a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Markout by trade size — does Guidestar keep profitable trades and shed toxic ones? ($1M pool)", fontweight="bold", fontsize=11)
    plt.tight_layout(); plt.savefig("plots/guidestar_backtest_by_size.png", dpi=115); plt.close()


def fig_breakeven(depth=1_000_000.0):
    mults = [1.0, 2.0, 4.0, 7.0, 11.0, 16.0]
    seeds_be = SEEDS[:8]
    finals = {nm: [] for nm in pools()}
    for m in mults:
        for nm, mk in pools().items():
            vals = []
            for s in seeds_be:
                fair, trades = run_pool(mk, depth, s, arrival_mult=m)
                recs = markout(fair, trades)
                vals.append(sum(r[2] for r in recs))
            finals[nm].append(np.mean(vals))
    fig, ax = plt.subplots(figsize=(9, 5))
    for nm in pools():
        y = np.array(finals[nm]); ax.plot(mults, y, "o-", color=COLORS[nm], lw=2, label=nm)
        # break-even crossing (linear interp where y crosses 0)
        for i in range(len(mults) - 1):
            if y[i] < 0 <= y[i + 1] or y[i] <= 0 < y[i + 1]:
                xb = mults[i] + (0 - y[i]) / (y[i + 1] - y[i]) * (mults[i + 1] - mults[i])
                ax.axvline(xb, color=COLORS[nm], ls=":", lw=1.2); ax.annotate(f"BE≈{xb:.1f}×", (xb, 0), color=COLORS[nm], fontsize=8)
                break
    ax.axhline(0, color="grey", lw=0.8); ax.set_xlabel("retail arrival multiplier (× base 0.46/blk)")
    ax.set_ylabel("final cumulative LP markout ($)"); ax.legend(fontsize=8)
    ax.set_title(f"Break-even: scale retail until LP markout crosses 0 (${depth/1e6:.1f}M pool, {len(seeds_be)} seeds)", fontweight="bold", fontsize=11)
    ax.grid(alpha=.25); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig("plots/guidestar_backtest_breakeven.png", dpi=115); plt.close()
    print("\nBreak-even sweep (final cumulative markout $ by retail multiplier):")
    print(f"{'mult':>5}", *[f"{nm.split(' (')[0]:>22}" for nm in pools()])
    for i, m in enumerate(mults):
        print(f"{m:>5.0f}", *[f"{finals[nm][i]:>22,.0f}" for nm in pools()])


def fig_volume(res_by_depth, depths):
    names = list(pools().keys()); xpos = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, d in enumerate(depths):
        vols = [res_by_depth[d][nm]["retail_vol"] for nm in names]
        ax.bar(xpos + (i - 0.5) * w, vols, w, label=f"${d/1e6:.1f}M depth",
               color=[COLORS[n] for n in names], alpha=0.55 + 0.45 * i, edgecolor="black", lw=0.5)
    ax.set_xticks(xpos); ax.set_xticklabels([n.split(" (")[0] for n in names], fontsize=9)
    ax.set_ylabel("retail volume captured ($, per sim)"); ax.legend(fontsize=8)
    ax.set_title("Retail volume captured by the candidate pool", fontweight="bold", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=.25, axis="y")
    plt.tight_layout(); plt.savefig("plots/guidestar_backtest_volume.png", dpi=115); plt.close()


def main():
    depths = [1_000_000.0, 500_000.0]
    res = {d: {nm: aggregate(mk, d) for nm, mk in pools().items()} for d in depths}
    D = 1_000_000.0
    fig_cumulative(res[D], D); fig_histograms(res[D]); fig_by_size(res[D]); fig_volume(res, depths)
    print(f"\n{'depth':>8} {'pool':40s} {'final cum markout $':>20s} {'retail vol $':>14s}")
    for d in depths:
        for nm, r in res[d].items():
            print(f"${d/1e6:>5.1f}M {nm:40s} {r['cum_tot'][:, -1].mean():>20,.0f} {r['retail_vol']:>14,.0f}")
    fig_breakeven(D)
    print("\nsaved plots/guidestar_backtest_{cumulative,histograms,by_size,breakeven,volume}.png")


if __name__ == "__main__":
    main()
