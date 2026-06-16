"""Why does Guidestar out-mark Nezlobin at base arrival? Empirical decomposition.

At $1M / §8 normalizer / base arrival / bottom-of-block-arb env, collect per-trade
data for Nezlobin, Guidestar, flat 5bp, flat 4.5/4.5 and split markout by SOURCE
(arb vs retail). Also record the effective cost each source pays (how far execution
sat from the pre-trade mid) — the LP's gross take. Prints a summary and caches
markout-by-source histograms for plotting.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arena_eval.exact_simple_amm.simulator import ExactSimpleAMMSimulator
from arena_eval.exact_simple_amm.strategies import FixedFeeStrategy
from scripts.calibration.nezlobin_backtest import (
    A, NORM_PHI, NORM_D, MARKOUT_S, STEP_S, PRIMARY_DEPTH, POOL_ORDER, make_strat, make_cfg,
)

CACHE = A / "nezlobin_intuition_cache.json"
SEEDS = tuple(range(40, 56))


def collect(name):
    """Per-trade (source, markout $, notional, effective-cost bps) across seeds at $1M."""
    src, mk, notl, cost = [], [], [], []
    for s in SEEDS:
        sim = ExactSimpleAMMSimulator(config=make_cfg(PRIMARY_DEPTH), submission_strategy=make_strat(name),
                                      normalizer_strategy=FixedFeeStrategy(NORM_PHI, NORM_PHI), seed=s)
        fair, ev = [], []
        while not sim.done:
            out = sim.step_once()
            fair.append(out["fair_price"])
            for e in out["trade_events"]:
                if e["venue"] == "submission":
                    ev.append((out["timestamp"], e["source"], e["trader_side"], e["amount_x"], e["amount_y"],
                               e["pre_spot_price"]))
        fair = np.asarray(fair); n = len(fair); off = MARKOUT_S / STEP_S
        for (t, source, side, ax, ay, pre) in ev:
            tf = t + off; i0 = int(np.floor(tf))
            if i0 + 1 >= n or ax <= 0:
                continue
            f15 = fair[i0] + (tf - i0) * (fair[i0 + 1] - fair[i0])
            m = (ay - ax * f15) if side == "buy_x" else (ax * f15 - ay)
            exec_px = ay / ax
            src.append(source); mk.append(m); notl.append(ay)
            cost.append(1e4 * abs(exec_px / pre - 1.0) if pre > 0 else 0.0)
    return (np.array(src), np.array(mk, dtype=float), np.array(notl, dtype=float), np.array(cost, dtype=float))


def main():
    bins = np.linspace(-60, 60, 61)
    out = {"meta": dict(seeds=len(SEEDS), depth=PRIMARY_DEPTH, pool_order=POOL_ORDER), "mk_bins": bins.tolist()}
    print(f"{'pool':30s} {'arb tot':>9} {'ret tot':>9} {'arb n':>7} {'ret n':>7} "
          f"{'arb mk bps':>10} {'ret mk bps':>10} {'arb cost':>9} {'ret cost':>9}")
    for nm in POOL_ORDER:
        src, mk, notl, cost = collect(nm)
        arb = src == "arb"; ret = src == "retail"
        ns = len(SEEDS)
        def hist(mask):
            b = 1e4 * mk[mask & (notl > 1.0)] / notl[mask & (notl > 1.0)]
            return np.histogram(b[np.isfinite(b)], bins=bins, density=True)[0].round(6).tolist()
        out[nm] = dict(
            arb_total=float(mk[arb].sum() / ns), retail_total=float(mk[ret].sum() / ns),
            arb_count=int(arb.sum() / ns), retail_count=int(ret.sum() / ns),
            arb_mk_hist=hist(arb), retail_mk_hist=hist(ret),
            arb_cost_bps=float(np.mean(cost[arb])) if arb.any() else 0.0,
            retail_cost_bps=float(np.mean(cost[ret])) if ret.any() else 0.0,
            arb_vol=float(notl[arb].sum() / ns), retail_vol=float(notl[ret].sum() / ns),
        )
        o = out[nm]
        mkbps = lambda mask: float(np.mean(1e4 * mk[mask & (notl > 1.0)] / notl[mask & (notl > 1.0)]))
        print(f"{nm:30s} {o['arb_total']:9.0f} {o['retail_total']:9.0f} {o['arb_count']:7d} {o['retail_count']:7d} "
              f"{mkbps(arb):10.2f} {mkbps(ret):10.2f} {o['arb_cost_bps']:9.2f} {o['retail_cost_bps']:9.2f}")
    CACHE.write_text(json.dumps(out))
    print(f"\nwrote {CACHE}")


if __name__ == "__main__":
    main()
