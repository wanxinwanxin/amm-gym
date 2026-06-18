"""Why a directional skew destroys the arb-direction persistence it tries to exploit.

The next top-of-block arb's direction is decided by where the next fair move r (bps)
lands relative to the no-arb band, which after the LAST arb is set by the offset o the
arb left and the resting fees (f_a, f_b):
    continue (same side as last arb)  if  r > o + f_a
    reverse                           if  r < o - f_b
    no arb                            otherwise

On the flat pool the arb leaves o = -f_a (mid one ask-fee below fair), so the CONTINUE
threshold o+f_a = 0 (fair sits at the continue edge -> primed to continue) while a
reverse needs the full spread. Skewing to 9/0 raises f_a to 9 but the offset does NOT
follow (the now-free bid side resets the mid toward fair, so o only reaches ~-2.5):
continue needs a big +move, a reverse needs only a small dip -> the asymmetry inverts and
persistence collapses.

This script measures, conditional on the LAST arb being a BUY, for several skew loads:
the offset o, the fees, the two thresholds, and the realized P(continue). It also caches
the per-block fair-return distribution so the notebook can show P(continue)/P(reverse) as
areas under the move distribution. Offset-immune signal (RememberedArbSkew), $1M / §8
normalizer / bottom-of-block-arb env, paired seeds.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arena_eval.exact_simple_amm.simulator import ExactSimpleAMMSimulator
from arena_eval.exact_simple_amm.strategies import FixedFeeStrategy
from scripts.calibration.nezlobin_backtest import A, NORM_PHI, PRIMARY_DEPTH, make_cfg
from scripts.calibration.nezlobin_skew_fix import RememberedArbSkew

CACHE = A / "nezlobin_persistence_intuition_cache.json"
SEEDS = tuple(range(40, 52))
LOADS = [0.0, 1.0 / 3, 2.0 / 3, 1.0]   # 4.5/4.5, 6/3, 7.5/1.5, 9/0


def collect(strat_factory):
    """Per block: fair, block-open mid (pre TOB arb), arb dir, resting fa/fb (bps)."""
    fair, mid, d, fa, fb = [], [], [], [], []
    for s in SEEDS:
        sim = ExactSimpleAMMSimulator(config=make_cfg(PRIMARY_DEPTH), submission_strategy=strat_factory(),
                                      normalizer_strategy=FixedFeeStrategy(NORM_PHI, NORM_PHI), seed=s)
        while not sim.done:
            out = sim.step_once()
            tob = next((e for e in out["trade_events"] if e["venue"] == "submission" and e["source"] == "arb"), None)
            fair.append(out["fair_price"])
            if tob is None:
                mid.append(np.nan); d.append(0); fa.append(np.nan); fb.append(np.nan)
            else:
                d.append(1 if tob["trader_side"] == "buy_x" else -1)
                mid.append(tob["pre_spot_price"])
                fa.append(1e4 * tob["pre_state"]["submission_ask_fee"])
                fb.append(1e4 * tob["pre_state"]["submission_bid_fee"])
        for L in (fair, mid, d, fa, fb):
            L.append(np.nan if L is not d else 0)   # seed break
    return (np.array(fair), np.array(mid), np.array(d), np.array(fa), np.array(fb))


def stats_after_buy(fair, mid, d, fa, fb):
    pf = np.concatenate([[np.nan], fair[:-1]]); pd = np.concatenate([[0], d[:-1]])
    o = 1e4 * (mid - pf) / pf
    r = 1e4 * (fair / pf - 1.0)
    m = (pd == 1) & np.isfinite(o) & (d != 0)              # blocks following a BUY arb, that have an arb
    o_buy = float(np.nanmean(o[m])); fa_b = float(np.nanmean(fa[m])); fb_b = float(np.nanmean(fb[m]))
    p_cont = float(np.mean(d[m] == 1))
    return dict(offset_bps=o_buy, fa_bps=fa_b, fb_bps=fb_b,
                t_continue_bps=o_buy + fa_b, t_reverse_bps=o_buy - fb_b,
                p_continue=p_cont, move_std_bps=float(np.nanstd(r[m])))


def main():
    out = {"meta": dict(seeds=len(SEEDS), depth=PRIMARY_DEPTH), "loads": LOADS, "by_load": []}
    # per-block fair-return distribution (same price process for all pools) for the area overlay
    fair, mid, d, fa, fb = collect(lambda: FixedFeeStrategy(4.5e-4, 4.5e-4))
    pf = np.concatenate([[np.nan], fair[:-1]]); r = 1e4 * (fair / pf - 1.0)
    r = r[np.isfinite(r)]
    edges = np.linspace(-30, 30, 121)
    out["move_hist"] = dict(edges=edges.tolist(),
                            density=np.histogram(r, bins=edges, density=True)[0].tolist(),
                            std_bps=float(np.std(r)))
    print(f"{'load':>5} {'fa/fb':>9} {'offset':>7} {'T_cont':>7} {'T_rev':>7} {'P(cont)':>8}")
    for ld in LOADS:
        if ld == 0.0:
            st = stats_after_buy(fair, mid, d, fa, fb)            # reuse flat run
        else:
            st = stats_after_buy(*collect(lambda ld=ld: RememberedArbSkew(load=ld)))
        st["load"] = ld
        out["by_load"].append(st)
        f = 4.5 + ld * 4.5
        print(f"{ld:5.2f} {f:4.1f}/{9 - f:3.1f} {st['offset_bps']:+7.2f} {st['t_continue_bps']:+7.2f} "
              f"{st['t_reverse_bps']:+7.2f} {st['p_continue']:8.3f}")
    CACHE.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {CACHE}")


if __name__ == "__main__":
    main()
