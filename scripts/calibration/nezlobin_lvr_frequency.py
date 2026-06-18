"""Does skewing reduce LVR by deterring arbs? Frequency vs LVR decomposition.

Intuition under test: a fee skewed toward the side where the next arb is likely should
DETER those arbs, and fewer arbs => less LVR. This script sweeps the skew load (offset-
immune RememberedArbSkew, flat..9/0) and records, per side (buy/sell arb): the arb COUNT
and the arb LVR (15s-forward markout $), plus retail markout. $1M / §8 normalizer /
bottom-of-block-arb env, paired seeds.

The result: the total no-arb band WIDTH is f_a + f_b = TS = 9 bps for EVERY split, so
skewing only slides the band's centre — it does not widen the band the price must cross
to be arbed. A light skew deters a few arbs (count dips ~11%) but LVR barely moves (~1%),
because the deterred mispricing is just corrected later/by a bigger arb; a heavy skew
frees the off-side (0 bps), which INCREASES arb count and lets arbs extract mispricing for
free, so LVR gets worse. LVR is governed by band width (=TS) and volatility, not arb count.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arena_eval.exact_simple_amm.simulator import ExactSimpleAMMSimulator
from arena_eval.exact_simple_amm.strategies import FixedFeeStrategy
from scripts.calibration.nezlobin_backtest import A, NORM_PHI, MARKOUT_S, STEP_S, PRIMARY_DEPTH, make_cfg
from scripts.calibration.nezlobin_skew_fix import RememberedArbSkew

CACHE = A / "nezlobin_lvr_frequency_cache.json"
SEEDS = tuple(range(40, 52))
LOADS = [0.0, 1.0 / 6, 1.0 / 3, 1.0 / 2, 2.0 / 3, 5.0 / 6, 1.0]


def measure(strat_factory):
    nbuy = nsell = 0
    lvr_buy = lvr_sell = ret = 0.0
    for s in SEEDS:
        sim = ExactSimpleAMMSimulator(config=make_cfg(PRIMARY_DEPTH), submission_strategy=strat_factory(),
                                      normalizer_strategy=FixedFeeStrategy(NORM_PHI, NORM_PHI), seed=s)
        fair, ev = [], []
        while not sim.done:
            out = sim.step_once()
            fair.append(out["fair_price"])
            for e in out["trade_events"]:
                if e["venue"] == "submission":
                    ev.append((out["timestamp"], e["source"], e["trader_side"], e["amount_x"], e["amount_y"]))
        fair = np.asarray(fair); n = len(fair); off = MARKOUT_S / STEP_S
        for (t, src, side, ax, ay) in ev:
            tf = t + off; i0 = int(np.floor(tf))
            if i0 + 1 >= n or ax <= 0:
                continue
            f15 = fair[i0] + (tf - i0) * (fair[i0 + 1] - fair[i0])
            m = (ay - ax * f15) if side == "buy_x" else (ax * f15 - ay)
            if src == "arb":
                if side == "buy_x":
                    nbuy += 1; lvr_buy += m
                else:
                    nsell += 1; lvr_sell += m
            else:
                ret += m
    ns = len(SEEDS)
    return dict(n_buy=nbuy / ns, n_sell=nsell / ns, n_arb=(nbuy + nsell) / ns,
                lvr_buy=lvr_buy / ns, lvr_sell=lvr_sell / ns, lvr_arb=(lvr_buy + lvr_sell) / ns,
                retail_mk=ret / ns)


def main():
    out = {"meta": dict(seeds=len(SEEDS), depth=PRIMARY_DEPTH, ts_bps=9.0), "loads": LOADS, "sweep": []}
    print(f"{'fa/fb':>9} {'arbs':>7} {'n_buy':>7} {'n_sell':>7} {'LVR$':>8} {'lvr_buy':>8} {'lvr_sell':>8} {'retail':>7}")
    for ld in LOADS:
        r = measure(lambda ld=ld: RememberedArbSkew(load=ld))
        r["load"] = ld
        out["sweep"].append(r)
        f = 4.5 + ld * 4.5
        print(f"{f:4.1f}/{9 - f:3.1f} {r['n_arb']:7.0f} {r['n_buy']:7.0f} {r['n_sell']:7.0f} "
              f"{r['lvr_arb']:8.1f} {r['lvr_buy']:8.1f} {r['lvr_sell']:8.1f} {r['retail_mk']:7.1f}")
    CACHE.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {CACHE}")


if __name__ == "__main__":
    main()
