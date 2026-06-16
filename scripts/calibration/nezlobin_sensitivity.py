"""Sensitivity of Nezlobin vs Guidestar vs flat baselines along three axes.

At the §8 normalizer, $1M depth, bottom-of-block-arb env (paired seeds, only fee
policy differs), sweep:
  - arrival  : retail arrival multiplier
  - size     : retail size-distribution mean multiplier (scale the quantiles)
  - vol      : price-volatility multiplier (scale the regime inverse-CDF returns)
recording the final 15s-forward LP markout (total/retail/arb) + retail volume per
pool. Writes analysis/weth_usdc_90d/nezlobin_sensitivity_cache.json.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arena_eval.exact_simple_amm.config import ExactSimpleAMMConfig
from arena_eval.exact_simple_amm.simulator import ExactSimpleAMMSimulator
from arena_eval.exact_simple_amm.strategies import FixedFeeStrategy
from scripts.calibration.nezlobin_backtest import (
    A, NORM_PHI, NORM_D, INIT_PX, N_STEPS, BASE_ARRIVAL, PRIMARY_DEPTH, ARB_PASSES,
    POOL_ORDER, make_strat, markout,
)

CACHE = A / "nezlobin_sensitivity_cache.json"
QUANT = str(A / "parent_order_usd_quantiles.csv")
INVCDF = str(A / "regimes_invcdf.csv")
SEEDS = tuple(range(40, 52))
ARR_MULTS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
MEAN_MULTS = [0.5, 1.0, 2.0, 4.0, 8.0]
VOL_MULTS = [0.5, 1.0, 2.0, 3.0]


def make_cfg(arrival_mult=1.0, quantiles_path=QUANT, invcdf_path=INVCDF):
    return ExactSimpleAMMConfig(
        n_steps=N_STEPS, initial_price=INIT_PX, initial_x=NORM_D / INIT_PX, initial_y=NORM_D,
        submission_liquidity_fraction=PRIMARY_DEPTH / NORM_D, evaluator_kind="real_data",
        price_process_kind="regime_switching", retail_flow_kind="empirical_usd_size",
        retail_arrival_rate=BASE_ARRIVAL * arrival_mult, retail_buy_prob=0.4627,
        regime_invcdf_path=invcdf_path,
        regime_transition_path=str(A / "regimes_transition_matrix.csv"),
        retail_usd_quantiles_path=quantiles_path,
        normalizer_tracks_fair=True, bottom_of_block_arb=True, arb_max_passes=ARB_PASSES,
    )


def run_final(name, arrival_mult=1.0, quantiles_path=QUANT, invcdf_path=INVCDF):
    tot = ret = arb = vol = 0.0
    for s in SEEDS:
        sim = ExactSimpleAMMSimulator(config=make_cfg(arrival_mult, quantiles_path, invcdf_path),
                                      submission_strategy=make_strat(name),
                                      normalizer_strategy=FixedFeeStrategy(NORM_PHI, NORM_PHI), seed=s)
        fair, trades = [], []
        while not sim.done:
            out = sim.step_once()
            fair.append(out["fair_price"])
            for ev in out["trade_events"]:
                if ev["venue"] == "submission":
                    trades.append((out["timestamp"], ev["source"], ev["trader_side"], ev["amount_x"], ev["amount_y"]))
        for (t, src, m, n_) in markout(np.asarray(fair), trades):
            tot += m
            if src == "retail":
                ret += m; vol += n_
            else:
                arb += m
    n = len(SEEDS)
    return dict(total=tot / n, retail=ret / n, arb=arb / n, volume=vol / n)


def _scaled_quants(mult, tmp):
    pct = np.genfromtxt(QUANT, delimiter=",", names=True, dtype=float)["pct"]
    sz = np.genfromtxt(QUANT, delimiter=",", names=True, dtype=float)["size_usd"]
    p = tmp / f"q_{mult}.csv"
    with open(p, "w") as f:
        f.write("pct,size_usd\n")
        for a, b in zip(pct, sz * mult):
            f.write(f"{a},{b}\n")
    return str(p)


def _scaled_invcdf(mult, tmp):
    rows = np.genfromtxt(INVCDF, delimiter=",", names=True, dtype=float)
    cols = rows.dtype.names
    p = tmp / f"v_{mult}.csv"
    with open(p, "w") as f:
        f.write(",".join(cols) + "\n")
        for i in range(len(rows[cols[0]])):
            vals = [rows["pct"][i]] + [rows[c][i] * mult for c in cols if c != "pct"]
            f.write(",".join(f"{v}" for v in vals) + "\n")
    return str(p)


def sweep(label, points, run_fn):
    out = {"mults": points}
    for nm in POOL_ORDER:
        out[nm] = {k: [] for k in ("total", "retail", "arb", "volume")}
    for x in points:
        for nm in POOL_ORDER:
            r = run_fn(nm, x)
            for k in ("total", "retail", "arb", "volume"):
                out[nm][k].append(round(r[k], 2))
        best = max(POOL_ORDER, key=lambda nm: out[nm]["total"][-1])
        print(f"  {label} {x:<5}: best={best[:9]:9s} " + " ".join(f"{nm[:4]}:{out[nm]['total'][-1]:7.0f}" for nm in POOL_ORDER))
    return out


def main():
    tmp = Path(tempfile.mkdtemp(prefix="nz_sens_"))
    print("arrival sweep ...")
    arrival = sweep("arr", ARR_MULTS, lambda nm, x: run_final(nm, arrival_mult=x))
    print("size sweep ...")
    qpaths = {m: _scaled_quants(m, tmp) for m in MEAN_MULTS}
    size = sweep("size", MEAN_MULTS, lambda nm, x: run_final(nm, quantiles_path=qpaths[x]))
    print("vol sweep ...")
    vpaths = {m: _scaled_invcdf(m, tmp) for m in VOL_MULTS}
    vol = sweep("vol", VOL_MULTS, lambda nm, x: run_final(nm, invcdf_path=vpaths[x]))

    cache = dict(
        meta=dict(seeds=len(SEEDS), depth=PRIMARY_DEPTH, pool_order=POOL_ORDER,
                  normalizer="§8 full-market φ=4.65bps D=$275M",
                  env="bottom_of_block_arb=True, arb_max_passes=%d" % ARB_PASSES),
        arrival=arrival, size=size, vol=vol,
    )
    CACHE.write_text(json.dumps(cache))
    print(f"\nwrote {CACHE} ({CACHE.stat().st_size/1e3:.0f} KB)")


if __name__ == "__main__":
    main()
