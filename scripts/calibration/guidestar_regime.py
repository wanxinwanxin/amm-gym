"""Joint arrival × size regime sweep for §11.

Grids the two sensitivity axes together — retail arrival multiplier × order-size
mean multiplier — and runs four candidate pools per cell, recording the final
15s-forward LP markout (mean per sim over seeds). Produces (i) the regime MAP that
visualizes the §10 intuition (where a low flat fee wins vs where Guidestar's
defense wins) and (ii) the test of the unifying FlowAwareGuidestar controller
(does one adaptive policy track the upper envelope of the two static regimes?).

Writes analysis/weth_usdc_90d/guidestar_regime_cache.json. Re-run offline to refresh.
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
from arena_eval.exact_simple_amm.guidestar_volatile import GuidestarVolatileStrategy
from arena_eval.exact_simple_amm.flow_aware_guidestar import FlowAwareGuidestarStrategy, SizeAwareGuidestarStrategy, UnifiedGuidestarStrategy

A = Path("analysis/weth_usdc_90d")
CACHE = A / "guidestar_regime_cache.json"
QUANT = str(A / "parent_order_usd_quantiles.csv")
NORM_PHI, NORM_D = 0.000465, 275.1e6
INIT_PX, STEP_S, MARKOUT_S = 100.0, 12.0, 15.0
FEE_INIT_BPS, BASE_ARRIVAL = 3.5, 98_676 / 216_000
N_STEPS, PRIMARY_DEPTH = 5000, 1_000_000.0
SEEDS = tuple(range(40, 52))                       # 12 paired seeds per cell
ARR_MULTS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
MEAN_MULTS = [0.5, 1.0, 2.0, 4.0, 8.0]
POOLS = ["Guidestar", "flat 3.5bp", "flat 5bp", "FlowAware", "SizeAware", "Unified"]


def make_strat(name):
    if name == "Guidestar":
        return GuidestarVolatileStrategy()
    if name == "flat 3.5bp":
        return FixedFeeStrategy(FEE_INIT_BPS / 1e4, FEE_INIT_BPS / 1e4)
    if name == "flat 5bp":
        return FixedFeeStrategy(5e-4, 5e-4)
    if name == "FlowAware":
        return FlowAwareGuidestarStrategy()
    if name == "SizeAware":
        return SizeAwareGuidestarStrategy()
    if name == "Unified":
        return UnifiedGuidestarStrategy()
    raise ValueError(name)


def make_cfg(arrival_mult, quantiles_path):
    return ExactSimpleAMMConfig(
        n_steps=N_STEPS, initial_price=INIT_PX, initial_x=NORM_D / INIT_PX, initial_y=NORM_D,
        submission_liquidity_fraction=PRIMARY_DEPTH / NORM_D, evaluator_kind="real_data",
        price_process_kind="regime_switching", retail_flow_kind="empirical_usd_size",
        retail_arrival_rate=BASE_ARRIVAL * arrival_mult, retail_buy_prob=0.4627,
        regime_invcdf_path=str(A / "regimes_invcdf.csv"),
        regime_transition_path=str(A / "regimes_transition_matrix.csv"),
        retail_usd_quantiles_path=quantiles_path, normalizer_tracks_fair=True,
    )


def markout(fair, trades):
    n = len(fair); off = MARKOUT_S / STEP_S
    tot = ret = arb = vol = 0.0
    for (t, src, side, ax, ay) in trades:
        tf = t + off; i0 = int(np.floor(tf))
        if i0 + 1 >= n:
            continue
        f15 = fair[i0] + (tf - i0) * (fair[i0 + 1] - fair[i0])
        m = (ay - ax * f15) if side == "buy_x" else (ax * f15 - ay)
        tot += m
        if src == "retail":
            ret += m; vol += ay
        else:
            arb += m
    return tot, ret, arb, vol


def run_cell(name, arrival_mult, quantiles_path):
    tot = ret = arb = vol = 0.0
    for s in SEEDS:
        sim = ExactSimpleAMMSimulator(config=make_cfg(arrival_mult, quantiles_path),
                                      submission_strategy=make_strat(name),
                                      normalizer_strategy=FixedFeeStrategy(NORM_PHI, NORM_PHI), seed=s)
        fair, trades = [], []
        while not sim.done:
            out = sim.step_once()
            fair.append(out["fair_price"])
            for ev in out["trade_events"]:
                if ev["venue"] == "submission":
                    trades.append((out["timestamp"], ev["source"], ev["trader_side"], ev["amount_x"], ev["amount_y"]))
        t, r, a, v = markout(np.asarray(fair), trades)
        tot += t; ret += r; arb += a; vol += v
    n = len(SEEDS)
    return tot / n, ret / n, arb / n, vol / n


def main():
    pct = np.genfromtxt(QUANT, delimiter=",", names=True, dtype=float)["pct"]
    sz = np.genfromtxt(QUANT, delimiter=",", names=True, dtype=float)["size_usd"]
    tmp = Path(tempfile.mkdtemp(prefix="gs_regime_"))
    quant_paths = {}
    for m in MEAN_MULTS:
        p = tmp / f"mean_{m}.csv"
        with open(p, "w") as f:
            f.write("pct,size_usd\n")
            for a, b in zip(pct, sz * m):
                f.write(f"{a},{b}\n")
        quant_paths[m] = str(p)

    grids = {nm: {k: np.zeros((len(MEAN_MULTS), len(ARR_MULTS))) for k in ("total", "retail", "arb", "volume")}
             for nm in POOLS}
    for i, mm in enumerate(MEAN_MULTS):
        for j, am in enumerate(ARR_MULTS):
            for nm in POOLS:
                t, r, a, v = run_cell(nm, am, quant_paths[mm])
                grids[nm]["total"][i, j] = t; grids[nm]["retail"][i, j] = r
                grids[nm]["arb"][i, j] = a; grids[nm]["volume"][i, j] = v
            best = max(POOLS, key=lambda nm: grids[nm]["total"][i, j])
            print(f"  mean×{mm:<4} arrival×{am:<5} best={best:11s} "
                  + " ".join(f"{nm[:4]}:{grids[nm]['total'][i,j]:7.0f}" for nm in POOLS))

    cache = dict(
        meta=dict(seeds=len(SEEDS), n_steps=N_STEPS, depth=PRIMARY_DEPTH, pools=POOLS,
                  arr_mults=ARR_MULTS, mean_mults=MEAN_MULTS, base_arrival=BASE_ARRIVAL,
                  normalizer="§8 full-market φ=4.65bps D=$275M"),
        grids={nm: {k: grids[nm][k].round(2).tolist() for k in grids[nm]} for nm in POOLS},
    )
    CACHE.write_text(json.dumps(cache))
    print(f"\nwrote {CACHE} ({CACHE.stat().st_size/1e3:.0f} KB)")


if __name__ == "__main__":
    main()
