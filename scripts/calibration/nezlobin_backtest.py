"""Backtest the doc-aligned Nezlobin Volatile-Hook pool across liquidity depths.

Like scripts/calibration/guidestar_backtest.py, but for the NezlobinDynamicFeeStrategy
(mainnet Volatile Hook spec) and swept over four depths — $0.5M / $1M / $2M / $5M —
against the §8 full-market normalizer (φ=4.65bps, D=$275M, held at fair).

The whole experiment runs in the bottom-of-block-arb environment (config
bottom_of_block_arb=True, arb_max_passes=3): a same-block backrun realigns the pool
to fair after retail — the leg that pays Nezlobin's intra-block surcharge. Every pool
is run in this SAME env on PAIRED seeds (identical fair path + retail stream); only
the submission fee policy differs. So the flat baselines here also get the same-block
backrun (and so their numbers differ from §9, which had no bottom arb).

Metric: 15s-forward LP markout (executed price vs fair 15s later), $ summed over every
swap (retail + arb) on the candidate pool. Writes a JSON cache the notebook/plots load.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arena_eval.exact_simple_amm.config import ExactSimpleAMMConfig
from arena_eval.exact_simple_amm.simulator import ExactSimpleAMMSimulator
from arena_eval.exact_simple_amm.strategies import FixedFeeStrategy
from arena_eval.exact_simple_amm.guidestar_volatile import GuidestarVolatileStrategy
from arena_eval.exact_simple_amm.nezlobin_dynamic_fee import NezlobinDynamicFeeStrategy

A = Path("analysis/weth_usdc_90d")
CACHE = A / "nezlobin_backtest_cache.json"
NORM_PHI, NORM_D = 0.000465, 275.1e6
INIT_PX, STEP_S, MARKOUT_S = 100.0, 12.0, 15.0
BASE_ARRIVAL = 98_676 / 216_000
SEEDS = tuple(range(40, 56))                 # 16 paired seeds
N_STEPS = 5000
DEPTHS = [0.5e6, 1.0e6, 2.0e6, 5.0e6]
PRIMARY_DEPTH = 1.0e6
ARB_PASSES = 3
POOL_ORDER = ["Nezlobin (doc spec)", "Guidestar (real params)",
              "flat 9bp (4.5/4.5, dyn off)", "flat 5bp (incumbent)"]


def make_strat(name):
    if name == POOL_ORDER[0]:
        return NezlobinDynamicFeeStrategy()
    if name == POOL_ORDER[1]:
        return GuidestarVolatileStrategy()
    if name == POOL_ORDER[2]:
        return FixedFeeStrategy(4.5e-4, 4.5e-4)
    return FixedFeeStrategy(5e-4, 5e-4)


def make_cfg(depth_usd):
    return ExactSimpleAMMConfig(
        n_steps=N_STEPS, initial_price=INIT_PX, initial_x=NORM_D / INIT_PX, initial_y=NORM_D,
        submission_liquidity_fraction=depth_usd / NORM_D, evaluator_kind="real_data",
        price_process_kind="regime_switching", retail_flow_kind="empirical_usd_size",
        retail_arrival_rate=BASE_ARRIVAL, retail_buy_prob=0.4627,
        regime_invcdf_path=str(A / "regimes_invcdf.csv"),
        regime_transition_path=str(A / "regimes_transition_matrix.csv"),
        retail_usd_quantiles_path=str(A / "parent_order_usd_quantiles.csv"),
        normalizer_tracks_fair=True, bottom_of_block_arb=True, arb_max_passes=ARB_PASSES,
    )


def run_pool(name, depth_usd, seed):
    sim = ExactSimpleAMMSimulator(config=make_cfg(depth_usd), submission_strategy=make_strat(name),
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


def aggregate(name, depth_usd, collect=False):
    cum_tot, cum_ret, cum_arb, rvol = [], [], [], []
    notl, mk, src = [], [], []
    for s in SEEDS:
        fair, trades = run_pool(name, depth_usd, s)
        st, sr, sa = (np.zeros(N_STEPS) for _ in range(3)); v = 0.0
        for (t, source, m, n_) in markout(fair, trades):
            st[t] += m
            (sr if source == "retail" else sa)[t] += m
            if source == "retail":
                v += n_
            if collect:
                notl.append(n_); mk.append(m); src.append(source)
        cum_tot.append(np.cumsum(st)); cum_ret.append(np.cumsum(sr)); cum_arb.append(np.cumsum(sa)); rvol.append(v)
    out = dict(cum_tot=np.array(cum_tot), cum_ret=np.array(cum_ret), cum_arb=np.array(cum_arb),
               retail_vol=float(np.mean(rvol)))
    if collect:
        out.update(notl=np.array(notl), mk=np.array(mk), src=np.array(src))
    return out


def main():
    res = {}
    for d in DEPTHS:
        for nm in POOL_ORDER:
            res[(d, nm)] = aggregate(nm, d, collect=(d == PRIMARY_DEPTH))
            f = res[(d, nm)]["cum_tot"][:, -1].mean()
            print(f"  ${d/1e6:>3.1f}M {nm:30s} final markout ${f:>9,.0f}  retail vol ${res[(d,nm)]['retail_vol']:>12,.0f}")

    ds = slice(0, N_STEPS, 10); steps = list(range(0, N_STEPS, 10))
    # full cumulative curves only at the primary depth
    cumulative = {nm: {c: {"mean": res[(PRIMARY_DEPTH, nm)][f"cum_{c}"].mean(0)[ds].round(3).tolist(),
                           "sd": res[(PRIMARY_DEPTH, nm)][f"cum_{c}"].std(0)[ds].round(3).tolist()}
                       for c in ("tot", "ret", "arb")} for nm in POOL_ORDER}
    finals = {f"{d:.0f}": {nm: {c: float(res[(d, nm)][f"cum_{c}"][:, -1].mean()) for c in ("tot", "ret", "arb")}
                           for nm in POOL_ORDER} for d in DEPTHS}
    volume = {f"{d:.0f}": {nm: res[(d, nm)]["retail_vol"] for nm in POOL_ORDER} for d in DEPTHS}

    # histogram + by-size at primary depth
    hbins = np.linspace(-150, 150, 81)
    histogram = {"bins": hbins.tolist()}
    edges = np.logspace(0, 6, 19); centers = np.sqrt(edges[:-1] * edges[1:])
    by_size = {"centers": centers.round(2).tolist()}
    for nm in POOL_ORDER:
        R = res[(PRIMARY_DEPTH, nm)]
        notl, mk, src = R["notl"], R["mk"], R["src"]
        b = 1e4 * mk[notl > 1.0] / notl[notl > 1.0]
        b = b[np.isfinite(b)]
        histogram[nm] = np.histogram(b, bins=hbins, density=True)[0].round(6).tolist()
        ret = src == "retail"
        rmean, tot, vol, alln = [], [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mr = ret & (notl >= lo) & (notl < hi); ma = (notl >= lo) & (notl < hi)
            rmean.append(float(np.mean(1e4 * mk[mr] / notl[mr])) if mr.sum() >= 30 else None)
            tot.append(float(mk[ma].sum() / len(SEEDS))); vol.append(float(notl[ma].sum() / len(SEEDS)))
            alln.append(int(ma.sum()))
        by_size[nm] = {"retail_mean_bps": rmean, "total_usd": tot, "volume": vol, "all_count": alln}

    cache = dict(
        meta=dict(seeds=len(SEEDS), n_steps=N_STEPS, step_seconds=STEP_S, markout_seconds=MARKOUT_S,
                  normalizer="§8 full-market φ=4.65bps D=$275M", depths=DEPTHS, primary_depth=PRIMARY_DEPTH,
                  pool_order=POOL_ORDER, base_arrival=BASE_ARRIVAL,
                  env="bottom_of_block_arb=True, arb_max_passes=%d (all pools)" % ARB_PASSES),
        cumulative_steps=steps, cumulative=cumulative, finals=finals, volume=volume,
        histogram=histogram, by_size=by_size,
    )
    CACHE.write_text(json.dumps(cache))
    print(f"\nwrote {CACHE} ({CACHE.stat().st_size/1e3:.0f} KB)")


if __name__ == "__main__":
    main()
