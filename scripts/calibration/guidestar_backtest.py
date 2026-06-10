"""Backtest a NEW (shallow) dynamic-fee pool vs fixed-fee baselines, against the
§8 full-market normalizer, on the calibrated realistic tape.

Computes the results and writes a compact JSON cache
(analysis/weth_usdc_90d/guidestar_backtest_cache.json); the notebook §9 loads the
cache and plots (via presentation/helpers.py) so Run-All stays fast. Re-run this
script offline to refresh the cache.

Metric: 15s-forward LP markout (executed price vs the fair price 15s later) in $,
summed over every swap (retail + arb) hitting the candidate pool. Paired seeds
(identical fair path + retail arrival stream; only the submission fee policy
differs). Pools at equal shallow depth: Guidestar volatile (real mainnet params,
3.5bp floor) / flat 3.5bp (= feeInit, "dynamics off") / flat 5bp (incumbent).
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

A = Path("analysis/weth_usdc_90d")
CACHE = A / "guidestar_backtest_cache.json"
NORM_PHI, NORM_D = 0.000465, 275.1e6
INIT_PX, STEP_S, MARKOUT_S = 100.0, 12.0, 15.0
FEE_INIT_BPS = 3.5
BASE_ARRIVAL = 98_676 / 216_000
SEEDS = tuple(range(40, 56))               # 16 paired seeds
SEEDS_BE = tuple(range(40, 48))            # 8 seeds for the break-even sweep
N_STEPS = 5000
DEPTHS = [1_000_000.0, 500_000.0]
PRIMARY_DEPTH = 1_000_000.0
BE_MULTS = [1.0, 2.0, 4.0, 7.0, 11.0, 16.0]
POOL_ORDER = ["Guidestar (real params, 3.5bp floor)", "flat 3.5bp (=feeInit, dynamics off)", "flat 5bp (incumbent)"]


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


def make_strat(name):
    fi = FEE_INIT_BPS / 1e4
    if name == POOL_ORDER[0]:
        return GuidestarVolatileStrategy()
    if name == POOL_ORDER[1]:
        return FixedFeeStrategy(fi, fi)
    return FixedFeeStrategy(5e-4, 5e-4)


def run_pool(name, depth_usd, seed, arrival_mult=1.0):
    sim = ExactSimpleAMMSimulator(config=make_cfg(depth_usd, arrival_mult), submission_strategy=make_strat(name),
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


def aggregate(name, depth_usd, arrival_mult=1.0):
    cum_tot, cum_ret, cum_arb, retail_vol = [], [], [], []
    notl, mk, src = [], [], []
    for s in SEEDS:
        fair, trades = run_pool(name, depth_usd, s, arrival_mult)
        st, sr, sa = (np.zeros(N_STEPS) for _ in range(3)); rv = 0.0
        for (t, source, m, n_) in markout(fair, trades):
            st[t] += m
            (sr if source == "retail" else sa)[t] += m
            if source == "retail":
                rv += n_
            notl.append(n_); mk.append(m); src.append(source)
        cum_tot.append(np.cumsum(st)); cum_ret.append(np.cumsum(sr)); cum_arb.append(np.cumsum(sa)); retail_vol.append(rv)
    return dict(cum_tot=np.array(cum_tot), cum_ret=np.array(cum_ret), cum_arb=np.array(cum_arb),
                retail_vol=float(np.mean(retail_vol)),
                notl=np.array(notl), mk=np.array(mk), src=np.array(src))


def build_cache():
    res = {d: {nm: aggregate(nm, d) for nm in POOL_ORDER} for d in DEPTHS}
    R = res[PRIMARY_DEPTH]
    ds = slice(0, N_STEPS, 10)                                   # downsample curves to 500 pts
    steps = list(range(0, N_STEPS, 10))
    cumulative = {nm: {comp: {"mean": R[nm][f"cum_{comp}"].mean(0)[ds].round(3).tolist(),
                              "sd": R[nm][f"cum_{comp}"].std(0)[ds].round(3).tolist()}
                       for comp in ["tot", "ret", "arb"]} for nm in POOL_ORDER}

    hbins = np.linspace(-150, 150, 81)
    histogram = {"bins": hbins.tolist()}
    for nm in POOL_ORDER:
        b = 1e4 * R[nm]["mk"][R[nm]["notl"] > 1.0] / R[nm]["notl"][R[nm]["notl"] > 1.0]
        b = b[np.isfinite(b)]
        histogram[nm] = (np.histogram(b, bins=hbins, density=True)[0]).round(6).tolist()

    edges = np.logspace(0, 6, 19); centers = np.sqrt(edges[:-1] * edges[1:])
    by_size = {"centers": centers.round(2).tolist()}
    for nm in POOL_ORDER:
        notl, mk, src = R[nm]["notl"], R[nm]["mk"], R[nm]["src"]
        ret = src == "retail"
        rmean, tot, vol = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mr = ret & (notl >= lo) & (notl < hi); ma = (notl >= lo) & (notl < hi)
            rmean.append(float(np.mean(1e4 * mk[mr] / notl[mr])) if mr.sum() >= 5 else None)
            tot.append(float(mk[ma].sum() / len(SEEDS))); vol.append(float(notl[ma].sum() / len(SEEDS)))
        by_size[nm] = {"retail_mean_bps": rmean, "total_usd": tot, "volume": vol}

    print("break-even sweep ...")
    finals = {nm: [] for nm in POOL_ORDER}
    for m in BE_MULTS:
        for nm in POOL_ORDER:
            vals = [sum(r[2] for r in markout(*run_pool(nm, PRIMARY_DEPTH, s, m))) for s in SEEDS_BE]
            finals[nm].append(float(np.mean(vals)))
    breakeven = {"mults": BE_MULTS, "finals": finals, "seeds": len(SEEDS_BE)}

    volume = {str(d): {nm: res[d][nm]["retail_vol"] for nm in POOL_ORDER} for d in DEPTHS}
    summary = {str(d): {nm: float(res[d][nm]["cum_tot"][:, -1].mean()) for nm in POOL_ORDER} for d in DEPTHS}

    cache = dict(
        meta=dict(seeds=len(SEEDS), n_steps=N_STEPS, step_seconds=STEP_S, markout_seconds=MARKOUT_S,
                  fee_init_bps=FEE_INIT_BPS, normalizer="§8 full-market φ=4.65bps D=$275M",
                  depths=DEPTHS, primary_depth=PRIMARY_DEPTH, pool_order=POOL_ORDER,
                  guidestar_params="mainnet volatile defaults (DeployGuidestar4 defaultHookParams)"),
        cumulative_steps=steps, cumulative=cumulative, histogram=histogram, by_size=by_size,
        breakeven=breakeven, volume=volume, summary=summary,
    )
    CACHE.write_text(json.dumps(cache))
    print(f"wrote {CACHE} ({CACHE.stat().st_size/1e3:.0f} KB)")
    for d in DEPTHS:
        for nm in POOL_ORDER:
            print(f"  ${d/1e6:.1f}M {nm:40s} final markout ${summary[str(d)][nm]:>10,.0f}  retail vol ${volume[str(d)][nm]:>12,.0f}")


if __name__ == "__main__":
    build_cache()
