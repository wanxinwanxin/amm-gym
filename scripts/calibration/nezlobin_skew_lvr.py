"""Does the top-of-block EMA directional-skew capture LVR? Isolate it and test the
mechanism's dependence on return autocorrelation (momentum).

The skew raises the resting fee in the direction of recent price impact, to make the
NEXT block's LVR-arb (which realigns to the new fair) pay more. That only helps if
recent moves predict the next move — i.e. if returns are autocorrelated. We swap an
AR(1) momentum price process (tunable ρ, vol matched to the regime process) onto the
sim and, at $1M / §8 normalizer / bottom-arb env, compare the arb/LVR loss of:
  flat 9bp (4.5/4.5)        — no skew (baseline)
  skew-only Nezlobin        — EMA skew only (surcharge + exception OFF)
  Nezlobin (full)           — reference
  Guidestar                 — reference (persistent defense)
ρ=0 is a martingale (≈ our calibrated WETH/USD market, autocorr≈0).
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arena_eval.exact_simple_amm.simulator import ExactSimpleAMMSimulator
from arena_eval.exact_simple_amm.strategies import FixedFeeStrategy
from arena_eval.exact_simple_amm.guidestar_volatile import GuidestarVolatileStrategy
from arena_eval.exact_simple_amm.nezlobin_dynamic_fee import NezlobinDynamicFeeStrategy
from scripts.calibration.nezlobin_backtest import A, NORM_PHI, INIT_PX, N_STEPS, PRIMARY_DEPTH, make_cfg, markout, run_pool

CACHE = A / "nezlobin_skew_lvr_cache.json"
SEEDS = tuple(range(40, 52))
RHOS = [0.0, 0.1, 0.2, 0.3, 0.5]
POOLS = ["flat 9bp", "skew-only", "Nezlobin", "Guidestar"]


class MomentumPriceProcess:
    """AR(1) log-return process: r_t = ρ·r_{t-1} + sqrt(1-ρ²)·ε, ε~N(0,σ). Lag-1 return
    autocorr = ρ, per-step std = σ. ρ=0 is a martingale."""
    def __init__(self, initial_price, rho, sigma, seed):
        self.current_price = float(initial_price); self.rho = float(rho); self.sigma = float(sigma)
        self.rng = np.random.default_rng(seed); self.r_prev = 0.0
    def step(self):
        eps = self.rng.normal(0.0, self.sigma)
        r = self.rho * self.r_prev + math.sqrt(max(1.0 - self.rho ** 2, 0.0)) * eps
        self.r_prev = r
        self.current_price *= math.exp(r)
        return self.current_price


def make_strat(name):
    if name == "flat 9bp":
        return FixedFeeStrategy(4.5e-4, 4.5e-4)
    if name == "skew-only":
        return NezlobinDynamicFeeStrategy(surcharge_on=False, exception_on=False)
    if name == "Nezlobin":
        return NezlobinDynamicFeeStrategy()
    return GuidestarVolatileStrategy()


def run(name, rho, sigma, seed):
    sim = ExactSimpleAMMSimulator(config=make_cfg(PRIMARY_DEPTH), submission_strategy=make_strat(name),
                                  normalizer_strategy=FixedFeeStrategy(NORM_PHI, NORM_PHI), seed=seed)
    sim.price_process = MomentumPriceProcess(INIT_PX, rho, sigma, seed)   # swap in momentum
    fair, trades = [], []
    while not sim.done:
        out = sim.step_once(); fair.append(out["fair_price"])
        for ev in out["trade_events"]:
            if ev["venue"] == "submission":
                trades.append((out["timestamp"], ev["source"], ev["trader_side"], ev["amount_x"], ev["amount_y"]))
    tot = ret = arb = 0.0
    for (t, src, m, n_) in markout(np.asarray(fair), trades):
        tot += m; (ret := ret) ; 
        if src == "retail": ret += m
        else: arb += m
    return tot, ret, arb


def main():
    # match the AR(1) innovation std to the regime process's realized per-step log-return std
    f, _ = run_pool("flat 5bp (incumbent)", PRIMARY_DEPTH, 40)
    sigma = float(np.std(np.diff(np.log(f))))
    print(f"matched per-step σ = {sigma*1e4:.1f} bps (from regime process)")
    res = {nm: {"total": [], "arb": [], "retail": []} for nm in POOLS}
    for rho in RHOS:
        agg = {nm: [0.0, 0.0, 0.0] for nm in POOLS}
        for nm in POOLS:
            for s in SEEDS:
                t, r, a = run(nm, rho, sigma, s)
                agg[nm][0] += t; agg[nm][1] += a; agg[nm][2] += r
        for nm in POOLS:
            n = len(SEEDS)
            res[nm]["total"].append(round(agg[nm][0] / n, 2))
            res[nm]["arb"].append(round(agg[nm][1] / n, 2))
            res[nm]["retail"].append(round(agg[nm][2] / n, 2))
        sk_cap = res["skew-only"]["arb"][-1] - res["flat 9bp"]["arb"][-1]
        print(f"  ρ={rho:<4}: flat arb {res['flat 9bp']['arb'][-1]:7.0f} | skew-only arb {res['skew-only']['arb'][-1]:7.0f} "
              f"(skew LVR capture {sk_cap:+.0f}) | Nezlobin arb {res['Nezlobin']['arb'][-1]:7.0f} | Guidestar arb {res['Guidestar']['arb'][-1]:7.0f}")
    cache = dict(meta=dict(seeds=len(SEEDS), depth=PRIMARY_DEPTH, rhos=RHOS, sigma_bps=round(sigma * 1e4, 2),
                           pool_order=POOLS, normalizer="§8 full-market"),
                 **{nm: res[nm] for nm in POOLS})
    CACHE.write_text(json.dumps(cache))
    print(f"wrote {CACHE}")


if __name__ == "__main__":
    main()
