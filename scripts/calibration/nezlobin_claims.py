"""Instrument the 'constant spread in most blocks' claim: record the Nezlobin pool's
top-of-block RESTING spread (f_a+f_b) per block across a sim, at $1M / §8 normalizer /
base arrival. Also record the per-block price impact (block-open-to-block-open move),
which drives the EMA skew + the big-PI exception."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arena_eval.exact_simple_amm.simulator import ExactSimpleAMMSimulator
from arena_eval.exact_simple_amm.strategies import FixedFeeStrategy
from scripts.calibration.nezlobin_backtest import A, NORM_PHI, PRIMARY_DEPTH, make_cfg
from arena_eval.exact_simple_amm.nezlobin_dynamic_fee import NezlobinDynamicFeeStrategy

CACHE = A / "nezlobin_claims_cache.json"
SEEDS = tuple(range(40, 48))

def main():
    spreads = []   # resting f_a+f_b per block (bps)
    for s in SEEDS:
        strat = NezlobinDynamicFeeStrategy()
        sim = ExactSimpleAMMSimulator(config=make_cfg(PRIMARY_DEPTH), submission_strategy=strat,
                                      normalizer_strategy=FixedFeeStrategy(NORM_PHI, NORM_PHI), seed=s)
        while not sim.done:
            sim.step_once()
            spreads.append((strat._fa + strat._fb) * 1e4)   # resting spread for the block just processed
    spreads = np.array(spreads)
    ts = 9.0
    edges = np.linspace(8.5, 30.0, 44)
    out = dict(
        ts_bps=ts, n_blocks=int(spreads.size),
        frac_at_ts=float(np.mean(np.abs(spreads - ts) < 0.05)),
        frac_widened=float(np.mean(spreads > ts + 0.05)),
        max_spread=float(spreads.max()), mean_spread=float(spreads.mean()),
        hist_edges=edges.tolist(),
        hist=np.histogram(spreads, bins=edges)[0].tolist(),
    )
    CACHE.write_text(json.dumps(out))
    print(f"blocks={out['n_blocks']}  frac at TS(9bp)={out['frac_at_ts']:.3f}  "
          f"frac widened={out['frac_widened']:.3f}  mean={out['mean_spread']:.2f}bp  max={out['max_spread']:.1f}bp")
    print(f"wrote {CACHE}")

if __name__ == "__main__":
    main()
