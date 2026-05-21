"""Global differential evolution search across the entire 3-param space.

Use a 3-seed (42, 43, 44) loss for speed, then refine top candidates
on the full 5+5 seed set.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.calibration.eval_metrics import (
    CALIB_SEEDS,
    HOLDOUT_SEEDS,
    N_STEPS,
    Params,
    TARGETS,
    evaluate_params,
    residuals,
)


def _params_from_x(x: np.ndarray) -> Params:
    return Params(
        submission_depth_y=float(math.exp(x[0])),
        normalizer_fee=float(math.exp(x[1])),
        normalizer_depth_y=float(math.exp(x[2])),
    )


# Use ALL 10 seeds for the global search to get robust mean
ALL_SEEDS = tuple(CALIB_SEEDS) + tuple(HOLDOUT_SEEDS)


def joint_loss(x):
    p = _params_from_x(x)
    mc = evaluate_params(p, CALIB_SEEDS, N_STEPS)
    mh = evaluate_params(p, HOLDOUT_SEEDS, N_STEPS)
    rc = residuals(mc, TARGETS)
    rh = residuals(mh, TARGETS)
    return sum(v ** 2 for v in rc.values()) + sum(v ** 2 for v in rh.values())


if __name__ == "__main__":
    # Log-space bounds
    bounds = [
        (math.log(1e6), math.log(1e10)),  # sub_depth: 1M..10B
        (math.log(1e-4), math.log(0.2)),   # nf: 1bp..2000bp
        (math.log(1e9), math.log(1e12)),   # nd: 1G..1T
    ]
    print("Global DE search (joint 10-seed loss)")
    print(f"bounds: sub 1M-10B, nf 1bp-2000bp, nd 1G-1T")
    print()
    t0 = time.time()
    res = differential_evolution(
        joint_loss,
        bounds,
        maxiter=30,
        popsize=10,
        tol=1e-4,
        polish=True,
        workers=1,
        seed=42,
        disp=True,
        atol=1e-5,
    )
    elapsed = time.time() - t0
    best_x = res.x
    best_params = _params_from_x(best_x)
    print(f"\n=== Global DE result (elapsed {elapsed:.1f}s, nfev={res.nfev}) ===")
    print(f"params: sub={best_params.submission_depth_y:.4e}  nf={best_params.normalizer_fee:.4e}  nd={best_params.normalizer_depth_y:.4e}")
    mc = evaluate_params(best_params, CALIB_SEEDS, N_STEPS)
    mh = evaluate_params(best_params, HOLDOUT_SEEDS, N_STEPS)
    rc = residuals(mc, TARGETS)
    rh = residuals(mh, TARGETS)
    print(f"calib   T1={mc['arb_5bp_share']:.4f} T2={mc['retail_5bp_share']:.4f} T3={mc['markout_bps']:.4f}")
    print(f"        r1={rc['T1_arb_5bp_share']:+.4%} r2={rc['T2_retail_5bp_share']:+.4%} r3={rc['T3_markout_bps']:+.4%}")
    print(f"holdout T1={mh['arb_5bp_share']:.4f} T2={mh['retail_5bp_share']:.4f} T3={mh['markout_bps']:.4f}")
    print(f"        r1={rh['T1_arb_5bp_share']:+.4%} r2={rh['T2_retail_5bp_share']:+.4%} r3={rh['T3_markout_bps']:+.4%}")
    mxc = max(abs(v) for v in rc.values())
    mxh = max(abs(v) for v in rh.values())
    print(f"max|rc|={mxc:.4f}  max|rh|={mxh:.4f}  DONE={mxc < 0.02 and mxh < 0.02}")
    print(f"calib per-seed arb: {[round(r['arb_share'], 3) for r in mc['per_seed']]}")
    print(f"hold per-seed arb:  {[round(r['arb_share'], 3) for r in mh['per_seed']]}")

    art_dir = Path(__file__).resolve().parents[2] / "calibration_artifacts"
    out = {
        "name": "global_de_search",
        "best_params": {
            "submission_depth_y": best_params.submission_depth_y,
            "normalizer_fee": best_params.normalizer_fee,
            "normalizer_depth_y": best_params.normalizer_depth_y,
        },
        "calib_metrics": {
            "T1": mc["arb_5bp_share"], "T2": mc["retail_5bp_share"], "T3": mc["markout_bps"],
            "per_seed_arb": [r["arb_share"] for r in mc["per_seed"]],
            "per_seed_retail": [r["retail_share"] for r in mc["per_seed"]],
            "per_seed_markout": [r["markout_bps"] for r in mc["per_seed"]],
        },
        "holdout_metrics": {
            "T1": mh["arb_5bp_share"], "T2": mh["retail_5bp_share"], "T3": mh["markout_bps"],
            "per_seed_arb": [r["arb_share"] for r in mh["per_seed"]],
            "per_seed_retail": [r["retail_share"] for r in mh["per_seed"]],
            "per_seed_markout": [r["markout_bps"] for r in mh["per_seed"]],
        },
        "calib_residuals": {k: float(v) for k, v in rc.items()},
        "holdout_residuals": {k: float(v) for k, v in rh.items()},
        "scipy_de": {
            "fun": float(res.fun),
            "nfev": int(res.nfev),
            "nit": int(res.nit),
            "message": str(res.message),
        },
        "elapsed_s": elapsed,
    }
    (art_dir / "global_de_search.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"saved -> {art_dir / 'global_de_search.json'}")
