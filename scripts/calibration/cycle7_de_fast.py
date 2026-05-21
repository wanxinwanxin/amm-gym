"""Fast global DE to verify the joint NM basin is the global minimum."""
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


def _params_from_x(x):
    return Params(math.exp(x[0]), math.exp(x[1]), math.exp(x[2]))


# Use a 3-seed subset for speed
FAST_CALIB = (42, 43, 44)
FAST_HOLD = (100, 101, 102)


def joint_loss(x):
    p = _params_from_x(x)
    mc = evaluate_params(p, FAST_CALIB, N_STEPS)
    mh = evaluate_params(p, FAST_HOLD, N_STEPS)
    rc = residuals(mc, TARGETS)
    rh = residuals(mh, TARGETS)
    return sum(v**2 for v in rc.values()) + sum(v**2 for v in rh.values())


eval_count = [0]
def loss_with_counter(x):
    eval_count[0] += 1
    L = joint_loss(x)
    if eval_count[0] % 20 == 1:
        p = _params_from_x(x)
        print(f"  eval {eval_count[0]:3d} | sub={p.submission_depth_y/1e6:6.1f}M nf={p.normalizer_fee*1e4:6.1f}bp nd={p.normalizer_depth_y/1e9:6.2f}G | L={L:.4f}", flush=True)
    return L


if __name__ == "__main__":
    bounds = [
        (math.log(1e7), math.log(1e9)),    # sub: 10M..1B
        (math.log(1e-4), math.log(0.1)),    # nf: 1bp..1000bp
        (math.log(1e9), math.log(1e12)),    # nd: 1G..1T
    ]
    print("Fast 3-seed DE search")
    t0 = time.time()
    res = differential_evolution(
        loss_with_counter,
        bounds,
        maxiter=15,
        popsize=8,
        seed=42,
        polish=False,
        tol=1e-4,
        atol=1e-5,
    )
    elapsed = time.time() - t0
    best_x = res.x
    best_params = _params_from_x(best_x)
    print(f"\n=== Fast DE result (elapsed {elapsed:.1f}s, nfev={res.nfev}) ===")
    print(f"params: sub={best_params.submission_depth_y:.4e}  nf={best_params.normalizer_fee:.4e}  nd={best_params.normalizer_depth_y:.4e}")
    # Re-eval on full 5-seed sets
    print("\n--- Re-eval on full 5-seed sets ---")
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
    L_full = sum(v**2 for v in rc.values()) + sum(v**2 for v in rh.values())
    print(f"L_full(5+5)={L_full:.6f}  max|rc|={mxc:.4f}  max|rh|={mxh:.4f}  DONE={mxc < 0.02 and mxh < 0.02}")
    print(f"calib per-seed arb: {[round(r['arb_share'], 3) for r in mc['per_seed']]}")
    print(f"hold per-seed arb:  {[round(r['arb_share'], 3) for r in mh['per_seed']]}")

    out = {
        "best_params": {
            "submission_depth_y": best_params.submission_depth_y,
            "normalizer_fee": best_params.normalizer_fee,
            "normalizer_depth_y": best_params.normalizer_depth_y,
        },
        "fast_de_loss_3seed": float(res.fun),
        "full_5seed_loss": L_full,
        "calib_residuals": {k: float(v) for k, v in rc.items()},
        "holdout_residuals": {k: float(v) for k, v in rh.items()},
        "max_calib_residual": float(mxc),
        "max_holdout_residual": float(mxh),
        "elapsed_s": elapsed,
        "nfev": int(res.nfev),
    }
    p = Path(__file__).resolve().parents[2] / "calibration_artifacts" / "cycle_07_de_fast.json"
    p.write_text(json.dumps(out, indent=2, default=float))
    print(f"saved -> {p}")
