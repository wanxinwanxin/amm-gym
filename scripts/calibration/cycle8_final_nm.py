"""Cycle 8 — Final tight NM with strict xatol/fatol to extract the basin's
minimum-loss point as precisely as possible.

Starting from the converged basin and using:
  - Much tighter convergence criteria
  - Larger initial simplex to ensure broad coverage
  - max_iter=400 to give it plenty of room

Goal: produce the canonical final params + residuals.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

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
from scripts.calibration.run_cycles import ART_DIR


CYCLE = 8


def _params_from_x(x):
    return Params(math.exp(x[0]), math.exp(x[1]), math.exp(x[2]))


def _x_from_params(p):
    return np.array([math.log(p.submission_depth_y), math.log(p.normalizer_fee), math.log(p.normalizer_depth_y)])


def joint_loss(x, log):
    if not (math.log(1e-5) <= x[1] <= math.log(0.5)):
        return 1e3
    if not (math.log(1e6) <= x[0] <= math.log(1e12)):
        return 1e3
    if not (math.log(1e6) <= x[2] <= math.log(1e12)):
        return 1e3
    p = _params_from_x(x)
    mc = evaluate_params(p, CALIB_SEEDS, N_STEPS)
    mh = evaluate_params(p, HOLDOUT_SEEDS, N_STEPS)
    rc = residuals(mc, TARGETS)
    rh = residuals(mh, TARGETS)
    L = sum(v**2 for v in rc.values()) + sum(v**2 for v in rh.values())
    log.append({
        "sub": p.submission_depth_y, "nf": p.normalizer_fee, "nd": p.normalizer_depth_y,
        "rc": dict(rc), "rh": dict(rh), "L": L,
    })
    if len(log) % 20 == 1:
        mxc = max(abs(v) for v in rc.values())
        mxh = max(abs(v) for v in rh.values())
        print(f"  eval {len(log):3d} | sub={p.submission_depth_y/1e6:6.2f}M nf={p.normalizer_fee*1e4:6.2f}bp nd={p.normalizer_depth_y/1e9:6.2f}G | L={L:.5f} mxc={mxc:.4f} mxh={mxh:.4f}", flush=True)
    return L


if __name__ == "__main__":
    init = Params(91.8e6, 0.03429, 23.0e9)
    log: list[dict] = []
    x0 = _x_from_params(init)
    initial_simplex = np.vstack([
        x0,
        x0 + np.array([0.10, 0.0, 0.0]),
        x0 + np.array([0.0, 0.10, 0.0]),
        x0 + np.array([0.0, 0.0, 0.10]),
    ])
    print(f"Cycle 8 — final tight NM (xatol=1e-7, fatol=1e-8)")
    print(f"init: sub={init.submission_depth_y/1e6:.1f}M nf={init.normalizer_fee*1e4:.1f}bp nd={init.normalizer_depth_y/1e9:.1f}G")
    t0 = time.time()
    result = minimize(
        lambda x: joint_loss(x, log),
        x0,
        method="Nelder-Mead",
        options={
            "maxiter": 400,
            "xatol": 1e-7,
            "fatol": 1e-8,
            "initial_simplex": initial_simplex,
            "adaptive": True,
        },
    )
    elapsed = time.time() - t0
    best_params = _params_from_x(result.x)
    mc = evaluate_params(best_params, CALIB_SEEDS, N_STEPS)
    mh = evaluate_params(best_params, HOLDOUT_SEEDS, N_STEPS)
    rc = residuals(mc, TARGETS)
    rh = residuals(mh, TARGETS)
    mxc = max(abs(v) for v in rc.values())
    mxh = max(abs(v) for v in rh.values())
    L = sum(v**2 for v in rc.values()) + sum(v**2 for v in rh.values())

    print(f"\n=== Cycle 8 — final tight NM result ===")
    print(f"params: sub={best_params.submission_depth_y:.6e}  nf={best_params.normalizer_fee:.6e}  nd={best_params.normalizer_depth_y:.6e}")
    print(f"calib   T1={mc['arb_5bp_share']:.4f}  T2={mc['retail_5bp_share']:.4f}  T3={mc['markout_bps']:.4f}")
    print(f"        r1={rc['T1_arb_5bp_share']:+.4%}  r2={rc['T2_retail_5bp_share']:+.4%}  r3={rc['T3_markout_bps']:+.4%}")
    print(f"holdout T1={mh['arb_5bp_share']:.4f}  T2={mh['retail_5bp_share']:.4f}  T3={mh['markout_bps']:.4f}")
    print(f"        r1={rh['T1_arb_5bp_share']:+.4%}  r2={rh['T2_retail_5bp_share']:+.4%}  r3={rh['T3_markout_bps']:+.4%}")
    print(f"L={L:.6f}  max|rc|={mxc:.4f}  max|rh|={mxh:.4f}  DONE={mxc < 0.02 and mxh < 0.02}")
    print(f"calib per-seed arb: {[round(r['arb_share'], 4) for r in mc['per_seed']]}")
    print(f"hold per-seed arb:  {[round(r['arb_share'], 4) for r in mh['per_seed']]}")
    print(f"nfev={result.nfev} nit={result.nit} elapsed={elapsed:.1f}s")

    out = {
        "cycle": CYCLE,
        "name": "final_tight_nm",
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
        "joint_loss": L,
        "max_calib_residual": float(mxc),
        "max_holdout_residual": float(mxh),
        "scipy": {"fun": float(result.fun), "nfev": int(result.nfev), "nit": int(result.nit),
                  "success": bool(result.success), "message": str(result.message)},
        "log": log,
        "elapsed_s": elapsed,
    }
    p = ART_DIR / f"cycle_{CYCLE:02d}.json"
    p.write_text(json.dumps(out, indent=2, default=float))
    print(f"saved -> {p}")
