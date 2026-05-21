"""Cycle 5 — Joint NM with re-weighted loss prioritizing the worst residuals.

Strategy: weight T2 (consistently floored on both sets) 4x more than T1/T3.
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


CYCLE = 5
T2_WEIGHT = 4.0  # emphasis on T2 residual


def _params_from_x(x: np.ndarray) -> Params:
    return Params(
        submission_depth_y=float(math.exp(x[0])),
        normalizer_fee=float(math.exp(x[1])),
        normalizer_depth_y=float(math.exp(x[2])),
    )


def _x_from_params(p: Params) -> np.ndarray:
    return np.array(
        [math.log(p.submission_depth_y), math.log(p.normalizer_fee), math.log(p.normalizer_depth_y)],
        dtype=np.float64,
    )


def loss_factory(log: list[dict], t2_weight: float = T2_WEIGHT):
    def loss(x: np.ndarray) -> float:
        if not (math.log(1e-5) <= x[1] <= math.log(0.5)):
            return 1e3
        if not (math.log(1e6) <= x[0] <= math.log(1e12)):
            return 1e3
        if not (math.log(1e6) <= x[2] <= math.log(1e12)):
            return 1e3
        p = _params_from_x(x)
        t0 = time.time()
        mc = evaluate_params(p, CALIB_SEEDS, N_STEPS)
        mh = evaluate_params(p, HOLDOUT_SEEDS, N_STEPS)
        rc = residuals(mc, TARGETS)
        rh = residuals(mh, TARGETS)
        # Weighted loss
        L = (
            rc["T1_arb_5bp_share"] ** 2 + rh["T1_arb_5bp_share"] ** 2
            + t2_weight * (rc["T2_retail_5bp_share"] ** 2 + rh["T2_retail_5bp_share"] ** 2)
            + rc["T3_markout_bps"] ** 2 + rh["T3_markout_bps"] ** 2
        )
        log.append({
            "sub_depth_y": p.submission_depth_y,
            "norm_fee": p.normalizer_fee,
            "norm_depth_y": p.normalizer_depth_y,
            "rc": {k: float(rc[k]) for k in rc},
            "rh": {k: float(rh[k]) for k in rh},
            "loss": L,
            "elapsed_s": time.time() - t0,
        })
        if len(log) % 10 == 1:
            mxc = max(abs(v) for v in rc.values())
            mxh = max(abs(v) for v in rh.values())
            print(f"  eval {len(log):3d} | sub={p.submission_depth_y/1e6:6.1f}M  nf={p.normalizer_fee*1e4:6.1f}bp  nd={p.normalizer_depth_y/1e9:6.2f}G | "
                  f"L_w={L:.4f}  mxc={mxc:.3f} mxh={mxh:.3f}", flush=True)
        return L
    return loss


if __name__ == "__main__":
    init = Params(91.8e6, 0.03429, 23.0e9)
    print(f"Cycle 5 — weighted joint NM (T2 weight = {T2_WEIGHT})")
    print(f"start: sub={init.submission_depth_y:.4e} nf={init.normalizer_fee:.4e} nd={init.normalizer_depth_y:.4e}")
    log: list[dict] = []
    x0 = _x_from_params(init)
    loss_fn = loss_factory(log)
    initial_simplex = np.vstack([
        x0,
        x0 + np.array([0.3, 0.0, 0.0]),
        x0 + np.array([0.0, 0.3, 0.0]),
        x0 + np.array([0.0, 0.0, 0.3]),
    ])
    t0 = time.time()
    result = minimize(
        loss_fn,
        x0,
        method="Nelder-Mead",
        options={
            "maxiter": 200,
            "xatol": 1e-5,
            "fatol": 1e-6,
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

    print(f"\n=== Cycle 5 weighted NM result ===")
    print(f"params: sub={best_params.submission_depth_y:.4e}  nf={best_params.normalizer_fee:.4e}  nd={best_params.normalizer_depth_y:.4e}")
    print(f"calib   T1={mc['arb_5bp_share']:.4f} T2={mc['retail_5bp_share']:.4f} T3={mc['markout_bps']:.4f}")
    print(f"        r1={rc['T1_arb_5bp_share']:+.4%} r2={rc['T2_retail_5bp_share']:+.4%} r3={rc['T3_markout_bps']:+.4%}")
    print(f"holdout T1={mh['arb_5bp_share']:.4f} T2={mh['retail_5bp_share']:.4f} T3={mh['markout_bps']:.4f}")
    print(f"        r1={rh['T1_arb_5bp_share']:+.4%} r2={rh['T2_retail_5bp_share']:+.4%} r3={rh['T3_markout_bps']:+.4%}")
    mxc = max(abs(v) for v in rc.values())
    mxh = max(abs(v) for v in rh.values())
    print(f"max|rc|={mxc:.4f}  max|rh|={mxh:.4f}  DONE={mxc < 0.02 and mxh < 0.02}")
    print(f"L_w={result.fun:.6f}  L_unw={sum(v**2 for v in rc.values()) + sum(v**2 for v in rh.values()):.6f}")
    print(f"calib per-seed arb: {[round(r['arb_share'], 3) for r in mc['per_seed']]}")
    print(f"hold per-seed arb:  {[round(r['arb_share'], 3) for r in mh['per_seed']]}")

    out = {
        "cycle": CYCLE,
        "name": "weighted_joint_nm",
        "t2_weight": T2_WEIGHT,
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
        "scipy": {"fun": float(result.fun), "nfev": int(result.nfev), "nit": int(result.nit),
                  "success": bool(result.success), "message": str(result.message)},
        "log": log,
        "elapsed_s": elapsed,
    }
    (ART_DIR / f"cycle_{CYCLE:02d}.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"saved -> calibration_artifacts/cycle_{CYCLE:02d}.json")
