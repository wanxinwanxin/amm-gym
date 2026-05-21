"""Cycle 4 — Refine from cycle 3 with tighter simplex + Powell-style scan."""
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


CYCLE = 4


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


def joint_loss_factory(log: list[dict]):
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
        L = sum(v ** 2 for v in rc.values()) + sum(v ** 2 for v in rh.values())
        entry = {
            "sub_depth_y": p.submission_depth_y,
            "norm_fee": p.normalizer_fee,
            "norm_depth_y": p.normalizer_depth_y,
            "calib": {**{k: float(rc[k]) for k in rc},
                      "T1": mc["arb_5bp_share"], "T2": mc["retail_5bp_share"], "T3": mc["markout_bps"]},
            "holdout": {**{k: float(rh[k]) for k in rh},
                        "T1": mh["arb_5bp_share"], "T2": mh["retail_5bp_share"], "T3": mh["markout_bps"]},
            "loss": L,
            "elapsed_s": time.time() - t0,
        }
        log.append(entry)
        max_rc = max(abs(v) for v in rc.values())
        max_rh = max(abs(v) for v in rh.values())
        if len(log) % 10 == 1:
            print(f"  eval {len(log):3d} | sub={p.submission_depth_y/1e6:6.1f}M  nf={p.normalizer_fee*1e4:6.1f}bp  nd={p.normalizer_depth_y/1e9:6.2f}G  | L={L:.4f}  max|rc|={max_rc:.3f}  max|rh|={max_rh:.3f}", flush=True)
        return L
    return loss


def run_nm(init: Params, simplex: float, max_iter: int, label: str):
    log: list[dict] = []
    x0 = _x_from_params(init)
    loss_fn = joint_loss_factory(log)
    initial_simplex = np.vstack([
        x0,
        x0 + np.array([simplex, 0.0, 0.0]),
        x0 + np.array([0.0, simplex, 0.0]),
        x0 + np.array([0.0, 0.0, simplex]),
    ])
    result = minimize(
        loss_fn,
        x0,
        method="Nelder-Mead",
        options={
            "maxiter": max_iter,
            "xatol": 1e-5,
            "fatol": 1e-6,
            "initial_simplex": initial_simplex,
            "adaptive": True,
        },
    )
    best_params = _params_from_x(result.x)
    mc = evaluate_params(best_params, CALIB_SEEDS, N_STEPS)
    mh = evaluate_params(best_params, HOLDOUT_SEEDS, N_STEPS)
    rc = residuals(mc, TARGETS)
    rh = residuals(mh, TARGETS)
    mxc = max(abs(v) for v in rc.values())
    mxh = max(abs(v) for v in rh.values())
    print(f"  [{label}] result: L={result.fun:.6f}  mxc={mxc:.4f}  mxh={mxh:.4f}  nfev={result.nfev}")
    return best_params, mc, mh, rc, rh, log, result


if __name__ == "__main__":
    # Run NM from a few starting variations around cycle 3 basin
    starts = [
        (Params(91.8e6, 0.034288, 23.0e9), 0.15, "tight_around_c3"),
        (Params(91.8e6, 0.034288, 23.0e9), 0.30, "wide_around_c3"),
        (Params(80e6, 0.04, 20e9), 0.30, "lower_sub"),
        (Params(110e6, 0.025, 25e9), 0.30, "higher_sub"),
        (Params(91.8e6, 0.04, 25e9), 0.20, "higher_nf"),
        (Params(91.8e6, 0.03, 30e9), 0.20, "lower_nf"),
    ]

    all_logs = []
    best_overall = None
    best_overall_L = 1e9
    for init, simplex, label in starts:
        print(f"\n--- Run: {label}, simplex={simplex} ---")
        bp, mc, mh, rc, rh, log, result = run_nm(init, simplex, 200, label)
        L = result.fun
        all_logs.append({"label": label, "init": (init.submission_depth_y, init.normalizer_fee, init.normalizer_depth_y), "log": log, "result": L, "params": (bp.submission_depth_y, bp.normalizer_fee, bp.normalizer_depth_y)})
        if L < best_overall_L:
            best_overall_L = L
            best_overall = (label, bp, mc, mh, rc, rh, result)

    # Report best across all runs
    print(f"\n\n=== Best of {len(starts)} restarts ===")
    label, bp, mc, mh, rc, rh, result = best_overall
    print(f"label: {label}")
    print(f"params: sub={bp.submission_depth_y:.4e}  nf={bp.normalizer_fee:.4e}  nd={bp.normalizer_depth_y:.4e}")
    print(f"calib   T1={mc['arb_5bp_share']:.4f} T2={mc['retail_5bp_share']:.4f} T3={mc['markout_bps']:.4f}")
    print(f"        r1={rc['T1_arb_5bp_share']:+.4%} r2={rc['T2_retail_5bp_share']:+.4%} r3={rc['T3_markout_bps']:+.4%}")
    print(f"holdout T1={mh['arb_5bp_share']:.4f} T2={mh['retail_5bp_share']:.4f} T3={mh['markout_bps']:.4f}")
    print(f"        r1={rh['T1_arb_5bp_share']:+.4%} r2={rh['T2_retail_5bp_share']:+.4%} r3={rh['T3_markout_bps']:+.4%}")
    print(f"loss={result.fun:.6f}")
    print(f"calib per-seed arb: {[round(r['arb_share'], 3) for r in mc['per_seed']]}")
    print(f"hold per-seed arb:  {[round(r['arb_share'], 3) for r in mh['per_seed']]}")
    mxc = max(abs(v) for v in rc.values())
    mxh = max(abs(v) for v in rh.values())
    print(f"max|rc|={mxc:.4f}  max|rh|={mxh:.4f}  DONE={mxc < 0.02 and mxh < 0.02}")

    out = {
        "cycle": CYCLE,
        "joint": True,
        "best_label": label,
        "best_params": {
            "submission_depth_y": bp.submission_depth_y,
            "normalizer_fee": bp.normalizer_fee,
            "normalizer_depth_y": bp.normalizer_depth_y,
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
        "all_runs": all_logs,
    }
    path = ART_DIR / f"cycle_{CYCLE:02d}.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"saved -> {path}")
