"""Nelder-Mead optimizer for the 3-param simple-AMM calibration.

Parameters live in log-space for stable steps:
  x[0] = log(submission_depth_y)
  x[1] = log(normalizer_fee)
  x[2] = log(normalizer_depth_y)

Loss = sum of squared *relative* residuals across T1, T2, T3.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Callable

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


def make_loss(seeds: tuple[int, ...], n_steps: int, log: list[dict]) -> Callable[[np.ndarray], float]:
    def loss(x: np.ndarray) -> float:
        p = _params_from_x(x)
        # Enforce bounds (normalizer_fee in [1e-5, 0.5]; depths in [1e6, 1e12])
        if not (math.log(1e-5) <= x[1] <= math.log(0.5)):
            return 1e3
        if not (math.log(1e6) <= x[0] <= math.log(1e12)):
            return 1e3
        if not (math.log(1e6) <= x[2] <= math.log(1e12)):
            return 1e3
        t0 = time.time()
        m = evaluate_params(p, seeds, n_steps)
        r = residuals(m, TARGETS)
        L = float(r["T1_arb_5bp_share"] ** 2 + r["T2_retail_5bp_share"] ** 2 + r["T3_markout_bps"] ** 2)
        entry = {
            "sub_depth_y": p.submission_depth_y,
            "norm_fee": p.normalizer_fee,
            "norm_depth_y": p.normalizer_depth_y,
            "arb_5bp_share": m["arb_5bp_share"],
            "retail_5bp_share": m["retail_5bp_share"],
            "markout_bps": m["markout_bps"],
            "r1": r["T1_arb_5bp_share"],
            "r2": r["T2_retail_5bp_share"],
            "r3": r["T3_markout_bps"],
            "loss": L,
            "elapsed_s": time.time() - t0,
        }
        log.append(entry)
        return L

    return loss


def run_nelder_mead(
    init_params: Params,
    seeds: tuple[int, ...] = CALIB_SEEDS,
    n_steps: int = N_STEPS,
    max_iter: int = 80,
    xatol: float = 1e-3,
    fatol: float = 1e-4,
) -> dict:
    log: list[dict] = []
    x0 = _x_from_params(init_params)
    loss_fn = make_loss(seeds, n_steps, log)
    # Initial simplex spread: ±0.3 in log space (~30%).
    initial_simplex = np.vstack([
        x0,
        x0 + np.array([0.4, 0.0, 0.0]),
        x0 + np.array([0.0, 0.4, 0.0]),
        x0 + np.array([0.0, 0.0, 0.4]),
    ])
    t0 = time.time()
    result = minimize(
        loss_fn,
        x0,
        method="Nelder-Mead",
        options={
            "maxiter": max_iter,
            "xatol": xatol,
            "fatol": fatol,
            "initial_simplex": initial_simplex,
            "adaptive": True,
        },
    )
    elapsed = time.time() - t0
    best_x = result.x
    best_params = _params_from_x(best_x)
    final_metrics = evaluate_params(best_params, seeds, n_steps)
    final_res = residuals(final_metrics, TARGETS)
    return {
        "best_params": best_params,
        "best_x": best_x.tolist(),
        "final_metrics": final_metrics,
        "final_residuals": final_res,
        "scipy_result": {
            "fun": float(result.fun),
            "nit": int(result.nit),
            "nfev": int(result.nfev),
            "success": bool(result.success),
            "message": str(result.message),
        },
        "log": log,
        "elapsed_s": elapsed,
    }


def report(opt_out: dict, label: str = "") -> None:
    p = opt_out["best_params"]
    m = opt_out["final_metrics"]
    r = opt_out["final_residuals"]
    print(f"--- {label} ---")
    print(f"  params: sub={p.submission_depth_y:.4e}  nf={p.normalizer_fee:.4e}  nd={p.normalizer_depth_y:.4e}")
    print(f"  T1 arb_5bp_share    = {m['arb_5bp_share']:.4f}  target {TARGETS['T1']:.4f}  residual {r['T1_arb_5bp_share']:+.3%}")
    print(f"  T2 retail_5bp_share = {m['retail_5bp_share']:.4f}  target {TARGETS['T2']:.4f}  residual {r['T2_retail_5bp_share']:+.3%}")
    print(f"  T3 markout_bps      = {m['markout_bps']:.4f}  target {TARGETS['T3']:.4f}  residual {r['T3_markout_bps']:+.3%}")
    print(f"  loss = {opt_out['scipy_result']['fun']:.6f}  nfev={opt_out['scipy_result']['nfev']}  elapsed={opt_out['elapsed_s']:.1f}s")


if __name__ == "__main__":
    init = Params(
        submission_depth_y=212_157_626,
        normalizer_fee=0.001,
        normalizer_depth_y=17_000_000_000,
    )
    out = run_nelder_mead(init, max_iter=80)
    report(out, label="Cycle 1 — Nelder-Mead from notebook baseline")
