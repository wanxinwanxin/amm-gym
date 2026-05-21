"""Robust optimization using a larger seed batch for stability.

Per-seed arb_share is bimodal (0 or 1) when the normalizer fee is high enough
to suppress normalizer arb in low-volatility paths. With only 5 seeds the
mean is highly variable. Use 20 seeds during optimization, then evaluate on
the canonical (42..46) and (100..104) sets.
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
from scripts.calibration.optimize import _params_from_x, _x_from_params

ART_DIR = _ROOT / "calibration_artifacts"
ART_DIR.mkdir(exist_ok=True)

ROBUST_SEEDS = tuple(range(200, 220))  # 20 seeds disjoint from calib/holdout


def loss_for(metrics: dict) -> float:
    r = residuals(metrics, TARGETS)
    return float(r["T1_arb_5bp_share"] ** 2 + r["T2_retail_5bp_share"] ** 2 + r["T3_markout_bps"] ** 2)


def make_loss(seeds: tuple[int, ...], n_steps: int, log: list[dict]):
    def loss(x: np.ndarray) -> float:
        if not (math.log(1e-5) <= x[1] <= math.log(0.5)):
            return 1e3
        if not (math.log(1e6) <= x[0] <= math.log(1e12)):
            return 1e3
        if not (math.log(1e6) <= x[2] <= math.log(1e12)):
            return 1e3
        p = _params_from_x(x)
        t0 = time.time()
        m = evaluate_params(p, seeds, n_steps)
        L = loss_for(m)
        entry = {
            "sub_depth_y": p.submission_depth_y,
            "norm_fee": p.normalizer_fee,
            "norm_depth_y": p.normalizer_depth_y,
            "arb_5bp_share": m["arb_5bp_share"],
            "retail_5bp_share": m["retail_5bp_share"],
            "markout_bps": m["markout_bps"],
            "loss": L,
            "elapsed_s": time.time() - t0,
        }
        log.append(entry)
        print(f"  L={L:7.4f}  T1={m['arb_5bp_share']:.3f} T2={m['retail_5bp_share']:.3f} T3={m['markout_bps']:7.3f}  | p=({p.submission_depth_y:.2e}, {p.normalizer_fee:.3e}, {p.normalizer_depth_y:.2e})", flush=True)
        return L

    return loss


def run_robust_nm(
    cycle: int,
    init: Params,
    *,
    seeds: tuple[int, ...] = ROBUST_SEEDS,
    n_steps: int = N_STEPS,
    max_iter: int = 80,
    simplex_spread: float = 0.4,
) -> dict:
    print(f"\n========== Robust Cycle {cycle} (seeds={len(seeds)}) ==========")
    print(f"start: sub={init.submission_depth_y:.3e}  nf={init.normalizer_fee:.3e}  nd={init.normalizer_depth_y:.3e}")
    log: list[dict] = []
    x0 = _x_from_params(init)
    loss_fn = make_loss(seeds, n_steps, log)
    initial_simplex = np.vstack([
        x0,
        x0 + np.array([simplex_spread, 0.0, 0.0]),
        x0 + np.array([0.0, simplex_spread, 0.0]),
        x0 + np.array([0.0, 0.0, simplex_spread]),
    ])
    t0 = time.time()
    result = minimize(
        loss_fn,
        x0,
        method="Nelder-Mead",
        options={
            "maxiter": max_iter,
            "xatol": 1e-3,
            "fatol": 1e-4,
            "initial_simplex": initial_simplex,
            "adaptive": True,
        },
    )
    elapsed = time.time() - t0
    best_x = result.x
    best_params = _params_from_x(best_x)

    print(f"\n-- Best params from robust optimization --")
    print(f"  sub={best_params.submission_depth_y:.4e}  nf={best_params.normalizer_fee:.4e}  nd={best_params.normalizer_depth_y:.4e}")

    # Evaluate on canonical calibration + holdout sets
    print("\n-- Calibration set (42..46) --")
    calib_m = evaluate_params(best_params, CALIB_SEEDS, n_steps)
    calib_r = residuals(calib_m, TARGETS)
    for k, v in calib_r.items():
        print(f"  {k}: residual {v:+.3%}")

    print("\n-- Held-out set (100..104) --")
    hold_m = evaluate_params(best_params, HOLDOUT_SEEDS, n_steps)
    hold_r = residuals(hold_m, TARGETS)
    for k, v in hold_r.items():
        print(f"  {k}: residual {v:+.3%}")

    print("\n-- Robust set (200..219) --")
    robust_m = evaluate_params(best_params, seeds, n_steps)
    robust_r = residuals(robust_m, TARGETS)
    for k, v in robust_r.items():
        print(f"  {k}: residual {v:+.3%}")

    out = {
        "cycle": cycle,
        "best_params": best_params,
        "calib_metrics": calib_m,
        "calib_residuals": calib_r,
        "holdout_metrics": hold_m,
        "holdout_residuals": hold_r,
        "robust_metrics": robust_m,
        "robust_residuals": robust_r,
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
    return out


def save_cycle(cycle: int, out: dict) -> Path:
    path = ART_DIR / f"cycle_{cycle:02d}.json"
    serial = {}
    for k, v in out.items():
        if k == "best_params":
            p = v
            serial[k] = {
                "submission_depth_y": p.submission_depth_y,
                "normalizer_fee": p.normalizer_fee,
                "normalizer_depth_y": p.normalizer_depth_y,
            }
        elif k.endswith("_metrics"):
            serial[k] = {kk: vv for kk, vv in v.items() if kk != "per_seed"}
            serial[k]["per_seed_arb_share"] = [r["arb_share"] for r in v["per_seed"]]
            serial[k]["per_seed_retail_share"] = [r["retail_share"] for r in v["per_seed"]]
            serial[k]["per_seed_markout_bps"] = [r["markout_bps"] for r in v["per_seed"]]
        elif k == "scipy_result" and "final_residuals" in v:
            serial[k] = v
        else:
            serial[k] = v
    # For backward compat with plot script, expose final_metrics+final_residuals as the calib values
    serial["final_metrics"] = serial.get("calib_metrics", serial.get("final_metrics", {}))
    serial["final_residuals"] = serial.get("calib_residuals", serial.get("final_residuals", {}))
    path.write_text(json.dumps(serial, indent=2, default=float))
    return path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--init-from", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--spread", type=float, default=0.4)
    parser.add_argument("--n-seeds", type=int, default=20)
    args = parser.parse_args()

    if args.init_from is not None:
        data = json.loads((ART_DIR / f"cycle_{args.init_from:02d}.json").read_text())
        bp = data["best_params"]
        init = Params(bp["submission_depth_y"], bp["normalizer_fee"], bp["normalizer_depth_y"])
    else:
        init = Params(1.5564e7, 4.8119e-2, 7.5660e10)

    seeds = tuple(range(200, 200 + args.n_seeds))
    out = run_robust_nm(args.cycle, init, seeds=seeds, max_iter=args.max_iter, simplex_spread=args.spread)
    path = save_cycle(args.cycle, out)
    print(f"\nSaved {path}")
