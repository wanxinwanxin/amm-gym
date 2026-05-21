"""Drive the calibration cycles end-to-end.

Saves per-cycle JSON snapshots into `calibration_artifacts/cycle_{n}.json`
so we don't lose state if a cycle stalls.
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
from scripts.calibration.optimize import (
    _params_from_x,
    _x_from_params,
    make_loss,
    report,
    run_nelder_mead,
)

ART_DIR = _ROOT / "calibration_artifacts"
ART_DIR.mkdir(exist_ok=True)


def save_cycle(cycle: int, payload: dict) -> Path:
    path = ART_DIR / f"cycle_{cycle:02d}.json"
    serializable = {}
    for k, v in payload.items():
        if k == "best_params":
            p = v
            serializable[k] = {
                "submission_depth_y": p.submission_depth_y,
                "normalizer_fee": p.normalizer_fee,
                "normalizer_depth_y": p.normalizer_depth_y,
            }
        elif k == "final_metrics":
            serializable[k] = {kk: vv for kk, vv in v.items() if kk not in ("per_seed",)}
            serializable[k]["per_seed_arb_share"] = [r["arb_share"] for r in v["per_seed"]]
            serializable[k]["per_seed_retail_share"] = [r["retail_share"] for r in v["per_seed"]]
            serializable[k]["per_seed_markout_bps"] = [r["markout_bps"] for r in v["per_seed"]]
        else:
            serializable[k] = v
    path.write_text(json.dumps(serializable, indent=2, default=float))
    return path


def all_residuals_within(metrics: dict, threshold: float = 0.02) -> tuple[bool, dict]:
    r = residuals(metrics, TARGETS)
    ok = max(abs(r[k]) for k in r) < threshold
    return ok, r


def run_one_nm_cycle(
    cycle: int,
    init_params: Params,
    seeds: tuple[int, ...] = CALIB_SEEDS,
    n_steps: int = N_STEPS,
    max_iter: int = 80,
    simplex_spread: float = 0.4,
) -> dict:
    print(f"\n========== Cycle {cycle} ==========")
    print(f"start params: sub={init_params.submission_depth_y:.4e}  "
          f"nf={init_params.normalizer_fee:.4e}  nd={init_params.normalizer_depth_y:.4e}")
    log: list[dict] = []
    x0 = _x_from_params(init_params)
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
    best_params = _params_from_x(result.x)
    final_metrics = evaluate_params(best_params, seeds, n_steps)
    final_res = residuals(final_metrics, TARGETS)
    out = {
        "cycle": cycle,
        "best_params": best_params,
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
        "init_params": {
            "submission_depth_y": init_params.submission_depth_y,
            "normalizer_fee": init_params.normalizer_fee,
            "normalizer_depth_y": init_params.normalizer_depth_y,
        },
    }
    report(out, label=f"Cycle {cycle}")
    save_cycle(cycle, out)
    return out


def eval_holdout(p: Params) -> dict:
    """Evaluate a parameter set on the held-out seeds."""
    m = evaluate_params(p, HOLDOUT_SEEDS, N_STEPS)
    r = residuals(m, TARGETS)
    return {"metrics": m, "residuals": r}


if __name__ == "__main__":
    # Default driver: starts from the prior cycle's params if cycle file exists.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--spread", type=float, default=0.4)
    parser.add_argument("--init-from", type=int, default=None,
                        help="Cycle JSON to seed from (default: cycle-1)")
    parser.add_argument("--n-steps", type=int, default=N_STEPS)
    args = parser.parse_args()

    init = None
    src_cycle = args.init_from if args.init_from is not None else (args.cycle - 1)
    src_path = ART_DIR / f"cycle_{src_cycle:02d}.json"
    if src_path.exists():
        data = json.loads(src_path.read_text())
        bp = data["best_params"]
        init = Params(bp["submission_depth_y"], bp["normalizer_fee"], bp["normalizer_depth_y"])
        print(f"seeded from {src_path}")
    else:
        init = Params(212_157_626, 0.001, 17e9)
        print("seeded from notebook baseline")

    out = run_one_nm_cycle(
        cycle=args.cycle,
        init_params=init,
        n_steps=args.n_steps,
        max_iter=args.max_iter,
        simplex_spread=args.spread,
    )
    # If passing, evaluate held-out
    ok, _ = all_residuals_within(out["final_metrics"])
    if ok:
        print("\n--- Calibration passes! Evaluating held-out seeds ---")
        hold = eval_holdout(out["best_params"])
        out["holdout"] = hold
        save_cycle(args.cycle, out)
        print(f"  T1 holdout = {hold['metrics']['arb_5bp_share']:.4f}  residual {hold['residuals']['T1_arb_5bp_share']:+.3%}")
        print(f"  T2 holdout = {hold['metrics']['retail_5bp_share']:.4f}  residual {hold['residuals']['T2_retail_5bp_share']:+.3%}")
        print(f"  T3 holdout = {hold['metrics']['markout_bps']:.4f}  residual {hold['residuals']['T3_markout_bps']:+.3%}")
