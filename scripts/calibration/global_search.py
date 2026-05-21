"""Global search via differential evolution to confirm the Pareto wall.

Uses 3 seeds (42, 43, 44) for speed during search, then re-evals the
best point on the full calibration + holdout seed sets.
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
    Params,
    TARGETS,
    evaluate_params,
    residuals,
)

ART_DIR = _ROOT / "calibration_artifacts"

# Log-space bounds.
BOUNDS = [
    (math.log(1e6), math.log(1e12)),    # sub_depth_y
    (math.log(1e-5), math.log(0.5)),    # norm_fee
    (math.log(1e7), math.log(2e12)),    # norm_depth_y
]


def make_loss(seeds, n_steps, log):
    iter_count = [0]

    def loss(x):
        p = Params(
            submission_depth_y=float(math.exp(x[0])),
            normalizer_fee=float(math.exp(x[1])),
            normalizer_depth_y=float(math.exp(x[2])),
        )
        m = evaluate_params(p, seeds, n_steps)
        r = residuals(m, TARGETS)
        L = float(r["T1_arb_5bp_share"] ** 2 + r["T2_retail_5bp_share"] ** 2 + r["T3_markout_bps"] ** 2)
        log.append({
            "sub": p.submission_depth_y, "nf": p.normalizer_fee, "nd": p.normalizer_depth_y,
            "T1": m["arb_5bp_share"], "T2": m["retail_5bp_share"], "T3": m["markout_bps"],
            "loss": L,
        })
        iter_count[0] += 1
        if iter_count[0] % 10 == 0:
            best = min(log, key=lambda d: d["loss"])
            print(f"  iter {iter_count[0]:4d}  best_L={best['loss']:.5f}  current_L={L:.5f}", flush=True)
        return L

    return loss


def main():
    seeds = (42, 43, 44)
    n_steps = 3000

    log = []
    loss_fn = make_loss(seeds, n_steps, log)
    print(f"Running differential evolution: seeds={seeds}, n_steps={n_steps}")
    t0 = time.time()
    result = differential_evolution(
        loss_fn,
        bounds=BOUNDS,
        maxiter=20,
        popsize=8,
        tol=1e-3,
        polish=False,
        seed=42,
        workers=1,
    )
    elapsed = time.time() - t0
    print(f"\nDE done in {elapsed:.1f}s, {result.nit} iter, {result.nfev} evals")

    best_x = result.x
    best_p = Params(math.exp(best_x[0]), math.exp(best_x[1]), math.exp(best_x[2]))
    print(f"\nBest params: sub={best_p.submission_depth_y:.4e}  nf={best_p.normalizer_fee:.4e}  nd={best_p.normalizer_depth_y:.4e}")

    # Now eval on canonical sets
    calib = evaluate_params(best_p, CALIB_SEEDS, 5000)
    cr = residuals(calib, TARGETS)
    print(f"-- Calibration set --")
    print(f"  T1 = {calib['arb_5bp_share']:.4f}  residual {cr['T1_arb_5bp_share']:+.3%}")
    print(f"  T2 = {calib['retail_5bp_share']:.4f}  residual {cr['T2_retail_5bp_share']:+.3%}")
    print(f"  T3 = {calib['markout_bps']:.4f}  residual {cr['T3_markout_bps']:+.3%}")

    hold = evaluate_params(best_p, HOLDOUT_SEEDS, 5000)
    hr = residuals(hold, TARGETS)
    print(f"-- Holdout set --")
    print(f"  T1 = {hold['arb_5bp_share']:.4f}  residual {hr['T1_arb_5bp_share']:+.3%}")
    print(f"  T2 = {hold['retail_5bp_share']:.4f}  residual {hr['T2_retail_5bp_share']:+.3%}")
    print(f"  T3 = {hold['markout_bps']:.4f}  residual {hr['T3_markout_bps']:+.3%}")

    # Save as cycle 6
    out = {
        "cycle": 6,
        "best_params": {
            "submission_depth_y": best_p.submission_depth_y,
            "normalizer_fee": best_p.normalizer_fee,
            "normalizer_depth_y": best_p.normalizer_depth_y,
        },
        "final_metrics": {
            "arb_5bp_share": calib["arb_5bp_share"],
            "retail_5bp_share": calib["retail_5bp_share"],
            "markout_bps": calib["markout_bps"],
            "per_seed_arb_share": [r["arb_share"] for r in calib["per_seed"]],
            "per_seed_retail_share": [r["retail_share"] for r in calib["per_seed"]],
            "per_seed_markout_bps": [r["markout_bps"] for r in calib["per_seed"]],
        },
        "final_residuals": cr,
        "holdout_metrics": {
            "arb_5bp_share": hold["arb_5bp_share"],
            "retail_5bp_share": hold["retail_5bp_share"],
            "markout_bps": hold["markout_bps"],
            "per_seed_arb_share": [r["arb_share"] for r in hold["per_seed"]],
            "per_seed_retail_share": [r["retail_share"] for r in hold["per_seed"]],
            "per_seed_markout_bps": [r["markout_bps"] for r in hold["per_seed"]],
        },
        "holdout_residuals": hr,
        "scipy_result": {
            "fun": float(result.fun),
            "nit": int(result.nit),
            "nfev": int(result.nfev),
            "success": bool(result.success),
            "message": "differential_evolution global search",
        },
        "elapsed_s": elapsed,
    }
    path = ART_DIR / "cycle_06.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
