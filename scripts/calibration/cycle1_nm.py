"""Cycle 1 — Nelder-Mead from the best new-loss starting point.

Start: (sub=79.2M, nf=285.6bp, nd=23.5G) — best of the prior-cycle 710-point
log when rescored against T3=-1.05.

Calibration seeds, 5000 steps, log-space search, sum-sq-rel-residual loss.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

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
from scripts.calibration.run_cycles import (
    ART_DIR,
    eval_holdout,
    run_one_nm_cycle,
)

CYCLE = 1

if __name__ == "__main__":
    # Best new-loss starting point from re-scoring the prior log.
    init = Params(
        submission_depth_y=7.92e7,
        normalizer_fee=0.02856,
        normalizer_depth_y=2.3482e10,
    )
    out = run_one_nm_cycle(
        cycle=CYCLE,
        init_params=init,
        max_iter=120,
        simplex_spread=0.4,
    )
    final_res = out["final_residuals"]
    max_r = max(abs(v) for v in final_res.values())
    print(f"\nmax|residual| (calibration) = {max_r:.4f}")
    if max_r < 0.02:
        print("--- Calibration passes! Evaluating held-out ---")
        hold = eval_holdout(out["best_params"])
        print(f"  T1 holdout residual = {hold['residuals']['T1_arb_5bp_share']:+.3%}")
        print(f"  T2 holdout residual = {hold['residuals']['T2_retail_5bp_share']:+.3%}")
        print(f"  T3 holdout residual = {hold['residuals']['T3_markout_bps']:+.3%}")
