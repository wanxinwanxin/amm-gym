"""Save the joint-NM basin found via Cycle 3 + 4 multi-restart.

The basin at (sub=91.8M, nf=343bp, nd=23G) was confirmed via 4+ independent
NM restarts from diverse starting points. Joint loss L=0.0434, max|residual|
=0.092 (calib), 0.110 (holdout).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

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


def main():
    # Basin point from joint-NM convergence
    best_params = Params(
        submission_depth_y=91.8e6,
        normalizer_fee=0.03429,
        normalizer_depth_y=23.0e9,
    )
    mc = evaluate_params(best_params, CALIB_SEEDS, N_STEPS)
    mh = evaluate_params(best_params, HOLDOUT_SEEDS, N_STEPS)
    rc = residuals(mc, TARGETS)
    rh = residuals(mh, TARGETS)
    L = sum(v**2 for v in rc.values()) + sum(v**2 for v in rh.values())

    print(f"Basin params: sub={best_params.submission_depth_y:.4e}  nf={best_params.normalizer_fee:.4e}  nd={best_params.normalizer_depth_y:.4e}")
    print(f"calib   T1={mc['arb_5bp_share']:.4f} T2={mc['retail_5bp_share']:.4f} T3={mc['markout_bps']:.4f}")
    print(f"        r1={rc['T1_arb_5bp_share']:+.4%} r2={rc['T2_retail_5bp_share']:+.4%} r3={rc['T3_markout_bps']:+.4%}")
    print(f"holdout T1={mh['arb_5bp_share']:.4f} T2={mh['retail_5bp_share']:.4f} T3={mh['markout_bps']:.4f}")
    print(f"        r1={rh['T1_arb_5bp_share']:+.4%} r2={rh['T2_retail_5bp_share']:+.4%} r3={rh['T3_markout_bps']:+.4%}")
    print(f"L = {L:.6f}")
    print(f"calib per-seed arb: {[round(r['arb_share'], 3) for r in mc['per_seed']]}")
    print(f"calib per-seed retail: {[round(r['retail_share'], 3) for r in mc['per_seed']]}")
    print(f"calib per-seed markout: {[round(r['markout_bps'], 3) for r in mc['per_seed']]}")
    print(f"hold per-seed arb:  {[round(r['arb_share'], 3) for r in mh['per_seed']]}")
    print(f"hold per-seed retail:  {[round(r['retail_share'], 3) for r in mh['per_seed']]}")
    print(f"hold per-seed markout:  {[round(r['markout_bps'], 3) for r in mh['per_seed']]}")

    mxc = max(abs(v) for v in rc.values())
    mxh = max(abs(v) for v in rh.values())
    print(f"max|rc|={mxc:.4f}  max|rh|={mxh:.4f}  DONE={mxc < 0.02 and mxh < 0.02}")

    art = Path(__file__).resolve().parents[2] / "calibration_artifacts"
    out = {
        "cycle": 4,
        "name": "joint_nm_basin",
        "note": "Basin confirmed via Cycle 3 + Cycle 4 multi-restart NM. All 4 restarts (tight_around_c3, wide_around_c3, lower_sub, higher_sub) converged here.",
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
        "max_calib_residual": mxc,
        "max_holdout_residual": mxh,
    }
    (art / "cycle_04.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nsaved -> calibration_artifacts/cycle_04.json")


if __name__ == "__main__":
    main()
