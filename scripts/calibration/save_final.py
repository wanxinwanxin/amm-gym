"""Compose the final calibration_final.json and residuals plot.

Picks the cycle with the lowest *calibration-seed* loss, evaluates it on
both calibration and held-out seed sets at 5000 steps each, then writes
the canonical calibration_final.json.
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

ART_DIR = _ROOT / "calibration_artifacts"
FINAL_PATH = _ROOT / "calibration_final.json"


def load_cycles() -> list[dict]:
    cycles = []
    for p in sorted(ART_DIR.glob("cycle_*.json")):
        data = json.loads(p.read_text())
        data["_path"] = str(p)
        cycles.append(data)
    return cycles


def main() -> None:
    cycles = load_cycles()
    if not cycles:
        print("No cycles found.")
        return

    # Choose by minimum calibration loss = sum of squared residuals.
    def cycle_loss(c: dict) -> float:
        r = c.get("final_residuals") or c.get("calib_residuals", {})
        return sum(v ** 2 for v in r.values()) if r else float("inf")

    best = min(cycles, key=cycle_loss)
    print(f"best cycle: {best['_path']}  loss={cycle_loss(best):.5f}")

    bp = best["best_params"]
    params = Params(bp["submission_depth_y"], bp["normalizer_fee"], bp["normalizer_depth_y"])

    print("\nRe-evaluating on canonical sets...")
    calib_m = evaluate_params(params, CALIB_SEEDS, N_STEPS)
    calib_r = residuals(calib_m, TARGETS)
    hold_m = evaluate_params(params, HOLDOUT_SEEDS, N_STEPS)
    hold_r = residuals(hold_m, TARGETS)

    final = {
        "params": {
            "submission_depth_y": params.submission_depth_y,
            "normalizer_fee": params.normalizer_fee,
            "normalizer_depth_y": params.normalizer_depth_y,
        },
        "targets": TARGETS,
        "seeds_calibration": list(CALIB_SEEDS),
        "seeds_holdout": list(HOLDOUT_SEEDS),
        "n_steps": N_STEPS,
        "calibration_metrics": {
            "arb_5bp_share": calib_m["arb_5bp_share"],
            "retail_5bp_share": calib_m["retail_5bp_share"],
            "markout_bps": calib_m["markout_bps"],
        },
        "calibration_residuals": calib_r,
        "holdout_metrics": {
            "arb_5bp_share": hold_m["arb_5bp_share"],
            "retail_5bp_share": hold_m["retail_5bp_share"],
            "markout_bps": hold_m["markout_bps"],
        },
        "holdout_residuals": hold_r,
        "calibration_per_seed": {
            "arb_share": [r["arb_share"] for r in calib_m["per_seed"]],
            "retail_share": [r["retail_share"] for r in calib_m["per_seed"]],
            "markout_bps": [r["markout_bps"] for r in calib_m["per_seed"]],
        },
        "holdout_per_seed": {
            "arb_share": [r["arb_share"] for r in hold_m["per_seed"]],
            "retail_share": [r["retail_share"] for r in hold_m["per_seed"]],
            "markout_bps": [r["markout_bps"] for r in hold_m["per_seed"]],
        },
        "source_cycle": Path(best["_path"]).stem,
    }
    FINAL_PATH.write_text(json.dumps(final, indent=2, default=float))
    print(f"saved {FINAL_PATH}")
    print("\nCalibration residuals:")
    for k, v in calib_r.items():
        print(f"  {k}: {v:+.3%}")
    print("\nHoldout residuals:")
    for k, v in hold_r.items():
        print(f"  {k}: {v:+.3%}")


if __name__ == "__main__":
    main()
