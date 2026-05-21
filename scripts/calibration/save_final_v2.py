"""Save the final calibration result to calibration_final.json
and refresh plots/calibration_residuals.png.

Final params: (sub=91.83M, nf=343bp, nd=23G) — the joint-NM basin minimum.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def main():
    final_params = Params(
        submission_depth_y=9.183091e7,
        normalizer_fee=3.428869e-2,
        normalizer_depth_y=2.299507e10,
    )
    print("Final params:")
    print(f"  submission_depth_y = {final_params.submission_depth_y:.6e}")
    print(f"  normalizer_fee     = {final_params.normalizer_fee:.6e}")
    print(f"  normalizer_depth_y = {final_params.normalizer_depth_y:.6e}")

    mc = evaluate_params(final_params, CALIB_SEEDS, N_STEPS)
    mh = evaluate_params(final_params, HOLDOUT_SEEDS, N_STEPS)
    rc = residuals(mc, TARGETS)
    rh = residuals(mh, TARGETS)
    print("\nCalibration set (seeds 42-46):")
    print(f"  T1 arb_5bp_share    = {mc['arb_5bp_share']:.6f}  target {TARGETS['T1']:.6f}  residual {rc['T1_arb_5bp_share']:+.4%}")
    print(f"  T2 retail_5bp_share = {mc['retail_5bp_share']:.6f}  target {TARGETS['T2']:.6f}  residual {rc['T2_retail_5bp_share']:+.4%}")
    print(f"  T3 markout_bps      = {mc['markout_bps']:.6f}  target {TARGETS['T3']:.6f}  residual {rc['T3_markout_bps']:+.4%}")
    print("\nHoldout set (seeds 100-104):")
    print(f"  T1 arb_5bp_share    = {mh['arb_5bp_share']:.6f}  target {TARGETS['T1']:.6f}  residual {rh['T1_arb_5bp_share']:+.4%}")
    print(f"  T2 retail_5bp_share = {mh['retail_5bp_share']:.6f}  target {TARGETS['T2']:.6f}  residual {rh['T2_retail_5bp_share']:+.4%}")
    print(f"  T3 markout_bps      = {mh['markout_bps']:.6f}  target {TARGETS['T3']:.6f}  residual {rh['T3_markout_bps']:+.4%}")

    out = {
        "status": "PARTIAL",
        "note": "Joint-NM basin minimum found; all 6 residuals under 12% but above 2% DONE threshold. T2 floors at ~11% on holdout due to simulator's closed-form router producing maximum 5bp share ~0.72 in the unsaturated regime; pushing higher requires either bimodal saturation (which breaks T1) or model changes (variable per-order routing) beyond the 3-parameter scope.",
        "params": {
            "submission_depth_y": final_params.submission_depth_y,
            "normalizer_fee": final_params.normalizer_fee,
            "normalizer_depth_y": final_params.normalizer_depth_y,
        },
        "targets": dict(TARGETS),
        "seeds_calibration": list(CALIB_SEEDS),
        "seeds_holdout": list(HOLDOUT_SEEDS),
        "n_steps": N_STEPS,
        "calibration_metrics": {
            "arb_5bp_share": mc["arb_5bp_share"],
            "retail_5bp_share": mc["retail_5bp_share"],
            "markout_bps": mc["markout_bps"],
        },
        "calibration_residuals": dict(rc),
        "holdout_metrics": {
            "arb_5bp_share": mh["arb_5bp_share"],
            "retail_5bp_share": mh["retail_5bp_share"],
            "markout_bps": mh["markout_bps"],
        },
        "holdout_residuals": dict(rh),
        "calibration_per_seed": {
            "arb_share": [r["arb_share"] for r in mc["per_seed"]],
            "retail_share": [r["retail_share"] for r in mc["per_seed"]],
            "markout_bps": [r["markout_bps"] for r in mc["per_seed"]],
        },
        "holdout_per_seed": {
            "arb_share": [r["arb_share"] for r in mh["per_seed"]],
            "retail_share": [r["retail_share"] for r in mh["per_seed"]],
            "markout_bps": [r["markout_bps"] for r in mh["per_seed"]],
        },
        "joint_loss": sum(v**2 for v in rc.values()) + sum(v**2 for v in rh.values()),
        "max_calib_residual": float(max(abs(v) for v in rc.values())),
        "max_holdout_residual": float(max(abs(v) for v in rh.values())),
        "source_cycle": "cycle_08_final_tight_nm",
    }

    final_path = _ROOT / "calibration_final.json"
    final_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nsaved -> {final_path}")

    # Build plots
    plots_dir = _ROOT / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Plot 1: Loss evolution across cycles
    art = _ROOT / "calibration_artifacts"
    cycles_data = []
    for i in range(1, 9):
        f = art / f"cycle_{i:02d}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        # Pick the loss + max residuals
        L = None
        mxc = None
        mxh = None
        if "joint_loss" in d:
            L = d["joint_loss"]
            mxc = d["max_calib_residual"]
            mxh = d["max_holdout_residual"]
        elif "scipy" in d and "fun" in d["scipy"]:
            L = d["scipy"]["fun"]
            mxc = max(abs(v) for v in d["calib_residuals"].values()) if "calib_residuals" in d else None
            mxh = max(abs(v) for v in d["holdout_residuals"].values()) if "holdout_residuals" in d else None
        cycles_data.append((i, L, mxc, mxh))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Top: per-cycle joint loss
    if cycles_data:
        cs = [c[0] for c in cycles_data]
        Ls = [c[1] for c in cycles_data if c[1] is not None]
        axes[0].plot(cs[:len(Ls)], Ls, marker="o", label="joint loss")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("Cycle")
        axes[0].set_ylabel("Sum of squared relative residuals (6 total)")
        axes[0].set_title("Joint loss evolution across cycles")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

    # Bottom: residuals at final params
    labels = ["T1 calib", "T2 calib", "T3 calib", "T1 hold", "T2 hold", "T3 hold"]
    rvals = [rc["T1_arb_5bp_share"], rc["T2_retail_5bp_share"], rc["T3_markout_bps"],
             rh["T1_arb_5bp_share"], rh["T2_retail_5bp_share"], rh["T3_markout_bps"]]
    colors = ["steelblue" if abs(r) < 0.02 else ("orange" if abs(r) < 0.15 else "crimson") for r in rvals]
    bars = axes[1].bar(labels, [100*r for r in rvals], color=colors)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].axhline(2, color="green", linestyle="--", alpha=0.5, label="DONE threshold (2%)")
    axes[1].axhline(-2, color="green", linestyle="--", alpha=0.5)
    axes[1].axhline(15, color="red", linestyle="--", alpha=0.5, label="BLOCKED threshold (15%)")
    axes[1].axhline(-15, color="red", linestyle="--", alpha=0.5)
    axes[1].set_ylabel("Relative residual (%)")
    axes[1].set_title("Final residuals at (sub=91.83M, nf=343bp, nd=23.0G)")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()
    for bar, r in zip(bars, rvals):
        axes[1].text(bar.get_x() + bar.get_width()/2, 100*r + (1 if r > 0 else -2),
                      f"{100*r:+.1f}%", ha="center", va="bottom" if r > 0 else "top", fontsize=9)

    fig.tight_layout()
    plot_path = plots_dir / "calibration_residuals.png"
    fig.savefig(plot_path, dpi=130)
    print(f"saved -> {plot_path}")


if __name__ == "__main__":
    main()
