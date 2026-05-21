"""Cycle 6 — Coordinate sweep around the joint basin.

Examine how max|residual| varies along each parameter direction from
(91.8M, 343bp, 23G), to see if there's a better-balanced point nearby.
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


BASIN_SUB = 91.8e6
BASIN_NF = 0.03429
BASIN_ND = 23.0e9


def eval_point(sub, nf, nd):
    p = Params(sub, nf, nd)
    mc = evaluate_params(p, CALIB_SEEDS, N_STEPS)
    mh = evaluate_params(p, HOLDOUT_SEEDS, N_STEPS)
    rc = residuals(mc, TARGETS)
    rh = residuals(mh, TARGETS)
    L = sum(v**2 for v in rc.values()) + sum(v**2 for v in rh.values())
    mxc = max(abs(v) for v in rc.values())
    mxh = max(abs(v) for v in rh.values())
    overall = max(mxc, mxh)
    return p, mc, mh, rc, rh, L, mxc, mxh, overall


if __name__ == "__main__":
    print("Cycle 6 — coordinate sweep around joint basin\n")
    print(f"basin: sub={BASIN_SUB/1e6:.1f}M  nf={BASIN_NF*1e4:.1f}bp  nd={BASIN_ND/1e9:.1f}G")
    print(f"{'param':>30} | {'cT1':>5} {'cT2':>5} {'cT3':>6} | {'hT1':>5} {'hT2':>5} {'hT3':>6} | "
          f"{'r1c':>7} {'r2c':>7} {'r3c':>7} | {'r1h':>7} {'r2h':>7} {'r3h':>7} | "
          f"{'mxc':>5} {'mxh':>5} | {'L':>6}")

    t0 = time.time()
    points = []
    # Sweep each coordinate
    for name, vals in [
        ("sub_M", [60, 70, 80, 85, 88, 91.8, 95, 100, 110, 120, 140, 170]),
        ("nf_bps", [200, 250, 280, 300, 320, 343, 360, 400, 450, 500, 600, 800]),
        ("nd_G", [5, 10, 15, 20, 23, 27, 35, 50, 75, 100, 150, 250]),
    ]:
        print(f"\n--- sweep over {name} ---")
        for v in vals:
            if name == "sub_M":
                sub, nf, nd = v * 1e6, BASIN_NF, BASIN_ND
                label = f"sub={v:6.1f}M"
            elif name == "nf_bps":
                sub, nf, nd = BASIN_SUB, v / 1e4, BASIN_ND
                label = f"nf={v:6.1f}bp"
            else:
                sub, nf, nd = BASIN_SUB, BASIN_NF, v * 1e9
                label = f"nd={v:6.1f}G"
            p, mc, mh, rc, rh, L, mxc, mxh, overall = eval_point(sub, nf, nd)
            print(f"{label:>30} | "
                  f"{mc['arb_5bp_share']:5.3f} {mc['retail_5bp_share']:5.3f} {mc['markout_bps']:6.2f} | "
                  f"{mh['arb_5bp_share']:5.3f} {mh['retail_5bp_share']:5.3f} {mh['markout_bps']:6.2f} | "
                  f"{rc['T1_arb_5bp_share']:+7.3f} {rc['T2_retail_5bp_share']:+7.3f} {rc['T3_markout_bps']:+7.3f} | "
                  f"{rh['T1_arb_5bp_share']:+7.3f} {rh['T2_retail_5bp_share']:+7.3f} {rh['T3_markout_bps']:+7.3f} | "
                  f"{mxc:5.3f} {mxh:5.3f} | {L:6.3f}",
                  flush=True)
            points.append({"sub": sub, "nf": nf, "nd": nd, "rc": dict(rc), "rh": dict(rh), "L": L})

    print(f"\nElapsed: {time.time()-t0:.1f}s")
    pts_sorted = sorted(points, key=lambda x: x["L"])
    print("\nTop 5 by L:")
    for p in pts_sorted[:5]:
        print(f"  sub={p['sub']/1e6:.1f}M nf={p['nf']*1e4:.1f}bp nd={p['nd']/1e9:.1f}G  L={p['L']:.4f}")
