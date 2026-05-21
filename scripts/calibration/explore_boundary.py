"""Explore the parameter neighborhood of Cycle 2 best to find a point where
the bimodal-region boundary lies between holdout seeds 102 and 103.

If we can find params where only 1 of (102, 103) hits arb_share=1.0,
holdout T1 should be near 0.337 (matching calibration set).
"""
from __future__ import annotations

import sys
import time
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


# Cycle 2 best: sub=102.4M, nf=486.7bp, nd=1.0e12
# Probe wider neighborhood to see how arb-share at each seed evolves.
def loss(rc, rh):
    return sum(v**2 for v in rc.values()) + sum(v**2 for v in rh.values())


grid = []
# Vary sub_depth and norm_fee around cycle 2 best
for sub_m in (60, 80, 100, 120, 150, 180):
    for nf_bps in (200, 300, 400, 500, 600):
        for nd_g in (50, 200, 1000):
            grid.append((sub_m, nf_bps, nd_g))

print(f"Probing {len(grid)} points near cycle 2 best for boundary effects on per-seed arb_share")
print(f"{'sub_M':>5} {'nf_bps':>6} {'nd_G':>6} | "
      f"{'cT1':>6} {'cT2':>6} {'cT3':>6} | {'hT1':>6} {'hT2':>6} {'hT3':>6} | "
      f"{'mxRc':>5} {'mxRh':>5} | {'L':>6} | {'1s_in_c':>7} {'1s_in_h':>7}")

t0 = time.time()
best = (1e9, None, None, None, None, None)
hits_done = []
for sub_m, nf_bps, nd_g in grid:
    p = Params(sub_m * 1e6, nf_bps / 1e4, nd_g * 1e9)
    mc = evaluate_params(p, CALIB_SEEDS, N_STEPS)
    mh = evaluate_params(p, HOLDOUT_SEEDS, N_STEPS)
    rc = residuals(mc, TARGETS)
    rh = residuals(mh, TARGETS)
    L = loss(rc, rh)
    mx_rc = max(abs(v) for v in rc.values())
    mx_rh = max(abs(v) for v in rh.values())
    ones_c = sum(1 for r in mc["per_seed"] if r["arb_share"] > 0.5)
    ones_h = sum(1 for r in mh["per_seed"] if r["arb_share"] > 0.5)
    flag = " DONE" if (mx_rc < 0.02 and mx_rh < 0.02) else ""
    if mx_rc < 0.02 and mx_rh < 0.02:
        hits_done.append((sub_m, nf_bps, nd_g, p, mc, mh, rc, rh))
    print(f"{sub_m:5d} {nf_bps:6d} {nd_g:6d} | "
          f"{mc['arb_5bp_share']:6.3f} {mc['retail_5bp_share']:6.3f} {mc['markout_bps']:6.2f} | "
          f"{mh['arb_5bp_share']:6.3f} {mh['retail_5bp_share']:6.3f} {mh['markout_bps']:6.2f} | "
          f"{mx_rc:5.2f} {mx_rh:5.2f} | {L:6.3f} | {ones_c:7d} {ones_h:7d}{flag}", flush=True)
    if L < best[0]:
        best = (L, p, mc, mh, rc, rh)

print(f"\nelapsed: {time.time()-t0:.1f}s")
print(f"\nBest joint loss point:")
L, p, mc, mh, rc, rh = best
print(f"  params: sub={p.submission_depth_y/1e6:.1f}M  nf={p.normalizer_fee*1e4:.1f}bp  nd={p.normalizer_depth_y/1e9:.1f}G")
print(f"  calib:   T1={mc['arb_5bp_share']:.4f}  T2={mc['retail_5bp_share']:.4f}  T3={mc['markout_bps']:.4f}")
print(f"           per-seed arb: {[round(r['arb_share'],3) for r in mc['per_seed']]}")
print(f"           residuals: r1={rc['T1_arb_5bp_share']:+.3%}  r2={rc['T2_retail_5bp_share']:+.3%}  r3={rc['T3_markout_bps']:+.3%}")
print(f"  holdout: T1={mh['arb_5bp_share']:.4f}  T2={mh['retail_5bp_share']:.4f}  T3={mh['markout_bps']:.4f}")
print(f"           per-seed arb: {[round(r['arb_share'],3) for r in mh['per_seed']]}")
print(f"           residuals: r1={rh['T1_arb_5bp_share']:+.3%}  r2={rh['T2_retail_5bp_share']:+.3%}  r3={rh['T3_markout_bps']:+.3%}")
print(f"  L={L:.4f}")
print()
if hits_done:
    print(f"DONE candidates: {len(hits_done)}")
    for sub_m, nf_bps, nd_g, p, mc, mh, rc, rh in hits_done:
        print(f"  ({sub_m}M, {nf_bps}bp, {nd_g}G)")
