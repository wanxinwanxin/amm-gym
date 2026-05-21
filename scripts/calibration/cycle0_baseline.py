"""Cycle 0 baseline against the revised T3 = -1.05 bps target.

Evaluates a handful of promising starting points from the prior-cycle log
on the canonical calibration seeds, plus a small fresh probe of the
sub_depth ~ 60-150M / norm_fee ~ 30-300 bps neighborhood that the prior
log identified as T2-feasible. Prints residuals against the new targets.
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
    N_STEPS,
    Params,
    TARGETS,
    evaluate_params,
    residuals,
)


def loss_for(m):
    r = residuals(m, TARGETS)
    return float(r["T1_arb_5bp_share"] ** 2 + r["T2_retail_5bp_share"] ** 2 + r["T3_markout_bps"] ** 2)


# Picks: best prior-log points by new-loss + a few hand-picked from the
# T2-feasible neighborhood the context hints at.
candidates = [
    # (sub_M, nf_bps, nd_G, label)
    (79.2, 285.6, 23.482, "best_prior_new_loss"),
    (58.2, 405.2, 24.772, "T1=0.327_T2=0.687"),
    (114.8, 187.7, 16.314, "near_120M_low_fee"),
    (121.3, 139.6, 27.912, "exactly_120M"),
    (86.9, 156.0, 31.192, "T2=0.62_T3=-1.5"),
    (100.0, 200.0, 25.0, "fresh_probe_a"),
    (140.0, 150.0, 20.0, "fresh_probe_b"),
    (90.0, 250.0, 30.0, "fresh_probe_c"),
    (75.0, 220.0, 20.0, "fresh_probe_d"),
    (60.0, 350.0, 25.0, "fresh_probe_e"),
]


print(f"Cycle 0 — evaluating {len(candidates)} candidates against revised targets")
print(f"  Targets: T1={TARGETS['T1']:.4f}, T2={TARGETS['T2']:.4f}, T3={TARGETS['T3']:.4f} bps")
print()
print(f"{'label':>26} | {'sub_M':>7} {'nf_bps':>7} {'nd_G':>6} | "
      f"{'T1':>6} {'T2':>6} {'T3':>8} | {'r1':>7} {'r2':>7} {'r3':>7} | {'L':>9}")

t0 = time.time()
results = []
for sub_m, nf_bps, nd_g, label in candidates:
    p = Params(sub_m * 1e6, nf_bps / 1e4, nd_g * 1e9)
    m = evaluate_params(p, CALIB_SEEDS, N_STEPS)
    r = residuals(m, TARGETS)
    L = loss_for(m)
    print(
        f"{label:>26} | "
        f"{sub_m:7.1f} {nf_bps:7.1f} {nd_g:6.2f} | "
        f"{m['arb_5bp_share']:6.3f} {m['retail_5bp_share']:6.3f} {m['markout_bps']:8.3f} | "
        f"{r['T1_arb_5bp_share']:+7.3f} {r['T2_retail_5bp_share']:+7.3f} {r['T3_markout_bps']:+7.3f} | "
        f"{L:9.4f}",
        flush=True,
    )
    results.append((label, p, m, r, L))

print(f"\nElapsed: {time.time() - t0:.1f}s")
results.sort(key=lambda x: x[4])
print(f"\nBest by new loss: {results[0][0]} -> L={results[0][4]:.4f}")
p = results[0][1]
print(f"  params: sub={p.submission_depth_y:.4e}  nf={p.normalizer_fee:.4e}  nd={p.normalizer_depth_y:.4e}")
