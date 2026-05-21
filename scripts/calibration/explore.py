"""Multi-start probe: evaluate a small grid of diverse starting points,
report metrics & loss to find better basins for NM.
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
    Params,
    TARGETS,
    evaluate_params,
    residuals,
)


def loss_for(m: dict) -> float:
    r = residuals(m, TARGETS)
    return float(r["T1_arb_5bp_share"]**2 + r["T2_retail_5bp_share"]**2 + r["T3_markout_bps"]**2)


SEEDS_FAST = (42, 43, 44)
N_STEPS_FAST = 3000


grid = []
# Probe combinations of (sub_depth_y, normalizer_fee, normalizer_depth_y)
for sub in (5e7, 1e8, 3e8, 1e9, 3e9, 1e10, 3e10):
    for nf in (0.001, 0.003, 0.005, 0.01, 0.02, 0.05):
        for nd in (1e9, 5e9, 2e10, 1e11):
            grid.append((sub, nf, nd))

print(f"Evaluating {len(grid)} grid points on seeds {SEEDS_FAST} with n_steps={N_STEPS_FAST}...")
print(f"{'sub_M':>10} {'nf_bps':>8} {'nd_B':>8} | {'T1':>6} {'T2':>6} {'T3':>8} | {'loss':>10}")

results = []
t0 = time.time()
for i, (sub, nf, nd) in enumerate(grid):
    p = Params(sub, nf, nd)
    m = evaluate_params(p, SEEDS_FAST, N_STEPS_FAST)
    L = loss_for(m)
    results.append((sub, nf, nd, m["arb_5bp_share"], m["retail_5bp_share"], m["markout_bps"], L))
    print(f"{sub/1e6:10.0f} {nf*1e4:8.1f} {nd/1e9:8.2f} | {m['arb_5bp_share']:6.3f} {m['retail_5bp_share']:6.3f} {m['markout_bps']:8.3f} | {L:10.4f}")

elapsed = time.time() - t0
print(f"\n{len(grid)} evals in {elapsed:.1f}s  ({elapsed/len(grid):.2f}s/eval)")

results.sort(key=lambda x: x[6])
print("\nTop 10 by loss:")
print(f"{'sub_M':>10} {'nf_bps':>8} {'nd_B':>8} | {'T1':>6} {'T2':>6} {'T3':>8} | {'loss':>10}")
for r in results[:10]:
    print(f"{r[0]/1e6:10.0f} {r[1]*1e4:8.1f} {r[2]/1e9:8.2f} | {r[3]:6.3f} {r[4]:6.3f} {r[5]:8.3f} | {r[6]:10.4f}")
