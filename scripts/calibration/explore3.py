"""Explore the deep-sub / balanced area more carefully."""

from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.calibration.eval_metrics import (
    Params,
    TARGETS,
    evaluate_params,
    residuals,
)


def loss_for(m):
    r = residuals(m, TARGETS)
    return float(r["T1_arb_5bp_share"] ** 2 + r["T2_retail_5bp_share"] ** 2 + r["T3_markout_bps"] ** 2)


SEEDS = (42, 43, 44, 45, 46)
N_STEPS = 5000


print("Probe T2-friendly (deeper sub) area with very-high norm fee:")
print(f"{'sub_M':>8} {'nf_bps':>8} {'nd_B':>8} | {'T1':>6} {'T2':>6} {'T3':>8} | {'loss':>10}")

t0 = time.time()
grid = []
# Try deeper sub pools to push T2 up, with high normalizer fees to keep T3 positive.
for sub_m in (50, 100, 200, 400, 800):
    for nf_pct in (0.05, 0.10, 0.20, 0.40):  # 5% to 40%
        for nd_b in (10, 50, 200, 500):
            grid.append((sub_m, nf_pct, nd_b))

for sub_m, nf, nd_b in grid:
    p = Params(sub_m * 1e6, nf, nd_b * 1e9)
    m = evaluate_params(p, SEEDS, N_STEPS)
    L = loss_for(m)
    mark = " ***" if L < 0.3 else (" .." if L < 0.5 else "")
    print(f"{sub_m:8d} {nf*1e4:8.1f} {nd_b:8d} | {m['arb_5bp_share']:6.3f} {m['retail_5bp_share']:6.3f} {m['markout_bps']:8.3f} | {L:10.4f}{mark}", flush=True)

print(f"\n{len(grid)} evals in {time.time()-t0:.1f}s")
