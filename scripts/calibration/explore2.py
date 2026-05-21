"""Probe a different region: medium-fee normalizer to see if balance is better."""

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

print("Probe normalizer near sub pool conditions (medium-fee variant):")
print(f"{'sub_M':>8} {'nf_bps':>8} {'nd_B':>8} | {'T1':>6} {'T2':>6} {'T3':>8} | {'loss':>10}")

t0 = time.time()
grid = []
# Pool depth where T2 can be reasonable
for sub in (200, 400, 800, 1500, 3000):  # sub in M
    for nf in (0.0005, 0.001, 0.002, 0.005, 0.01):  # 5 to 100 bps
        for nd in (1, 3, 10):  # nd in B
            grid.append((sub, nf, nd))

for sub_m, nf, nd_b in grid:
    p = Params(sub_m * 1e6, nf, nd_b * 1e9)
    m = evaluate_params(p, SEEDS, N_STEPS)
    L = loss_for(m)
    mark = " ***" if L < 0.5 else ""
    print(f"{sub_m:8d} {nf*1e4:8.1f} {nd_b:8d} | {m['arb_5bp_share']:6.3f} {m['retail_5bp_share']:6.3f} {m['markout_bps']:8.3f} | {L:10.4f}{mark}", flush=True)

print(f"\n{len(grid)} evals in {time.time()-t0:.1f}s")
