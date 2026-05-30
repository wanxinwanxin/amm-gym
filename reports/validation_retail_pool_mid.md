# Retail-only held-out validation — impact-curve framework

Calibration step (not run here): fit V2 (φ, depth) of the normalizer pool to
the empirical non-5bp router-routed WETH/USDC impact curve (pool_mid_pre
referenced). 5bp pool metrics never enter the fit.

This script runs the simulator with:
- submission = 5bp on-chain frozen (fee=0.0005, virtual_depth_y=$212.16M)
- normalizer = **plan_b**: φ = **1.91 bps**, depth = **$49.5M**

and compares the three retail-only validation metrics to on-chain reality.

## Retail flow split

| Metric | Real | Sim (mean ± std) |
|--------|------|-------------------|
| Volume share @5bp | 49.49% | 59.71% ± 4.97pp |
| Fee share @5bp    | 17.15% | 79.35% ± 3.28pp |

## Retail markout_15s on the 5bp pool — USD-weighted

| Metric | Real (USD-w) | Sim (USD-w) |
|--------|--------------|-------------|
| mean   | +6.889 bps | +61.247 bps |
| p1 | -9.39 | -6.44 |
| p5 | -4.27 | -1.61 |
| p25 | +2.83 | +3.98 |
| p50 | +6.16 | +11.27 |
| p75 | +10.07 | +36.97 |
| p95 | +18.47 | +263.22 |
| p99 | +23.53 | +263.22 |

Convention: markout_15s is LP-positive (a positive value means LP profited
from the trade after the 15-second look-ahead). Real data uses 15s; the sim
uses the next-block fair_price (≈12s, the closest available proxy).

## Artifacts
- Per-trade sim markouts: `analysis/weth_usdc_90d/markout_5bp_pool_sim_retail.csv`
- Validation plot: `plots/validation_retail_pool_mid.png`
