# Retail-only held-out validation — impact-curve framework

Calibration step (not run here): fit V2 (φ, depth) of the normalizer pool to
the empirical non-5bp router-routed WETH/USDC impact curve (pool_mid_pre
referenced). 5bp pool metrics never enter the fit.

This script runs the simulator with:
- submission = 5bp on-chain frozen (fee=0.0005, virtual_depth_y=$212.16M)
- normalizer = **plan_b**: φ = **1.29 bps**, depth = **$56.1M**

and compares the three retail-only validation metrics to on-chain reality.

## Retail flow split

| Metric | Real | Sim (mean ± std) |
|--------|------|-------------------|
| Volume share @5bp | 49.49% | 61.47% ± 8.86pp |
| Fee share @5bp    | 17.15% | 85.64% ± 4.62pp |

## Retail markout_15s on the 5bp pool — USD-weighted

| Metric | Real (USD-w) | Sim (USD-w) |
|--------|--------------|-------------|
| mean   | +6.889 bps | +139.404 bps |
| p1 | -9.39 | -2.97 |
| p5 | -4.27 | +0.93 |
| p25 | +2.83 | +15.66 |
| p50 | +6.16 | +152.88 |
| p75 | +10.07 | +235.57 |
| p95 | +18.47 | +266.87 |
| p99 | +23.53 | +266.87 |

Convention: markout_15s is LP-positive (a positive value means LP profited
from the trade after the 15-second look-ahead). Real data uses 15s; the sim
uses the next-block fair_price (≈12s, the closest available proxy).

## Artifacts
- Per-trade sim markouts: `analysis/weth_usdc_90d/markout_5bp_pool_sim_retail.csv`
- Validation plot: `plots/validation_retail_pool_mid.png`
