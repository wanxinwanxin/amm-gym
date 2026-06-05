# Retail-only held-out validation — impact-curve framework

Calibration step (not run here): fit V2 (φ, depth) of the normalizer pool to
the empirical non-5bp router-routed WETH/USDC impact curve (pool_mid_pre
referenced). 5bp pool metrics never enter the fit.

This script runs the simulator with:
- submission = 5bp on-chain frozen (fee=0.0005, virtual_depth_y=$212.16M)
- normalizer = **plan_b**: φ = **3.67 bps**, depth = **$73.4M**

and compares the three retail-only validation metrics to on-chain reality.

## Retail flow split

| Metric | Real | Sim (mean ± std) |
|--------|------|-------------------|
| Volume share @5bp | 45.89% | 55.09% ± 1.02pp |
| Fee share @5bp    | 17.36% | 62.53% ± 0.97pp |

## Retail markout_15s on the 5bp pool — USD-weighted

| Metric | Real (USD-w) | Sim (USD-w) |
|--------|--------------|-------------|
| mean   | +9.706 bps | +5.018 bps |
| p1 | -6.11 | -11.75 |
| p5 | -1.27 | -4.19 |
| p25 | +3.58 | +1.03 |
| p50 | +7.97 | +4.43 |
| p75 | +13.39 | +9.13 |
| p95 | +22.75 | +15.63 |
| p99 | +64.30 | +22.85 |

Convention: markout_15s is LP-positive (a positive value means LP profited
from the trade after the 15-second look-ahead). Real data uses 15s; the sim
uses the next-block fair_price (≈12s, the closest available proxy).

## Artifacts
- Per-trade sim markouts: `analysis/weth_usdc_90d/markout_5bp_pool_sim_retail.csv`
- Validation plot: `plots/validation_retail_pool_mid.png`
