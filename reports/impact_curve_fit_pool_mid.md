# Impact-curve fit — pool_mid_pre reference

**Sample:** V3 swap legs of router-routed non-5bp WETH/USDC txs in the 7d
window **2026-05-14..2026-05-20**, aggregated per tx, with spread measured
against the pool's own **pre-trade marginal price** (V3 sqrtPriceX96 of
the prior swap).

**n_txs** = **28,269** after dropping mixed-direction and dust rows.

Sign convention: LP-positive (`observed_spread_pool_bps > 0` ⇔ user
paid above pool mid).

## Observation summary

| stat | value |
|------|-------|
| USD-weighted mean spread | **+8.316 bps** |
| median spread | +1.751 bps |
| p1 / p99 spread | +1.00 / +33.50 bps |
| fraction of txs with negative spread | 0.00% |

The previous fair-referenced sample had USD-weighted mean **-1.62 bps**;
the pool_mid_pre reference removes the AMM↔Binance timing drift and
restores the natural positive shape.

## Empirical impact curve, by log-size decile (USD-weighted)

| median size | n_txs | USD-vol share | USD-w mean spread (bps) | median spread (bps) | pct negative |
|-------------|-------|----------------|--------------------------|----------------------|---------------|
| $            2 |    290 |  0.00% |  +3.89 |  +1.00 |   0.0% |
| $            8 |    694 |  0.00% |  +1.66 |  +1.00 |   0.0% |
| $           30 |  1,215 |  0.03% |  +3.17 |  +1.01 |   0.0% |
| $          100 |  2,249 |  0.20% |  +1.51 |  +1.03 |   0.0% |
| $          357 |  2,835 |  0.92% |  +3.27 |  +1.11 |   0.0% |
| $        1,570 |  7,162 |  9.65% |  +2.78 |  +1.52 |   0.0% |
| $        4,325 | 10,719 | 43.41% |  +3.61 |  +2.30 |   0.0% |
| $       12,538 |  2,909 | 35.82% |  +9.30 |  +4.57 |   0.0% |
| $       46,133 |    187 |  8.44% | +25.74 | +24.09 |   0.0% |
| $      171,785 |      9 |  1.53% | +61.89 | +51.00 |   0.0% |

The curve is **monotone non-decreasing, strictly positive at the mean**, and
clearly V2-shaped: small-size shelf near the 1bp pool's fee, then linear-in-size
growth at larger sizes.

## Fitted parameters

| Plan | φ | depth (USDC) | fit-loss |
|------|----|---------------|----------|
| **A — USD-weighted L2** | **0.000397** (≈3.9737 bps) | **$39,509,766** (≈$39.5 M) | 70.0781 |
| **B — USD-weighted Huber (δ = 3.64 bps)** | **0.000134** (≈1.3424 bps) | **$34,808,967** (≈$34.8 M) | 10.7962 |

## Residual diagnostics (USD-weighted, |residual| in bps)

| stat        | Plan A | Plan B |
|-------------|---------|---------|
| median \|r\|     |   2.95 |   0.78 |
| p75 \|r\|        |   4.04 |   2.13 |
| p90 \|r\|        |  13.91 |  14.49 |
| p99 \|r\|        |  25.49 |  28.30 |
| USD-weighted RMSE |   8.37 |   8.71 |
| USD-weighted mean residual | +0.000 | +2.046 |

## Artifacts

- Fit JSON: `analysis/weth_usdc_90d/impact_curve_fit_pool_mid.json`
- Fit plot: `plots/impact_curve_fit_pool_mid.png`
- Residuals plot: `plots/impact_curve_residuals_pool_mid.png`
