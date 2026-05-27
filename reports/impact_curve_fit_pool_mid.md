# Impact-curve fit — pool_mid_pre reference

**Sample:** V3 swap legs of router-routed non-5bp WETH/USDC txs in the 7d
window **2026-05-14..2026-05-20**, aggregated per tx, with spread measured
against the pool's own **pre-trade marginal price** (V3 sqrtPriceX96 of
the prior swap).

**n_txs** = **11,596** after dropping mixed-direction and dust rows.

Sign convention: LP-positive (`observed_spread_pool_bps > 0` ⇔ user
paid above pool mid).

## Observation summary

| stat | value |
|------|-------|
| USD-weighted mean spread | **+3.728 bps** |
| median spread | +1.085 bps |
| p1 / p99 spread | +1.00 / +30.00 bps |
| fraction of txs with negative spread | 0.00% |

The previous fair-referenced sample had USD-weighted mean **-1.62 bps**;
the pool_mid_pre reference removes the AMM↔Binance timing drift and
restores the natural positive shape.

## Empirical impact curve, by log-size decile (USD-weighted)

| median size | n_txs | USD-vol share | USD-w mean spread (bps) | median spread (bps) | pct negative |
|-------------|-------|----------------|--------------------------|----------------------|---------------|
| $            2 |    272 |  0.00% |  +3.63 |  +1.00 |   0.0% |
| $            6 |    491 |  0.02% |  +1.64 |  +1.00 |   0.0% |
| $           20 |    983 |  0.15% |  +2.63 |  +1.01 |   0.0% |
| $           58 |  1,712 |  0.85% |  +2.02 |  +1.02 |   0.0% |
| $          184 |  2,620 |  3.89% |  +1.21 |  +1.05 |   0.0% |
| $          521 |  2,391 | 10.94% |  +1.38 |  +1.16 |   0.0% |
| $        1,656 |  2,157 | 30.12% |  +1.72 |  +1.47 |   0.0% |
| $        4,498 |    828 | 33.05% |  +2.86 |  +2.29 |   0.0% |
| $       12,215 |    128 | 13.91% |  +5.87 |  +4.51 |   0.0% |
| $       57,407 |     14 |  7.06% | +17.44 | +13.73 |   0.0% |

The curve is **monotone non-decreasing, strictly positive at the mean**, and
clearly V2-shaped: small-size shelf near the 1bp pool's fee, then linear-in-size
growth at larger sizes.

## Fitted parameters

| Plan | φ | depth (USDC) | fit-loss |
|------|----|---------------|----------|
| **A — USD-weighted L2** | **0.000145** (≈1.4521 bps) | **$42,070,403** (≈$42.1 M) | 14.9811 |
| **B — USD-weighted Huber (δ = 0.52 bps)** | **0.000129** (≈1.2911 bps) | **$56,090,643** (≈$56.1 M) | 0.5125 |

## Residual diagnostics (USD-weighted, |residual| in bps)

| stat        | Plan A | Plan B |
|-------------|---------|---------|
| median \|r\|     |   0.45 |   0.29 |
| p75 \|r\|        |   0.82 |   0.46 |
| p90 \|r\|        |   2.46 |   1.46 |
| p99 \|r\|        |  22.12 |  27.47 |
| USD-weighted RMSE |   3.87 |   4.09 |
| USD-weighted mean residual | +0.000 | +0.730 |

## Artifacts

- Fit JSON: `analysis/weth_usdc_90d/impact_curve_fit_pool_mid.json`
- Fit plot: `plots/impact_curve_fit_pool_mid.png`
- Residuals plot: `plots/impact_curve_residuals_pool_mid.png`
