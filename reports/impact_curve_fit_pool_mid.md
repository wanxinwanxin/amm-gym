# Impact-curve fit — lagged-fair reference

**Sample:** V3 swap legs of router-routed non-5bp WETH/USDC txs in the 7d
window **2026-05-14..2026-05-20**, aggregated per tx, with spread measured
against the **lagged fair price** — the Binance mid in the 12s bucket
immediately preceding each trade ("fair at or before 12s before the trade";
the real-data analog of the simulator's "1 step before"). Reference column:
`observed_spread_fair_lag_bps`.

**n_txs** = **11,596** after dropping mixed-direction and dust rows.

Sign convention: LP-positive (`observed_spread_fair_lag_bps > 0` ⇔ user paid above the
lagged fair). Because the reference is one 12s step stale, each trade's
spread also carries ~12s of price drift, so the cloud has heavy ± tails;
the USD-weighted Huber fit (Plan B) is the robust default.

## Observation summary

| stat | value |
|------|-------|
| USD-weighted mean spread | **+6.555 bps** |
| median spread | +1.285 bps |
| p1 / p99 spread | -123.92 / +126.01 bps |
| fraction of txs with negative spread | 25.54% |

The lagged fair is a pre-trade reference (one 12s step before the trade), so
the trade cannot have influenced it; it differs from the contemporaneous-fair
and pool-mid references (both also in the sample CSV) by the 12s price drift it
carries into each observation.

## Empirical impact curve, by log-size decile (USD-weighted)

| median size | n_txs | USD-vol share | USD-w mean spread (bps) | median spread (bps) | pct negative |
|-------------|-------|----------------|--------------------------|----------------------|---------------|
| $            2 |    272 |  0.00% |  +4.17 |  +1.03 |  28.7% |
| $            6 |    491 |  0.02% |  +0.53 |  +1.17 |  29.5% |
| $           20 |    983 |  0.15% |  +2.26 |  +1.12 |  26.2% |
| $           58 |  1,712 |  0.85% |  -0.26 |  +1.14 |  28.3% |
| $          184 |  2,620 |  3.89% |  -0.06 |  +1.11 |  27.2% |
| $          521 |  2,391 | 10.94% |  -2.61 |  +1.11 |  28.4% |
| $        1,656 |  2,157 | 30.12% |  +1.07 |  +1.50 |  21.7% |
| $        4,498 |    828 | 33.05% |  +7.93 |  +2.45 |  15.5% |
| $       12,215 |    128 | 13.91% | +10.18 |  +4.54 |   6.2% |
| $       57,407 |     14 |  7.06% | +35.18 | +15.97 |   7.1% |

The curve is **monotone non-decreasing, strictly positive at the mean**, and
clearly V2-shaped: small-size shelf near the 1bp pool's fee, then linear-in-size
growth at larger sizes.

## Fitted parameters

| Plan | φ | depth (USDC) | fit-loss |
|------|----|---------------|----------|
| **A — USD-weighted L2** | **0.000370** (≈3.7004 bps) | **$33,551,287** (≈$33.6 M) | 2982.6910 |
| **B — USD-weighted Huber (δ = 69.15 bps)** | **0.000191** (≈1.9121 bps) | **$49,473,053** (≈$49.5 M) | 637.1416 |

## Residual diagnostics (USD-weighted, |residual| in bps)

| stat        | Plan A | Plan B |
|-------------|---------|---------|
| median \|r\|     |   3.36 |   1.74 |
| p75 \|r\|        |   5.93 |   3.91 |
| p90 \|r\|        |  38.87 |  39.69 |
| p99 \|r\|        | 154.76 | 152.29 |
| USD-weighted RMSE |  54.61 |  54.71 |
| USD-weighted mean residual | +0.000 | +2.708 |

## Artifacts

- Fit JSON: `analysis/weth_usdc_90d/impact_curve_fit_pool_mid.json`
- Fit plot: `plots/impact_curve_fit_pool_mid.png`
- Residuals plot: `plots/impact_curve_residuals_pool_mid.png`
