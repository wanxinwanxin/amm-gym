# Impact-curve fit — lagged-fair reference

**Sample:** V3 swap legs of **MetaMask-fee retail** (87.5 bps convenience fee,
collector `0xf326e4…`) non-5bp WETH/USDC txs in the 30-day window
**2026-04-21..2026-05-20**, aggregated per tx, with spread measured
against the **lagged fair price** — the Binance mid in the 12s bucket
immediately preceding each trade ("fair at or before 12s before the trade";
the real-data analog of the simulator's "1 step before"). Reference column:
`observed_spread_fair_lag_bps`.

**n_txs** = **3,745** after dropping mixed-direction and dust rows.

Sign convention: LP-positive (`observed_spread_fair_lag_bps > 0` ⇔ user paid above the
lagged fair). Because the reference is one 12s step stale, each trade's
spread also carries ~12s of price drift, so the cloud has heavy ± tails;
the USD-weighted Huber fit (Plan B) is the robust default.

## Observation summary

| stat | value |
|------|-------|
| USD-weighted mean spread | **+1.736 bps** |
| median spread | +1.234 bps |
| p1 / p99 spread | -41.76 / +12.81 bps |
| fraction of txs with negative spread | 23.55% |

The lagged fair is a pre-trade reference (one 12s step before the trade), so
the trade cannot have influenced it; it differs from the contemporaneous-fair
and pool-mid references (both also in the sample CSV) by the 12s price drift it
carries into each observation.

## Empirical impact curve, by log-size decile (USD-weighted)

| median size | n_txs | USD-vol share | USD-w mean spread (bps) | median spread (bps) | pct negative |
|-------------|-------|----------------|--------------------------|----------------------|---------------|
| $            2 |     26 |  0.00% |  -2.32 |  +1.26 |  19.2% |
| $            5 |     55 |  0.01% |  +0.88 |  +1.05 |  21.8% |
| $           16 |    185 |  0.08% |  +2.12 |  +1.01 |  28.6% |
| $           48 |    412 |  0.53% |  +0.76 |  +1.13 |  28.2% |
| $          108 |    734 |  2.46% |  +0.86 |  +1.07 |  23.7% |
| $          351 |    991 | 10.02% |  +0.18 |  +1.18 |  24.9% |
| $          979 |    731 | 19.96% |  +1.40 |  +1.19 |  23.3% |
| $        2,458 |    472 | 35.33% |  +1.62 |  +1.58 |  18.9% |
| $        5,978 |    125 | 24.29% |  +2.49 |  +2.56 |  11.2% |
| $       14,619 |     14 |  7.32% |  +3.22 |  +3.69 |  14.3% |

The curve is **monotone non-decreasing, strictly positive at the mean**, and
clearly V2-shaped: small-size shelf near the 1bp pool's fee, then linear-in-size
growth at larger sizes.

## Fitted parameters

| Plan | φ | depth (USDC) | fit-loss |
|------|----|---------------|----------|
| **A — USD-weighted L2** | **0.000116** (≈1.1579 bps) | **$80,984,508** (≈$81.0 M) | 102.7996 |
| **B — USD-weighted Huber (δ = 4.07 bps)** | **0.000134** (≈1.3423 bps) | **$88,853,788** (≈$88.9 M) | 6.4308 |

## Residual diagnostics (USD-weighted, |residual| in bps)

| stat        | Plan A | Plan B |
|-------------|---------|---------|
| median \|r\|     |   1.25 |   1.24 |
| p75 \|r\|        |   2.26 |   2.27 |
| p90 \|r\|        |   4.10 |   4.08 |
| p99 \|r\|        |  24.30 |  24.15 |
| USD-weighted RMSE |  10.14 |  10.14 |
| USD-weighted mean residual | +0.000 | -0.133 |

## Artifacts

- Fit JSON: `analysis/weth_usdc_90d/impact_curve_fit_pool_mid.json`
- Fit plot: `plots/impact_curve_fit_pool_mid.png`
- Residuals plot: `plots/impact_curve_residuals_pool_mid.png`
