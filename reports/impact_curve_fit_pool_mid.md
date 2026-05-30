# Impact-curve fit — lagged-fair reference

**Sample:** V3 swap legs of **MetaMask-fee retail** (87.5 bps convenience fee,
collector `0xf326e4…`) non-5bp WETH/USDC txs in the 30-day window
**2026-04-21..2026-05-20**, aggregated per tx, with spread measured
against the **lagged fair price** — the Binance mid in the 12s bucket
immediately preceding each trade ("fair at or before 12s before the trade";
the real-data analog of the simulator's "1 step before"). Reference column:
`observed_spread_fair_lag_bps`.

**n_txs** = **69,627** after dropping mixed-direction and dust rows.

Sign convention: LP-positive (`observed_spread_fair_lag_bps > 0` ⇔ user paid above the
lagged fair). Because the reference is one 12s step stale, each trade's
spread also carries ~12s of price drift, so the cloud has heavy ± tails;
the USD-weighted Huber fit (Plan B) is the robust default.

## Observation summary

| stat | value |
|------|-------|
| USD-weighted mean spread | **+10.657 bps** |
| median spread | +1.374 bps |
| p1 / p99 spread | -20.67 / +82.68 bps |
| fraction of txs with negative spread | 21.89% |

The lagged fair is a pre-trade reference (one 12s step before the trade), so
the trade cannot have influenced it; it differs from the contemporaneous-fair
and pool-mid references (both also in the sample CSV) by the 12s price drift it
carries into each observation.

## Empirical impact curve, by log-size decile (USD-weighted)

| median size | n_txs | USD-vol share | USD-w mean spread (bps) | median spread (bps) | pct negative |
|-------------|-------|----------------|--------------------------|----------------------|---------------|
| $            2 |  1,026 |  0.00% |  +1.28 |  +1.24 |  23.4% |
| $            8 |  2,644 |  0.02% |  +1.08 |  +1.15 |  25.0% |
| $           27 |  5,665 |  0.14% |  +0.29 |  +1.03 |  26.8% |
| $           99 | 12,978 |  1.11% |  +0.99 |  +1.09 |  24.8% |
| $          322 | 17,614 |  5.35% |  +1.31 |  +1.12 |  25.2% |
| $        1,057 | 16,736 | 17.37% |  +2.50 |  +1.35 |  21.4% |
| $        3,459 | 10,230 | 34.52% |  +8.24 |  +2.05 |  14.1% |
| $       10,792 |  2,399 | 26.47% | +15.25 |  +4.09 |   5.7% |
| $       39,988 |    316 | 12.52% | +20.24 |  +8.82 |   0.6% |
| $      116,996 |     19 |  2.51% | +28.92 | +17.29 |   0.0% |

The curve is **monotone non-decreasing, strictly positive at the mean**, and
clearly V2-shaped: small-size shelf near the 1bp pool's fee, then linear-in-size
growth at larger sizes.

## Fitted parameters

| Plan | φ | depth (USDC) | fit-loss |
|------|----|---------------|----------|
| **A — USD-weighted L2** | **0.000808** (≈8.0762 bps) | **$63,897,340** (≈$63.9 M) | 6631.2835 |
| **B — USD-weighted Huber (δ = 9.84 bps)** | **0.000218** (≈2.1817 bps) | **$63,913,448** (≈$63.9 M) | 80.5953 |

## Residual diagnostics (USD-weighted, |residual| in bps)

| stat        | Plan A | Plan B |
|-------------|---------|---------|
| median \|r\|     |   6.85 |   1.73 |
| p75 \|r\|        |   8.35 |   3.34 |
| p90 \|r\|        |  11.86 |   8.55 |
| p99 \|r\|        | 161.48 | 167.23 |
| USD-weighted RMSE |  81.43 |  81.65 |
| USD-weighted mean residual | +0.000 | +5.901 |

## Artifacts

- Fit JSON: `analysis/weth_usdc_90d/impact_curve_fit_pool_mid.json`
- Fit plot: `plots/impact_curve_fit_pool_mid.png`
- Residuals plot: `plots/impact_curve_residuals_pool_mid.png`
