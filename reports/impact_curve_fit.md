# Impact-curve fit (Phase 2)

**Sample:** non-5bp portion of router-routed WETH/USDC transactions,
7-day window **2026-05-14..2026-05-20**, n_txs = **34,603** after dropping
mixed-direction and dust rows. Sign convention: LP-positive
(observed_spread_bps > 0 means the user paid above Binance mid).
**57.3% of txs have negative observed spread** (better-than-fair fills);
those were kept in the fit per spec.

The hypothetical V2 (constant-product) "other pool" has spread
`spread_hyp_bps(S) = 10_000 * (φ + (1-φ) * S / depth) / (1-φ)`. It is by
construction **strictly non-negative and monotonically increasing in S**.

## Empirical impact curve, by log-size decile (USD-weighted)

| median size | n_txs | USD-vol share | USD-w mean spread (bps) | median spread (bps) | pct negative |
|-------------|-------|----------------|--------------------------|----------------------|---------------|
| $            2 |    579 |  0.00% |  +3.44 |  +0.87 |  37.8% |
| $            8 |  1,264 |  0.01% |  +1.67 |  +0.71 |  38.9% |
| $           30 |  2,241 |  0.06% |  +5.50 |  +1.18 |  28.7% |
| $          104 |  3,426 |  0.30% |  +6.10 |  +1.05 |  32.3% |
| $          361 |  5,765 |  1.76% |  +2.52 |  -0.21 |  52.4% |
| $        1,682 |  8,446 | 11.17% |  +1.60 |  -0.06 |  52.0% |
| $        4,888 | 10,155 | 42.70% |  +0.27 |  -0.79 |  75.3% |
| $       14,629 |  2,537 | 33.04% |  -0.99 |  -2.23 |  85.5% |
| $       53,540 |    181 |  8.90% |  -1.16 |  -4.90 |  71.3% |
| $      272,624 |      9 |  2.06% | -11.19 | -10.46 |  55.6% |

The empirical curve is **non-monotonic and goes negative at large sizes**:
small-size txs (< $1k) cluster around +1 to +5 bps (retail), the meaty
$1k-$10k bucket (which carries ~30% of USD volume) sits near 0, and the
$10k+ buckets (~60% of USD volume) trend strongly negative
(arb / SOR / price-improvement). This is **not a V2 shape** — a V2 has
spread monotonically increasing in size.

## Fitted parameters

| Plan | φ | depth (USDC) | fit-loss |
|------|----|---------------|----------|
| **A — USD-weighted L2** | **0.000000** (≈0.0000 bps) | **$100,000,000,000** (≈$100.0 B) | 207.0372 |
| **B — USD-weighted Huber (δ = 7.02 bps)** | **0.000000** (≈0.0000 bps) | **$100,000,000,000** (≈$100.0 B) | 19.6104 |

Both fits converge to the **lower φ boundary and upper depth boundary**.
This is the correct L-BFGS-B optimum given a non-negative V2 model and
a USD-weighted target that is itself slightly **negative** (
USD-weighted overall mean spread = **-0.304 bps**): the closest
non-negative V2 to a USD-weighted-negative cloud is the limiting
zero-spread V2, i.e., depth → ∞, φ → 0.

This is **diagnostic**, not a numeric failure: it tells us a single
aggregated V2 cannot represent the rest-of-world side because the
rest-of-world side **systematically improves trader execution** in the
USD-volume-dominant size range. That is the empirical signal a V2
cannot produce.

## Residual diagnostics (USD-weighted, |residual| in bps)

| stat        | Plan A | Plan B |
|-------------|---------|---------|
| median \|r\|     |   1.98 |   1.98 |
| p75 \|r\|        |   4.02 |   4.02 |
| p90 \|r\|        |   8.51 |   8.51 |
| p99 \|r\|        |  42.90 |  42.90 |
| USD-weighted RMSE |  14.39 |  14.39 |
| USD-weighted mean residual | -0.304 | -0.304 |

Plan A and Plan B converge to the same boundary point and so produce
identical residual statistics — both methods agree that no interior
V2 has a smaller USD-weighted error than the limiting zero-spread V2.

## Defensibility judgment

**Neither Plan A nor Plan B produces an interior V2 fit** — the
USD-weighted-mean of the empirical cloud is negative at every size
above ~$10k, and a non-negative V2 cannot match. The shape of the
empirical impact curve (non-monotonic, negative at scale) is what a
V2 cannot represent. This is the expected outcome the Phase 1
diagnostic plot foreshadowed: heavy negative tail at large sizes.

**Recommendation for Phase 3:** validate both plans (they happen to
be the same boundary fit, so the validation collapses to one sim
run). The validation is informative regardless: it will quantify
*how* mismatched the predicted 5bp metrics become when the
rest-of-world side is modeled as effectively-infinite-depth /
zero-fee. That mismatch is the diagnostic the user asked for — it
characterizes the V2-aggregation misspecification rather than
producing a number-fit. Phase 4 surfaces the candidate fixes
(e.g., 2-pool decomposition of the other side).

## Artifacts

- Fit JSON: `analysis/weth_usdc_90d/impact_curve_fit.json`
- Fit plot: `plots/impact_curve_fit.png`
- Residuals plot: `plots/impact_curve_residuals.png`
