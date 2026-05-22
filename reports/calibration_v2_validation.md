# Calibration v2 — validation against held-out 5bp targets

**Framework reminder:**

- We FROZE the 5bp Uniswap V3 pool's on-chain parameters
  (`fee = 0.0005`, `virtual_depth_y = $212,157,626` USDC).
  These were NOT optimization targets.
- We FIT the hypothetical V2 "other pool" `(φ, depth)` against
  the empirical impact cloud of the non-5bp WETH/USDC universe.
  That fit never saw any 5bp metric.
- We now RUN the simulator with `submission = 5bp-frozen`,
  `normalizer = (φ, depth)`, and compare the simulator's predicted
  5bp metrics to the real held-out targets:
  - `arb_5bp_share = 0.337330` (7d, on-chain)
  - `retail_5bp_share = 0.782049` (7d, on-chain)
  - `markout_5bp_bps = -1.05` (7d USD-weighted, LP-positive)

If the simulator predicts those targets within tolerance, the
framework has demonstrated predictive ability without circularity.

## Validation results (mean across 5 seeds per set)

Plan A: φ = **0.000000** (≈0.0000 bps), depth = **$100.00 B**
Plan B: φ = **0.000000** (≈0.0000 bps), depth = **$100.00 B**

| Metric | Real | Plan A · calib | Plan A · held-out | Plan B · calib | Plan B · held-out |
|--------|------|----------------|-------------------|----------------|-------------------|
| T1 · arb_share_5bp | +0.3373 | +0.0009 (-99.7%) | +0.0008 (-99.8%) | +0.0009 (-99.7%) | +0.0008 (-99.8%) |
| T2 · retail_share_5bp | +0.7820 | +0.0003 (-100.0%) | +0.0004 (-99.9%) | +0.0003 (-100.0%) | +0.0004 (-99.9%) |
| T3 · markout_bps | -1.0500 | -16.7140 (+1491.8%) | -20.3526 (+1838.3%) | -16.7140 (+1491.8%) | -20.3526 (+1838.3%) |

Largest held-out residual: Plan A = **1838.3%**, Plan B = **1838.3%**.
Outcome at the 20% tolerance bar: **FAILED**.

## Per-seed detail — Plan A (held-out)

| seed | arb_share | retail_share | markout_bps |
|------|-----------|--------------|--------------|
| 100 | 0.0009 | 0.0004 | -23.843 |
| 101 | 0.0008 | 0.0003 | -17.157 |
| 102 | 0.0007 | 0.0004 | -3.074 |
| 103 | 0.0010 | 0.0005 | -24.184 |
| 104 | 0.0008 | 0.0004 | -33.505 |

## Per-seed detail — Plan B (held-out)

| seed | arb_share | retail_share | markout_bps |
|------|-----------|--------------|--------------|
| 100 | 0.0009 | 0.0004 | -23.843 |
| 101 | 0.0008 | 0.0003 | -17.157 |
| 102 | 0.0007 | 0.0004 | -3.074 |
| 103 | 0.0010 | 0.0005 | -24.184 |
| 104 | 0.0008 | 0.0004 | -33.505 |

## Plain-English judgment

Both plans collapse to the same boundary fit (`φ → 0`, `depth → $100 B`)
because the USD-weighted center of the empirical non-5bp impact cloud
is slightly negative (-0.30 bps) and a non-negative V2 can only get to
zero. In the simulator that boundary fit acts as an effectively
infinite-liquidity zero-fee "other pool". Predictably, the 5bp pool
loses essentially all flow to the other pool: the predicted 5bp arb
and retail shares fall well below the real-world ~33% and ~78%.

The markout prediction (T3) is the most informative residual: with no
arb flow taking liquidity from the submission pool, the simulator
cannot reproduce the empirical ~-1 bps LP markout. The mismatch is a
direct fingerprint of the V2 functional misspecification, not of bad
parameter values.

## Conclusion

The predictive framework is methodologically sound, but a single
aggregated V2 is the wrong model for the rest-of-world non-5bp pool
side. The user's hypothesis behind running this experiment is
confirmed by what the validation reveals: a single-V2 aggregation is
too coarse for the real multi-tier WETH/USDC universe (V3 1bp, V4
0-fee + 1-30bp, V2 30bp, Balancer, etc.). See Phase 4 below for
candidate model refinements.

---

# Phase 4 — Candidate model misspecifications

Per the spec, we are **not** tuning `(φ, depth)` to make the 5bp
metrics match; that would re-introduce circularity. Instead we
surface the model changes most likely to close the gap, with a
brief argument for each.

## 1. Single-V2 aggregation is too coarse (PRIMARY)

This is the misspecification the Phase 1+2 plots already point at.
The empirical non-5bp impact curve is **non-monotonic** with a
positive small-size shelf (~+2 to +6 bps) and a negative large-size
tail (-1 to -8 bps USD-weighted). A V2 has spread monotonically
**increasing** in size and **always non-negative**. The shape is
unreachable with a single CP pool.

**Candidate fix:** model the other side as **two pools** instead
of one. A natural split for WETH/USDC:

- **Tight tier** (matches V3-1bp + V4-0bp + Curve-style stable
  routes): small fee (≤ 1 bp), large depth (~$500M-$1.5B virtual).
  Captures the negative-spread tail at large sizes (these tiers are
  where SOR routes when the size is big enough to dominate
  individual-leg fees).
- **Wide tier** (matches V3-30bp, V2-30bp, Balancer): larger fee
  (~10-30 bps), shallower depth (~$50-100M virtual). Captures the
  positive-spread shelf at small/mid sizes.

The simulator's `OrderRouter` already supports only two AMMs
(submission + normalizer), so the cleanest implementation is:
  - keep `submission` = 5bp pool (frozen),
  - replace the single `normalizer` with a **composite** other-pool
    object whose `quote_buy_x` / `quote_sell_x` are the two-tier
    optimal split,
  - fit `(φ_tight, depth_tight, φ_wide, depth_wide)` to the
    same empirical impact cloud.

This is a four-parameter fit but the parameters are still
calibrated only from non-5bp empirical impact — the 5bp metrics
remain held out.

## 2. Router splitter has a structural ceiling near 0.72 retail share

The closed-form 2-AMM router `split_buy_two_amms` /
`split_sell_two_amms` derives the optimal y-split (or x-split) from
the relative `sqrt(x · γ · y)` of the two pools, which is the
*marginal-equal-price* split for two CP pools with linear fees.
Empirically the sibling calibration agent observed this can't push
the submission-pool share above ~0.72 in the unsaturated regime no
matter how the depths are set (above 0.72 the small pool saturates
and the router falls back to single-pool).

The real-world retail 5bp share is **0.782 > 0.72**, so the
simulator's router *physically cannot* match T2 with any choice of
two-V2 depths. To reach 0.78 we likely need:

- a non-closed-form router that picks per-order venues
  probabilistically (some user retail flows are quoted only against
  one pool, e.g., wallets pre-routing to UniswapX or the universal
  router), or
- a router that pre-allocates a configurable retail-stickiness
  fraction to the 5bp pool (a "retail stickiness" knob) calibrated
  from a separate empirical primitive (e.g., per-pool retail
  flow-share in the dex_trades data) — note that *that* primitive
  is also held out from the 5bp pool's *markout* and *arb*
  metrics, so calibrating router stickiness from it remains
  circularity-free.

## 3. Arb model is per-AMM, no cross-pool arb

The simulator's `Arbitrageur.execute_arb` is closed-form against a
single AMM vs the fair price. It does not consider cross-pool
arbitrage between the submission and normalizer pools. Real-world
arb flow is significantly cross-pool: large arb traders rebalance
both the 5bp pool and the 1bp pool against Binance in the same
block, and the spec'd negative-spread large trades in the
empirical sample are precisely those arb flows.

**Candidate fix:** add a cross-pool arbitrageur that searches the
joint price surface for the optimal pair of trades that
simultaneously rebalances both pools against the fair price. This
also changes the arb-volume split (T1) — likely pulling more arb
volume to the deeper pool, which is the direction we need (real
T1 = 0.337 means the 5bp pool gets ~1/3 of arb volume, currently
the sim produces ~0).

## 4. Retail USD-size distribution mismatch on non-5bp

The simulator currently resamples retail USD sizes from
`parent_order_usd_quantiles.csv`, which was extracted from
**all-pool parent orders** in the same router-routed universe.
Restricted to **non-5bp** parent orders, the size distribution is
different (data from this calibration sample):

| pct | sim CSV (all pools) | empirical non-5bp portion | ratio |
|-----|----------------------|------------------------------|--------|
| p10 | $14.95               | $40.91                       | 2.7×  |
| p25 | $100.17              | $251.57                      | 2.5×  |
| p50 | $701.17              | $1,633.71                    | 2.3×  |
| p75 | $2,872.61            | $4,442.62                    | 1.5×  |
| p90 | $14,136.25           | $9,161.66                    | 0.65× |
| p95 | $37,572.15           | $13,350.24                   | 0.36× |
| p99 | $145,360.62          | $29,441.67                   | 0.20× |

The non-5bp portion has a **wider middle** and a **thinner upper
tail**. That's consistent with the SOR routing the very large
parent orders disproportionately to the 5bp pool (because it
has the lowest fee for the deep range), so the residual on the
non-5bp side is a "compressed" version of the full distribution.

A composite (two-tier other pool) model is the cleanest fix;
swapping the retail USD quantile file to a non-5bp-only file is a
smaller, complementary change.

## Recommended next step

The most impactful single change is **#1 (two-tier other pool)**.
It directly addresses the impact-curve shape mismatch identified
in Phase 2, and the four parameters are still calibrated from
held-out non-5bp empirical data. Changes #2-#4 are additive and
should be considered in order of remaining residual after #1.

## Artifacts

- Sim results JSON: `analysis/weth_usdc_90d/calibration_v2_final.json`
- Validation plot: `plots/validation_sim_vs_real.png`
