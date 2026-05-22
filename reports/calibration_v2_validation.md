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
0-fee + 1-30bp, V2 30bp, Balancer, etc.). See Phase 4 for candidate
model refinements.

## Artifacts

- Sim results JSON: `analysis/weth_usdc_90d/calibration_v2_final.json`
- Validation plot: `plots/validation_sim_vs_real.png`
