-- §4 NORMALIZER calibration sample — the impact curve of EVERYTHING-EXCEPT-THE-5BP-POOL
-- across ALL venues (Uniswap V3+V4+V2, Fluid, Swaap, Pancake, Balancer, Curve, …),
-- built entirely from uniswap-labs.research.markout_prod.
--
-- This is the competitor the held-out 5bp pool is VALIDATED against (§5). The 5bp
-- pool competes against ALL other pools on the pair, not just the V3 ones, so the
-- normalizer must be all NON-5bp venues — earlier this sample was non-5bp **V3
-- only**, which dropped the ~10% of non-5bp flow on V4/Fluid/Balancer/Curve/etc.
-- and biased the fit.
--
-- This is identical to allpools_impact_sample_30d.sql (the §8 full-market sample)
-- EXCEPT the calibration legs EXCLUDE the 5bp pool. So §4 (this) and §8 differ only
-- by whether the 5bp pool is in the normalizer:
--   §4 = all venues  MINUS the 5bp pool  (5bp is the submission pool being validated)
--   §8 = all venues  INCLUDING the 5bp pool  (a new strategy is the submission pool)
--
-- Same all-venue rationale as §8: the calibration target is the LAGGED BINANCE MID
-- (observed_spread_fair_lag_bps), which needs only the routed order amounts
-- (effective_exec = ΣUSDC/ΣWETH, side) + the fair grid — NOT any per-pool mid — so
-- there is no reason to restrict to V3, and no pool-mid column is emitted.
-- (Per-pool-mid sandwich detection therefore stays a V3-subset diagnostic.)
--
-- The fair grid is still built from ALL WETH/USDC benchmarks (incl. the 5bp pool's
-- legs) — the Binance mid is a market-wide reference; only the CALIBRATION legs drop
-- the 5bp pool. Cohort: strict retail (Uniswap FE ∪ MetaMask 87.5bps). 30d window.

DECLARE start_date DATE DEFAULT DATE '2026-04-21';
DECLARE end_date   DATE DEFAULT DATE '2026-05-20';
DECLARE start_ts   TIMESTAMP DEFAULT TIMESTAMP '2026-04-21 00:00:00';
DECLARE end_ts     TIMESTAMP DEFAULT TIMESTAMP '2026-05-20 23:59:59';
DECLARE grid_start TIMESTAMP DEFAULT TIMESTAMP '2026-04-20 23:00:00';
DECLARE mm_fee_addr STRING DEFAULT '0xf326e4de8f66a0bdc0970b79e0924e33c79f1915';
DECLARE pool_5bp    STRING DEFAULT '0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640';

WITH constants AS (
  SELECT
    '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2' AS weth,
    '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48' AS usdc
),
mm_fee_txs AS (
  SELECT DISTINCT LOWER(TRANSACTION_HASH) AS tx_hash
  FROM `uniswap-allium.ethereum.assets_erc20_token_transfers`
  WHERE DATE(BLOCK_TIMESTAMP) BETWEEN start_date AND end_date
    AND LOWER(TO_ADDRESS) = mm_fee_addr
),
uni_fe_txs AS (
  SELECT DISTINCT LOWER(transaction_hash) AS tx_hash
  FROM `uniswap-labs.core.swaps`
  WHERE report_date BETWEEN DATE_SUB(start_date, INTERVAL 1 DAY) AND DATE_ADD(end_date, INTERVAL 1 DAY)
    AND LOWER(chain_name) = 'ethereum'
    AND is_complete
    AND transaction_hash IS NOT NULL
),
retail_txs AS (
  SELECT tx_hash FROM uni_fe_txs
  UNION DISTINCT
  SELECT tx_hash FROM mm_fee_txs
),
-- All WETH/USDC swaps across every venue (padded window), oriented to canonical
-- units. Feeds the fair grid (all legs) and the calibration legs (strict subset).
mk_all AS (
  SELECT
    LOWER(transaction_hash) AS tx_hash,
    log_index,
    block_timestamp,
    LOWER(liquidity_pool_address) AS pool,
    project,
    usd_amount AS leg_usd,
    CASE
      WHEN LOWER(token_sold_address) = (SELECT usdc FROM constants)
       AND LOWER(token_bought_address) = (SELECT weth FROM constants) THEN 1   -- buy eth
      WHEN LOWER(token_sold_address) = (SELECT weth FROM constants)
       AND LOWER(token_bought_address) = (SELECT usdc FROM constants) THEN -1  -- sell eth
      ELSE NULL
    END AS side,
    CASE
      WHEN LOWER(token_sold_address)   = (SELECT weth FROM constants) THEN token_sold_amount
      WHEN LOWER(token_bought_address) = (SELECT weth FROM constants) THEN token_bought_amount
    END AS weth_amt,
    CASE
      WHEN LOWER(token_sold_address)   = (SELECT usdc FROM constants) THEN token_sold_amount
      WHEN LOWER(token_bought_address) = (SELECT usdc FROM constants) THEN token_bought_amount
    END AS usdc_amt,
    CASE
      WHEN LOWER(token_sold_address) = (SELECT weth FROM constants)
       AND LOWER(token_bought_address) = (SELECT usdc FROM constants) THEN benchmark
      WHEN LOWER(token_sold_address) = (SELECT usdc FROM constants)
       AND LOWER(token_bought_address) = (SELECT weth FROM constants) THEN SAFE_DIVIDE(1.0, benchmark)
      ELSE NULL
    END AS bench_usdc_per_weth
  FROM `uniswap-labs.research.markout_prod`
  WHERE chain = 'ethereum'
    AND block_date BETWEEN DATE_SUB(start_date, INTERVAL 1 DAY) AND end_date
    AND (
      (LOWER(token_sold_address)   = (SELECT weth FROM constants)
        AND LOWER(token_bought_address) = (SELECT usdc FROM constants))
      OR
      (LOWER(token_sold_address)   = (SELECT usdc FROM constants)
        AND LOWER(token_bought_address) = (SELECT weth FROM constants))
    )
),
-- 12s Binance-mid grid (from ALL WETH/USDC benchmarks, incl. 5bp legs), forward-filled.
fair_bucket_obs AS (
  SELECT
    TIMESTAMP_SECONDS(DIV(UNIX_SECONDS(block_timestamp), 12) * 12) AS bucket_ts,
    bench_usdc_per_weth AS mid
  FROM mk_all
  WHERE bench_usdc_per_weth IS NOT NULL AND bench_usdc_per_weth > 0
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY TIMESTAMP_SECONDS(DIV(UNIX_SECONDS(block_timestamp), 12) * 12)
    ORDER BY block_timestamp DESC
  ) = 1
),
grid AS (
  SELECT DISTINCT TIMESTAMP_SECONDS(DIV(UNIX_SECONDS(g), 12) * 12) AS grid_ts
  FROM UNNEST(GENERATE_TIMESTAMP_ARRAY(grid_start, end_ts, INTERVAL 12 SECOND)) AS g
),
grid_filled AS (
  SELECT
    g.grid_ts,
    LAST_VALUE(o.mid IGNORE NULLS) OVER (
      ORDER BY g.grid_ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS fair_mid
  FROM grid AS g
  LEFT JOIN fair_bucket_obs AS o ON o.bucket_ts = g.grid_ts
),
-- Calibration legs: strict-retail cohort, EXCLUDING the 5bp pool (the submission
-- pool being validated), inside the window, valid amounts.
legs AS (
  SELECT m.*
  FROM mk_all AS m
  JOIN retail_txs AS rt ON m.tx_hash = rt.tx_hash
  WHERE m.block_timestamp BETWEEN start_ts AND end_ts
    AND m.pool != pool_5bp
    AND m.side IS NOT NULL
    AND m.weth_amt > 0 AND m.usdc_amt > 0
    AND m.leg_usd IS NOT NULL AND m.leg_usd > 0
),
agg AS (
  SELECT
    tx_hash,
    MIN(block_timestamp)                                               AS block_timestamp,
    COUNT(*)                                                           AS n_legs,
    SUM(leg_usd)                                                       AS size_usd,
    SUM(weth_amt)                                                      AS sum_weth,
    SUM(usdc_amt)                                                      AS sum_usdc,
    SUM(CAST(side AS FLOAT64) * weth_amt)                              AS net_signed_weth,
    SAFE_DIVIDE(
      SUM(IF(bench_usdc_per_weth IS NOT NULL, bench_usdc_per_weth * leg_usd, 0.0)),
      SUM(IF(bench_usdc_per_weth IS NOT NULL, leg_usd, 0.0))
    )                                                                  AS fair_price_blended,
    SUM(IF(bench_usdc_per_weth IS NOT NULL, leg_usd, 0.0))             AS fair_usd_covered,
    COUNT(DISTINCT pool)                                               AS n_distinct_pools,
    COUNT(DISTINCT side)                                               AS n_distinct_sides,
    COUNT(DISTINCT project)                                            AS n_distinct_venues
  FROM legs
  GROUP BY tx_hash
),
filtered AS (
  SELECT
    tx_hash,
    block_timestamp,
    n_legs,
    n_distinct_pools,
    n_distinct_venues,
    size_usd,
    SAFE_DIVIDE(sum_usdc, sum_weth)         AS effective_exec_usdc_per_weth,
    fair_price_blended,
    SAFE_DIVIDE(fair_usd_covered, size_usd) AS fair_coverage_frac,
    CASE WHEN net_signed_weth > 0 THEN 1 WHEN net_signed_weth < 0 THEN -1 ELSE NULL END AS side,
    n_distinct_sides
  FROM agg
  WHERE n_legs >= 1 AND size_usd > 0 AND sum_weth > 0 AND sum_usdc > 0
),
with_lag AS (
  SELECT f.*, gf.fair_mid AS fair_lag_price
  FROM filtered AS f
  LEFT JOIN grid_filled AS gf
    ON gf.grid_ts = TIMESTAMP_SECONDS((DIV(UNIX_SECONDS(f.block_timestamp), 12) - 1) * 12)
)
SELECT
  tx_hash,
  block_timestamp,
  size_usd,
  n_legs,
  n_distinct_pools,
  n_distinct_venues,
  side,
  n_distinct_sides,
  effective_exec_usdc_per_weth,
  fair_price_blended,
  fair_coverage_frac,
  fair_lag_price,
  CASE
    WHEN fair_price_blended IS NULL OR fair_price_blended = 0 THEN NULL
    ELSE 10000.0 * (effective_exec_usdc_per_weth - fair_price_blended) * CAST(side AS FLOAT64) / fair_price_blended
  END AS observed_spread_fair_bps,
  CASE
    WHEN fair_lag_price IS NULL OR fair_lag_price = 0 THEN NULL
    ELSE 10000.0 * (effective_exec_usdc_per_weth - fair_lag_price) * CAST(side AS FLOAT64) / fair_lag_price
  END AS observed_spread_fair_lag_bps
FROM with_lag
WHERE side IS NOT NULL
ORDER BY block_timestamp;
