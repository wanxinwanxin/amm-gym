-- Build the non-5bp price-impact sample for STRICTLY-RETAIL WETH/USDC
-- transactions, emitting THREE spread references side-by-side:
--   (a) observed_spread_pool_bps      -- referenced to pre-trade POOL mid (V3 sqrtPriceX96)
--   (b) observed_spread_fair_bps      -- referenced to CONTEMPORANEOUS fair (Binance benchmark
--                                        joined per-swap from markout_prod, ~at the trade's block)
--   (c) observed_spread_fair_lag_bps  -- referenced to LAGGED fair: the Binance mid in the 12s
--                                        bucket immediately preceding the trade's own 12s bucket
--                                        (i.e. "fair at or before 12s before the trade"; the
--                                        real-data analog of the simulator's "1 step before").
--
-- RETAIL FILTER (the key change): instead of the broad 19-router cohort heuristic,
-- this restricts to transactions that paid a **MetaMask Swaps 87.5 bps convenience
-- fee** — a transfer to the MetaMask fee collector
-- 0xf326e4de8f66a0bdc0970b79e0924e33c79f1915 (Etherscan "MetaMask: DS Proxy";
-- empirically receives a clean ~87.9 bps cut of WETH/USDC swaps). A swap that paid
-- this fee is unambiguously a human retail user on the MetaMask front-end, which is
-- exactly the population the normalizer pool is meant to compete for.
--
-- The Uniswap-interface half of the strict filter is DEFERRED: Uniswap Labs dropped
-- the web-interface swap fee in late 2025, so the fee-transfer heuristic no longer
-- isolates interface flow. The canonical on-chain signature for interface-originated
-- txs is being confirmed (Pinky thread) before it is added here.
--
-- The calibration (scripts/calibration/fit_impact_curve_pool_mid.py) fits the V2
-- normalizer (φ, depth) against (c) — spread vs pre-trade fair. (a)/(b) are kept as
-- diagnostics.
--
-- LAGGED-FAIR SOURCE (no Binance book table for 2026): reconstructed from
-- markout_prod.benchmark over ALL WETH/USDC swaps (last obs per 12s bucket,
-- forward-filled); each trade reads the bucket one step (12s) before its own.
--
-- WINDOW: 30 days, 2026-04-21..2026-05-20 (ending at the validation week). Widened
-- from 7 days because the strict fee filter cuts the sample ~6x; 30 days restores a
-- few thousand observations for a stable (φ, depth) fit under the noisy lagged ref.
--
-- Scope: Uniswap V3 swap events on WETH/USDC pools, excluding the 5bp pool. V3 is
-- ~90% of non-5bp WETH/USDC volume; V4/V2/others dropped (~10%, separate pool-mid
-- reconstruction, not material for the diagnostic).
--
-- Pool-mid (V3): sqrtPriceX96 in the Swap log is POST-swap; pool_mid_PRE for swap N
-- = POST for the previous swap on the same pool. LAG over (pool ORDER BY block,
-- log_index); first swap per pool in the window is dropped.
--
-- Per-tx aggregation: all non-5bp V3 legs rolled into one (size_usd, spread) point;
-- side = sign of net signed WETH (mixed-direction txs dropped); effective_exec =
-- sum(USDC)/sum(WETH); blended references are USD-weighted across legs.

DECLARE start_date DATE DEFAULT DATE '2026-04-21';
DECLARE end_date   DATE DEFAULT DATE '2026-05-20';
DECLARE start_ts   TIMESTAMP DEFAULT TIMESTAMP '2026-04-21 00:00:00';
DECLARE end_ts     TIMESTAMP DEFAULT TIMESTAMP '2026-05-20 23:59:59';
-- 1-hour pad so trades near the window start can look back one 12s step.
DECLARE grid_start TIMESTAMP DEFAULT TIMESTAMP '2026-04-20 23:00:00';
-- MetaMask Swaps fee collector (87.5 bps convenience fee).
DECLARE mm_fee_addr STRING DEFAULT '0xf326e4de8f66a0bdc0970b79e0924e33c79f1915';

WITH constants AS (
  SELECT
    '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2' AS weth,
    '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48' AS usdc,
    '0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640' AS pool_5bp
),
-- Transactions that paid the MetaMask 87.5 bps fee (transfer to the fee collector).
-- This IS the strict-retail filter — it replaces the old 19-router cohort join.
mm_fee_txs AS (
  SELECT DISTINCT LOWER(TRANSACTION_HASH) AS tx_hash
  FROM `uniswap-allium.ethereum.assets_erc20_token_transfers`
  WHERE DATE(BLOCK_TIMESTAMP) BETWEEN start_date AND end_date
    AND LOWER(TO_ADDRESS) = mm_fee_addr
),
-- V3 swap events restricted to WETH/USDC pools (any orientation), excluding the 5bp pool.
v3_swaps AS (
  SELECT
    BLOCK_NUMBER                            AS block_number,
    BLOCK_TIMESTAMP                         AS block_timestamp,
    LOWER(TRANSACTION_HASH)                 AS tx_hash,
    LOG_INDEX                               AS log_index,
    LOWER(LIQUIDITY_POOL_ADDRESS)           AS pool,
    LOWER(TOKEN0_ADDRESS)                   AS token0_addr,
    LOWER(TOKEN1_ADDRESS)                   AS token1_addr,
    TOKEN0_AMOUNT                           AS t0_amt,
    TOKEN1_AMOUNT                           AS t1_amt,
    USD_AMOUNT                              AS leg_usd,
    CAST(SQRT_PRICE_X96 AS FLOAT64)         AS sqrt_post,
    LAG(CAST(SQRT_PRICE_X96 AS FLOAT64)) OVER (
      PARTITION BY LOWER(LIQUIDITY_POOL_ADDRESS)
      ORDER BY BLOCK_NUMBER, LOG_INDEX
    )                                       AS sqrt_pre,
    LOWER(TRANSACTION_TO_ADDRESS)           AS tx_to,
    FEE                                     AS fee_tier
  FROM `uniswap-allium.ethereum.dex_uniswap_v3_events`
  WHERE BLOCK_TIMESTAMP BETWEEN start_ts AND end_ts
    AND EVENT = 'swap'
    AND LOWER(LIQUIDITY_POOL_ADDRESS) != (SELECT pool_5bp FROM constants)
    AND (
      (LOWER(TOKEN0_ADDRESS) = (SELECT usdc FROM constants) AND LOWER(TOKEN1_ADDRESS) = (SELECT weth FROM constants))
      OR
      (LOWER(TOKEN0_ADDRESS) = (SELECT weth FROM constants) AND LOWER(TOKEN1_ADDRESS) = (SELECT usdc FROM constants))
    )
),
-- Orient each leg into canonical units (WETH amount, USDC amount, side, pool_mid_pre USDC/WETH)
oriented AS (
  SELECT
    s.block_number,
    s.block_timestamp,
    s.tx_hash,
    s.log_index,
    s.pool,
    s.fee_tier,
    s.leg_usd,
    s.tx_to,
    CASE
      WHEN s.token0_addr = (SELECT usdc FROM constants) AND s.t0_amt > 0 THEN 1
      WHEN s.token0_addr = (SELECT usdc FROM constants) AND s.t0_amt < 0 THEN -1
      WHEN s.token0_addr = (SELECT weth FROM constants) AND s.t0_amt < 0 THEN 1
      WHEN s.token0_addr = (SELECT weth FROM constants) AND s.t0_amt > 0 THEN -1
      ELSE NULL
    END AS side,
    CASE
      WHEN s.token0_addr = (SELECT weth FROM constants) THEN ABS(s.t0_amt)
      WHEN s.token0_addr = (SELECT usdc FROM constants) THEN ABS(s.t1_amt)
    END AS weth_amt,
    CASE
      WHEN s.token0_addr = (SELECT usdc FROM constants) THEN ABS(s.t0_amt)
      WHEN s.token0_addr = (SELECT weth FROM constants) THEN ABS(s.t1_amt)
    END AS usdc_amt,
    CASE
      WHEN s.sqrt_pre IS NULL THEN NULL
      WHEN s.token0_addr = (SELECT usdc FROM constants) THEN
        SAFE_DIVIDE(POW(10.0, 12), POW(s.sqrt_pre / POW(2.0, 96), 2))
      WHEN s.token0_addr = (SELECT weth FROM constants) THEN
        POW(s.sqrt_pre / POW(2.0, 96), 2) * POW(10.0, 12)
    END AS pool_mid_pre_usdc_per_weth,
    s.sqrt_pre,
    s.sqrt_post
  FROM v3_swaps AS s
),
-- Filter legs: keep MetaMask-fee txs, with valid amounts and pool_mid_pre
legs AS (
  SELECT
    o.tx_hash,
    o.block_number,
    o.block_timestamp,
    o.log_index,
    o.pool,
    o.fee_tier,
    o.leg_usd,
    o.side,
    o.weth_amt,
    o.usdc_amt,
    o.pool_mid_pre_usdc_per_weth
  FROM oriented AS o
  JOIN mm_fee_txs AS m
    ON o.tx_hash = m.tx_hash
  WHERE o.side IS NOT NULL
    AND o.weth_amt > 0
    AND o.usdc_amt > 0
    AND o.leg_usd IS NOT NULL AND o.leg_usd > 0
    AND o.pool_mid_pre_usdc_per_weth IS NOT NULL
    AND o.pool_mid_pre_usdc_per_weth > 0
),
-- All WETH/USDC markout_prod rows in the (padded) window: source of BOTH the
-- per-swap contemporaneous fair AND the reconstructed 12s Binance-mid grid.
mk_weth_usdc AS (
  SELECT
    LOWER(transaction_hash) AS tx_hash,
    log_index,
    block_timestamp         AS mk_ts,
    benchmark,
    LOWER(token_sold_address)   AS sold_addr,
    LOWER(token_bought_address) AS bought_addr
  FROM `uniswap-labs.research.markout_prod`
  WHERE chain = 'ethereum'
    AND block_date BETWEEN DATE_SUB(start_date, INTERVAL 1 DAY) AND end_date
    AND benchmark IS NOT NULL
    AND (
      (LOWER(token_sold_address)   = (SELECT weth FROM constants)
        AND LOWER(token_bought_address) = (SELECT usdc FROM constants))
      OR
      (LOWER(token_sold_address)   = (SELECT usdc FROM constants)
        AND LOWER(token_bought_address) = (SELECT weth FROM constants))
    )
),
mk_oriented AS (
  SELECT
    tx_hash,
    log_index,
    mk_ts,
    CASE
      WHEN sold_addr = (SELECT weth FROM constants) AND bought_addr = (SELECT usdc FROM constants) THEN benchmark
      WHEN sold_addr = (SELECT usdc FROM constants) AND bought_addr = (SELECT weth FROM constants) THEN SAFE_DIVIDE(1.0, benchmark)
      ELSE NULL
    END AS bench_usdc_per_weth
  FROM mk_weth_usdc
),
markout_oriented AS (
  SELECT tx_hash, log_index, bench_usdc_per_weth AS leg_benchmark_usdc_per_weth
  FROM mk_oriented
),
fair_bucket_obs AS (
  SELECT
    TIMESTAMP_SECONDS(DIV(UNIX_SECONDS(mk_ts), 12) * 12) AS bucket_ts,
    bench_usdc_per_weth                                   AS mid
  FROM mk_oriented
  WHERE bench_usdc_per_weth IS NOT NULL AND bench_usdc_per_weth > 0
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY TIMESTAMP_SECONDS(DIV(UNIX_SECONDS(mk_ts), 12) * 12)
    ORDER BY mk_ts DESC
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
  LEFT JOIN fair_bucket_obs AS o
    ON o.bucket_ts = g.grid_ts
),
legs_with_fair AS (
  SELECT
    l.tx_hash, l.block_number, l.block_timestamp, l.log_index, l.pool, l.fee_tier,
    l.leg_usd, l.side, l.weth_amt, l.usdc_amt,
    l.pool_mid_pre_usdc_per_weth,
    mo.leg_benchmark_usdc_per_weth
  FROM legs AS l
  LEFT JOIN markout_oriented AS mo
    USING (tx_hash, log_index)
),
agg AS (
  SELECT
    tx_hash,
    MIN(block_timestamp)                                                AS block_timestamp,
    COUNT(*)                                                            AS n_legs,
    SUM(leg_usd)                                                        AS size_usd,
    SUM(weth_amt)                                                       AS sum_weth,
    SUM(usdc_amt)                                                       AS sum_usdc,
    SUM(CAST(side AS FLOAT64) * weth_amt)                               AS net_signed_weth,
    SAFE_DIVIDE(SUM(pool_mid_pre_usdc_per_weth * leg_usd), SUM(leg_usd))  AS pool_mid_pre_blended,
    SAFE_DIVIDE(
      SUM(IF(leg_benchmark_usdc_per_weth IS NOT NULL, leg_benchmark_usdc_per_weth * leg_usd, 0.0)),
      SUM(IF(leg_benchmark_usdc_per_weth IS NOT NULL, leg_usd, 0.0))
    )                                                                   AS fair_price_blended,
    SUM(IF(leg_benchmark_usdc_per_weth IS NOT NULL, leg_usd, 0.0))      AS fair_usd_covered,
    COUNT(DISTINCT pool)                                                AS n_distinct_pools,
    COUNT(DISTINCT side)                                                AS n_distinct_sides,
    ANY_VALUE(fee_tier)                                                 AS any_fee_tier
  FROM legs_with_fair
  GROUP BY tx_hash
),
filtered AS (
  SELECT
    tx_hash,
    block_timestamp,
    n_legs,
    n_distinct_pools,
    any_fee_tier,
    size_usd,
    SAFE_DIVIDE(sum_usdc, sum_weth)                AS effective_exec_usdc_per_weth,
    pool_mid_pre_blended,
    fair_price_blended,
    SAFE_DIVIDE(fair_usd_covered, size_usd)        AS fair_coverage_frac,
    CASE
      WHEN net_signed_weth > 0 THEN 1
      WHEN net_signed_weth < 0 THEN -1
      ELSE NULL
    END                                            AS side,
    n_distinct_sides
  FROM agg
  WHERE n_legs >= 1
    AND size_usd > 0
    AND sum_weth > 0
    AND sum_usdc > 0
    AND pool_mid_pre_blended IS NOT NULL
    AND pool_mid_pre_blended > 0
),
with_lag AS (
  SELECT
    f.*,
    gf.fair_mid AS fair_lag_price
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
  side,
  n_distinct_sides,
  any_fee_tier AS fee_tier_sample,
  effective_exec_usdc_per_weth,
  pool_mid_pre_blended,
  fair_price_blended,
  fair_coverage_frac,
  fair_lag_price,
  10000.0 * (effective_exec_usdc_per_weth - pool_mid_pre_blended) * CAST(side AS FLOAT64)
    / pool_mid_pre_blended AS observed_spread_pool_bps,
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
