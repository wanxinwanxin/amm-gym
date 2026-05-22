-- Build the non-5bp price-impact sample for router-routed WETH/USDC transactions.
--
-- Purpose:
--   For each interface-routed (router-originated) WETH/USDC tx in the 7-day
--   window 2026-05-14..2026-05-20, aggregate the non-5bp portion of the
--   tx (all non-5bp legs combined) into a single (size_usd,
--   observed_spread_bps) data point.
--
--   These points become the calibration sample for the hypothetical
--   constant-product "other pool" that the simulator will treat as the
--   aggregated rest-of-world liquidity.
--
-- Data source:
--   `uniswap-labs.research.markout_prod` contains per-leg Binance benchmark
--   midprices (trade-block alignment, NOT next-block: the `benchmark` column
--   is the Binance mid at the swap's block timestamp).
--
-- Filters:
--   - WETH = 0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2
--   - USDC = 0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48
--   - 7-day window: 2026-05-14..2026-05-20 (matches reports/5bp_markout_investigation.md
--     7d row, the held-out validation window for the calibration)
--   - tx_to ∈ 19 router list from
--     analysis/weth_usdc_90d/sql/router_parent_order_size_windows.sql
--   - Exclude legs where pool = 5bp pool (0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640)
--   - Drop tx with mixed-direction non-5bp legs
--   - Drop tx with zero or missing non-5bp USD volume
--
-- Per-tx aggregation:
--   - side: +1 if net non-5bp ETH bought > 0, -1 if net ETH sold > 0
--   - size_usd: sum(usd_amount) over non-5bp legs
--   - effective_exec_price = (sum USDC across non-5bp legs) / (sum WETH across non-5bp legs)
--   - fair_price_at_block = USD-volume-weighted average of leg benchmark prices
--     (each leg's benchmark is the Binance mid at the trade block, already
--     converted to the same buy-per-sell currency convention as the leg's
--     executed price; we re-orient all legs to USDC/WETH using leg direction)
--   - observed_spread_bps = 10_000 * (effective_exec - fair) * side / fair
--     (SIGNED — negative spreads are kept, no abs())

WITH
constants AS (
  SELECT
    '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2' AS weth,
    '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48' AS usdc,
    '0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640' AS pool_5bp
),
routers AS (
  SELECT addr FROM UNNEST([
    '0xef1c6e67703c7bd7107eed8303fbe6ec2554bf6b',
    '0x3fc91a3afd70395cd496c647d5a6cc9d4b2b7fad',
    '0x66a9893cc07d91d95644aedd05d03f95e1dba8af',
    '0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45',
    '0xe592427a0aece92de3edee1f18e0157c05861564',
    '0x7a250d5630b4cf539739df2c5dacb4c659f2488d',
    '0x1111111254fb6c44bac0bed2854e76f90643097d',
    '0x1111111254eeb25477b68fb85ed929f73a960582',
    '0x111111125421ca6dc452d289314280a0f8842a65',
    '0xdef1c0ded9bec7f1a1670819833240f027b25eff',
    '0x0000000000001ff3684f28c67538d4d072c22734',
    '0x9008d19f58aabd9ed0d60971565aa8510560ab41',
    '0xdef171fe48cf0115b1d80b88dc8eab59176fee57',
    '0x6a000f20005980200259b80c5102003040001068',
    '0x6131b5fae19ea4f9d964eac0408e4408b66337b5',
    '0x617dee16b86534a5d792a4d7a62fb491b544111e',
    '0xbdb3ba9ffe392549e1f8658dd2630c141fdf47b6',
    '0x51c72848c68a965f66fa7a88855f9f7784502a7f',
    '0x881d40237659c251811cec9c364ef91dc08d300c'
  ]) AS addr
),
raw_legs AS (
  SELECT
    LOWER(m.transaction_hash) AS tx_hash,
    m.block_timestamp,
    LOWER(m.liquidity_pool_address) AS pool,
    LOWER(m.token_sold_address) AS sold_addr,
    LOWER(m.token_bought_address) AS bought_addr,
    m.token_sold_amount AS sold_amt,
    m.token_bought_amount AS bought_amt,
    -- Pre-resolved USD value (mirrors COALESCE pattern in router_parent_order_size_windows.sql)
    COALESCE(
      NULLIF(m.usd_amount, 0.0),
      NULLIF(m.usd_sold_amount, 0.0),
      NULLIF(m.usd_bought_amount, 0.0)
    ) AS leg_usd,
    -- benchmark is Binance mid at the swap's block, in token_bought per token_sold units.
    -- We re-orient to USDC-per-WETH below.
    m.benchmark AS leg_benchmark,
    m.executed_price_with_binance AS leg_exec_price,
    -- ETH side from token direction
    CASE
      WHEN LOWER(m.token_sold_address) = (SELECT usdc FROM constants)
       AND LOWER(m.token_bought_address) = (SELECT weth FROM constants) THEN 1
      WHEN LOWER(m.token_sold_address) = (SELECT weth FROM constants)
       AND LOWER(m.token_bought_address) = (SELECT usdc FROM constants) THEN -1
      ELSE NULL
    END AS leg_side
  FROM `uniswap-labs.research.markout_prod` AS m
  JOIN routers AS r
    ON LOWER(m.transaction_to_address) = r.addr
  WHERE m.chain = 'ethereum'
    AND m.block_date BETWEEN DATE '2026-05-14' AND DATE '2026-05-20'
    AND (
      (LOWER(m.token_sold_address) = (SELECT weth FROM constants) AND LOWER(m.token_bought_address) = (SELECT usdc FROM constants))
      OR
      (LOWER(m.token_sold_address) = (SELECT usdc FROM constants) AND LOWER(m.token_bought_address) = (SELECT weth FROM constants))
    )
    AND m.benchmark IS NOT NULL
    AND m.executed_price_with_binance IS NOT NULL
),
-- Convert each leg to canonical USDC-per-WETH units for both executed and benchmark.
-- For sell_eth legs (sold WETH, bought USDC): exec_usdc_per_weth = bought_amt / sold_amt = leg_exec_price (already USDC/WETH)
-- For buy_eth legs (sold USDC, bought WETH): leg_exec_price is bought / sold = WETH/USDC, so usdc_per_weth = 1 / leg_exec_price
-- Same logic for benchmark.
oriented_legs AS (
  SELECT
    tx_hash,
    block_timestamp,
    pool,
    leg_usd,
    leg_side,
    -- WETH amount (always positive)
    CASE leg_side WHEN 1 THEN bought_amt WHEN -1 THEN sold_amt END AS weth_amt,
    -- USDC amount (always positive)
    CASE leg_side WHEN 1 THEN sold_amt WHEN -1 THEN bought_amt END AS usdc_amt,
    -- Executed USDC/WETH
    CASE
      WHEN leg_side = -1 THEN leg_exec_price  -- sold WETH -> got USDC, exec is USDC/WETH already
      WHEN leg_side = 1 THEN 1.0 / NULLIF(leg_exec_price, 0)  -- sold USDC -> got WETH, exec is WETH/USDC, invert
    END AS leg_exec_usdc_per_weth,
    -- Benchmark USDC/WETH (Binance mid at trade block)
    CASE
      WHEN leg_side = -1 THEN leg_benchmark
      WHEN leg_side = 1 THEN 1.0 / NULLIF(leg_benchmark, 0)
    END AS leg_benchmark_usdc_per_weth,
    -- 5bp/non-5bp flag
    (pool != (SELECT pool_5bp FROM constants)) AS is_non5bp
  FROM raw_legs
  WHERE leg_side IS NOT NULL
    AND leg_usd IS NOT NULL
    AND leg_usd > 0
    AND bought_amt > 0
    AND sold_amt > 0
),
-- Per-tx aggregation, restricted to NON-5bp legs only
non5bp_agg AS (
  SELECT
    tx_hash,
    MIN(block_timestamp) AS block_timestamp,
    COUNT(*) AS n_non5bp_legs,
    SUM(leg_usd) AS sum_leg_usd,
    SUM(weth_amt) AS sum_weth,
    SUM(usdc_amt) AS sum_usdc,
    -- Net signed ETH across non-5bp legs (in WETH units, positive = net buy, negative = net sell)
    SUM(CAST(leg_side AS FLOAT64) * weth_amt) AS net_signed_weth,
    -- USD-weighted average benchmark USDC/WETH (fair price at block)
    SAFE_DIVIDE(SUM(leg_benchmark_usdc_per_weth * leg_usd), SUM(leg_usd)) AS fair_price_at_block,
    -- For diagnostic: distinct directions
    COUNT(DISTINCT leg_side) AS n_distinct_sides,
    -- Sum signed for tie-breaking when distinct sides > 1
    SUM(leg_side) AS sum_side_raw
  FROM oriented_legs
  WHERE is_non5bp
  GROUP BY tx_hash
),
-- Apply per-tx constraints and compute observed_spread_bps
filtered AS (
  SELECT
    tx_hash,
    block_timestamp,
    n_non5bp_legs,
    sum_leg_usd AS size_usd,
    fair_price_at_block,
    -- side: sign of net_signed_weth; drop ties
    CASE
      WHEN net_signed_weth > 0 THEN 1
      WHEN net_signed_weth < 0 THEN -1
      ELSE NULL
    END AS side,
    -- effective_exec_price (USDC per WETH) = sum(USDC) / sum(WETH)
    SAFE_DIVIDE(sum_usdc, sum_weth) AS effective_exec_price,
    n_distinct_sides
  FROM non5bp_agg
  WHERE n_non5bp_legs >= 1
    AND sum_leg_usd > 0
    AND sum_weth > 0
    AND sum_usdc > 0
    AND fair_price_at_block IS NOT NULL
    AND fair_price_at_block > 0
)
SELECT
  tx_hash,
  block_timestamp,
  size_usd,
  10000.0 * (effective_exec_price - fair_price_at_block) * CAST(side AS FLOAT64) / fair_price_at_block AS observed_spread_bps,
  n_non5bp_legs,
  side,
  n_distinct_sides
FROM filtered
WHERE side IS NOT NULL
ORDER BY block_timestamp;
