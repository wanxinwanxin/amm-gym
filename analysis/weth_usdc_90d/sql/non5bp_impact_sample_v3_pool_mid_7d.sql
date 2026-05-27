-- Build the non-5bp price-impact sample for router-routed WETH/USDC transactions,
-- emitting BOTH references side-by-side:
--   (a) observed_spread_pool_bps    -- referenced to pre-trade POOL mid (V3 sqrtPriceX96)
--   (b) observed_spread_fair_bps    -- referenced to FAIR price (Binance benchmark at block)
--
-- Rationale:
--   V2/V3 spread formula is in pool's-own-frame: exec_price vs pool_mid_pre.
--   Calibrating against pool-mid removes drift between AMM mid and Binance mid,
--   which is the most likely source of the negative-spread tail we observed
--   in the previous (fair-referenced) sample.
--
-- Scope:
--   - Universe: Uniswap V3 swap events on WETH/USDC pools, excluding the 5bp pool.
--     V3 is ~90% of non-5bp router-routed WETH/USDC volume (probed 2026-05-14
--     30-min slice). V4 (~6%) and V2/Fluid/Curve (~4%) are dropped — they would
--     require separate pool-mid reconstruction and are not material for the
--     diagnostic comparison.
--   - 7-day window: 2026-05-14..2026-05-20 (same as previous fair-only sample).
--   - tx_to ∈ 19 router list (matches non5bp_impact_sample_7d.sql).
--
-- Pool-mid computation (V3):
--   sqrtPriceX96 emitted by V3 Swap log is POST-swap.
--   pool_mid_PRE for swap N = pool_mid_POST for the previous swap on same pool
--     (Mint/Burn between swaps don't change sqrtPriceX96).
--   We use LAG(sqrt_price_X96) OVER (PARTITION BY pool ORDER BY block_number, log_index).
--   First swap per pool in the window has no LAG and is dropped.
--   Decimal scaling (USDC=6, WETH=18):
--     If TOKEN0=USDC, TOKEN1=WETH:
--       price_raw = (sqrt/2^96)^2 = WETH_raw / USDC_raw
--       USDC_human / WETH_human = (10^(18-6)) / price_raw = 10^12 / price_raw
--     If TOKEN0=WETH, TOKEN1=USDC:
--       price_raw = (sqrt/2^96)^2 = USDC_raw / WETH_raw
--       USDC_human / WETH_human = price_raw * 10^12
--
-- Per-tx aggregation:
--   - All non-5bp V3 legs in the tx are rolled up into a single (size_usd, spread) point.
--   - side: sign of net signed WETH across legs; tx with mixed directions are dropped.
--   - effective_exec = sum(USDC) / sum(WETH)
--   - pool_mid_pre_blended = USD-weighted average of leg pool_mid_pre  (USDC/WETH)
--   - fair_price_blended  = USD-weighted average of leg benchmark      (USDC/WETH)
--   - observed_spread_pool_bps = 1e4 * (effective_exec - pool_mid_pre_blended) * side / pool_mid_pre_blended
--   - observed_spread_fair_bps = 1e4 * (effective_exec - fair_price_blended)   * side / fair_price_blended
--     (matches previous query's convention — signed, no abs())

DECLARE start_date DATE DEFAULT DATE '2026-05-14';
DECLARE end_date   DATE DEFAULT DATE '2026-05-20';
DECLARE start_ts   TIMESTAMP DEFAULT TIMESTAMP '2026-05-14 00:00:00';
DECLARE end_ts     TIMESTAMP DEFAULT TIMESTAMP '2026-05-20 23:59:59';

WITH constants AS (
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
    '0xdef171fe48cf0115b1d80b88dc8eab59176fee57',
    '0x6a000f20005980200259b80c5102003040001068',
    '0x6131b5fae19ea4f9d964eac0408e4408b66337b5',
    '0x617dee16b86534a5d792a4d7a62fb491b544111e',
    '0x881d40237659c251811cec9c364ef91dc08d300c',
    '0xcf5540fffcdc3d510b18bfca6d2b9987b0772559',
    '0x6352a56caadc4f1e25cd6c75970fa768a3304e64',
    '0x1231deb6f5749ef6ce6943a275a1d3e7486f4eae'
  ]) AS addr
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
    -- side: +1 if buyer bought WETH (pool sent WETH out, received USDC)
    CASE
      WHEN s.token0_addr = (SELECT usdc FROM constants) AND s.t0_amt > 0 THEN 1   -- pool received USDC
      WHEN s.token0_addr = (SELECT usdc FROM constants) AND s.t0_amt < 0 THEN -1  -- pool sent USDC
      WHEN s.token0_addr = (SELECT weth FROM constants) AND s.t0_amt < 0 THEN 1   -- pool sent WETH
      WHEN s.token0_addr = (SELECT weth FROM constants) AND s.t0_amt > 0 THEN -1  -- pool received WETH
      ELSE NULL
    END AS side,
    -- |WETH amount|
    CASE
      WHEN s.token0_addr = (SELECT weth FROM constants) THEN ABS(s.t0_amt)
      WHEN s.token0_addr = (SELECT usdc FROM constants) THEN ABS(s.t1_amt)
    END AS weth_amt,
    -- |USDC amount|
    CASE
      WHEN s.token0_addr = (SELECT usdc FROM constants) THEN ABS(s.t0_amt)
      WHEN s.token0_addr = (SELECT weth FROM constants) THEN ABS(s.t1_amt)
    END AS usdc_amt,
    -- pool_mid_pre in USDC/WETH human units; NULL for first swap per pool in window
    CASE
      WHEN s.sqrt_pre IS NULL THEN NULL
      WHEN s.token0_addr = (SELECT usdc FROM constants) THEN
        -- (sqrt/2^96)^2 = WETH_raw/USDC_raw; USDC_human/WETH_human = 10^12 / that
        SAFE_DIVIDE(POW(10.0, 12), POW(s.sqrt_pre / POW(2.0, 96), 2))
      WHEN s.token0_addr = (SELECT weth FROM constants) THEN
        -- (sqrt/2^96)^2 = USDC_raw/WETH_raw; USDC_human/WETH_human = that * 10^12
        POW(s.sqrt_pre / POW(2.0, 96), 2) * POW(10.0, 12)
    END AS pool_mid_pre_usdc_per_weth,
    s.sqrt_pre,
    s.sqrt_post
  FROM v3_swaps AS s
),
-- Filter legs: keep router-routed, with valid amounts and pool_mid_pre
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
  JOIN routers AS r
    ON o.tx_to = r.addr
  WHERE o.side IS NOT NULL
    AND o.weth_amt > 0
    AND o.usdc_amt > 0
    AND o.leg_usd IS NOT NULL AND o.leg_usd > 0
    AND o.pool_mid_pre_usdc_per_weth IS NOT NULL
    AND o.pool_mid_pre_usdc_per_weth > 0
),
-- Attach Binance benchmark per (tx_hash, log_index) from markout_prod for side-by-side fair reference
markout AS (
  SELECT
    LOWER(transaction_hash) AS tx_hash,
    log_index,
    benchmark AS leg_benchmark,
    LOWER(token_sold_address)   AS sold_addr,
    LOWER(token_bought_address) AS bought_addr
  FROM `uniswap-labs.research.markout_prod`
  WHERE chain = 'ethereum'
    AND block_date BETWEEN start_date AND end_date
    AND benchmark IS NOT NULL
),
-- Orient markout benchmark to USDC/WETH per leg (same convention as previous query)
markout_oriented AS (
  SELECT
    m.tx_hash,
    m.log_index,
    -- leg_benchmark is in (bought per sold) units; orient to USDC/WETH
    CASE
      -- sold WETH, bought USDC: benchmark is USDC/WETH already
      WHEN m.sold_addr = (SELECT weth FROM constants) AND m.bought_addr = (SELECT usdc FROM constants) THEN m.leg_benchmark
      -- sold USDC, bought WETH: benchmark is WETH/USDC, invert
      WHEN m.sold_addr = (SELECT usdc FROM constants) AND m.bought_addr = (SELECT weth FROM constants) THEN SAFE_DIVIDE(1.0, m.leg_benchmark)
      ELSE NULL
    END AS leg_benchmark_usdc_per_weth
  FROM markout AS m
),
-- Join V3 swap legs with markout benchmark
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
-- Per-tx aggregation: sum legs, USD-weighted blended references, signed net WETH for side
agg AS (
  SELECT
    tx_hash,
    MIN(block_timestamp)                                                AS block_timestamp,
    COUNT(*)                                                            AS n_legs,
    SUM(leg_usd)                                                        AS size_usd,
    SUM(weth_amt)                                                       AS sum_weth,
    SUM(usdc_amt)                                                       AS sum_usdc,
    SUM(CAST(side AS FLOAT64) * weth_amt)                               AS net_signed_weth,
    -- USD-weighted blended pool_mid_pre (USDC/WETH)
    SAFE_DIVIDE(SUM(pool_mid_pre_usdc_per_weth * leg_usd), SUM(leg_usd))  AS pool_mid_pre_blended,
    -- USD-weighted blended fair benchmark (USDC/WETH), only over legs that joined markout
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
  10000.0 * (effective_exec_usdc_per_weth - pool_mid_pre_blended) * CAST(side AS FLOAT64)
    / pool_mid_pre_blended AS observed_spread_pool_bps,
  CASE
    WHEN fair_price_blended IS NULL OR fair_price_blended = 0 THEN NULL
    ELSE 10000.0 * (effective_exec_usdc_per_weth - fair_price_blended) * CAST(side AS FLOAT64) / fair_price_blended
  END AS observed_spread_fair_bps
FROM filtered
WHERE side IS NOT NULL
ORDER BY block_timestamp;
