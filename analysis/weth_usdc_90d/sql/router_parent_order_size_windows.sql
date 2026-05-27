-- Parent-order USD size distributions for router-originated WETH/USDC flow.
--
-- Purpose:
--   Estimate empirical retail order sizes for WETH/USDC using router-originated
--   dex_trades rows, comparing the last 6 months, 1 year, and 2 years.
--
-- Parent-order heuristic:
--   1. Keep WETH/USDC swap-event rows whose transaction_to_address is a known
--      router.
--   2. Group candidate legs by transaction hash.
--   3. Infer ETH side from token direction:
--        USDC -> WETH = buy_eth
--        WETH -> USDC = sell_eth
--   4. Strict mode keeps only same-side, non-batch-router transactions with
--      1-4 WETH/USDC legs and positive USD notional. Size is sum(USD_AMOUNT)
--      across WETH/USDC legs in the transaction.
--   5. Loose mode keeps same-side positive-USD transactions without the batch
--      router and leg-count filters. It is included only as a sensitivity check.
--
-- Output:
--   Long percentile table with duplicated diagnostics per window/mode/side_group.
--   pct is in [0, 100] at 0.01 percentile increments (10000-bucket approx.
--   quantiles). The denser tail is necessary because the simulator uses
--   linear interpolation between quantile points to sample order sizes,
--   and the original 0.1-percentile grid had a single bin (p99.9 → p100)
--   spanning ~$217K to ~$7M, which made the sim over-sample $1M+ trades by
--   ~17×. Denser anchors at p99.91…p99.99 close that gap.

WITH
windows AS (
  SELECT '6m' AS window_name,
         DATE_SUB(CURRENT_DATE(), INTERVAL 6 MONTH) AS start_date,
         DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY) AS end_date
  UNION ALL
  SELECT '1y',
         DATE_SUB(CURRENT_DATE(), INTERVAL 1 YEAR),
         DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
  UNION ALL
  SELECT '2y',
         DATE_SUB(CURRENT_DATE(), INTERVAL 2 YEAR),
         DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
),
routers AS (
  SELECT * FROM UNNEST([
    STRUCT('0xef1c6e67703c7bd7107eed8303fbe6ec2554bf6b' AS addr, 'Uniswap Universal Router v1' AS router_name, FALSE AS is_batch_router),
    STRUCT('0x3fc91a3afd70395cd496c647d5a6cc9d4b2b7fad' AS addr, 'Uniswap Universal Router v1.2' AS router_name, FALSE AS is_batch_router),
    STRUCT('0x66a9893cc07d91d95644aedd05d03f95e1dba8af' AS addr, 'Uniswap Universal Router v2' AS router_name, FALSE AS is_batch_router),
    STRUCT('0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45' AS addr, 'Uniswap SwapRouter02' AS router_name, FALSE AS is_batch_router),
    STRUCT('0xe592427a0aece92de3edee1f18e0157c05861564' AS addr, 'Uniswap SwapRouter v3' AS router_name, FALSE AS is_batch_router),
    STRUCT('0x7a250d5630b4cf539739df2c5dacb4c659f2488d' AS addr, 'Uniswap V2Router02' AS router_name, FALSE AS is_batch_router),
    STRUCT('0x1111111254fb6c44bac0bed2854e76f90643097d' AS addr, '1inch v4' AS router_name, FALSE AS is_batch_router),
    STRUCT('0x1111111254eeb25477b68fb85ed929f73a960582' AS addr, '1inch v5' AS router_name, FALSE AS is_batch_router),
    STRUCT('0x111111125421ca6dc452d289314280a0f8842a65' AS addr, '1inch v6' AS router_name, FALSE AS is_batch_router),
    STRUCT('0xdef1c0ded9bec7f1a1670819833240f027b25eff' AS addr, '0x ExchangeProxy' AS router_name, FALSE AS is_batch_router),
    STRUCT('0x0000000000001ff3684f28c67538d4d072c22734' AS addr, '0x AllowanceHolder' AS router_name, FALSE AS is_batch_router),
    STRUCT('0xdef171fe48cf0115b1d80b88dc8eab59176fee57' AS addr, 'Paraswap v5' AS router_name, FALSE AS is_batch_router),
    STRUCT('0x6a000f20005980200259b80c5102003040001068' AS addr, 'Paraswap v6' AS router_name, FALSE AS is_batch_router),
    STRUCT('0x6131b5fae19ea4f9d964eac0408e4408b66337b5' AS addr, 'Kyber MetaAggregationRouter v2' AS router_name, FALSE AS is_batch_router),
    STRUCT('0x617dee16b86534a5d792a4d7a62fb491b544111e' AS addr, 'Kyber AggregationRouter classic' AS router_name, FALSE AS is_batch_router),
    STRUCT('0x881d40237659c251811cec9c364ef91dc08d300c' AS addr, 'MetaMask Swap router' AS router_name, FALSE AS is_batch_router),
    STRUCT('0xcf5540fffcdc3d510b18bfca6d2b9987b0772559' AS addr, 'Odos Router V2' AS router_name, FALSE AS is_batch_router),
    STRUCT('0x6352a56caadc4f1e25cd6c75970fa768a3304e64' AS addr, 'OpenOcean Exchange Proxy' AS router_name, FALSE AS is_batch_router),
    STRUCT('0x1231deb6f5749ef6ce6943a275a1d3e7486f4eae' AS addr, 'LI.FI Diamond' AS router_name, FALSE AS is_batch_router)
  ])
),
constants AS (
  SELECT
    '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2' AS weth,
    '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48' AS usdc
),
raw_router_legs AS (
  SELECT
    DATE(t.BLOCK_TIMESTAMP) AS block_date,
    t.BLOCK_TIMESTAMP,
    CAST(t.BLOCK_NUMBER AS INT64) AS block_number,
    CAST(t.LOG_INDEX AS INT64) AS log_index,
    LOWER(t.TRANSACTION_HASH) AS transaction_hash,
    LOWER(t.TRANSACTION_TO_ADDRESS) AS router_address,
    r.router_name,
    r.is_batch_router,
    LOWER(t.LIQUIDITY_POOL_ADDRESS) AS liquidity_pool_address,
    LOWER(t.TOKEN_SOLD_ADDRESS) AS token_sold_address,
    LOWER(t.TOKEN_BOUGHT_ADDRESS) AS token_bought_address,
    CASE
      WHEN LOWER(t.TOKEN_SOLD_ADDRESS) = c.usdc
       AND LOWER(t.TOKEN_BOUGHT_ADDRESS) = c.weth THEN 'buy_eth'
      WHEN LOWER(t.TOKEN_SOLD_ADDRESS) = c.weth
       AND LOWER(t.TOKEN_BOUGHT_ADDRESS) = c.usdc THEN 'sell_eth'
      ELSE NULL
    END AS eth_side,
    COALESCE(
      NULLIF(t.USD_AMOUNT, 0.0),
      NULLIF(t.USD_SOLD_AMOUNT, 0.0),
      NULLIF(t.USD_BOUGHT_AMOUNT, 0.0)
    ) AS leg_usd
  FROM `uniswap-allium.ethereum.dex_trades` AS t
  JOIN routers AS r
    ON LOWER(t.TRANSACTION_TO_ADDRESS) = r.addr
  CROSS JOIN constants AS c
  WHERE t.BLOCK_TIMESTAMP >= TIMESTAMP(DATE_SUB(CURRENT_DATE(), INTERVAL 2 YEAR))
    AND t.BLOCK_TIMESTAMP < TIMESTAMP(CURRENT_DATE())
    AND (
      (LOWER(t.TOKEN_SOLD_ADDRESS) = c.weth AND LOWER(t.TOKEN_BOUGHT_ADDRESS) = c.usdc)
      OR
      (LOWER(t.TOKEN_SOLD_ADDRESS) = c.usdc AND LOWER(t.TOKEN_BOUGHT_ADDRESS) = c.weth)
    )
),
windowed_legs AS (
  SELECT
    w.window_name,
    w.start_date,
    w.end_date,
    l.*
  FROM raw_router_legs AS l
  JOIN windows AS w
    ON l.block_date BETWEEN w.start_date AND w.end_date
),
parent_candidates AS (
  SELECT
    window_name,
    start_date,
    end_date,
    transaction_hash,
    MIN(BLOCK_TIMESTAMP) AS first_seen_at,
    MIN(block_number) AS first_block_number,
    COUNT(*) AS weth_usdc_leg_count,
    COUNT(DISTINCT liquidity_pool_address) AS pool_count,
    COUNT(DISTINCT router_address) AS router_count,
    LOGICAL_OR(is_batch_router) AS has_batch_router,
    MIN(eth_side) AS min_eth_side,
    MAX(eth_side) AS max_eth_side,
    COUNT(DISTINCT eth_side) AS side_count,
    SUM(IFNULL(leg_usd, 0.0)) AS parent_usd,
    COUNTIF(leg_usd IS NULL OR leg_usd <= 0.0) AS missing_or_nonpositive_usd_legs,
    ARRAY_AGG(DISTINCT router_name IGNORE NULLS ORDER BY router_name LIMIT 5) AS router_names
  FROM windowed_legs
  GROUP BY window_name, start_date, end_date, transaction_hash
),
classified AS (
  SELECT
    *,
    CASE
      WHEN side_count != 1 THEN 'mixed_eth_direction'
      WHEN parent_usd <= 0.0 OR missing_or_nonpositive_usd_legs > 0 THEN 'missing_or_nonpositive_usd'
      WHEN has_batch_router THEN 'batch_router'
      WHEN weth_usdc_leg_count > 4 THEN 'too_many_weth_usdc_legs'
      ELSE 'accepted_strict'
    END AS strict_status,
    CASE
      WHEN side_count = 1 AND parent_usd > 0.0 THEN 'accepted_loose'
      WHEN side_count != 1 THEN 'mixed_eth_direction'
      ELSE 'missing_or_nonpositive_usd'
    END AS loose_status,
    min_eth_side AS eth_side
  FROM parent_candidates
),
orders AS (
  SELECT
    window_name,
    start_date,
    end_date,
    'strict' AS mode,
    eth_side,
    parent_usd,
    weth_usdc_leg_count
  FROM classified
  WHERE strict_status = 'accepted_strict'
  UNION ALL
  SELECT
    window_name,
    start_date,
    end_date,
    'loose' AS mode,
    eth_side,
    parent_usd,
    weth_usdc_leg_count
  FROM classified
  WHERE loose_status = 'accepted_loose'
),
orders_with_side_groups AS (
  SELECT window_name, start_date, end_date, mode, 'all' AS side_group, parent_usd, weth_usdc_leg_count
  FROM orders
  UNION ALL
  SELECT window_name, start_date, end_date, mode, eth_side AS side_group, parent_usd, weth_usdc_leg_count
  FROM orders
),
diagnostics AS (
  SELECT
    c.window_name,
    c.start_date,
    c.end_date,
    DATE_DIFF(c.end_date, c.start_date, DAY) + 1 AS horizon_days,
    COUNT(*) AS candidate_parent_tx_count,
    SUM(weth_usdc_leg_count) AS raw_weth_usdc_leg_count,
    COUNTIF(strict_status = 'accepted_strict') AS strict_parent_count,
    SUM(IF(strict_status = 'accepted_strict', parent_usd, 0.0)) AS strict_parent_usd,
    COUNTIF(strict_status = 'accepted_strict' AND weth_usdc_leg_count = 1) AS strict_single_leg_parent_count,
    COUNTIF(strict_status = 'accepted_strict' AND weth_usdc_leg_count > 1) AS strict_multi_leg_parent_count,
    COUNTIF(strict_status = 'mixed_eth_direction') AS strict_excluded_mixed_direction_count,
    COUNTIF(strict_status = 'missing_or_nonpositive_usd') AS strict_excluded_missing_usd_count,
    COUNTIF(strict_status = 'batch_router') AS strict_excluded_batch_router_count,
    COUNTIF(strict_status = 'too_many_weth_usdc_legs') AS strict_excluded_many_leg_count,
    COUNTIF(loose_status = 'accepted_loose') AS loose_parent_count,
    SUM(IF(loose_status = 'accepted_loose', parent_usd, 0.0)) AS loose_parent_usd
  FROM classified AS c
  GROUP BY c.window_name, c.start_date, c.end_date
),
distributions AS (
  SELECT
    window_name,
    start_date,
    end_date,
    mode,
    side_group,
    COUNT(*) AS parent_count,
    SUM(parent_usd) AS total_parent_usd,
    AVG(parent_usd) AS mean_parent_usd,
    AVG(IF(weth_usdc_leg_count = 1, 1.0, 0.0)) AS single_leg_share,
    APPROX_QUANTILES(parent_usd, 10000) AS size_usd_quantiles
  FROM orders_with_side_groups
  GROUP BY window_name, start_date, end_date, mode, side_group
),
percentile_rows AS (
  SELECT
    d.window_name,
    d.start_date,
    d.end_date,
    DATE_DIFF(d.end_date, d.start_date, DAY) + 1 AS horizon_days,
    d.mode,
    d.side_group,
    CAST(pct_index AS FLOAT64) / 100.0 AS pct,
    size_usd,
    d.parent_count,
    d.total_parent_usd,
    d.mean_parent_usd,
    d.single_leg_share,
    1.0 - d.single_leg_share AS multi_leg_share
  FROM distributions AS d
  CROSS JOIN UNNEST(d.size_usd_quantiles) AS size_usd WITH OFFSET AS pct_index
)
SELECT
  p.window_name,
  p.start_date,
  p.end_date,
  p.horizon_days,
  p.mode,
  p.side_group,
  p.pct,
  p.size_usd,
  p.parent_count,
  p.total_parent_usd,
  p.mean_parent_usd,
  p.single_leg_share,
  p.multi_leg_share,
  dg.candidate_parent_tx_count,
  dg.raw_weth_usdc_leg_count,
  dg.strict_parent_count,
  dg.strict_parent_usd,
  dg.strict_single_leg_parent_count,
  dg.strict_multi_leg_parent_count,
  dg.strict_excluded_mixed_direction_count,
  dg.strict_excluded_missing_usd_count,
  dg.strict_excluded_batch_router_count,
  dg.strict_excluded_many_leg_count,
  dg.loose_parent_count,
  dg.loose_parent_usd
FROM percentile_rows AS p
JOIN diagnostics AS dg
  USING (window_name, start_date, end_date, horizon_days)
ORDER BY
  CASE p.window_name WHEN '6m' THEN 1 WHEN '1y' THEN 2 WHEN '2y' THEN 3 ELSE 99 END,
  p.mode,
  p.side_group,
  p.pct;
