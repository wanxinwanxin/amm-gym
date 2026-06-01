-- Parent-order USD size distribution for STRICTLY-RETAIL WETH/USDC flow (30d).
--
-- Same cohort as the §4 calibration sample: transactions from a known end-user
-- front-end — Uniswap first-party FE (web / mobile / extension, via the
-- x-request-source surface tag materialized in uniswap-labs.core.swaps) UNION
-- MetaMask Swaps (87.5 bps fee transfer to 0xf326e4…). This replaces the broad
-- 19-router cohort used by router_parent_order_size_windows.sql (kept for
-- robustness checks). Window matches calibration: 2026-04-21..2026-05-20.
--
-- Parent order: group a strict-retail tx's WETH/USDC dex_trades legs; size =
-- sum(USD) across those legs; eth_side from token direction (USDC->WETH = buy,
-- WETH->USDC = sell). Parent-order quality filter (same as the old "strict mode",
-- minus the batch-router flag which the strict cohort makes moot): same-side,
-- 1-4 WETH/USDC legs, positive USD.
--
-- Output: 10001-point parent_usd quantiles for side_group='all', with the
-- arrival/buy diagnostics (parent_count, buy/sell counts, horizon_days)
-- repeated on every row so the puller can read them.

DECLARE start_date DATE      DEFAULT DATE '2026-04-21';
DECLARE end_date   DATE      DEFAULT DATE '2026-05-20';
DECLARE start_ts   TIMESTAMP DEFAULT TIMESTAMP '2026-04-21 00:00:00';
DECLARE end_ts     TIMESTAMP DEFAULT TIMESTAMP '2026-05-20 23:59:59';
DECLARE mm_fee_addr STRING   DEFAULT '0xf326e4de8f66a0bdc0970b79e0924e33c79f1915';

WITH constants AS (
  SELECT
    '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2' AS weth,
    '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48' AS usdc
),
-- Strict-retail tx set (identical definition to the calibration pull).
uni_fe AS (
  SELECT DISTINCT LOWER(transaction_hash) AS tx
  FROM `uniswap-labs.core.swaps`
  WHERE report_date BETWEEN DATE_SUB(start_date, INTERVAL 1 DAY) AND DATE_ADD(end_date, INTERVAL 1 DAY)
    AND LOWER(chain_name) = 'ethereum' AND is_complete AND transaction_hash IS NOT NULL
),
mm_fee AS (
  SELECT DISTINCT LOWER(TRANSACTION_HASH) AS tx
  FROM `uniswap-allium.ethereum.assets_erc20_token_transfers`
  WHERE DATE(BLOCK_TIMESTAMP) BETWEEN start_date AND end_date
    AND LOWER(TO_ADDRESS) = mm_fee_addr
),
retail_txs AS (
  SELECT tx FROM uni_fe UNION DISTINCT SELECT tx FROM mm_fee
),
raw_legs AS (
  SELECT
    LOWER(t.TRANSACTION_HASH) AS transaction_hash,
    CASE
      WHEN LOWER(t.TOKEN_SOLD_ADDRESS) = c.usdc AND LOWER(t.TOKEN_BOUGHT_ADDRESS) = c.weth THEN 'buy_eth'
      WHEN LOWER(t.TOKEN_SOLD_ADDRESS) = c.weth AND LOWER(t.TOKEN_BOUGHT_ADDRESS) = c.usdc THEN 'sell_eth'
      ELSE NULL
    END AS eth_side,
    COALESCE(NULLIF(t.USD_AMOUNT, 0.0), NULLIF(t.USD_SOLD_AMOUNT, 0.0), NULLIF(t.USD_BOUGHT_AMOUNT, 0.0)) AS leg_usd
  FROM `uniswap-allium.ethereum.dex_trades` AS t
  JOIN retail_txs AS rt ON LOWER(t.TRANSACTION_HASH) = rt.tx
  CROSS JOIN constants AS c
  WHERE t.BLOCK_TIMESTAMP BETWEEN start_ts AND end_ts
    AND (
      (LOWER(t.TOKEN_SOLD_ADDRESS) = c.weth AND LOWER(t.TOKEN_BOUGHT_ADDRESS) = c.usdc)
      OR
      (LOWER(t.TOKEN_SOLD_ADDRESS) = c.usdc AND LOWER(t.TOKEN_BOUGHT_ADDRESS) = c.weth)
    )
),
parent AS (
  SELECT
    transaction_hash,
    COUNT(*) AS weth_usdc_leg_count,
    COUNT(DISTINCT eth_side) AS side_count,
    MIN(eth_side) AS eth_side,
    SUM(IFNULL(leg_usd, 0.0)) AS parent_usd,
    COUNTIF(leg_usd IS NULL OR leg_usd <= 0.0) AS missing_usd_legs
  FROM raw_legs
  GROUP BY transaction_hash
),
strict_orders AS (
  SELECT eth_side, parent_usd
  FROM parent
  WHERE side_count = 1
    AND parent_usd > 0.0
    AND missing_usd_legs = 0
    AND weth_usdc_leg_count BETWEEN 1 AND 4
),
diag AS (
  SELECT
    COUNT(*) AS parent_count,
    COUNTIF(eth_side = 'buy_eth') AS buy_eth_count,
    COUNTIF(eth_side = 'sell_eth') AS sell_eth_count,
    DATE_DIFF(end_date, start_date, DAY) + 1 AS horizon_days,
    APPROX_QUANTILES(parent_usd, 10000) AS q
  FROM strict_orders
)
SELECT
  '30d_strict_retail' AS window_name,
  'strict' AS mode,
  'all' AS side_group,
  CAST(pct_index AS FLOAT64) / 100.0 AS pct,
  size_usd,
  d.parent_count,
  d.buy_eth_count,
  d.sell_eth_count,
  d.horizon_days
FROM diag AS d
CROSS JOIN UNNEST(d.q) AS size_usd WITH OFFSET AS pct_index
ORDER BY pct;
