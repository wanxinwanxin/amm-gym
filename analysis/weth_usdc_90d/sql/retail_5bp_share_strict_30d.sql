-- Strict-retail WETH/USDC volume & fee share at the 5bp pool vs everywhere-else,
-- over the 30d calibration window (2026-04-21..2026-05-20). Cohort = the same
-- strict retail_txs set as §4 calibration: Uniswap first-party FE (core.swaps
-- surface tag) UNION MetaMask 87.5bps-fee txs. Replaces the broad 19-router
-- share SQL (retail_5bp_volume_fee_share.sql, kept for robustness).
--
-- Denominator = ALL strict-retail WETH/USDC volume/fees across every venue
-- (the total retail flow the two-pool model routes between submission=5bp and
-- normalizer=everything-else). Output: pool_group ∈ {5bp, other}, volume_usd, fees_usd.

DECLARE start_date  DATE      DEFAULT DATE '2026-04-21';
DECLARE end_date    DATE      DEFAULT DATE '2026-05-20';
DECLARE start_ts    TIMESTAMP DEFAULT TIMESTAMP '2026-04-21 00:00:00';
DECLARE end_ts      TIMESTAMP DEFAULT TIMESTAMP '2026-05-20 23:59:59';
DECLARE mm_fee_addr STRING    DEFAULT '0xf326e4de8f66a0bdc0970b79e0924e33c79f1915';

WITH constants AS (
  SELECT
    '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2' AS weth,
    '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48' AS usdc,
    '0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640' AS pool_5bp
),
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
)
SELECT
  CASE WHEN LOWER(t.LIQUIDITY_POOL_ADDRESS) = c.pool_5bp THEN '5bp' ELSE 'other' END AS pool_group,
  SUM(COALESCE(NULLIF(t.USD_AMOUNT, 0.0), NULLIF(t.USD_SOLD_AMOUNT, 0.0), NULLIF(t.USD_BOUGHT_AMOUNT, 0.0))) AS volume_usd,
  SUM(t.TRANSACTION_FEES_USD) AS fees_usd,
  COUNT(*) AS n_legs
FROM `uniswap-allium.ethereum.dex_trades` AS t
JOIN retail_txs AS rt ON LOWER(t.TRANSACTION_HASH) = rt.tx
CROSS JOIN constants AS c
WHERE t.BLOCK_TIMESTAMP BETWEEN start_ts AND end_ts
  AND (
    (LOWER(t.TOKEN_SOLD_ADDRESS) = c.weth AND LOWER(t.TOKEN_BOUGHT_ADDRESS) = c.usdc)
    OR
    (LOWER(t.TOKEN_SOLD_ADDRESS) = c.usdc AND LOWER(t.TOKEN_BOUGHT_ADDRESS) = c.weth)
  )
GROUP BY pool_group
ORDER BY pool_group;
