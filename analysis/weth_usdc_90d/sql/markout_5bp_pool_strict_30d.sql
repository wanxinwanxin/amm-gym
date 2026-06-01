-- Per-swap markouts for STRICT-RETAIL trades on the 5bp WETH/USDC pool, 30d
-- (2026-04-21..2026-05-20). Cohort = the same strict retail_txs set as §4
-- calibration: Uniswap first-party FE (core.swaps surface tag) UNION MetaMask
-- 87.5bps-fee txs. Replaces the broad 19-router markout build (kept for robustness).
--
-- markout_15s is the LP-profitability reference (15s after the trade); markout_next
-- (next-block mid) is pulled for trader-t-cost context. Output is per-swap so the
-- notebook can do USD-weighted distributions.

DECLARE start_date  DATE   DEFAULT DATE '2026-04-21';
DECLARE end_date    DATE   DEFAULT DATE '2026-05-20';
DECLARE mm_fee_addr STRING DEFAULT '0xf326e4de8f66a0bdc0970b79e0924e33c79f1915';

WITH uni_fe AS (
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
  m.block_date,
  m.usd_amount,
  m.markout_next * 1e4 AS markout_next_bps,
  m.markout_15s  * 1e4 AS markout_15s_bps
FROM `uniswap-labs.research.markout_prod` AS m
JOIN retail_txs AS rt ON LOWER(m.transaction_hash) = rt.tx
WHERE m.block_date BETWEEN start_date AND end_date
  AND m.liquidity_pool_address = '0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640'
  AND m.markout_next IS NOT NULL
  AND m.markout_15s  IS NOT NULL
  AND m.usd_amount   IS NOT NULL
ORDER BY m.block_date;
