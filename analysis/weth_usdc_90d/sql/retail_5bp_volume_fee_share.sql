-- Retail (router-routed) WETH/USDC volume and fee aggregated by 5bp-pool vs
-- everywhere-else, over a single day. The driver script issues this query
-- per day and aggregates client-side to stay under the bytes-billed cap.
--
-- The denominator includes ALL router-routed WETH/USDC volume across every
-- venue (V3, V4, V2, Curve, Balancer, …) — i.e., the total retail flow that
-- the simulator's two-pool model has to route between submission (5bp) and
-- normalizer (everything else).
--
-- Output columns:
--   pool_group     ∈ { '5bp', 'other' }
--   volume_usd     SUM(USD_AMOUNT) for retail legs
--   fees_usd       SUM(TRANSACTION_FEES_USD) for retail legs
--   n_legs         COUNT(*)

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
    '0xdef171fe48cf0115b1d80b88dc8eab59176fee57',
    '0x6a000f20005980200259b80c5102003040001068',
    '0x6131b5fae19ea4f9d964eac0408e4408b66337b5',
    '0x617dee16b86534a5d792a4d7a62fb491b544111e',
    '0x881d40237659c251811cec9c364ef91dc08d300c',
    '0xcf5540fffcdc3d510b18bfca6d2b9987b0772559',
    '0x6352a56caadc4f1e25cd6c75970fa768a3304e64',
    '0x1231deb6f5749ef6ce6943a275a1d3e7486f4eae'
  ]) AS addr
)
SELECT
  CASE WHEN LOWER(t.LIQUIDITY_POOL_ADDRESS) = c.pool_5bp THEN '5bp' ELSE 'other' END
    AS pool_group,
  SUM(COALESCE(NULLIF(t.USD_AMOUNT, 0.0),
               NULLIF(t.USD_SOLD_AMOUNT, 0.0),
               NULLIF(t.USD_BOUGHT_AMOUNT, 0.0)))   AS volume_usd,
  SUM(t.TRANSACTION_FEES_USD)                       AS fees_usd,
  COUNT(*)                                          AS n_legs
FROM `uniswap-allium.ethereum.dex_trades` AS t
JOIN routers      AS r ON LOWER(t.TRANSACTION_TO_ADDRESS) = r.addr
CROSS JOIN constants AS c
WHERE DATE(t.BLOCK_TIMESTAMP) = @target_date
  AND (
        (LOWER(t.TOKEN_SOLD_ADDRESS)   = c.weth AND LOWER(t.TOKEN_BOUGHT_ADDRESS) = c.usdc)
     OR (LOWER(t.TOKEN_SOLD_ADDRESS)   = c.usdc AND LOWER(t.TOKEN_BOUGHT_ADDRESS) = c.weth)
      )
GROUP BY pool_group
ORDER BY pool_group
