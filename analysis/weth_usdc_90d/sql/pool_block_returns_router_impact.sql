WITH
routers AS (
  SELECT addr FROM UNNEST([
        '0xef1c6e67703c7bd7107eed8303fbe6ec2554bf6b',  -- Uniswap Universal Router v1
    '0x3fc91a3afd70395cd496c647d5a6cc9d4b2b7fad',  -- Uniswap Universal Router v1.2
    '0x66a9893cc07d91d95644aedd05d03f95e1dba8af',  -- Uniswap Universal Router v2
    '0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45',  -- Uniswap SwapRouter02
    '0xe592427a0aece92de3edee1f18e0157c05861564',  -- Uniswap SwapRouter v3
    '0x7a250d5630b4cf539739df2c5dacb4c659f2488d',  -- Uniswap V2Router02
    '0x1111111254fb6c44bac0bed2854e76f90643097d',  -- 1inch v4
    '0x1111111254eeb25477b68fb85ed929f73a960582',  -- 1inch v5
    '0x111111125421ca6dc452d289314280a0f8842a65',  -- 1inch v6
    '0xdef1c0ded9bec7f1a1670819833240f027b25eff',  -- 0x ExchangeProxy
    '0x0000000000001ff3684f28c67538d4d072c22734',  -- 0x AllowanceHolder
    '0xdef171fe48cf0115b1d80b88dc8eab59176fee57',  -- Paraswap v5
    '0x6a000f20005980200259b80c5102003040001068',  -- Paraswap v6
    '0x6131b5fae19ea4f9d964eac0408e4408b66337b5',  -- Kyber MetaAggregationRouter v2
    '0x617dee16b86534a5d792a4d7a62fb491b544111e',  -- Kyber AggregationRouter classic
    '0x881d40237659c251811cec9c364ef91dc08d300c',  -- MetaMask Swap router
    '0xcf5540fffcdc3d510b18bfca6d2b9987b0772559',  -- Odos Router V2
    '0x6352a56caadc4f1e25cd6c75970fa768a3304e64',  -- OpenOcean Exchange Proxy
    '0x1231deb6f5749ef6ce6943a275a1d3e7486f4eae',  -- LI.FI Diamond
  ]) AS addr
),
raw AS (
  SELECT BLOCK_NUMBER, LOG_INDEX, LOWER(TRANSACTION_TO_ADDRESS) AS tx_to,
         SAFE_CAST(JSON_VALUE(EXTRA_FIELDS, '$.sqrt_price_x96') AS FLOAT64) AS sx
  FROM `uniswap-allium.ethereum.dex_trades`
  WHERE DATE(BLOCK_TIMESTAMP) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
                                  AND DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
    AND LIQUIDITY_POOL_ADDRESS = '0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640'
),
priced AS (
  SELECT BLOCK_NUMBER, LOG_INDEX, tx_to,
         1e12 / POW(sx / POW(2.0, 96), 2) AS mid
  FROM raw WHERE sx IS NOT NULL AND sx > 0
),
seq AS (
  SELECT BLOCK_NUMBER, LOG_INDEX, tx_to, mid,
         LAG(mid) OVER (ORDER BY BLOCK_NUMBER, LOG_INDEX) AS pre_mid,
         ROW_NUMBER() OVER (PARTITION BY BLOCK_NUMBER ORDER BY LOG_INDEX DESC) AS rn_last
  FROM priced
),
block_last AS (
  SELECT BLOCK_NUMBER, mid AS last_mid FROM seq WHERE rn_last = 1
),
grid AS (
  SELECT blk FROM UNNEST(GENERATE_ARRAY(
    (SELECT MIN(BLOCK_NUMBER) FROM block_last),
    (SELECT MAX(BLOCK_NUMBER) FROM block_last)
  )) AS blk
),
dense AS (
  SELECT g.blk,
         LAST_VALUE(b.last_mid IGNORE NULLS) OVER (
           ORDER BY g.blk ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
         ) AS mid
  FROM grid g LEFT JOIN block_last b ON g.blk = b.BLOCK_NUMBER
),
block_rets AS (
  SELECT SAFE.LN(mid / LAG(mid) OVER (ORDER BY blk)) AS log_ret FROM dense
),
block_rets_clean AS (SELECT log_ret FROM block_rets WHERE log_ret IS NOT NULL),
router_impacts AS (
  SELECT SAFE.LN(mid / pre_mid) AS log_impact
  FROM seq
  WHERE pre_mid IS NOT NULL AND pre_mid > 0 AND mid > 0
    AND tx_to IN (SELECT addr FROM routers)
),
n_router AS (
  SELECT COUNT(*) AS n_router_swaps FROM seq
  WHERE tx_to IN (SELECT addr FROM routers)
),
n_all AS (SELECT COUNT(*) AS n_all_swaps FROM seq)
SELECT
  'block_returns' AS kind,
  COUNT(*) AS n,
  AVG(log_ret) AS mean_val,
  STDDEV(log_ret) AS stddev_val,
  MIN(log_ret) AS min_val,
  MAX(log_ret) AS max_val,
  APPROX_QUANTILES(log_ret, 1000) AS q200,
  (SELECT n_all_swaps FROM n_all) AS n_all_swaps,
  (SELECT n_router_swaps FROM n_router) AS n_router_swaps
FROM block_rets_clean
UNION ALL
SELECT
  'router_impacts',
  COUNT(*),
  AVG(log_impact),
  STDDEV(log_impact),
  MIN(log_impact),
  MAX(log_impact),
  APPROX_QUANTILES(log_impact, 1000),
  (SELECT n_all_swaps FROM n_all),
  (SELECT n_router_swaps FROM n_router)
FROM router_impacts
