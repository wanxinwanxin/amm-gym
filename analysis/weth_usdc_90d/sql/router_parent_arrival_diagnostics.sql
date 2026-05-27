-- Arrival-process diagnostics for WETH/USDC router-parent transactions (6m window).
--
-- Goal: assess whether parent-order arrivals are Poisson-like.
--   - Fano factor (Var/Mean) at multiple bucket sizes:
--       Poisson => Fano == 1.0 at every scale.
--       Pure overdispersion (e.g., negative-binomial) => Fano > 1, ~flat across scales.
--       Clustering / autocorrelation => Fano grows with bucket size.
--   - Mean rate by hour-of-day (UTC) and by day-of-week (BQ DAYOFWEEK: 1=Sun..7=Sat).
--   - Lag-1/5/30 autocorrelation of per-12s counts (12s/60s/360s lags).
--
-- Output: ~40 rows tagged by `metric`. Client computes Fano / ACF from raw sums.

WITH
constants AS (
  SELECT
    '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2' AS weth,
    '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48' AS usdc
),
routers AS (
  SELECT * FROM UNNEST([
    STRUCT('0xef1c6e67703c7bd7107eed8303fbe6ec2554bf6b', FALSE),
    STRUCT('0x3fc91a3afd70395cd496c647d5a6cc9d4b2b7fad', FALSE),
    STRUCT('0x66a9893cc07d91d95644aedd05d03f95e1dba8af', FALSE),
    STRUCT('0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45', FALSE),
    STRUCT('0xe592427a0aece92de3edee1f18e0157c05861564', FALSE),
    STRUCT('0x7a250d5630b4cf539739df2c5dacb4c659f2488d', FALSE),
    STRUCT('0x1111111254fb6c44bac0bed2854e76f90643097d', FALSE),
    STRUCT('0x1111111254eeb25477b68fb85ed929f73a960582', FALSE),
    STRUCT('0x111111125421ca6dc452d289314280a0f8842a65', FALSE),
    STRUCT('0xdef1c0ded9bec7f1a1670819833240f027b25eff', FALSE),
    STRUCT('0x0000000000001ff3684f28c67538d4d072c22734', FALSE),
    STRUCT('0xdef171fe48cf0115b1d80b88dc8eab59176fee57', FALSE),
    STRUCT('0x6a000f20005980200259b80c5102003040001068', FALSE),
    STRUCT('0x6131b5fae19ea4f9d964eac0408e4408b66337b5', FALSE),
    STRUCT('0x617dee16b86534a5d792a4d7a62fb491b544111e', FALSE),
    STRUCT('0x881d40237659c251811cec9c364ef91dc08d300c', FALSE),
    STRUCT('0xcf5540fffcdc3d510b18bfca6d2b9987b0772559', FALSE),
    STRUCT('0x6352a56caadc4f1e25cd6c75970fa768a3304e64', FALSE),
    STRUCT('0x1231deb6f5749ef6ce6943a275a1d3e7486f4eae', FALSE)
  ])
),
window_bounds AS (
  SELECT
    TIMESTAMP(DATE_SUB(CURRENT_DATE(), INTERVAL 6 MONTH)) AS start_ts,
    TIMESTAMP(CURRENT_DATE()) AS end_ts
),
raw_router_legs AS (
  SELECT
    t.BLOCK_TIMESTAMP,
    LOWER(t.TRANSACTION_HASH) AS tx_hash,
    r.is_batch,
    CASE
      WHEN LOWER(t.TOKEN_SOLD_ADDRESS) = c.usdc AND LOWER(t.TOKEN_BOUGHT_ADDRESS) = c.weth THEN 'buy_eth'
      WHEN LOWER(t.TOKEN_SOLD_ADDRESS) = c.weth AND LOWER(t.TOKEN_BOUGHT_ADDRESS) = c.usdc THEN 'sell_eth'
      ELSE NULL
    END AS eth_side,
    COALESCE(NULLIF(t.USD_AMOUNT, 0.0), NULLIF(t.USD_SOLD_AMOUNT, 0.0), NULLIF(t.USD_BOUGHT_AMOUNT, 0.0)) AS leg_usd
  FROM `uniswap-allium.ethereum.dex_trades` AS t
  JOIN routers AS r ON LOWER(t.TRANSACTION_TO_ADDRESS) = r.addr
  CROSS JOIN constants AS c
  CROSS JOIN window_bounds AS w
  WHERE t.BLOCK_TIMESTAMP >= w.start_ts
    AND t.BLOCK_TIMESTAMP <  w.end_ts
    AND ((LOWER(t.TOKEN_SOLD_ADDRESS) = c.weth AND LOWER(t.TOKEN_BOUGHT_ADDRESS) = c.usdc)
      OR (LOWER(t.TOKEN_SOLD_ADDRESS) = c.usdc AND LOWER(t.TOKEN_BOUGHT_ADDRESS) = c.weth))
),
parent_candidates AS (
  SELECT
    tx_hash,
    MIN(BLOCK_TIMESTAMP) AS first_seen_at,
    LOGICAL_OR(is_batch) AS has_batch,
    COUNT(*) AS leg_count,
    MIN(eth_side) AS min_side,
    MAX(eth_side) AS max_side,
    SUM(IFNULL(leg_usd, 0.0)) AS parent_usd,
    COUNTIF(leg_usd IS NULL OR leg_usd <= 0.0) AS bad_legs
  FROM raw_router_legs
  GROUP BY tx_hash
),
strict_parents AS (
  SELECT first_seen_at AS ts
  FROM parent_candidates
  WHERE min_side IS NOT NULL
    AND min_side = max_side
    AND NOT has_batch
    AND leg_count BETWEEN 1 AND 4
    AND parent_usd > 0
    AND bad_legs = 0
),
window_seconds AS (
  SELECT TIMESTAMP_DIFF(end_ts, start_ts, SECOND) AS total_seconds,
         UNIX_SECONDS(start_ts) AS start_unix
  FROM window_bounds
),
events_with_idx AS (
  SELECT
    UNIX_SECONDS(ts) - (SELECT start_unix FROM window_seconds) AS sec_offset,
    ts
  FROM strict_parents
),
fano_lookup AS (
  SELECT 12   AS bucket_seconds UNION ALL
  SELECT 60   UNION ALL
  SELECT 300  UNION ALL
  SELECT 3600
),
per_bucket_aggregated AS (
  SELECT
    f.bucket_seconds,
    DIV(e.sec_offset, f.bucket_seconds) AS bucket_idx,
    COUNT(*) AS n
  FROM events_with_idx e
  CROSS JOIN fano_lookup f
  GROUP BY bucket_seconds, bucket_idx
),
fano_stats AS (
  SELECT
    bucket_seconds,
    SUM(n) AS event_count,
    SUM(n * n) AS sum_n_squared,
    DIV((SELECT total_seconds FROM window_seconds), bucket_seconds) AS total_buckets
  FROM per_bucket_aggregated
  GROUP BY bucket_seconds
),
per_hour_of_day AS (
  SELECT EXTRACT(HOUR FROM ts AT TIME ZONE 'UTC') AS hour_of_day,
         COUNT(*) AS event_count
  FROM strict_parents
  GROUP BY hour_of_day
),
per_dow AS (
  SELECT EXTRACT(DAYOFWEEK FROM ts AT TIME ZONE 'UTC') AS dow,
         COUNT(*) AS event_count
  FROM strict_parents
  GROUP BY dow
),
per_12s AS (
  SELECT DIV(sec_offset, 12) AS bucket_idx,
         COUNT(*) AS n
  FROM events_with_idx
  GROUP BY bucket_idx
),
acf_lags AS (
  SELECT 1 AS lag UNION ALL
  SELECT 5 UNION ALL
  SELECT 30
),
acf_pairs AS (
  SELECT a.lag, SUM(p.n * p_lag.n) AS cross_sum
  FROM per_12s p
  CROSS JOIN acf_lags a
  JOIN per_12s p_lag ON p_lag.bucket_idx = p.bucket_idx - a.lag
  GROUP BY a.lag
)
SELECT 'fano' AS metric, CAST(bucket_seconds AS STRING) AS key,
       CAST(event_count AS FLOAT64) AS v1,
       CAST(sum_n_squared AS FLOAT64) AS v2,
       CAST(total_buckets AS FLOAT64) AS v3
FROM fano_stats
UNION ALL
SELECT 'hour_of_day', CAST(hour_of_day AS STRING), CAST(event_count AS FLOAT64), NULL, NULL
FROM per_hour_of_day
UNION ALL
SELECT 'dow', CAST(dow AS STRING), CAST(event_count AS FLOAT64), NULL, NULL
FROM per_dow
UNION ALL
SELECT 'acf', CAST(lag AS STRING), CAST(cross_sum AS FLOAT64), NULL, NULL
FROM acf_pairs
UNION ALL
SELECT 'window', 'totals',
       CAST((SELECT COUNT(*) FROM strict_parents) AS FLOAT64),
       CAST((SELECT total_seconds FROM window_seconds) AS FLOAT64),
       NULL
ORDER BY metric, key
