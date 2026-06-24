-- Time-ordered 12-second mid series for Binance ETHUSDT, 2 full years
-- (2023-11-07 .. 2025-11-06). Same downsampling as regime_distributions.sql
-- (§2.1 of REGIME_EXTRACTION.md): one row per 12s bucket = the last snapshot
-- in the bucket, mid = (best_ask + best_bid)/2. The difference: this query
-- returns the RAW CONTIGUOUS SERIES (ts, mid) instead of aggregating it into
-- per-regime quantiles. Used by the historical-replay arb-only lab to slice
-- real ~17h windows (5000 x 12s blocks) and replay them as the fair path —
-- same marginal CDF as the regime process, but the actual serial structure
-- (autocorrelated jumps, vol clustering) the regime model assumes away.
--
-- Scans ~45 GB (~$0.28 on-demand). Bill to a project with >1GB headroom, e.g.
--   bq --project_id=uniswap-allium query --nouse_legacy_sql < this.sql
-- or via scripts/calibration/pull_binance_12s_series.py (downloads to parquet).
WITH binned AS (
  SELECT
    TIMESTAMP_SECONDS(DIV(UNIX_SECONDS(timestamp), 12) * 12) AS bucket_ts,
    CAST((asks_0_price + bids_0_price) / 2 AS FLOAT64) AS mid
  FROM `uniswap-labs.cex.binance_book_snapshot_5_ETHUSDT`
  WHERE DATE(timestamp) BETWEEN '2023-11-07' AND '2025-11-06'
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY TIMESTAMP_SECONDS(DIV(UNIX_SECONDS(timestamp), 12) * 12)
    ORDER BY timestamp DESC
  ) = 1
)
SELECT UNIX_SECONDS(bucket_ts) AS ts, mid
FROM binned
ORDER BY bucket_ts
