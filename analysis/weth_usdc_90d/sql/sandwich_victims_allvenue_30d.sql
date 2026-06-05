-- All-venue WETH/USDC sandwich VICTIMS over the 30d window (2026-04-21..05-20),
-- reconstructed from the Allium heuristic sandwich table.
--
-- uniswap-allium.ethereum.dex_sandwich_trades stores the ATTACKER legs only
-- (DIRECTION = 'front' / 'back'); TRANSACTION_HASH is that leg's tx and
-- SANDWICH_TRANSACTION_HASH / SANDWICH_TRANSACTION_INDEX point at the paired leg.
-- A victim is any OTHER swap on the same pool+block whose transaction_index sits
-- strictly between the front and back indices. We rebuild victims by joining the
-- front/back windows to all WETH/USDC swaps in dex_trades.
--
-- This is the all-venue complement to the V3-only, pool-mid sandwich diagnostic
-- (Chart 3b): it catches sandwiches on V4 / Pancake / Curve / Balancer / V2 too.
-- Caveat: Uniswap V4 swaps all carry the PoolManager singleton as
-- LIQUIDITY_POOL_ADDRESS, so the pool+block+index-between match can occasionally
-- catch a same-block V4 swap on a different poolId; front/back indices are usually
-- adjacent so this is rare.

DECLARE start_ts TIMESTAMP DEFAULT TIMESTAMP '2026-04-21 00:00:00';
DECLARE end_ts   TIMESTAMP DEFAULT TIMESTAMP '2026-05-21 00:00:00';
DECLARE weth STRING DEFAULT '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2';
DECLARE usdc STRING DEFAULT '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48';

WITH pairs AS (   -- one row per sandwich attack (the front leg carries both indices)
  SELECT
    PROTOCOL,
    LOWER(LIQUIDITY_POOL_ADDRESS) AS pool,
    CAST(BLOCK_NUMBER AS INT64)   AS bn,
    LEAST(CAST(TRANSACTION_INDEX AS INT64), CAST(SANDWICH_TRANSACTION_INDEX AS INT64))    AS lo_idx,
    GREATEST(CAST(TRANSACTION_INDEX AS INT64), CAST(SANDWICH_TRANSACTION_INDEX AS INT64)) AS hi_idx
  FROM `uniswap-allium.ethereum.dex_sandwich_trades`
  WHERE DIRECTION = 'front'
    AND BLOCK_TIMESTAMP >= start_ts AND BLOCK_TIMESTAMP < end_ts
    AND (
      (LOWER(TOKEN_SOLD_ADDRESS)=weth AND LOWER(TOKEN_BOUGHT_ADDRESS)=usdc) OR
      (LOWER(TOKEN_SOLD_ADDRESS)=usdc AND LOWER(TOKEN_BOUGHT_ADDRESS)=weth)
    )
),
attacker_tx AS (  -- attacker legs themselves are not victims
  SELECT DISTINCT LOWER(TRANSACTION_HASH) AS tx
  FROM `uniswap-allium.ethereum.dex_sandwich_trades`
  WHERE BLOCK_TIMESTAMP >= start_ts AND BLOCK_TIMESTAMP < end_ts
    AND (
      (LOWER(TOKEN_SOLD_ADDRESS)=weth AND LOWER(TOKEN_BOUGHT_ADDRESS)=usdc) OR
      (LOWER(TOKEN_SOLD_ADDRESS)=usdc AND LOWER(TOKEN_BOUGHT_ADDRESS)=weth)
    )
),
swaps AS (        -- all WETH/USDC swaps in the window
  SELECT
    LOWER(TRANSACTION_HASH)       AS tx,
    LOWER(LIQUIDITY_POOL_ADDRESS) AS pool,
    CAST(BLOCK_NUMBER AS INT64)   AS bn,
    CAST(TRANSACTION_INDEX AS INT64) AS idx,
    PROTOCOL,
    USD_AMOUNT
  FROM `uniswap-allium.ethereum.dex_trades`
  WHERE BLOCK_TIMESTAMP >= start_ts AND BLOCK_TIMESTAMP < end_ts
    AND (
      (LOWER(TOKEN_SOLD_ADDRESS)=weth AND LOWER(TOKEN_BOUGHT_ADDRESS)=usdc) OR
      (LOWER(TOKEN_SOLD_ADDRESS)=usdc AND LOWER(TOKEN_BOUGHT_ADDRESS)=weth)
    )
)
SELECT DISTINCT
  s.tx               AS victim_tx,
  s.PROTOCOL         AS protocol,
  s.pool             AS pool,
  s.USD_AMOUNT       AS usd_amount
FROM swaps AS s
JOIN pairs AS p
  ON s.pool = p.pool AND s.bn = p.bn
 AND s.idx > p.lo_idx AND s.idx < p.hi_idx
WHERE s.tx NOT IN (SELECT tx FROM attacker_tx);
