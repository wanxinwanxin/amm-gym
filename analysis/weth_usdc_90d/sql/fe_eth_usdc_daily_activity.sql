-- Daily completed Uniswap front-end swaps between ETH/WETH and USDC on Ethereum mainnet:
-- distinct transactions, USD notional shown to the user, distinct wallets, by UTC day.
-- Source of analysis/weth_usdc_90d/fe_eth_usdc_daily_activity.csv, consumed by
-- scripts/calibration/fit_retail_vol_elasticity.py (joined to the day's realized
-- Binance ETHUSDT 12 s volatility to fit the retail arrival / size elasticities in
-- arena_eval/exact_simple_amm/config.py). ~11 GB scan over the two-year window.
SELECT
  DATE(event_time)                    AS day_utc,
  COUNT(DISTINCT transaction_hash)    AS swaps,
  SUM(usd_amount_shown_to_user)       AS usd,
  COUNT(DISTINCT wallet_address)      AS wallets
FROM `uniswap-labs.core.swaps`
WHERE report_date BETWEEN DATE '2024-11-06' AND DATE '2026-09-22'   -- partition filter, one day of slack each side
  AND chain_id = 1
  AND is_complete
  AND transaction_hash IS NOT NULL
  -- either leg is ETH/WETH, the other USDC
  AND LOWER(token_in_address)  IN ('0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2', 'native', '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48')
  AND LOWER(token_out_address) IN ('0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2', 'native', '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48')
  AND NOT (LOWER(token_in_address)  IN ('0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2', 'native')
       AND LOWER(token_out_address) IN ('0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2', 'native'))   -- ETH<->WETH wraps
  AND NOT (LOWER(token_in_address)  = '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48'
       AND LOWER(token_out_address) = '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48')
  AND DATE(event_time) BETWEEN DATE '2024-11-07' AND DATE '2026-09-21'
GROUP BY day_utc
ORDER BY day_utc
