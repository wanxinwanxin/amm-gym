# Retail flow vs volatility: calibration of the elastic retail generator

Question: does more retail arrive, and does it trade larger, on volatile days? The
exact simulator's retail generators used one Poisson rate and one size distribution on
every step regardless of the price path. That is a fidelity gap for any comparison in
which one fee design reacts to volatility and another does not: the time-average quote
of the reacting design is then not the quote the retail who actually show up see.

## Data

- Activity: completed Uniswap front-end swaps between ETH/WETH and USDC on Ethereum
  mainnet, per UTC day, 2024-11-07 .. 2026-09-21. Distinct transactions, USD notional
  shown to the user, distinct wallets. `uniswap-labs.core.swaps`, query in
  `sql/fe_eth_usdc_daily_activity.sql`, result in `fe_eth_usdc_daily_activity.csv`.
- Volatility: Binance ETHUSDT 12 s mids, std of contiguous 12 s log returns per UTC day,
  in bps per 12 s step. `binance_eth_day_vol.csv` (1,012 days, 2023-11-07 .. 2026-09-21;
  recomputing from the 12 s parquet matches to 0.001 bps).
- Join: 664 days.

Script: `scripts/calibration/fit_retail_vol_elasticity.py`. Output:
`retail_vol_elasticity_fit.json`.

## Fit

`log(activity_d) = a + e * log(vol_d)`, OLS. "Detrended" removes a centred 61-day
rolling median from both series in logs (front-end usage has secular trends unrelated
to volatility).

| series | elasticity (raw) | corr | elasticity (detrended) | corr |
|---|---|---|---|---|
| swaps | 0.49 | 0.60 | 0.45 | 0.63 |
| usd | 1.20 | 0.64 | 1.21 | 0.67 |
| wallets | 0.49 | 0.59 | 0.44 | 0.66 |

Regimes = volatility terciles of the 2023-11-07 .. 2025-11-06 replay sample
(calm <= 2.98, mid <= 3.99, volatile > 3.99 bps/step):

| regime | days | vol bps | swaps/day | USD/day | wallets/day |
|---|---|---|---|---|---|
| calm | 212 | 2.27 | 669 | $4.3m | 519 |
| mid | 196 | 3.48 | 834 | $7.2m | 651 |
| volatile | 256 | 5.35 | 1,049 | $12.1m | 814 |

Volatile / calm: swaps 1.57x, USD 2.82x, wallets 1.57x. Last 12 months only: 1.46x,
2.04x, 1.43x.

## Reading

- More people trade on volatile days (wallets and swaps both ~0.45), and each trades
  larger (USD elasticity 1.2, so mean size elasticity ~0.75).
- The relationship holds within the last year, so it is not a trend artefact.
- Day-level only. Intraday elasticity is not measured here; the simulator applies the
  same exponents to a 6 h-half-life EWMA of the step volatility, clipped to
  [0.25, 4] x the reference.

## Simulator parameters

`ExactSimpleAMMConfig`:

| field | value | source |
|---|---|---|
| `retail_arrival_vol_elasticity` | 0.45 | detrended swap-count elasticity |
| `retail_size_vol_elasticity` | 0.75 | USD elasticity 1.2 minus count elasticity 0.45 |
| `retail_vol_reference_bps` | 3.7 | mean day vol of the replay sample (3.74); multiplier averages ~1 over days |
| `retail_vol_half_life_steps` | 1800 | 6 h at 12 s; not fitted |
| `retail_vol_ratio_min/max` | 0.25 / 4.0 | clip; not fitted |

Defaults are 0 / 0: existing configs are unchanged bit for bit. Opt in with
`ExactSimpleAMMConfig.real_data_from_seed(seed, "empirical_usd_size", vol_elastic_retail=True)`
or set the fields directly.

Implication for level-matched comparisons: with constant retail, a design that quotes
wide when volatile and tight when calm is matched to a flat pool on its time-average
quote; with elastic retail it is matched on the quote its (more numerous, larger)
volatile-day customers see, and the volume it keeps on calm days counts for less.
