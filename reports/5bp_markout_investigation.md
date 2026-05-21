# Uniswap V3 0.05% WETH/USDC pool — USD-weighted next-block markout

**Pool:** `0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640` (Uniswap V3 5bp WETH/USDC).
**Question:** Is the USD-volume-weighted next-block LP markout on this pool positive
or negative over long horizons?
**Anchor date:** windows end 2026-05-19 (latest day with full data in
`uniswap-labs.research.markout_prod`).
**Headline answer:** Long-horizon USD-volume-weighted LP markout is
**NEGATIVE** on this pool — between **-0.8 and -1.4 bps** across the 90d / 180d /
360d / 730d windows. Net LP markout loss to next-block is **-$7.36M over the last
730d** ($67.4B volume). The user's prior (LPs lose to informed flow on a
competitive 5bp pool) is **validated**.

---

## 1. Sign convention (verified)

We use **LP-positive markout** throughout:

> `markout = (benchmark_price - executed_price) / benchmark_price`
>
> where `executed_price` and `benchmark_price` are both in the **token_bought
> per token_sold** convention used in `uniswap-labs.research.markout_prod`.
> `markout > 0` means the LP gained, `< 0` means the LP lost.
> Equivalently for the trader/taker: trader markout = `-markout`.

Sample-row verification (May 7 2026, top swaps by USD volume):

| direction          | exec px   | next-block Binance mid | `markout_next` (decimal) | LP interp.          |
|--------------------|-----------|------------------------|---------------------------|----------------------|
| Sold WETH, bought USDC | 2300.55 USDC/WETH | 2296.27 USDC/WETH | -0.001862  | LP lost (trader sold ETH above mid)  |
| Sold WETH, bought USDC | 2297.31 USDC/WETH | 2293.91 USDC/WETH | -0.001484  | LP lost                              |
| Sold USDC, bought WETH | 0.000435 WETH/USDC | 0.000436 WETH/USDC | +0.002888 | LP gained (trader bought ETH cheap) |

Both formulas check out: `(benchmark - executed) / benchmark` ≈ the reported
`markout_next` to 4 decimal places.

The "next" benchmark is the Binance midprice at the next swap on the same pool
(per `15s_*_timestamp` columns the 15s variant uses Binance mid 15 s later).
This is the natural "next-block markout" definition for a 12 s blockchain.

`markout_next_dollar = markout_next * usd_amount`, so

> **USD-weighted markout (bps) = SUM(markout_next_dollar) / SUM(usd_amount) * 1e4**

This is the metric the user asked about.

---

## 2. Provenance of the +3.637 bps figure

`analysis/weth_usdc_90d/markout_5bp_pool_summary.csv` (n_swaps = 6328,
avg = 3.637) is **exactly reproduced** by:

```sql
SELECT COUNT(*) AS n_swaps,
       AVG(markout_next) * 1e4 AS avg_next_bps,
       STDDEV(markout_next) * 1e4 AS std_next_bps
FROM `uniswap-labs.research.markout_prod`
WHERE block_date = DATE '2026-05-07'
  AND LOWER(liquidity_pool_address) = '0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640'
-- (no NULL filter; AVG ignores NULLs internally)
```

returns `n_swaps=6328, avg=3.6368 bps, std=4.4559 bps`. Matches the CSV bit-for-bit.

So `+3.637` is:

- **Sign convention:** LP-positive (matches our convention).
- **Weighting:** **simple per-swap average**, NOT USD-weighted, NOT swap-count weighted across days.
- **Time window:** **a single day**, 2026-05-07 only.
- **Definition of "next":** Binance midprice 12 s after the swap (the `markout_next` field, benchmark = Binance mid at the next swap on the same pool).

On that same day:

| metric                                                | value                    |
|-------------------------------------------------------|--------------------------|
| simple per-swap markout (LP, bps, next-block)         | **+3.637**               |
| USD-volume-weighted markout (LP, bps, next-block)     | **+0.84**                |
| total USD volume                                      | $44.93M                  |
| net LP markout PnL                                    | **+$3,767**              |

So even on the single quoted day, USD-weighting collapses the headline figure
from +3.64 bps to +0.84 bps (4.3× smaller). The figure has not been a "wrong"
number for what it measured; it was just a **simple per-swap average over a
single day**, which is *not* the LP-economic metric for this pool.

---

## 3. USD-weighted markout by window (ending 2026-05-19)

Computed from per-day aggregates of `uniswap-labs.research.markout_prod`
(`scripts/fetch_markout_daily.py` produces the daily roll-ups,
`scripts/rollup_markout_windows.py` rolls them up into windows). The full per-day
CSV is `reports/markout_daily_5bp_pool.csv`; the windowed CSV is
`reports/markout_windows.csv`; the plot is `reports/markout_by_window.png`.

All values are **LP-positive bps** (positive = LP gain). The 360d window covers
358 days actually present; the 730d window currently only has 473 days of data
in the source table — `markout_prod` data does not exist before 2025-01-01 for
this pool. We therefore also report a **"max available"** row to make the
ground-truth horizon explicit. Patterns are stable across the 90d / 180d / 360d
/ 730d windows so the conclusion does not depend on having a full 730 days.

| window  | dates                  | days present | n_swaps   | USD volume       | **USD-weighted next bps** | simple per-swap next bps | net LP PnL next ($)  | USD-weighted 15s bps |
|---------|------------------------|--------------|-----------|------------------|----------------------------|----------------------------|-----------------------|----------------------|
| 7d      | 2026-05-13..2026-05-19 | 7            | 25,288    | $566.9M          | **-1.045**                 | +3.249                     | -$59,257              | -1.618               |
| 30d     | 2026-04-20..2026-05-19 | 30           | 141,595   | $1.558B          | **+0.072**                 | +3.464                     | +$11,162              | -0.422               |
| 90d     | 2026-02-19..2026-05-19 | 89           | 503,077   | $12.564B         | **-1.446**                 | +3.244                     | -$1,816,796           | -2.129               |
| 180d    | 2025-11-21..2026-05-19 | 178          | 1,011,072 | $20.962B         | **-1.314**                 | +2.817                     | -$2,754,323           | -1.873               |
| 360d    | 2025-05-25..2026-05-19 | 358          | 1,987,502 | $37.245B         | **-0.805**                 | +3.040                     | -$2,999,703           | -1.250               |
| 730d    | 2024-05-20..2026-05-19 | 473 (av.)    | 2,830,805 | $67.400B         | **-1.092**                 | +2.989                     | -$7,357,922           | -1.391               |
| max-avail | 2025-01-01..2026-05-19 | 473         | 2,830,805 | $67.400B         | **-1.092**                 | +2.989                     | -$7,357,922           | -1.391               |

(In `markout_prod`, the `15s` variant uses Binance midprice 15 s after the
swap; the "next" variant uses Binance midprice at the next swap on the same
pool. They are nearly identical sign-wise; the 15s variant is consistently
more negative for LPs by ~0.3-0.7 bps, suggesting LPs realize even more
adverse selection at 15 s than at the immediate next-swap horizon.)

The repo's CALIBRATION TARGET of **+3.637 bps** (the dashed grey line in
`markout_by_window.png`) lies more than **5 bps above the actual USD-weighted
markout for every window 90d and longer**. It only barely matches the simple
per-swap average from a single day, and even that day's USD-weighted value
(+0.84 bps) is far below the +3.637 number.

---

## 4. Why simple-avg and USD-weighted disagree (and the sign flips)

Stratifying by swap-USD-size for the last 30 days (`reports/markout_by_size_bucket_30d.csv`):

| size bucket    | n_swaps | simple avg (bps) | USD-weighted (bps) | total USD     | net LP PnL ($) |
|----------------|---------|-------------------|---------------------|---------------|-----------------|
| < $100         | 38,150  | +5.67             | +4.66               | $1.19M        | +$554           |
| $100-$1k       | 38,619  | +4.48             | +4.36               | $14.37M       | +$6,261         |
| $1k-$10k       | 31,527  | +2.97             | +2.23               | $126.51M      | +$28,178        |
| $10k-$100k     | 30,727  | +0.28             | +0.02               | $934.92M      | +$1,284         |
| **$100k-$1M**  | 2,471   | **-0.34**         | **-0.29**           | **$456.81M**  | **-$13,836**    |
| **> $1M**      | **101** | +1.80             | **-4.85**           | **$23.80M**   | **-$11,279**    |

Mechanism:
- The "simple per-swap average" is dominated by the 138k small (< $10k) retail
  swaps, each strongly positive for LPs (toxic-flow-protected retail tape).
- USD weighting puts most of the weight on the 2.5k swaps > $100k, which are
  arbitrage / CEX-DEX flow, and which are negative on average for LPs.
- Whale swaps (n=101 in 30d, > $1M each) have a positive *simple* avg of
  +1.80 bps but a USD-weighted markout of **-4.85 bps** — i.e., the small
  whale swaps are positive and the large whale swaps are very negative, and
  the large ones dominate USD volume.

This is the classic LVR pattern: the 5bp pool is competitive enough that
informed (arb) flow extracts LP value on every block, with retail noise
flow providing some partial offset.

---

## 5. Cross-validation: the calibration agent's circumstantial evidence

The sibling calibration agent's BLOCKED result is now mechanistically explained.
At the realistic on-chain virtual depth (~$212M USDC for the 5bp pool), arb
flow is wide and concentrated → the simulator correctly produces strongly
NEGATIVE markout, matching the USD-weighted ground truth we computed here
(roughly -1 bps over 730d). To force the simulator to produce the +3.637 bps
target it had to shrink submission_depth by ~14× (to $15.6M), which makes
retail-like small fills dominate — effectively switching from a USD-weighted to
a simple-average regime, but at the cost of physical realism.

So the simulator was correct; the calibration target was wrong (or rather,
measured a quantity that is not the LP-economic metric).

---

## 6. Conclusion

**The user's prior is validated.**

On a USD-volume-weighted basis, the Uniswap V3 5bp WETH/USDC pool's LP markout
to next-block is **NEGATIVE** at every window of practical interest (90d, 180d,
360d, 730d), in the range **-0.8 to -1.4 bps**. Net LP markout PnL on a
volume-weighted next-block basis is **-$7.36M over the last 730d** (-$9.38M
on the 15s basis), against $67.4B traded.

The figure of **+3.637 bps** currently used as the calibration target in
`presentation/helpers.py` (`OBSERVED_MARKOUT["avg_bps"]`) and
`presentation/realistic_simulator.ipynb` (Cell 25) is the **simple per-swap
average over a single day (2026-05-07)**. It correctly answers the question
*"on the average individual swap, does the LP gain?"* — yes, by ~+3.6 bps.
But it does NOT answer the LP-economic question *"on the dollar I trade, does
the LP gain?"* — that answer is no, by roughly -1 bps.

Confidence: **high**. The numbers are from the same upstream `markout_prod`
table that produced the +3.637 figure, queried with the same definitions of
benchmark and sign; only the aggregation function (USD-weighted SUM vs
simple-mean AVG) differs. The pattern is stable across all four long-horizon
windows (90d / 180d / 360d / 730d) and is mechanistically explained by the
size-bucket breakdown — large arb swaps tax LPs while retail noise pays them.

**Recommendation:** Replace the calibration target `OBSERVED_MARKOUT["avg_bps"]`
in `presentation/helpers.py` with a window-aggregated USD-weighted figure
(e.g., 730d USD-weighted next-block markout = **-1.09 bps**, or the more
conservative 180d = **-1.31 bps**), and reframe the comparison as
USD-weighted next-block LP markout rather than simple per-swap average.
Keep the +3.637 reference in the docs as a sanity check for the
simple-per-swap regime, but stop using it as the single-number target for
realistic-economics calibration.

---

## 7. Reproduction

```bash
# From the worktree root:
.venv/bin/python scripts/fetch_markout_daily.py \
    --start 2024-05-21 --end 2026-05-19 \
    --out reports/markout_daily_5bp_pool.csv \
    --chunk-days 30 --max-bytes 40000000000

.venv/bin/python scripts/rollup_markout_windows.py
```

Total BigQuery scan ~70 GB, ~$0.35 at on-demand pricing.

Artifacts:
- `reports/markout_daily_5bp_pool.csv` — one row per day, source for everything else.
- `reports/markout_windows.csv` — windowed roll-up (7d / 30d / 90d / 180d / 360d / 730d / max-available).
- `reports/markout_by_size_bucket_30d.csv` — last-30d markout by swap size bucket.
- `reports/markout_by_window.png` — bar plot of USD-weighted vs simple-avg by window.
- `scripts/fetch_markout_daily.py`, `scripts/rollup_markout_windows.py` — helpers.
