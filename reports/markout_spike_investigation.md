# Markout-Spike Investigation (§5 of `calibration_three_targets.ipynb`)

## Problem

Two narrow, tall spikes appear in the §5 simulated per-trade markout histogram that do not appear in the empirical distribution. The user asked: where are they, what causes them, and should we fix or annotate?

## Numerical location

Captured the full 24,794-trade markout array via `scripts/_spike_capture.py`
(5 held-out seeds x 5,000 steps, calibrated params from
`calibration_artifacts/final_calibration.json`).

High-resolution histograms (`scripts/_spike_analyze.py`):

| bin | density | observation |
|-----|---------|-------------|
| `[+0.000, +0.500) bps` | 14.18 % | spike 1 |
| `[+10.000, +10.500) bps` | 12.98 % | spike 2 |
| neighbouring bins | ~3.1 % each | smooth-uniform background |

At 0.05 bps resolution, both spikes are pinned at the **upper edge of the first
bin** straddling +0 and +10 bps respectively (1421 and 1282 samples in those two
50-microbps slivers, ~5 % of the whole distribution each).

So the spikes are at **exactly +0 bps and exactly +10 bps**, with sharp decay
away from those points.

## Mechanism

`scripts/_spike_mechanism.py` partitions retail trades on the submission pool
by `(arb-direction-on-this-block, retail-side)`:

```
   arb_dir    retail     count    mode_bps    median       p1      p99
   buy_arb     buy_x      2122      +0.025    +0.090   -8.075  +27.807
   buy_arb    sell_x      2295     +10.025   +10.086   +1.030  +26.270
  sell_arb     buy_x      2141     +10.025   +10.093   +3.378  +29.731
  sell_arb    sell_x      2348      +0.025    +0.099   -8.393  +20.256
    no_arb     buy_x      7685  (uniform on [0,10])
    no_arb    sell_x      8203  (uniform on [0,10])
```

When arb fires on the submission pool in a block, it pins the pool's spot price
to the **no-arb boundary** of fair: `fair * (1 - fee)` after a `buy_arb`, or
`fair / (1 - fee)` after a `sell_arb`. The first marginal retail unit then has
a deterministic markout against the **same-block** `fair_price`:

- `buy_arb` + retail `buy_x`: marginal cost per X = `spot / gamma = fair * (1 - fee) / (1 - fee) = fair` -> markout = **0 bps**.
- `buy_arb` + retail `sell_x`: marginal LP bid per X = `spot * gamma = fair * (1 - fee)^2 ≈ fair * (1 - 2*fee)` -> markout = **+2 * fee = +10 bps**.
- `sell_arb` + retail `buy_x`: marginal cost per X = `spot / gamma = fair / (1 - fee)^2 ≈ fair * (1 + 2*fee)` -> markout = **+2 * fee = +10 bps**.
- `sell_arb` + retail `sell_x`: marginal LP bid per X = `spot * gamma = fair / (1 - fee) * (1 - fee) = fair` -> markout = **0 bps**.

For tiny retail trades (`amount_y` near zero), the markout collapses to the
boundary; for larger retail trades, slippage moves it away. Hence the
**accumulation at +0 and +10 bps** when arb fires (36 % of blocks). The
no-arb sub-population is approximately **uniform on [0, 10]** because the
spot is somewhere inside the no-arb band when retail arrives, again pinning
the markout into the [0, 10] interval.

## Hypothesis verdict

- **Hypothesis A** (spike at +5 bps from zero-return blocks): **rejected**.
  The +5 bps cluster does exist but as a soft median peak, not a spike. The
  two narrow spikes are at +0 and +10 bps, set by the **fee-band edge** of
  the post-arb spot, not by zero-return blocks.
- **Hypothesis B** (degenerate-arb / `MIN_AMOUNT` branch): rejected. Arb
  fires productively when it fires; the `MIN_AMOUNT` gate is only hit on
  trivially small trades that don't dominate the histogram.
- **Hypothesis C** (USD-quantile-sampler flat-segment artifact): rejected.
  The quantile grid is dense (1000 points) and the modes survive even when
  conditioned on a single `(arb_dir, retail_side)` cell.
- **Hypothesis D** (regime-switching discrete returns x deterministic arb):
  the closed-form arb fires every block where spot != fair (which is every
  block, since `fair_price` is a fresh draw each block). This *does*
  contribute, but the proximate mechanism is the same-block markout
  reference combined with the fee-band edge.

## Root cause is the markout *reference price*, not the simulator dynamics

The simulator's per-trade markout in `presentation/helpers.py::run_calibrated_sim`
uses the **same-block** `fair_price` as the LP markout reference:

```python
edge = ev["amount_y"] - ev["amount_x"] * step["fair_price"]   # for retail buy_x
markout_bps = edge / ev["amount_y"] * 10_000
```

But the empirical reference column on the same plot is
`analysis/weth_usdc_90d/markout_5bp_pool_percentiles.csv`'s **`markout_next_bps`**
column - a **next-block** mid. T3 in the calibration is `7d USD-weighted
markout` measured the same way (next-block).

So the plot was an apples-to-oranges comparison: empirical is next-block,
simulated is same-block. The simulated same-block metric is geometrically
bounded by the fee-band edges (+0 and +10 bps for tiny trades), which is
exactly where the spikes sit. The empirical next-block metric incorporates
the next-block return shock and is unbounded -> no spikes.

## Disposition

**Fix.** Change `run_calibrated_sim` (in `presentation/helpers.py`) to record
each retail trade's markout against the **next block's** `fair_price`. This
is one buffered-deferred computation per trade and yields a clean
apples-to-apples comparison against the empirical `markout_next_bps`.

Validated via `scripts/_spike_nextblock.py`:

| metric | same-block | next-block |
|--------|-----------:|-----------:|
| n trades | 24,794 | 24,787 (last block dropped: no next-block) |
| mean | +5.92 bps | +5.91 bps |
| median | +5.64 bps | +5.55 bps |
| p5 | +0.00 bps | -3.25 bps |
| p95 | +12.47 bps | +15.98 bps |
| 0-bps bin density | 14.2 % | 4.5 % (now uniform with neighbours) |
| 10-bps bin density | 13.0 % | 4.0 % (now uniform with neighbours) |
| supports negative markouts | no (hard floor at +0) | yes |

Spikes are gone. The aggregate (USD-weighted) markout barely moves
(+5.92 vs +5.91 bps), so the calibrated parameters remain on the
target-residual frontier and §4 / T3 are unaffected.

## Footprint of the fix

- `presentation/helpers.py::run_calibrated_sim` updated to track each retail
  trade's `(amount_x, amount_y, trader_side)` and compute markout on the
  *next* `step_once` using the new fair_price. The last block's pending
  trades are dropped (no next block available; ~0.03 % of the sample).
- `aggregate_markout_bps` and `retail_edge_sub` in the return dict continue
  to use the simulator's same-block metric (so they remain consistent with
  T3's calibration residual; no change in §4).
- No change to the simulator core. No change to the calibration result.
- Re-run `tests/test_simple_amm_search.py` to confirm parity.

## Why this is *not* purely an annotation

The spikes are geometrically inherent to the *same-block* markout reference
on a fee-bounded AMM with arb-fires-every-block. They are NOT a simulator
bug. But the same-block reference is also **not what the empirical column
measures**. So the right fix is to align the simulator's reported metric
with the empirical metric (next-block), which is the canonical LP-markout
definition used elsewhere in the repo (`reports/markout_windows.csv`,
T3 calibration target).
