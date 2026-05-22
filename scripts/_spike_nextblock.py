"""Compute the per-trade markout using NEXT-BLOCK fair_price as reference."""
from __future__ import annotations

import sys
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKTREE / "presentation"))
sys.path.insert(0, str(WORKTREE))

import numpy as np  # noqa: E402

from helpers import _build_sim  # noqa: E402
from calibration_helpers import load_final  # noqa: E402


def main() -> None:
    final = load_final()
    p = final["params"]
    seeds = tuple(final["seeds_holdout"])
    n_steps = int(final["n_steps"])

    next_block_markouts: list[float] = []
    same_block_markouts: list[float] = []

    for seed in seeds:
        sim = _build_sim(
            p["normalizer_fee"],
            p["normalizer_depth_y"],
            p["submission_depth_y"],
            0.0005,
            n_steps,
            seed,
        )

        # Run step-by-step, collect retail trades on submission, then on the next
        # step capture the new fair_price and compute markout against it.
        pending_trades: list[tuple[float, float, str]] = []  # (amount_x, amount_y, side)
        while not sim.done:
            step = sim.step_once()
            # First, finalise any pending trades from the previous block using the
            # current block's fair price as the next-block reference.
            new_fair = step["fair_price"]
            for amount_x, amount_y, side in pending_trades:
                if side == "sell_x":
                    edge_next = amount_x * new_fair - amount_y
                else:
                    edge_next = amount_y - amount_x * new_fair
                if amount_y > 0:
                    next_block_markouts.append(edge_next / amount_y * 10_000)
            pending_trades.clear()

            # Now collect this block's retail trades on submission with same-block markout
            for ev in step["trade_events"]:
                if ev["source"] != "retail" or ev["venue"] != "submission":
                    continue
                fair = step["fair_price"]
                if ev["trader_side"] == "sell_x":
                    edge = ev["amount_x"] * fair - ev["amount_y"]
                else:
                    edge = ev["amount_y"] - ev["amount_x"] * fair
                if ev["amount_y"] > 0:
                    same_block_markouts.append(edge / ev["amount_y"] * 10_000)
                pending_trades.append((ev["amount_x"], ev["amount_y"], ev["trader_side"]))

    sb = np.asarray(same_block_markouts)
    nb = np.asarray(next_block_markouts)
    print(f"same-block markouts:  n={sb.size:,}  mean={sb.mean():+.3f}  median={np.median(sb):+.3f}  "
          f"p5={np.percentile(sb,5):+.3f}  p95={np.percentile(sb,95):+.3f}")
    print(f"next-block markouts:  n={nb.size:,}  mean={nb.mean():+.3f}  median={np.median(nb):+.3f}  "
          f"p5={np.percentile(nb,5):+.3f}  p95={np.percentile(nb,95):+.3f}")

    # Histogram for next-block: do the spikes vanish?
    print('\nNext-block histogram, 0.5 bps bins in [-25, 25]:')
    edges = np.arange(-25, 25.5, 0.5)
    counts, edges = np.histogram(nb, bins=edges)
    n_in_range = counts.sum()
    for c, e0, e1 in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * min(80, c // 20)
        pct = c / nb.size * 100
        print(f"  [{e0:+7.2f},{e1:+7.2f})  {c:6d}  {pct:5.2f}%  {bar}")
    print(f"\nOf {nb.size} markouts, {n_in_range} ({n_in_range/nb.size*100:.1f}%) in [-25, 25].")

    np.save(WORKTREE / "reports" / "spike_markouts_next.npy", nb)
    print(f"saved {nb.size:,} next-block markouts to reports/spike_markouts_next.npy")


if __name__ == "__main__":
    main()
