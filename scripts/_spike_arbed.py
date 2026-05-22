"""Tight characterisation of the arb-then-retail spike at 0 and 10 bps."""
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

    arbed_buy = []      # arb fired this block, then retail buys X
    arbed_sell = []     # arb fired this block, then retail sells X
    after_buy_arb_buy_x = []
    after_buy_arb_sell_x = []
    after_sell_arb_buy_x = []
    after_sell_arb_sell_x = []

    for seed in seeds:
        sim = _build_sim(
            p["normalizer_fee"],
            p["normalizer_depth_y"],
            p["submission_depth_y"],
            0.0005,
            n_steps,
            seed,
        )
        while not sim.done:
            step = sim.step_once()
            sub_arb_dir = None
            for ev in step["trade_events"]:
                if ev["source"] != "arb" or ev["venue"] != "submission":
                    continue
                sub_arb_dir = ev["trader_side"]
                break
            if sub_arb_dir is None:
                continue

            for ev in step["trade_events"]:
                if ev["source"] != "retail" or ev["venue"] != "submission":
                    continue
                fair = step["fair_price"]
                if ev["trader_side"] == "sell_x":
                    edge = ev["amount_x"] * fair - ev["amount_y"]
                    bps = edge / ev["amount_y"] * 10_000 if ev["amount_y"] > 0 else 0.0
                    arbed_sell.append(bps)
                    if sub_arb_dir == "buy_x":
                        after_buy_arb_sell_x.append(bps)
                    else:
                        after_sell_arb_sell_x.append(bps)
                else:
                    edge = ev["amount_y"] - ev["amount_x"] * fair
                    bps = edge / ev["amount_y"] * 10_000 if ev["amount_y"] > 0 else 0.0
                    arbed_buy.append(bps)
                    if sub_arb_dir == "buy_x":
                        after_buy_arb_buy_x.append(bps)
                    else:
                        after_sell_arb_buy_x.append(bps)

    def describe(name: str, arr_list: list[float]) -> None:
        arr = np.asarray(arr_list)
        print(f"\n{name}: {arr.size:,} samples; min/p1/p50/p99/max = "
              f"{arr.min():+.4f}/{np.percentile(arr,1):+.4f}/"
              f"{np.median(arr):+.4f}/{np.percentile(arr,99):+.4f}/{arr.max():+.4f}")
        # Very fine bins near the expected modes
        if "sell_arb" in name and "sell_x" in name:
            center, span = 0.0, 0.5
        elif "buy_arb" in name and "buy_x" in name:
            center, span = 0.0, 0.5
        else:
            center, span = 10.0, 0.5
        edges = np.arange(center - span, center + span + 0.025, 0.025)
        counts, edges = np.histogram(arr, bins=edges)
        for c, e0, e1 in zip(counts, edges[:-1], edges[1:]):
            if c >= 5:
                print(f"  [{e0:+.4f},{e1:+.4f})  {c}")

    describe("after_buy_arb -> retail buy_x", after_buy_arb_buy_x)
    describe("after_sell_arb -> retail buy_x", after_sell_arb_buy_x)
    describe("after_buy_arb -> retail sell_x", after_buy_arb_sell_x)
    describe("after_sell_arb -> retail sell_x", after_sell_arb_sell_x)


if __name__ == "__main__":
    main()
