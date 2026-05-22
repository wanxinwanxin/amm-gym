"""Drill into the no_arb sub-population to characterise its distribution."""
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

    no_arb_buy: list[float] = []
    no_arb_sell: list[float] = []

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
            sub_arb_fired = any(
                ev["source"] == "arb" and ev["venue"] == "submission"
                for ev in step["trade_events"]
            )
            if sub_arb_fired:
                continue
            for ev in step["trade_events"]:
                if ev["source"] != "retail" or ev["venue"] != "submission":
                    continue
                fair = step["fair_price"]
                if ev["trader_side"] == "sell_x":
                    edge = ev["amount_x"] * fair - ev["amount_y"]
                    bps = edge / ev["amount_y"] * 10_000 if ev["amount_y"] > 0 else 0.0
                    no_arb_sell.append(bps)
                else:
                    edge = ev["amount_y"] - ev["amount_x"] * fair
                    bps = edge / ev["amount_y"] * 10_000 if ev["amount_y"] > 0 else 0.0
                    no_arb_buy.append(bps)

    for name, arr in (("no_arb_buy_x", np.asarray(no_arb_buy)),
                      ("no_arb_sell_x", np.asarray(no_arb_sell))):
        print(f"\n{name}: {arr.size:,} samples")
        print(f"  range: {arr.min():+.3f} .. {arr.max():+.3f} bps")
        # 0.5 bps bins
        edges = np.arange(-5, 20, 0.5)
        counts, edges = np.histogram(arr, bins=edges)
        for c, e0, e1 in zip(counts, edges[:-1], edges[1:]):
            bar = "#" * min(60, c // 10)
            print(f"  [{e0:+6.1f},{e1:+6.1f})  {c:5d}  {bar}")


if __name__ == "__main__":
    main()
