"""Validate the spike mechanism: split markouts by (arb_dir, retail_side)."""
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

    submission_fee = 0.0005

    # Buckets by (arb_dir, retail_dir)
    buckets: dict[tuple[str, str], list[float]] = {
        ("buy_arb", "buy_x"): [],
        ("buy_arb", "sell_x"): [],
        ("sell_arb", "buy_x"): [],
        ("sell_arb", "sell_x"): [],
        ("no_arb", "buy_x"): [],
        ("no_arb", "sell_x"): [],
    }
    arb_fired_steps = {"sub": 0, "norm": 0, "neither": 0, "either": 0}
    total_steps = 0

    for seed in seeds:
        sim = _build_sim(
            p["normalizer_fee"],
            p["normalizer_depth_y"],
            p["submission_depth_y"],
            submission_fee,
            n_steps,
            seed,
        )
        while not sim.done:
            step = sim.step_once()
            total_steps += 1
            # Did arb fire on submission?  Look at trade_events for venue=submission, source=arb
            arb_dir_sub = "no_arb"
            for ev in step["trade_events"]:
                if ev["source"] != "arb":
                    continue
                if ev["venue"] != "submission":
                    continue
                arb_dir_sub = "buy_arb" if ev["trader_side"] == "buy_x" else "sell_arb"
                arb_fired_steps["sub"] += 1
                break
            else:
                arb_fired_steps["neither"] += 1

            # Then retail trades on submission
            for ev in step["trade_events"]:
                if ev["source"] != "retail" or ev["venue"] != "submission":
                    continue
                fair = step["fair_price"]
                if ev["trader_side"] == "sell_x":
                    edge = ev["amount_x"] * fair - ev["amount_y"]
                else:
                    edge = ev["amount_y"] - ev["amount_x"] * fair
                vol_y = ev["amount_y"]
                if vol_y > 0:
                    bps = edge / vol_y * 10_000
                    buckets[(arb_dir_sub, ev["trader_side"])].append(bps)

    print(f"Total steps: {total_steps}.  Arb fired on submission: {arb_fired_steps['sub']} "
          f"({arb_fired_steps['sub']/total_steps*100:.2f}%).  No arb: {arb_fired_steps['neither']} "
          f"({arb_fired_steps['neither']/total_steps*100:.2f}%).\n")
    print(f"{'arb_dir':>10}  {'retail':>8}  {'count':>8}  {'mode_bps':>10}  {'median':>8}  {'p1':>7}  {'p99':>7}")
    for key, vals in buckets.items():
        arr = np.asarray(vals)
        if arr.size == 0:
            print(f"{key[0]:>10}  {key[1]:>8}  {0:>8}  (empty)")
            continue
        # mode = bin center with highest count using 0.05 bps bins
        edges = np.arange(-5, 20, 0.05)
        counts, edges = np.histogram(arr, bins=edges)
        idx = counts.argmax()
        mode = 0.5 * (edges[idx] + edges[idx + 1])
        print(f"{key[0]:>10}  {key[1]:>8}  {arr.size:>8}  {mode:>+10.3f}  "
              f"{np.median(arr):>+8.3f}  {np.percentile(arr,1):>+7.3f}  {np.percentile(arr,99):>+7.3f}")


if __name__ == "__main__":
    main()
