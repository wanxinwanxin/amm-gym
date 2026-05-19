#!/usr/bin/env python3
"""Run the 2D calibration search and save results.

Usage: python presentation/run_calibration_2d.py
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arena_eval.exact_simple_amm.config import ExactSimpleAMMConfig
from arena_eval.exact_simple_amm.simulator import ExactSimpleAMMSimulator
from arena_eval.exact_simple_amm.strategies import FixedFeeStrategy

ANALYSIS_DIR = Path(__file__).resolve().parent.parent / "analysis" / "weth_usdc_90d"
REAL_POOL_FEE = 0.0005
REAL_VIRTUAL_USDC = 212_157_626.44
TARGET_VOL_SHARE = 0.311
TARGET_FEE_SHARE = 0.155


def build_sim(normalizer_fee, normalizer_depth_y, submission_depth_y=REAL_VIRTUAL_USDC,
              submission_fee=REAL_POOL_FEE, n_steps=5000, seed=42):
    initial_price = 100.0
    norm_y = normalizer_depth_y
    norm_x = norm_y / initial_price
    frac = submission_depth_y / norm_y

    cfg = ExactSimpleAMMConfig(
        n_steps=n_steps, initial_price=initial_price,
        initial_x=norm_x, initial_y=norm_y,
        submission_liquidity_fraction=frac,
        evaluator_kind="real_data",
        price_process_kind="regime_switching",
        retail_flow_kind="empirical_usd_size",
        retail_arrival_rate=1_294_178 / 1_303_200,
        retail_buy_prob=0.4842,
        regime_invcdf_path=str(ANALYSIS_DIR / "regimes_invcdf.csv"),
        regime_transition_path=str(ANALYSIS_DIR / "regimes_transition_matrix.csv"),
        retail_usd_quantiles_path=str(ANALYSIS_DIR / "parent_order_usd_quantiles.csv"),
    )
    return ExactSimpleAMMSimulator(
        config=cfg,
        submission_strategy=FixedFeeStrategy(bid_fee=submission_fee, ask_fee=submission_fee),
        normalizer_strategy=FixedFeeStrategy(bid_fee=normalizer_fee, ask_fee=normalizer_fee),
        seed=seed,
    )


def measure(normalizer_fee, depth, submission_fee=REAL_POOL_FEE, n_steps=5000, seeds=(42, 43)):
    total_sub_vol = total_norm_vol = 0.0
    for seed in seeds:
        sim = build_sim(normalizer_fee, depth, n_steps=n_steps, seed=seed)
        sim.run()
        res = sim.result()
        total_sub_vol += res.retail_volume_submission_y
        total_norm_vol += res.retail_volume_normalizer_y
    total_vol = total_sub_vol + total_norm_vol
    vol_share = total_sub_vol / total_vol if total_vol > 0 else 0.5
    sub_fees = total_sub_vol * submission_fee
    norm_fees = total_norm_vol * normalizer_fee
    total_fees = sub_fees + norm_fees
    fee_share = sub_fees / total_fees if total_fees > 0 else 0.5
    return {"vol_share": vol_share, "fee_share": fee_share}


def main():
    fee_grid = [0.0003, 0.0005, 0.001, 0.003, 0.005, 0.007, 0.008, 0.009, 0.01, 0.012, 0.015, 0.02, 0.03]
    n_steps = 5000
    seeds = (42, 43)
    full_log = []

    start = time.time()
    for nf in fee_grid:
        print(f"\n--- Normalizer fee = {nf * 1e4:.1f} bps ---")
        lo, hi = 1e7, 50e9
        for iteration in range(15):
            mid = math.sqrt(lo * hi)
            shares = measure(nf, mid, n_steps=n_steps, seeds=seeds)
            entry = {
                "normalizer_fee": nf,
                "normalizer_fee_bps": nf * 1e4,
                "depth_y": mid, "depth_M": mid / 1e6,
                "vol_share": shares["vol_share"],
                "fee_share": shares["fee_share"],
                "vol_share_error": shares["vol_share"] - TARGET_VOL_SHARE,
                "fee_share_error": shares["fee_share"] - TARGET_FEE_SHARE,
                "iteration": iteration,
            }
            full_log.append(entry)
            print(f"  [{iteration}] depth=${mid/1e6:,.1f}M → vol={shares['vol_share']*100:.1f}% fee={shares['fee_share']*100:.1f}%")
            if abs(shares["vol_share"] - TARGET_VOL_SHARE) < 0.005:
                break
            if shares["vol_share"] > TARGET_VOL_SHARE:
                lo = mid
            else:
                hi = mid

    elapsed = time.time() - start

    # Find best
    converged = {}
    for e in full_log:
        nf = e["normalizer_fee"]
        if nf not in converged or e["iteration"] > converged[nf]["iteration"]:
            converged[nf] = e
    converged_list = sorted(converged.values(), key=lambda x: x["normalizer_fee"])
    best = min(converged_list, key=lambda x: abs(x["fee_share_error"]))

    print(f"\n{'='*60}")
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"\nBest calibration:")
    print(f"  Normalizer fee:   {best['normalizer_fee_bps']:.1f} bps")
    print(f"  Normalizer depth: ${best['depth_M']:.1f}M")
    print(f"  Vol share: {best['vol_share']*100:.1f}% (target: {TARGET_VOL_SHARE*100:.1f}%)")
    print(f"  Fee share: {best['fee_share']*100:.1f}% (target: {TARGET_FEE_SHARE*100:.1f}%)")

    print(f"\nAll converged points:")
    for p in converged_list:
        marker = " ← BEST" if p == best else ""
        print(f"  fee={p['normalizer_fee_bps']:6.1f}bp  depth=${p['depth_M']:8.1f}M  "
              f"vol={p['vol_share']*100:5.1f}%  fee_share={p['fee_share']*100:5.1f}%{marker}")

    results = {"best": best, "converged": converged_list, "full_log": full_log, "elapsed_seconds": elapsed}
    out_path = Path(__file__).parent / "calibration_2d_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
