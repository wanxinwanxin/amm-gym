"""Calibration metric evaluation harness for realistic simple-AMM simulator.

Exposes `evaluate_params(params, seeds, n_steps)` returning a dict with:
- arb_5bp_share (T1)
- retail_5bp_share (T2)
- markout_bps (T3) — volume-weighted instant LP markout on submission pool,
  combining retail + arb flow.

Used by the calibration driver in the same directory.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

# Ensure the worktree root is on sys.path so the editable arena_eval resolves locally.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arena_eval.exact_simple_amm.config import ExactSimpleAMMConfig
from arena_eval.exact_simple_amm.simulator import ExactSimpleAMMSimulator
from arena_eval.exact_simple_amm.strategies import FixedFeeStrategy

ANALYSIS_DIR = Path(__file__).resolve().parents[2] / "analysis" / "weth_usdc_90d"

SUBMISSION_FEE = 0.0005  # fixed 5 bps
EMPIRICAL_PARENT_ORDER_ARRIVAL_RATE = 1_294_178 / 1_303_200
EMPIRICAL_PARENT_ORDER_BUY_PROB = 0.4842


@dataclass(frozen=True)
class Params:
    submission_depth_y: float
    normalizer_fee: float
    normalizer_depth_y: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.submission_depth_y, self.normalizer_fee, self.normalizer_depth_y)


def _build_config(params: Params, n_steps: int) -> ExactSimpleAMMConfig:
    initial_price = 100.0
    norm_y = float(params.normalizer_depth_y)
    norm_x = norm_y / initial_price
    frac = float(params.submission_depth_y) / norm_y
    return ExactSimpleAMMConfig(
        n_steps=n_steps,
        initial_price=initial_price,
        initial_x=norm_x,
        initial_y=norm_y,
        submission_liquidity_fraction=frac,
        evaluator_kind="real_data",
        price_process_kind="regime_switching",
        retail_flow_kind="empirical_usd_size",
        retail_arrival_rate=EMPIRICAL_PARENT_ORDER_ARRIVAL_RATE,
        retail_buy_prob=EMPIRICAL_PARENT_ORDER_BUY_PROB,
        regime_invcdf_path=str(ANALYSIS_DIR / "regimes_invcdf.csv"),
        regime_transition_path=str(ANALYSIS_DIR / "regimes_transition_matrix.csv"),
        retail_usd_quantiles_path=str(ANALYSIS_DIR / "parent_order_usd_quantiles.csv"),
    )


def _run_one(params: Params, seed: int, n_steps: int) -> dict[str, float]:
    cfg = _build_config(params, n_steps)
    sim = ExactSimpleAMMSimulator(
        config=cfg,
        submission_strategy=FixedFeeStrategy(bid_fee=SUBMISSION_FEE, ask_fee=SUBMISSION_FEE),
        normalizer_strategy=FixedFeeStrategy(
            bid_fee=float(params.normalizer_fee), ask_fee=float(params.normalizer_fee)
        ),
        seed=int(seed),
    )
    res = sim.run()
    arb_total = res.arb_volume_submission_y + res.arb_volume_normalizer_y
    retail_total = res.retail_volume_submission_y + res.retail_volume_normalizer_y
    sub_total_y = res.arb_volume_submission_y + res.retail_volume_submission_y
    return {
        "arb_share": (res.arb_volume_submission_y / arb_total) if arb_total > 0 else float("nan"),
        "retail_share": (res.retail_volume_submission_y / retail_total) if retail_total > 0 else float("nan"),
        "markout_bps": (res.edge_submission / sub_total_y * 10_000.0) if sub_total_y > 0 else float("nan"),
        "arb_volume_submission_y": res.arb_volume_submission_y,
        "arb_volume_normalizer_y": res.arb_volume_normalizer_y,
        "retail_volume_submission_y": res.retail_volume_submission_y,
        "retail_volume_normalizer_y": res.retail_volume_normalizer_y,
        "edge_submission": res.edge_submission,
        "edge_normalizer": res.edge_normalizer,
        "retail_edge_submission": res.retail_edge_submission,
        "arb_loss_submission": res.arb_loss_submission,
    }


def evaluate_params(
    params: Params,
    seeds: Iterable[int],
    n_steps: int = 5000,
) -> dict[str, float]:
    """Return mean metrics across seeds plus per-seed values."""
    per_seed = [_run_one(params, s, n_steps) for s in seeds]
    arb_shares = np.array([r["arb_share"] for r in per_seed])
    retail_shares = np.array([r["retail_share"] for r in per_seed])
    markouts = np.array([r["markout_bps"] for r in per_seed])
    return {
        "arb_5bp_share": float(np.nanmean(arb_shares)),
        "retail_5bp_share": float(np.nanmean(retail_shares)),
        "markout_bps": float(np.nanmean(markouts)),
        "per_seed": per_seed,
        "seeds": tuple(int(s) for s in seeds),
        "n_steps": int(n_steps),
        "params": {
            "submission_depth_y": float(params.submission_depth_y),
            "normalizer_fee": float(params.normalizer_fee),
            "normalizer_depth_y": float(params.normalizer_depth_y),
        },
    }


def residuals(metrics: dict[str, float], targets: dict[str, float]) -> dict[str, float]:
    return {
        "T1_arb_5bp_share": (metrics["arb_5bp_share"] - targets["T1"]) / targets["T1"],
        "T2_retail_5bp_share": (metrics["retail_5bp_share"] - targets["T2"]) / targets["T2"],
        "T3_markout_bps": (metrics["markout_bps"] - targets["T3"]) / targets["T3"],
    }


TARGETS = {
    "T1": 0.337330,  # arb 5bp share (USD-weighted, 7d BigQuery 2026-05-14..2026-05-20)
    "T2": 0.782049,  # retail 5bp share (USD-weighted, same window)
    "T3": -1.05,     # USD-volume-weighted next-block LP markout bps (same window).
                     # Revised 2026-05-21 from prior +3.637 (which was a single-day
                     # simple per-swap average, dominated by small retail swaps).
}

CALIB_SEEDS = (42, 43, 44, 45, 46)
HOLDOUT_SEEDS = (100, 101, 102, 103, 104)
N_STEPS = 5000


if __name__ == "__main__":
    init = Params(
        submission_depth_y=212_157_626,
        normalizer_fee=0.001,
        normalizer_depth_y=17_000_000_000,
    )
    metrics = evaluate_params(init, CALIB_SEEDS, N_STEPS)
    res = residuals(metrics, TARGETS)
    print("BASELINE — calibration seeds:")
    print(f"  T1 arb_5bp_share    = {metrics['arb_5bp_share']:.4f}  (target {TARGETS['T1']:.4f}, residual {res['T1_arb_5bp_share']:+.3%})")
    print(f"  T2 retail_5bp_share = {metrics['retail_5bp_share']:.4f}  (target {TARGETS['T2']:.4f}, residual {res['T2_retail_5bp_share']:+.3%})")
    print(f"  T3 markout_bps      = {metrics['markout_bps']:.4f}  (target {TARGETS['T3']:.4f}, residual {res['T3_markout_bps']:+.3%})")
