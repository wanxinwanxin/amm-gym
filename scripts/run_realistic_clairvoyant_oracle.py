"""Benchmark clairvoyant controllers on realistic empirical-impact tapes."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arena_eval.exact_simple_amm import (
    ExactSimpleAMMConfig,
    FixedFeeClairvoyantController,
    GreedyStepOracleController,
    StructuredRetailOracleController,
    run_realistic_clairvoyant_batch,
)


def _seed_range(start: int, count: int) -> tuple[int, ...]:
    return tuple(range(start, start + count))


def _config_factory(n_steps: int | None, submission_liquidity_fraction: float):
    def factory(seed: int) -> ExactSimpleAMMConfig:
        config = ExactSimpleAMMConfig.real_data_from_seed(seed)
        updates = {"submission_liquidity_fraction": submission_liquidity_fraction}
        if n_steps is not None:
            updates["n_steps"] = n_steps
        return replace(config, **updates)

    return factory


def _batch_payload(result) -> dict[str, object]:
    return {
        "score": result.score,
        "edge_mean_submission": result.edge_mean_submission,
        "edge_mean_normalizer": result.edge_mean_normalizer,
        "edge_advantage_mean": result.edge_advantage_mean,
        "pnl_mean_submission": result.pnl_mean_submission,
        "pnl_mean_normalizer": result.pnl_mean_normalizer,
        "pnl_advantage_mean": result.pnl_advantage_mean,
        "retail_edge_mean_submission": result.retail_edge_mean_submission,
        "retail_edge_mean_normalizer": result.retail_edge_mean_normalizer,
        "arb_loss_mean_submission": result.arb_loss_mean_submission,
        "arb_loss_mean_normalizer": result.arb_loss_mean_normalizer,
        "annualized_edge_return_mean_submission": result.annualized_edge_return_mean_submission,
        "annualized_retail_edge_return_mean_submission": result.annualized_retail_edge_return_mean_submission,
        "annualized_arb_loss_return_mean_submission": result.annualized_arb_loss_return_mean_submission,
        "retail_markout_bps_mean_submission": result.retail_markout_bps_mean_submission,
        "arb_markout_bps_mean_submission": result.arb_markout_bps_mean_submission,
        "initial_value_mean": result.initial_value_mean,
        "episode_seconds_mean": result.episode_seconds_mean,
        "metadata": dict(result.metadata),
    }


def _controller_factory(name: str, *, fee_grid_size: int, baseline_fee: float):
    if name == "greedy_step":
        return lambda: GreedyStepOracleController(fee_grid_size=fee_grid_size, fallback_fee=baseline_fee)
    if name == "structured_retail":
        return lambda: StructuredRetailOracleController(fee_grid_size=fee_grid_size, fallback_fee=baseline_fee)
    raise ValueError(f"Unsupported controller: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run clairvoyant oracle benchmarks on realistic tapes")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=32)
    parser.add_argument("--n-steps", type=int, default=None, help="Optional override for shorter diagnostic runs")
    parser.add_argument("--baseline-fee", type=float, default=0.003)
    parser.add_argument(
        "--submission-liquidity-fraction",
        type=float,
        default=1.0,
        help="Initial submission-pool liquidity as a fraction of the normalizer pool",
    )
    parser.add_argument("--controller", choices=("greedy_step", "structured_retail"), default="structured_retail")
    parser.add_argument("--fee-grid-size", type=int, default=1001)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    seeds = _seed_range(args.seed_start, args.seed_count)
    config_factory = _config_factory(args.n_steps, args.submission_liquidity_fraction)

    baseline = run_realistic_clairvoyant_batch(
        lambda: FixedFeeClairvoyantController(args.baseline_fee, args.baseline_fee),
        seeds,
        normalizer_fee=args.baseline_fee,
        config_factory=config_factory,
    )
    controller_factory = _controller_factory(
        args.controller,
        fee_grid_size=args.fee_grid_size,
        baseline_fee=args.baseline_fee,
    )
    oracle = run_realistic_clairvoyant_batch(
        controller_factory,
        seeds,
        normalizer_fee=args.baseline_fee,
        config_factory=config_factory,
    )

    payload = {
        "seeds": list(seeds),
        "evaluator_kind": "real_data",
        "n_steps_override": args.n_steps,
        "baseline_fee": args.baseline_fee,
        "submission_liquidity_fraction": args.submission_liquidity_fraction,
        "controller": args.controller,
        "fee_grid_size": args.fee_grid_size,
        "fixed_fee": _batch_payload(baseline),
        f"{args.controller}_oracle": _batch_payload(oracle),
    }

    print(json.dumps(payload, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
