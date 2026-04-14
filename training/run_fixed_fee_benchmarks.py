"""CLI entrypoint for the fixed-fee benchmark sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

from training.fixed_fee_benchmarks import (
    DEFAULT_FEE_SWEEP_BPS,
    format_summary_table,
    run_fixed_fee_sweep,
    save_benchmark_result,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a fixed-fee benchmark sweep against the current env"
    )
    parser.add_argument("--num-seeds", type=_positive_int, default=5)
    parser.add_argument("--episode-length", type=_positive_int, default=500)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    result = run_fixed_fee_sweep(
        num_seeds=args.num_seeds,
        episode_length=args.episode_length,
        fee_bps_values=DEFAULT_FEE_SWEEP_BPS,
    )

    print(format_summary_table(result))

    if args.output is not None:
        save_benchmark_result(result, args.output)
        print(f"\nSaved benchmark output to {args.output}")


if __name__ == "__main__":
    main()

