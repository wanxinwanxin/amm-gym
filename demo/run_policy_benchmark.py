"""Benchmark research and builder-facing policies across fixed scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from demo.presets import named_schedules
from training.eval import benchmark_scenarios, default_seed_split, evaluate_policy_across_scenarios
from training.eval.plots import save_edge_advantage_plot
from training.policies import research_benchmark_policies


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark hand-authored ladder policies")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--scenario", default="regime_shift", choices=sorted(benchmark_scenarios()))
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--plot", type=Path, default=None)
    return parser


def evaluate_policy(
    policy_name: str,
    policy,
    schedule_name: str,
    schedule: tuple[tuple[int, float], ...] | None,
    steps: int,
    window_size: int,
    seeds: int,
) -> dict[str, float | str]:
    del schedule, steps, seeds
    scenario_name = schedule_name
    result = evaluate_policy_across_scenarios(
        policy_name=policy_name,
        policy=policy,
        scenario_names=(scenario_name,),
        split_name="validation",
        window_size=window_size,
        seed_split=default_seed_split(),
    )
    summary = result["scenarios"][scenario_name]["summary"]
    return {
        "policy": policy_name,
        "schedule": scenario_name,
        "mean_reward": float(summary["total_reward"]["mean"]),
        "mean_edge": float(summary["edge"]["mean"]),
        "mean_pnl": float(summary["pnl"]["mean"]),
        "mean_execution_volume_y": float(summary["execution_volume_y"]["mean"]),
        "mean_execution_count": 0.0,
    }


def main() -> None:
    args = build_parser().parse_args()
    rows: list[dict[str, object]] = []
    scenario = args.scenario
    for policy_name, policy in research_benchmark_policies().items():
        result = evaluate_policy_across_scenarios(
            policy_name=policy_name,
            policy=policy,
            scenario_names=(scenario,),
            split_name=args.split,
            window_size=args.window_size,
        )
        summary = result["scenarios"][scenario]["summary"]
        rows.append(
            {
                "policy": policy_name,
                "scenario": scenario,
                "edge_advantage_mean": summary["edge_advantage_mean"],
                "edge_advantage_win_rate": summary["edge_advantage_win_rate"],
                "mean_edge": summary["edge"]["mean"],
                "mean_edge_normalizer": summary["edge_normalizer"]["mean"],
                "mean_pnl": summary["pnl"]["mean"],
                "mean_pnl_normalizer": summary["pnl_normalizer"]["mean"],
                "mean_retail_share": summary["retail_share"]["mean"],
                "mean_arb_share": summary["arb_share"]["mean"],
            }
        )

    rows.sort(key=lambda row: float(row["edge_advantage_mean"]), reverse=True)
    print(json.dumps(rows, indent=2))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2))
    if args.plot is not None:
        save_edge_advantage_plot(args.plot, rows)


if __name__ == "__main__":
    main()
