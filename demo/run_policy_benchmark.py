"""Builder-facing benchmark over hand-authored ladder policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from amm_gym import AMMFeeEnv
from amm_gym.baselines import benchmark_depth_policies
from demo.presets import (
    DEFAULT_DEMO_STEPS,
    DEFAULT_WINDOW_SIZE,
    build_hackathon_demo_config,
    named_schedules,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark hand-authored ladder policies")
    parser.add_argument("--steps", type=int, default=DEFAULT_DEMO_STEPS)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
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
    rewards: list[float] = []
    edges: list[float] = []
    pnls: list[float] = []
    volumes: list[float] = []
    counts: list[float] = []

    for seed in range(seeds):
        env = AMMFeeEnv(
            config=build_hackathon_demo_config(
                seed=seed,
                steps=steps,
                schedule=schedule,
            ),
            window_size=window_size,
        )
        obs, _ = env.reset(seed=seed)
        total_reward = 0.0
        total_volume = 0.0
        total_count = 0.0
        terminated = False
        truncated = False
        info: dict[str, float] = {}

        while not (terminated or truncated):
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_volume += float(info["execution_volume_y"])
            total_count += float(info["execution_count"])

        rewards.append(total_reward)
        edges.append(float(info["edge"]))
        pnls.append(float(info["pnl"]))
        volumes.append(total_volume)
        counts.append(total_count)

    return {
        "policy": policy_name,
        "schedule": schedule_name,
        "mean_reward": float(np.mean(rewards)),
        "mean_edge": float(np.mean(edges)),
        "mean_pnl": float(np.mean(pnls)),
        "mean_execution_volume_y": float(np.mean(volumes)),
        "mean_execution_count": float(np.mean(counts)),
    }


def format_table(rows: list[dict[str, float | str]]) -> str:
    headers = [
        "policy",
        "schedule",
        "mean_reward",
        "mean_edge",
        "mean_pnl",
        "mean_execution_volume_y",
        "mean_execution_count",
    ]
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            value = row[header]
            text = f"{value:.4f}" if isinstance(value, float) else str(value)
            widths[header] = max(widths[header], len(text))

    lines = []
    lines.append("  ".join(header.ljust(widths[header]) for header in headers))
    lines.append("  ".join("-" * widths[header] for header in headers))
    for row in rows:
        rendered = []
        for header in headers:
            value = row[header]
            text = f"{value:.4f}" if isinstance(value, float) else str(value)
            rendered.append(text.ljust(widths[header]))
        lines.append("  ".join(rendered))
    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()
    rows: list[dict[str, float | str]] = []
    for schedule_name, schedule in named_schedules().items():
        for policy_name, policy in benchmark_depth_policies().items():
            rows.append(
                evaluate_policy(
                    policy_name=policy_name,
                    policy=policy,
                    schedule_name=schedule_name,
                    schedule=schedule,
                    steps=args.steps,
                    window_size=args.window_size,
                    seeds=args.seeds,
                )
            )

    print(format_table(rows))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
