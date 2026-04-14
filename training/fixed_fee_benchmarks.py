"""Fixed-fee benchmark sweep for the training worktree.

This module keeps the first-pass benchmark intentionally small: it evaluates
static symmetric fee policies only and reports aggregate outcome statistics.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from amm_gym import AMMFeeEnv
from amm_gym.sim.engine import SimConfig


DEFAULT_FEE_SWEEP_BPS: tuple[int, ...] = (0, 10, 20, 30, 40, 50, 60)


@dataclass(frozen=True)
class FeeBenchmarkRow:
    fee_bps: int
    applied_fee_bps: float
    mean_reward: float
    std_reward: float
    mean_edge: float
    std_edge: float
    mean_pnl: float
    std_pnl: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class FeeBenchmarkResult:
    num_seeds: int
    episode_length: int
    rows: list[FeeBenchmarkRow]

    def to_payload(self) -> dict[str, object]:
        return {
            "num_seeds": self.num_seeds,
            "episode_length": self.episode_length,
            "rows": [row.as_dict() for row in self.rows],
        }


def _static_symmetric_action(fee_bps: int) -> np.ndarray:
    fee = float(fee_bps) / 10_000.0
    return np.array([fee, fee], dtype=np.float32)


def run_single_episode(
    *,
    fee_bps: int,
    seed: int,
    episode_length: int,
    window_size: int = 10,
) -> dict[str, float]:
    if episode_length <= 0:
        raise ValueError("episode_length must be positive")

    env = AMMFeeEnv(config=SimConfig(n_steps=episode_length), window_size=window_size)
    try:
        _, info = env.reset(seed=seed)

        action = _static_symmetric_action(fee_bps)
        total_reward = 0.0
        terminated = False
        truncated = False
        applied_fee_bps = float(fee_bps)

        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(action)
            del obs
            total_reward += float(reward)
            applied_fee_bps = float(info["bid_fee"]) * 10_000.0

        return {
            "reward": total_reward,
            "edge": float(info["edge"]),
            "pnl": float(info["pnl"]),
            "applied_fee_bps": applied_fee_bps,
        }
    finally:
        env.close()


def run_fixed_fee_sweep(
    *,
    num_seeds: int,
    episode_length: int,
    fee_bps_values: Sequence[int] = DEFAULT_FEE_SWEEP_BPS,
    window_size: int = 10,
) -> FeeBenchmarkResult:
    if num_seeds <= 0:
        raise ValueError("num_seeds must be positive")
    if episode_length <= 0:
        raise ValueError("episode_length must be positive")

    fee_bps_values = tuple(fee_bps_values)
    if not fee_bps_values:
        raise ValueError("fee_bps_values must not be empty")

    rows: list[FeeBenchmarkRow] = []
    seeds = list(range(num_seeds))

    for fee_bps in fee_bps_values:
        episode_metrics = [
            run_single_episode(
                fee_bps=fee_bps,
                seed=seed,
                episode_length=episode_length,
                window_size=window_size,
            )
            for seed in seeds
        ]

        rewards = np.asarray([m["reward"] for m in episode_metrics], dtype=np.float64)
        edges = np.asarray([m["edge"] for m in episode_metrics], dtype=np.float64)
        pnls = np.asarray([m["pnl"] for m in episode_metrics], dtype=np.float64)
        applied_fees = np.asarray(
            [m["applied_fee_bps"] for m in episode_metrics], dtype=np.float64
        )

        rows.append(
            FeeBenchmarkRow(
                fee_bps=int(fee_bps),
                applied_fee_bps=float(applied_fees.mean()),
                mean_reward=float(rewards.mean()),
                std_reward=float(rewards.std(ddof=0)),
                mean_edge=float(edges.mean()),
                std_edge=float(edges.std(ddof=0)),
                mean_pnl=float(pnls.mean()),
                std_pnl=float(pnls.std(ddof=0)),
            )
        )

    return FeeBenchmarkResult(
        num_seeds=num_seeds,
        episode_length=episode_length,
        rows=rows,
    )


def format_summary_table(result: FeeBenchmarkResult) -> str:
    headers = (
        "fee_bps",
        "applied_bps",
        "mean_reward",
        "std_reward",
        "mean_edge",
        "std_edge",
        "mean_pnl",
        "std_pnl",
    )
    widths = {
        "fee_bps": 8,
        "applied_bps": 11,
        "mean_reward": 12,
        "std_reward": 11,
        "mean_edge": 11,
        "std_edge": 10,
        "mean_pnl": 10,
        "std_pnl": 9,
    }

    lines = [
        "Fixed-fee benchmark sweep",
        f"  num_seeds={result.num_seeds} episode_length={result.episode_length}",
        "",
        " ".join(name.rjust(widths[name]) for name in headers),
    ]
    for row in result.rows:
        lines.append(
            " ".join(
                [
                    f"{row.fee_bps:>{widths['fee_bps']}d}",
                    f"{row.applied_fee_bps:>{widths['applied_bps']}.2f}",
                    f"{row.mean_reward:>{widths['mean_reward']}.6f}",
                    f"{row.std_reward:>{widths['std_reward']}.6f}",
                    f"{row.mean_edge:>{widths['mean_edge']}.6f}",
                    f"{row.std_edge:>{widths['std_edge']}.6f}",
                    f"{row.mean_pnl:>{widths['mean_pnl']}.6f}",
                    f"{row.std_pnl:>{widths['std_pnl']}.6f}",
                ]
            )
        )
    return "\n".join(lines)


def save_benchmark_result(result: FeeBenchmarkResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.to_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

