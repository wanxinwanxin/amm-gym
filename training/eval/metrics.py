"""Episode metrics and relative-to-normalizer summaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from statistics import mean, median
from typing import Callable

import numpy as np


Policy = Callable[[np.ndarray], np.ndarray]
EnvFactory = Callable[[], object]


@dataclass
class EpisodeMetrics:
    seed: int
    total_reward: float
    edge: float
    edge_normalizer: float
    edge_advantage: float
    pnl: float
    pnl_normalizer: float
    pnl_advantage: float
    execution_volume_y: float
    execution_volume_y_normalizer: float
    retail_volume_y: float
    retail_volume_y_normalizer: float
    arb_volume_y: float
    arb_volume_y_normalizer: float
    retail_share: float
    arb_share: float
    max_abs_imbalance: float
    mean_abs_imbalance: float


def score_episode_metrics(
    metrics: EpisodeMetrics,
    objective: str,
) -> float:
    if objective == "reward":
        return float(metrics.total_reward)
    if objective == "edge":
        return float(metrics.edge)
    if objective == "edge_advantage":
        return float(metrics.edge_advantage)
    if objective == "pnl":
        return float(metrics.pnl)
    if objective == "pnl_advantage":
        return float(metrics.pnl_advantage)
    if objective == "balanced":
        return float(
            metrics.edge_advantage
            + 0.05 * metrics.pnl_advantage
            + 0.5 * metrics.retail_share
            - 10.0 * metrics.mean_abs_imbalance
        )
    raise ValueError(f"unknown objective `{objective}`")


def evaluate_episode(env_factory: EnvFactory, policy: Policy, *, seed: int) -> EpisodeMetrics:
    reset_policy = getattr(policy, "reset", None)
    if callable(reset_policy):
        reset_policy()
    env = env_factory()
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    execution_volume_y = 0.0
    execution_volume_y_normalizer = 0.0
    retail_volume_y = 0.0
    retail_volume_y_normalizer = 0.0
    arb_volume_y = 0.0
    arb_volume_y_normalizer = 0.0
    imbalance_abs_sum = 0.0
    max_abs_imbalance = 0.0
    n_steps = 0

    terminated = False
    truncated = False
    info: dict[str, float] = {}
    while not (terminated or truncated):
        action = np.asarray(policy(obs), dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        execution_volume_y += float(info["execution_volume_y"])
        execution_volume_y_normalizer += float(info["execution_volume_y_normalizer"])
        retail_volume_y += float(info["retail_volume_y"])
        retail_volume_y_normalizer += float(info["retail_volume_y_normalizer"])
        arb_volume_y += float(info["arb_volume_y"])
        arb_volume_y_normalizer += float(info["arb_volume_y_normalizer"])
        ws = env.window_size
        imbalance = abs(float(obs[ws + 2]))
        imbalance_abs_sum += imbalance
        max_abs_imbalance = max(max_abs_imbalance, imbalance)
        n_steps += 1

    retail_total = retail_volume_y + retail_volume_y_normalizer
    arb_total = arb_volume_y + arb_volume_y_normalizer
    edge = float(info["edge"])
    edge_normalizer = float(info["edge_normalizer"])
    pnl = float(info["pnl"])
    pnl_normalizer = float(info["pnl_normalizer"])
    return EpisodeMetrics(
        seed=int(seed),
        total_reward=total_reward,
        edge=edge,
        edge_normalizer=edge_normalizer,
        edge_advantage=edge - edge_normalizer,
        pnl=pnl,
        pnl_normalizer=pnl_normalizer,
        pnl_advantage=pnl - pnl_normalizer,
        execution_volume_y=execution_volume_y,
        execution_volume_y_normalizer=execution_volume_y_normalizer,
        retail_volume_y=retail_volume_y,
        retail_volume_y_normalizer=retail_volume_y_normalizer,
        arb_volume_y=arb_volume_y,
        arb_volume_y_normalizer=arb_volume_y_normalizer,
        retail_share=retail_volume_y / retail_total if retail_total > 0 else 0.0,
        arb_share=arb_volume_y / arb_total if arb_total > 0 else 0.0,
        max_abs_imbalance=max_abs_imbalance,
        mean_abs_imbalance=imbalance_abs_sum / max(n_steps, 1),
    )


def aggregate_episode_metrics(metrics: list[EpisodeMetrics]) -> dict[str, object]:
    if not metrics:
        raise ValueError("metrics must be non-empty")

    summary: dict[str, object] = {
        "n_episodes": len(metrics),
        "per_seed": [asdict(metric) for metric in metrics],
    }
    numeric_fields = [field.name for field in fields(EpisodeMetrics) if field.name != "seed"]
    for name in numeric_fields:
        values = np.asarray([getattr(metric, name) for metric in metrics], dtype=np.float64)
        summary[name] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std": float(values.std()),
            "p10": float(np.percentile(values, 10)),
            "p90": float(np.percentile(values, 90)),
        }

    edge_advantages = [metric.edge_advantage for metric in metrics]
    summary["edge_advantage_win_rate"] = float(mean(1.0 if value > 0.0 else 0.0 for value in edge_advantages))
    summary["edge_advantage_mean"] = float(mean(edge_advantages))
    summary["edge_advantage_median"] = float(median(edge_advantages))
    return summary
