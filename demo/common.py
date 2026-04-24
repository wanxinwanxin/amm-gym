"""Shared rollout collection helpers for demo scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from amm_gym import AMMChallengeEnv, AMMFeeEnv


Policy = Callable[[np.ndarray], np.ndarray]


@dataclass
class RolloutTrace:
    steps: list[int]
    fair_prices: list[float]
    submission_spots: list[float]
    normalizer_spots: list[float]
    active_sigmas: list[float]
    edges: list[float]
    normalizer_edges: list[float]
    pnls: list[float]
    normalizer_pnls: list[float]
    rewards: list[float]
    imbalances: list[float]
    execution_counts: list[float]
    execution_volumes: list[float]
    normalizer_execution_counts: list[float]
    normalizer_execution_volumes: list[float]
    retail_volumes: list[float]
    normalizer_retail_volumes: list[float]
    arb_volumes: list[float]
    normalizer_arb_volumes: list[float]
    net_flows: list[float]
    ask_near: list[float]
    ask_far: list[float]
    bid_near: list[float]
    bid_far: list[float]
    actions: list[np.ndarray]
    bid_band_depths: list[np.ndarray]
    ask_band_depths: list[np.ndarray]
    band_centers_bps: np.ndarray
    band_widths_bps: np.ndarray
    final_info: dict[str, float]
    total_reward: float


def collect_rollout(env: AMMFeeEnv | AMMChallengeEnv, policy: Policy, *, seed: int) -> RolloutTrace:
    obs, _ = env.reset(seed=seed)

    trace = RolloutTrace(
        steps=[],
        fair_prices=[],
        submission_spots=[],
        normalizer_spots=[],
        active_sigmas=[],
        edges=[],
        normalizer_edges=[],
        pnls=[],
        normalizer_pnls=[],
        rewards=[],
        imbalances=[],
        execution_counts=[],
        execution_volumes=[],
        normalizer_execution_counts=[],
        normalizer_execution_volumes=[],
        retail_volumes=[],
        normalizer_retail_volumes=[],
        arb_volumes=[],
        normalizer_arb_volumes=[],
        net_flows=[],
        ask_near=[],
        ask_far=[],
        bid_near=[],
        bid_far=[],
        actions=[],
        bid_band_depths=[],
        ask_band_depths=[],
        band_centers_bps=np.zeros(0, dtype=np.float64),
        band_widths_bps=np.zeros(0, dtype=np.float64),
        final_info={},
        total_reward=0.0,
    )

    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = np.asarray(policy(obs), dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        ws = env.window_size
        trace.steps.append(int(info["step"]))
        trace.fair_prices.append(float(env.engine.current_fair_price))
        trace.submission_spots.append(float(info["spot_price"]))
        trace.normalizer_spots.append(float(env.engine.amm_norm.spot_price))
        trace.active_sigmas.append(float(env.engine._sigma_for_step(env.engine.current_step - 1)))
        trace.edges.append(float(info["edge"]))
        trace.normalizer_edges.append(float(info["edge_normalizer"]))
        trace.pnls.append(float(info["pnl"]))
        trace.normalizer_pnls.append(float(info["pnl_normalizer"]))
        trace.rewards.append(float(reward))
        trace.imbalances.append(float(obs[ws + 2]))
        trace.execution_counts.append(float(info["execution_count"]))
        trace.execution_volumes.append(float(info["execution_volume_y"]))
        trace.normalizer_execution_counts.append(float(info["execution_count_normalizer"]))
        trace.normalizer_execution_volumes.append(float(info["execution_volume_y_normalizer"]))
        trace.retail_volumes.append(float(info["retail_volume_y"]))
        trace.normalizer_retail_volumes.append(float(info["retail_volume_y_normalizer"]))
        trace.arb_volumes.append(float(info["arb_volume_y"]))
        trace.normalizer_arb_volumes.append(float(info["arb_volume_y_normalizer"]))
        trace.net_flows.append(float(info["net_flow_y"]))
        trace.ask_near.append(float(info.get("ask_near_depth_y", info.get("ask_spread_bps", 0.0))))
        trace.ask_far.append(float(info.get("ask_far_depth_y", info.get("ask_capacity_x", 0.0))))
        trace.bid_near.append(float(info.get("bid_near_depth_y", info.get("bid_spread_bps", 0.0))))
        trace.bid_far.append(float(info.get("bid_far_depth_y", info.get("bid_capacity_x", 0.0))))
        trace.actions.append(action.copy())
        ladder = env.engine.amm_agent
        if hasattr(ladder, "_band_lower") and hasattr(ladder, "band_rel"):
            centers_bps = 0.5 * (ladder._band_lower + ladder.band_rel) * 10_000.0
            widths_bps = (ladder.band_rel - ladder._band_lower) * 10_000.0
            trace.band_centers_bps = centers_bps.copy()
            trace.band_widths_bps = widths_bps.copy()
            trace.bid_band_depths.append((ladder.bid_depth_x * ladder.reference_price).copy())
            trace.ask_band_depths.append(ladder.ask_depth_y.copy())
        else:
            trace.band_centers_bps = np.asarray([0.0, 1.0], dtype=np.float64)
            trace.band_widths_bps = np.asarray([1.0, 1.0], dtype=np.float64)
            trace.bid_band_depths.append(np.asarray([info.get("bid_spread_bps", 0.0)], dtype=np.float64))
            trace.ask_band_depths.append(np.asarray([info.get("ask_spread_bps", 0.0)], dtype=np.float64))
        trace.total_reward += float(reward)
        trace.final_info = dict(info)

    return trace
