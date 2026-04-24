from __future__ import annotations

import numpy as np
import pytest

from amm_gym import AMMFeeEnv
from amm_gym.sim.engine import SimConfig
from amm_gym.sim.venues import VenueSpec
from training.eval import aggregate_episode_metrics, evaluate_policy_across_scenarios
from training.eval.metrics import evaluate_episode
from training.eval.metrics import score_episode_metrics
from training.policies import research_benchmark_policies
from training.policies.mlp import MLPPolicySpace


def test_episode_metrics_include_relative_fields():
    policy = research_benchmark_policies()["balanced"]

    def env_factory() -> AMMFeeEnv:
        return AMMFeeEnv(config=SimConfig(n_steps=15), window_size=6)

    metrics = evaluate_episode(env_factory, policy, seed=3)

    assert metrics.seed == 3
    assert metrics.edge_advantage == metrics.edge - metrics.edge_normalizer
    assert 0.0 <= metrics.retail_share <= 1.0
    assert 0.0 <= metrics.arb_share <= 1.0


def test_aggregate_metrics_tracks_edge_advantage_statistics():
    policy = research_benchmark_policies()["balanced"]

    def env_factory() -> AMMFeeEnv:
        return AMMFeeEnv(config=SimConfig(n_steps=12), window_size=6)

    episodes = [evaluate_episode(env_factory, policy, seed=seed) for seed in (1, 2)]
    summary = aggregate_episode_metrics(episodes)

    assert summary["n_episodes"] == 2
    assert "edge_advantage_mean" in summary
    assert "edge_advantage_win_rate" in summary
    assert isinstance(summary["per_seed"], list)


def test_benchmark_runner_is_deterministic_for_fixed_split():
    policy = research_benchmark_policies()["inventory_skew"]
    first = evaluate_policy_across_scenarios(
        policy_name="inventory_skew",
        policy=policy,
        scenario_names=("constant_low_vol",),
        split_name="validation",
        window_size=6,
    )
    second = evaluate_policy_across_scenarios(
        policy_name="inventory_skew",
        policy=policy,
        scenario_names=("constant_low_vol",),
        split_name="validation",
        window_size=6,
    )

    assert first == second


def test_mlp_policy_space_param_dim_is_positive():
    env = AMMFeeEnv(config=SimConfig(n_steps=10), window_size=6)
    space = MLPPolicySpace.from_env(env, hidden_sizes=(8, 4))

    assert space.param_dim > 0


def test_score_episode_metrics_balanced_objective_is_finite():
    policy = research_benchmark_policies()["balanced"]

    def env_factory() -> AMMFeeEnv:
        return AMMFeeEnv(config=SimConfig(n_steps=10), window_size=6)

    metrics = evaluate_episode(env_factory, policy, seed=5)
    score = score_episode_metrics(metrics, "balanced")

    assert isinstance(score, float)


def test_symmetric_cpmm_comparison_is_a_tie():
    cpmm_submission = VenueSpec(
        kind="cpmm",
        name="submission_cpmm",
        reserve_x=100.0,
        reserve_y=10_000.0,
        bid_fee=0.003,
        ask_fee=0.003,
        controllable=False,
    )
    cpmm_benchmark = VenueSpec(
        kind="cpmm",
        name="benchmark_cpmm",
        reserve_x=100.0,
        reserve_y=10_000.0,
        bid_fee=0.003,
        ask_fee=0.003,
        controllable=False,
    )

    def env_factory() -> AMMFeeEnv:
        return AMMFeeEnv(
            config=SimConfig(
                n_steps=40,
                seed=13,
                submission_venue=cpmm_submission,
                benchmark_venue=cpmm_benchmark,
            ),
            window_size=6,
        )

    metrics = evaluate_episode(env_factory, lambda obs: np.zeros(6, dtype=np.float32), seed=13)

    assert metrics.edge_advantage == pytest.approx(0.0, abs=1e-9)
    assert metrics.pnl_advantage == pytest.approx(0.0, abs=1e-9)
    assert metrics.retail_share == pytest.approx(0.5, abs=1e-9)
    assert metrics.arb_share == pytest.approx(0.5, abs=1e-9)


def test_symmetric_ladder_comparison_is_a_tie():
    action = (0.15, -0.2, 0.05, 0.15, -0.2, 0.05)
    ladder_submission = VenueSpec(
        kind="depth_ladder",
        name="submission",
        reserve_x=100.0,
        reserve_y=10_000.0,
        band_bps=(2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0),
        base_notional_y=1_000.0,
        controllable=True,
    )
    ladder_benchmark = VenueSpec(
        kind="depth_ladder",
        name="benchmark_ladder",
        reserve_x=100.0,
        reserve_y=10_000.0,
        band_bps=(2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0),
        base_notional_y=1_000.0,
        controllable=False,
        fixed_action=action,
    )

    def env_factory() -> AMMFeeEnv:
        return AMMFeeEnv(
            config=SimConfig(
                n_steps=40,
                seed=17,
                submission_venue=ladder_submission,
                benchmark_venue=ladder_benchmark,
            ),
            window_size=6,
        )

    metrics = evaluate_episode(
        env_factory,
        lambda obs: np.asarray(action, dtype=np.float32),
        seed=17,
    )

    assert metrics.edge_advantage == pytest.approx(0.0, abs=1e-7)
    assert metrics.pnl_advantage == pytest.approx(0.0, abs=1e-7)
    assert metrics.retail_share == pytest.approx(0.5, abs=1e-7)
    assert metrics.arb_share == pytest.approx(0.5, abs=1e-7)
