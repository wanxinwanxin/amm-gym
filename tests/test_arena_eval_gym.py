from __future__ import annotations

import numpy as np
import pytest

from arena_eval.exact_simple_amm import ExactSimpleAMMConfig
from arena_eval.gym import ExactSimpleAMMGymEnv


def test_gym_env_reset_and_step_shapes():
    env = ExactSimpleAMMGymEnv(config=ExactSimpleAMMConfig(n_steps=12))
    obs, info = env.reset(seed=5)

    assert obs.shape == env.observation_space.shape
    assert info["edge"] == 0.0

    next_obs, reward, terminated, truncated, step_info = env.step(np.array([0.01, 0.02], dtype=np.float32))
    assert next_obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "edge" in step_info
    assert "submission_trade" in step_info


def test_gym_env_is_seed_deterministic():
    env_a = ExactSimpleAMMGymEnv(config=ExactSimpleAMMConfig(n_steps=10))
    env_b = ExactSimpleAMMGymEnv(config=ExactSimpleAMMConfig(n_steps=10))

    obs_a, _ = env_a.reset(seed=9)
    obs_b, _ = env_b.reset(seed=9)
    np.testing.assert_allclose(obs_a, obs_b)

    rewards_a = []
    rewards_b = []
    action = np.array([0.003, 0.003], dtype=np.float32)
    for _ in range(10):
        _, reward_a, _, _, _ = env_a.step(action)
        _, reward_b, _, _, _ = env_b.step(action)
        rewards_a.append(reward_a)
        rewards_b.append(reward_b)
    np.testing.assert_allclose(rewards_a, rewards_b)


def test_gym_env_reward_sums_to_final_edge():
    env = ExactSimpleAMMGymEnv(config=ExactSimpleAMMConfig(n_steps=15))
    env.reset(seed=13)

    total_reward = 0.0
    final_info = None
    for _ in range(15):
        _, reward, _, _, final_info = env.step(np.array([0.003, 0.003], dtype=np.float32))
        total_reward += reward

    assert final_info is not None
    assert total_reward == pytest.approx(final_info["edge"], rel=1e-6)


def test_gym_env_does_not_expose_hidden_fair_price():
    env = ExactSimpleAMMGymEnv(config=ExactSimpleAMMConfig(n_steps=6))
    env.reset(seed=4)
    _, _, _, _, info = env.step(np.array([0.003, 0.003], dtype=np.float32))

    assert "fair_price" not in info
