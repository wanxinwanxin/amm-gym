"""Tests for the Gymnasium environment."""

import numpy as np
import pytest

from amm_gym.env import ACTION_DIM, ACTION_MAX, ACTION_MIN, AMMFeeEnv
from amm_gym.baselines import StaticDepthPolicy
from amm_gym.sim.engine import SimConfig


@pytest.fixture
def short_config():
    return SimConfig(n_steps=100, seed=42)


class TestEnvAPI:
    def test_reset_returns_obs_and_info(self, short_config):
        env = AMMFeeEnv(config=short_config)
        obs, info = env.reset(seed=42)
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32
        assert "edge" in info

    def test_action_space_is_six_dimensional(self, short_config):
        env = AMMFeeEnv(config=short_config)
        assert env.action_space.shape == (ACTION_DIM,)
        np.testing.assert_array_equal(env.action_space.low, np.full(ACTION_DIM, ACTION_MIN))
        np.testing.assert_array_equal(env.action_space.high, np.full(ACTION_DIM, ACTION_MAX))

    def test_step_returns_correct_tuple(self, short_config):
        env = AMMFeeEnv(config=short_config)
        env.reset(seed=42)
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "edge" in info
        assert "ask_near_depth_y" in info

    def test_action_clipping_is_reflected_in_observation(self, short_config):
        env = AMMFeeEnv(config=short_config)
        obs, _ = env.reset(seed=42)
        action = np.array([2.0, -2.0, 1.5, -1.5, 2.0, -2.0], dtype=np.float32)
        obs, _, _, _, _ = env.step(action)
        ws = env.window_size
        expected = np.clip(action, ACTION_MIN, ACTION_MAX)
        np.testing.assert_allclose(obs[ws + 7 : ws + 13], expected)

    def test_deterministic_with_seed(self, short_config):
        env = AMMFeeEnv(config=short_config)
        action = np.zeros(ACTION_DIM, dtype=np.float32)

        env.reset(seed=42)
        rewards_a = [env.step(action)[1] for _ in range(50)]

        env.reset(seed=42)
        rewards_b = [env.step(action)[1] for _ in range(50)]

        np.testing.assert_array_equal(rewards_a, rewards_b)

    def test_reset_without_seed_produces_fresh_episode(self, short_config):
        env = AMMFeeEnv(config=short_config)
        action = np.zeros(ACTION_DIM, dtype=np.float32)

        env.reset(seed=42)
        rewards_seeded = [env.step(action)[1] for _ in range(10)]

        env.reset()
        rewards_unseeded = [env.step(action)[1] for _ in range(10)]

        assert rewards_unseeded != rewards_seeded


class TestEnvBehavior:
    def test_reward_sums_to_final_edge(self):
        env = AMMFeeEnv(config=SimConfig(n_steps=300, seed=42))
        env.reset(seed=42)

        total_reward = 0.0
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        for _ in range(300):
            _, reward, _, _, info = env.step(action)
            total_reward += reward

        assert total_reward == pytest.approx(info["edge"], rel=1e-6)

    def test_zero_retail_rate_yields_zero_flow(self):
        config = SimConfig(n_steps=20, retail_arrival_rate=0.0, seed=42)
        env = AMMFeeEnv(config=config)
        env.reset(seed=42)
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        for _ in range(20):
            _, _, _, _, _ = env.step(action)
            assert env._ema_exec_count == 0.0
            assert env._ema_exec_volume == 0.0

    def test_more_aggressive_depth_policy_changes_outcomes(self):
        config = SimConfig(n_steps=500, seed=42)

        env = AMMFeeEnv(config=config)
        obs, _ = env.reset(seed=42)
        conservative = StaticDepthPolicy(
            bid_scale=-0.6, ask_scale=-0.6, bid_decay=0.6, ask_decay=0.6
        )
        for _ in range(500):
            obs, _, _, _, info_cons = env.step(conservative(obs))

        env2 = AMMFeeEnv(config=config)
        obs, _ = env2.reset(seed=42)
        aggressive = StaticDepthPolicy(
            bid_scale=0.6, ask_scale=0.6, bid_decay=-0.3, ask_decay=-0.3
        )
        for _ in range(500):
            obs, _, _, _, info_aggr = env2.step(aggressive(obs))

        assert info_cons["execution_volume_y"] != info_aggr["execution_volume_y"]

    def test_current_true_price_is_not_exposed_in_info(self):
        env = AMMFeeEnv(config=SimConfig(n_steps=10, seed=42))
        env.reset(seed=42)
        _, _, _, _, info = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
        assert "fair_price" not in info

    def test_current_sigma_is_not_exposed_in_info(self):
        env = AMMFeeEnv(
            config=SimConfig(
                n_steps=10,
                seed=42,
                volatility_schedule=((0, 0.001), (5, 0.004)),
            )
        )
        env.reset(seed=42)
        _, _, _, _, info = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
        assert "active_sigma" not in info

    def test_volatility_schedule_is_seed_reproducible(self):
        config = SimConfig(
            n_steps=40,
            volatility_schedule=((0, 0.001), (20, 0.004)),
        )
        env_a = AMMFeeEnv(config=config)
        env_b = AMMFeeEnv(config=config)

        obs_a, _ = env_a.reset(seed=11)
        obs_b, _ = env_b.reset(seed=11)
        np.testing.assert_allclose(obs_a, obs_b)

        rewards_a = []
        rewards_b = []
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        for _ in range(40):
            _, reward_a, _, _, _ = env_a.step(action)
            _, reward_b, _, _, _ = env_b.step(action)
            rewards_a.append(reward_a)
            rewards_b.append(reward_b)

        np.testing.assert_allclose(rewards_a, rewards_b)
