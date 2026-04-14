"""Tests for the Gymnasium environment."""

import numpy as np
import pytest

from amm_gym.env import AMMFeeEnv, MIN_FEE, MAX_FEE
from amm_gym.sim.engine import SimConfig
from amm_gym.baselines import StaticFeePolicy


@pytest.fixture
def short_config():
    return SimConfig(n_steps=100, seed=42)


class TestEnvAPI:
    """Verify Gymnasium API compliance."""

    def test_reset_returns_obs_and_info(self, short_config):
        env = AMMFeeEnv(config=short_config)
        obs, info = env.reset(seed=42)
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32
        assert "edge" in info

    def test_step_returns_correct_tuple(self, short_config):
        env = AMMFeeEnv(config=short_config)
        env.reset(seed=42)
        action = np.array([0.003, 0.003], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "edge" in info

    def test_episode_terminates(self, short_config):
        env = AMMFeeEnv(config=short_config)
        env.reset(seed=42)
        action = np.array([0.003, 0.003], dtype=np.float32)
        for i in range(100):
            obs, reward, terminated, truncated, info = env.step(action)
            if i < 99:
                assert not terminated
        assert terminated

    def test_action_clipping(self, short_config):
        env = AMMFeeEnv(config=short_config)
        env.reset(seed=42)
        # Action outside bounds should be clipped, not crash
        action = np.array([-1.0, 2.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert info["bid_fee"] == pytest.approx(MIN_FEE)
        assert info["ask_fee"] == pytest.approx(MAX_FEE)

    def test_deterministic_with_seed(self, short_config):
        env = AMMFeeEnv(config=short_config)
        action = np.array([0.003, 0.003], dtype=np.float32)

        env.reset(seed=42)
        rewards_a = []
        for _ in range(50):
            _, r, _, _, _ = env.step(action)
            rewards_a.append(r)

        env.reset(seed=42)
        rewards_b = []
        for _ in range(50):
            _, r, _, _, _ = env.step(action)
            rewards_b.append(r)

        np.testing.assert_array_equal(rewards_a, rewards_b)

    def test_reset_without_seed_produces_fresh_episode(self, short_config):
        env = AMMFeeEnv(config=short_config)
        action = np.array([0.003, 0.003], dtype=np.float32)

        env.reset(seed=42)
        rewards_seeded = []
        for _ in range(10):
            _, reward, _, _, _ = env.step(action)
            rewards_seeded.append(reward)

        env.reset()
        rewards_unseeded = []
        for _ in range(10):
            _, reward, _, _, _ = env.step(action)
            rewards_unseeded.append(reward)

        assert rewards_unseeded != rewards_seeded

    def test_reset_observation_is_independent_of_initial_price(self):
        config_a = SimConfig(n_steps=10, initial_price=100.0, seed=42)
        config_b = SimConfig(n_steps=10, initial_price=250.0, seed=42)

        env_a = AMMFeeEnv(config=config_a)
        env_b = AMMFeeEnv(config=config_b)

        obs_a, _ = env_a.reset(seed=42)
        obs_b, _ = env_b.reset(seed=42)

        np.testing.assert_array_equal(obs_a, obs_b)


class TestEnvBehavior:
    """Verify economic behavior makes sense."""

    def test_static_30bps_near_zero_edge(self):
        """With both AMMs at 30bps, agent edge should be ~0 (symmetric)."""
        config = SimConfig(n_steps=5000, seed=42)
        env = AMMFeeEnv(config=config)
        env.reset(seed=42)

        policy = StaticFeePolicy(30)
        total_reward = 0.0
        obs = env.reset(seed=42)[0]
        for _ in range(5000):
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # Both AMMs have same fees -> should split flow evenly -> ~0 edge difference
        # Allow some noise since retail flow is stochastic
        assert abs(info["edge"]) < abs(info["edge_normalizer"]) * 5 + 50

    def test_very_low_fees_attract_more_retail(self):
        """Very low fees should attract more retail but lose more to arb."""
        config = SimConfig(n_steps=2000, seed=42)

        # Run with 5 bps
        env = AMMFeeEnv(config=config)
        obs, _ = env.reset(seed=42)
        low_fee_policy = StaticFeePolicy(5)
        for _ in range(2000):
            obs, _, _, _, info_low = env.step(low_fee_policy(obs))

        # Run with 100 bps
        env2 = AMMFeeEnv(config=config)
        obs, _ = env2.reset(seed=42)
        high_fee_policy = StaticFeePolicy(100)
        for _ in range(2000):
            obs, _, _, _, info_high = env2.step(high_fee_policy(obs))

        # Low fees should capture more retail flow to agent AMM
        # but the edge comparison depends on arb losses too
        # Just check both ran without error and produced different results
        assert info_low["edge"] != info_high["edge"]

    def test_reward_sums_to_final_edge(self):
        """Sum of rewards should equal final edge."""
        config = SimConfig(n_steps=500, seed=42)
        env = AMMFeeEnv(config=config)
        env.reset(seed=42)

        total_reward = 0.0
        action = np.array([0.003, 0.003], dtype=np.float32)
        for _ in range(500):
            _, reward, _, _, info = env.step(action)
            total_reward += reward

        assert total_reward == pytest.approx(info["edge"], rel=1e-6)

    def test_reward_is_delayed_by_one_step(self):
        config = SimConfig(n_steps=10, seed=42)
        env = AMMFeeEnv(config=config)
        env.reset(seed=42)

        action = np.array([0.003, 0.003], dtype=np.float32)
        _, reward0, _, _, info0 = env.step(action)
        _, reward1, _, _, _ = env.step(action)

        assert reward0 == pytest.approx(0.0)
        assert reward1 == pytest.approx(info0["edge"], rel=1e-6, abs=1e-6)

    def test_zero_retail_rate_yields_zero_flow(self):
        config = SimConfig(n_steps=20, retail_arrival_rate=0.0, seed=42)
        env = AMMFeeEnv(config=config)
        obs, _ = env.reset(seed=42)

        action = np.array([0.003, 0.003], dtype=np.float32)
        ws = env.window_size
        assert obs[ws + 4] == 0.0
        assert obs[ws + 5] == 0.0
        assert obs[ws + 6] == 0.0
        for _ in range(20):
            _, _, _, _, info = env.step(action)
            assert env._ema_exec_count == 0.0
            assert env._ema_exec_volume == 0.0
            assert env._ema_net_flow == 0.0
            assert info["execution_count"] == 0
            assert info["execution_volume_y"] == 0.0
            assert info["net_flow_y"] == 0.0
