"""Gymnasium environments for AMM market-making."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from amm_gym.sim.engine import SimConfig, SimulationEngine
from amm_gym.sim.quote_surface import QUOTE_SURFACE_ACTION_DIM, QUOTE_SURFACE_STATE_DIM
from amm_gym.sim.venues import VenueSpec


ACTION_DIM = 6
ACTION_MIN = -1.0
ACTION_MAX = 1.0
CHALLENGE_ACTION_DIM = QUOTE_SURFACE_ACTION_DIM


class _BaseAMMEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: SimConfig | None = None, window_size: int = 10) -> None:
        super().__init__()
        self.config = config or SimConfig()
        self.window_size = window_size
        self.engine: SimulationEngine | None = None
        self._price_history: deque[float] = deque(maxlen=window_size + 1)
        self._return_history: deque[float] = deque(maxlen=window_size)
        self._vol_window: deque[float] = deque(maxlen=50)
        self._ema_exec_volume = 0.0
        self._ema_exec_count = 0.0
        self._ema_net_flow = 0.0
        self._ema_alpha = 0.1
        self._prev_edge = 0.0
        self._pending_reward = 0.0
        self._initial_value = 0.0
        self._last_action = np.zeros(self.action_dim, dtype=np.float32)

    @property
    def action_dim(self) -> int:
        raise NotImplementedError

    @property
    def obs_dim(self) -> int:
        raise NotImplementedError

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        engine_config = self._make_engine_config(seed)
        self.engine = SimulationEngine(engine_config)
        self._initial_value = (
            engine_config.initial_x * engine_config.initial_price + engine_config.initial_y
        )

        self._price_history.clear()
        self._return_history.clear()
        self._vol_window.clear()
        self._price_history.append(engine_config.initial_price)

        self._ema_exec_volume = 0.0
        self._ema_exec_count = 0.0
        self._ema_net_flow = 0.0
        self._prev_edge = 0.0
        self._pending_reward = 0.0
        self._last_action = np.zeros(self.action_dim, dtype=np.float32)

        obs = self._get_obs()
        info = self._initial_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self.engine is not None, "Call reset() first"

        clipped = np.clip(np.asarray(action, dtype=np.float32), ACTION_MIN, ACTION_MAX)
        self.engine.set_agent_action(clipped)
        self._last_action = clipped.astype(np.float32)

        result = self.engine.step()

        self._price_history.append(result.fair_price)
        if len(self._price_history) >= 2:
            prices = list(self._price_history)
            log_ret = float(np.log(prices[-1] / prices[-2]))
            self._return_history.append(log_ret)
            self._vol_window.append(log_ret)

        alpha = self._ema_alpha
        exec_volume = result.execution_volume_y.get("submission", 0.0)
        exec_count = result.execution_count.get("submission", 0)
        net_flow_y = result.net_flow_y.get("submission", 0.0)
        self._ema_exec_volume = (1 - alpha) * self._ema_exec_volume + alpha * exec_volume
        self._ema_exec_count = (1 - alpha) * self._ema_exec_count + alpha * exec_count
        self._ema_net_flow = (1 - alpha) * self._ema_net_flow + alpha * net_flow_y

        current_edge = result.edges.get("submission", 0.0)
        current_reward = current_edge - self._prev_edge
        reward = self._pending_reward
        if self.engine.done:
            reward += current_reward
        self._pending_reward = current_reward
        self._prev_edge = current_edge

        obs = self._get_obs()
        info = self._build_info(result, net_flow_y)
        return obs, float(reward), self.engine.done, False, info

    def _make_engine_config(self, seed: int | None) -> SimConfig:
        return replace(self.config, seed=seed)

    def _initial_info(self) -> dict[str, float]:
        return {"edge": 0.0, "pnl": 0.0, "step": 0}

    def _build_info(self, result, net_flow_y: float) -> dict[str, Any]:
        info = {
            "edge": result.edges.get("submission", 0.0),
            "edge_benchmark": result.edges.get("benchmark", 0.0),
            "edge_normalizer": result.edges.get("normalizer", 0.0),
            "pnl": result.pnls.get("submission", 0.0),
            "pnl_benchmark": result.pnls.get("benchmark", 0.0),
            "pnl_normalizer": result.pnls.get("normalizer", 0.0),
            "spot_price": result.spot_prices.get("submission", 0.0),
            "spot_price_benchmark": result.spot_prices.get("benchmark", 0.0),
            "spot_price_normalizer": result.spot_prices.get("normalizer", 0.0),
            "step": result.timestamp,
            "execution_count": result.execution_count.get("submission", 0),
            "execution_count_benchmark": result.execution_count.get("benchmark", 0),
            "execution_count_normalizer": result.execution_count.get("normalizer", 0),
            "execution_volume_y": result.execution_volume_y.get("submission", 0.0),
            "execution_volume_y_benchmark": result.execution_volume_y.get("benchmark", 0.0),
            "execution_volume_y_normalizer": result.execution_volume_y.get("normalizer", 0.0),
            "retail_volume_y": result.retail_volume_y.get("submission", 0.0),
            "retail_volume_y_benchmark": result.retail_volume_y.get("benchmark", 0.0),
            "retail_volume_y_normalizer": result.retail_volume_y.get("normalizer", 0.0),
            "arb_volume_y": result.arb_volume_y.get("submission", 0.0),
            "arb_volume_y_benchmark": result.arb_volume_y.get("benchmark", 0.0),
            "arb_volume_y_normalizer": result.arb_volume_y.get("normalizer", 0.0),
            "net_flow_y": net_flow_y,
            **result.ladder_depth_y,
        }
        return info

    def _rolling_vol(self) -> float:
        if len(self._vol_window) < 2:
            return 0.0
        return float(np.std(list(self._vol_window)))

    def _common_obs_prefix(self) -> np.ndarray:
        assert self.engine is not None
        ws = self.window_size
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        returns = list(self._return_history)
        tail = returns[-ws:]
        for i, value in enumerate(tail):
            obs[ws - len(tail) + i] = value

        amm = self.engine.amm_agent
        init_x = max(self.engine.config.initial_x, 1.0)
        init_y = max(self.engine.config.initial_y, 1.0)
        init_val = max(self._initial_value, 1.0)

        obs[ws] = amm.reserve_x / init_x
        obs[ws + 1] = amm.reserve_y / init_y
        total_reserves = obs[ws] + obs[ws + 1]
        obs[ws + 2] = (
            (obs[ws] - obs[ws + 1]) / total_reserves if total_reserves > 0 else 0.0
        )
        obs[ws + 3] = self._prev_edge / init_val
        obs[ws + 4] = self._ema_exec_volume / init_val
        obs[ws + 5] = self._ema_exec_count / max(self.engine.config.retail_arrival_rate, 1.0)
        obs[ws + 6] = self._ema_net_flow / init_val
        return obs


class AMMFeeEnv(_BaseAMMEnv):
    """Gymnasium environment for depth-ladder market making."""

    def __init__(
        self,
        config: SimConfig | None = None,
        window_size: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, window_size=window_size)
        self.action_space = spaces.Box(
            low=np.float32(ACTION_MIN),
            high=np.float32(ACTION_MAX),
            shape=(ACTION_DIM,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size + 15,), dtype=np.float32
        )

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    @property
    def obs_dim(self) -> int:
        return self.window_size + 15

    def _get_obs(self) -> np.ndarray:
        assert self.engine is not None
        ws = self.window_size
        obs = self._common_obs_prefix()
        obs[ws + 7 : ws + 13] = self._last_action
        obs[ws + 13] = self._rolling_vol()
        obs[ws + 14] = self.engine.current_step / max(self.engine.config.n_steps, 1)
        return obs


class AMMChallengeEnv(_BaseAMMEnv):
    """Challenge-oriented env with a stateful quote-surface submission venue."""

    def __init__(
        self,
        config: SimConfig | None = None,
        window_size: int = 10,
        benchmark_fee_bps_range: tuple[int, int] = (30, 80),
        benchmark_liquidity_range: tuple[float, float] = (0.4, 2.0),
        retail_arrival_rate_range: tuple[float, float] = (0.4, 1.2),
        retail_mean_size_range: tuple[float, float] = (12.0, 28.0),
        sigma_range: tuple[float, float] = (0.0001, 0.0070),
        **kwargs: Any,
    ) -> None:
        base_config = config or SimConfig(
            n_steps=512,
            retail_arrival_rate=0.8,
            retail_mean_size=18.0,
            retail_size_sigma=0.7,
        )
        super().__init__(config=base_config, window_size=window_size)
        self.benchmark_fee_bps_range = benchmark_fee_bps_range
        self.benchmark_liquidity_range = benchmark_liquidity_range
        self.retail_arrival_rate_range = retail_arrival_rate_range
        self.retail_mean_size_range = retail_mean_size_range
        self.sigma_range = sigma_range
        self._sampled_benchmark_fee_bps = float(sum(benchmark_fee_bps_range) / 2.0)
        self._sampled_benchmark_liquidity_mult = float(sum(benchmark_liquidity_range) / 2.0)
        self.action_space = spaces.Box(
            low=np.float32(ACTION_MIN),
            high=np.float32(ACTION_MAX),
            shape=(CHALLENGE_ACTION_DIM,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size + 29,), dtype=np.float32
        )

    @property
    def action_dim(self) -> int:
        return CHALLENGE_ACTION_DIM

    @property
    def obs_dim(self) -> int:
        return self.window_size + 29

    def _make_engine_config(self, seed: int | None) -> SimConfig:
        rng = np.random.default_rng(seed)
        sampled_sigma = float(rng.uniform(*self.sigma_range))
        sampled_arrival = float(rng.uniform(*self.retail_arrival_rate_range))
        sampled_mean_size = float(rng.uniform(*self.retail_mean_size_range))
        sampled_fee_bps = int(rng.integers(self.benchmark_fee_bps_range[0], self.benchmark_fee_bps_range[1] + 1))
        sampled_liquidity = float(rng.uniform(*self.benchmark_liquidity_range))

        self._sampled_benchmark_fee_bps = float(sampled_fee_bps)
        self._sampled_benchmark_liquidity_mult = float(sampled_liquidity)
        benchmark_spec = VenueSpec(
            kind="cpmm",
            name="normalizer",
            reserve_x=self.config.initial_x * sampled_liquidity,
            reserve_y=self.config.initial_y * sampled_liquidity,
            bid_fee=sampled_fee_bps / 10_000.0,
            ask_fee=sampled_fee_bps / 10_000.0,
            controllable=False,
        )
        submission_spec = VenueSpec(
            kind="quote_surface",
            name="submission",
            reserve_x=self.config.initial_x,
            reserve_y=self.config.initial_y,
            controllable=True,
        )

        sampled_config = replace(
            self.config,
            seed=seed,
            gbm_sigma=sampled_sigma,
            retail_arrival_rate=sampled_arrival,
            retail_mean_size=sampled_mean_size,
            submission_venue=submission_spec,
            benchmark_venue=benchmark_spec,
        )
        return sampled_config

    def _initial_info(self) -> dict[str, float]:
        info = super()._initial_info()
        info["benchmark_fee_bps"] = self._sampled_benchmark_fee_bps
        info["benchmark_liquidity_mult"] = self._sampled_benchmark_liquidity_mult
        return info

    def _build_info(self, result, net_flow_y: float) -> dict[str, Any]:
        info = super()._build_info(result, net_flow_y)
        info["benchmark_fee_bps"] = self._sampled_benchmark_fee_bps
        info["benchmark_liquidity_mult"] = self._sampled_benchmark_liquidity_mult
        info["submission_edge_score"] = info["edge"]
        return info

    def _get_obs(self) -> np.ndarray:
        assert self.engine is not None
        ws = self.window_size
        obs = self._common_obs_prefix()
        obs[ws + 7 : ws + 7 + CHALLENGE_ACTION_DIM] = self._last_action

        venue_state = np.zeros(QUOTE_SURFACE_STATE_DIM, dtype=np.float32)
        if hasattr(self.engine.amm_agent, "state"):
            venue_state = np.asarray(self.engine.amm_agent.state, dtype=np.float32)
        obs[ws + 7 + CHALLENGE_ACTION_DIM : ws + 11 + CHALLENGE_ACTION_DIM] = venue_state
        obs[ws + 11 + CHALLENGE_ACTION_DIM] = self._sampled_benchmark_fee_bps / 100.0
        obs[ws + 12 + CHALLENGE_ACTION_DIM] = self._sampled_benchmark_liquidity_mult
        obs[ws + 13 + CHALLENGE_ACTION_DIM] = self._rolling_vol()
        obs[ws + 14 + CHALLENGE_ACTION_DIM] = self.engine.current_step / max(
            self.engine.config.n_steps, 1
        )
        return obs
