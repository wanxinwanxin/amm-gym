"""Gymnasium environment for dynamic AMM liquidity placement."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from amm_gym.sim.engine import SimConfig, SimulationEngine


ACTION_DIM = 6
ACTION_MIN = -1.0
ACTION_MAX = 1.0


class AMMFeeEnv(gym.Env):
    """Gymnasium environment for depth-ladder market making."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: SimConfig | None = None,
        window_size: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.config = config or SimConfig()
        self.window_size = window_size

        self.action_space = spaces.Box(
            low=np.float32(ACTION_MIN),
            high=np.float32(ACTION_MAX),
            shape=(ACTION_DIM,),
            dtype=np.float32,
        )

        # [0:ws] recent log returns
        # [ws] reserve_x normalized by initial_x
        # [ws+1] reserve_y normalized by initial_y
        # [ws+2] reserve imbalance
        # [ws+3] lagged edge normalized
        # [ws+4] EMA execution volume normalized
        # [ws+5] EMA execution count normalized
        # [ws+6] EMA signed net flow normalized
        # [ws+7:ws+13] last posted action
        # [ws+13] rolling volatility
        # [ws+14] step fraction
        obs_dim = window_size + 15
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

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
        self._last_action = np.zeros(ACTION_DIM, dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        engine_config = replace(self.config, seed=seed)
        self.engine = SimulationEngine(engine_config)
        self._initial_value = (
            self.config.initial_x * self.config.initial_price + self.config.initial_y
        )

        self._price_history.clear()
        self._return_history.clear()
        self._vol_window.clear()
        self._price_history.append(self.config.initial_price)

        self._ema_exec_volume = 0.0
        self._ema_exec_count = 0.0
        self._ema_net_flow = 0.0
        self._prev_edge = 0.0
        self._pending_reward = 0.0
        self._last_action.fill(0.0)

        obs = self._get_obs()
        info = {"edge": 0.0, "pnl": 0.0, "step": 0}
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

        terminated = self.engine.done
        truncated = False

        obs = self._get_obs()
        info = {
            "edge": current_edge,
            "edge_normalizer": result.edges.get("normalizer", 0.0),
            "pnl": result.pnls.get("submission", 0.0),
            "pnl_normalizer": result.pnls.get("normalizer", 0.0),
            "spot_price": result.spot_prices.get("submission", 0.0),
            "step": result.timestamp,
            "execution_count": exec_count,
            "execution_volume_y": exec_volume,
            "net_flow_y": net_flow_y,
            **result.ladder_depth_y,
        }
        return obs, float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        assert self.engine is not None

        ws = self.window_size
        obs = np.zeros(ws + 15, dtype=np.float32)

        returns = list(self._return_history)
        tail = returns[-ws:]
        for i, value in enumerate(tail):
            obs[ws - len(tail) + i] = value

        amm = self.engine.amm_agent
        init_x = max(self.config.initial_x, 1.0)
        init_y = max(self.config.initial_y, 1.0)
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
        obs[ws + 7 : ws + 13] = self._last_action

        if len(self._vol_window) >= 2:
            obs[ws + 13] = float(np.std(list(self._vol_window)))
        obs[ws + 14] = self.engine.current_step / max(self.engine.config.n_steps, 1)
        return obs
