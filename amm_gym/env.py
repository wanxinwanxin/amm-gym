"""Gymnasium environment for AMM fee setting.

Observation: compact vector of public market state plus lagged price history.
Action: (bid_fee, ask_fee) continuous.
Reward: one-step delayed change in agent AMM edge.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from amm_gym.sim.engine import SimConfig, SimulationEngine


# Fee bounds (in decimal): 1 bps to 1000 bps
MIN_FEE = 0.0001
MAX_FEE = 0.10


class AMMFeeEnv(gym.Env):
    """Gymnasium environment for dynamic AMM fee setting.

    The agent controls bid/ask fees on a constant-product AMM competing
    with a fixed-fee normalizer AMM for retail flow. Arbitrageurs trade
    both AMMs to fair price each step.
    """

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

        # Action: (bid_fee, ask_fee)
        self.action_space = spaces.Box(
            low=np.float32(MIN_FEE),
            high=np.float32(MAX_FEE),
            shape=(2,),
            dtype=np.float32,
        )

        # Observation: compact market-state vector
        # [0:window_size] recent log-returns
        # [ws] reserve_x (normalized by initial X)
        # [ws+1] reserve_y (normalized by initial Y)
        # [ws+2] reserve imbalance [-1, 1]
        # [ws+3] lagged edge so far (normalized)
        # [ws+4] EMA of executed volume (normalized)
        # [ws+5] EMA of execution count
        # [ws+6] EMA of signed net Y flow (normalized)
        # [ws+7] current bid fee
        # [ws+8] current ask fee
        # [ws+9] volatility estimate
        # [ws+10] step fraction [0, 1]
        obs_dim = window_size + 11
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.engine: SimulationEngine | None = None
        self._price_history: deque[float] = deque(maxlen=window_size + 1)
        self._return_history: deque[float] = deque(maxlen=window_size)
        self._vol_window: deque[float] = deque(maxlen=50)

        # EMA state for observable execution stats
        self._ema_exec_volume = 0.0
        self._ema_exec_count = 0.0
        self._ema_net_flow = 0.0
        self._ema_alpha = 0.1

        self._prev_edge = 0.0
        self._pending_reward = 0.0
        self._initial_value = 0.0

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
            self.config.initial_x * self.config.initial_price
            + self.config.initial_y
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

        obs = self._get_obs()
        info = {"edge": 0.0, "pnl": 0.0, "step": 0}
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self.engine is not None, "Call reset() first"

        # Clip and apply action
        bid_fee = float(np.clip(action[0], MIN_FEE, MAX_FEE))
        ask_fee = float(np.clip(action[1], MIN_FEE, MAX_FEE))
        self.engine.set_agent_fees(bid_fee, ask_fee)

        # Step the simulation
        result = self.engine.step()

        # Update price history
        self._price_history.append(result.fair_price)
        if len(self._price_history) >= 2:
            prices = list(self._price_history)
            log_ret = np.log(prices[-1] / prices[-2])
            self._return_history.append(log_ret)
            self._vol_window.append(log_ret)

        # Update execution EMA stats
        alpha = self._ema_alpha
        agent_exec_volume = result.execution_volume_y.get("submission", 0.0)
        agent_exec_count = result.execution_count.get("submission", 0)
        agent_net_flow = result.net_flow_y.get("submission", 0.0)
        self._ema_exec_volume = (
            (1 - alpha) * self._ema_exec_volume + alpha * agent_exec_volume
        )
        self._ema_exec_count = (
            (1 - alpha) * self._ema_exec_count + alpha * agent_exec_count
        )
        self._ema_net_flow = (
            (1 - alpha) * self._ema_net_flow + alpha * agent_net_flow
        )

        # Reward is delayed by one step to avoid exposing same-step markout.
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
            "bid_fee": bid_fee,
            "ask_fee": ask_fee,
            "execution_count": agent_exec_count,
            "execution_volume_y": agent_exec_volume,
            "net_flow_y": agent_net_flow,
        }

        return obs, float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        assert self.engine is not None

        ws = self.window_size
        obs = np.zeros(ws + 11, dtype=np.float32)

        # Recent log-returns (zero-padded if not enough history)
        returns = list(self._return_history)
        for i, r in enumerate(returns[-ws:]):
            obs[ws - len(returns[-ws:]) + i] = r

        amm = self.engine.amm_agent
        init_x = max(self.config.initial_x, 1.0)
        init_y = max(self.config.initial_y, 1.0)
        init_val = max(self._initial_value, 1.0)

        # Public reserves, normalized without using the current fair price.
        obs[ws] = amm.reserve_x / init_x
        obs[ws + 1] = amm.reserve_y / init_y

        # Reserve imbalance from public quantities only.
        total_reserves = obs[ws] + obs[ws + 1]
        obs[ws + 2] = (
            (obs[ws] - obs[ws + 1]) / total_reserves if total_reserves > 0 else 0.0
        )

        # Lagged edge so far (normalized)
        obs[ws + 3] = self._prev_edge / init_val

        # Observable execution stats (EMA, normalized)
        obs[ws + 4] = self._ema_exec_volume / init_val
        obs[ws + 5] = self._ema_exec_count / max(
            self.engine.config.retail_arrival_rate, 1.0
        )
        obs[ws + 6] = self._ema_net_flow / init_val

        # Current fees
        obs[ws + 7] = amm.fees.bid_fee
        obs[ws + 8] = amm.fees.ask_fee

        # Volatility estimate (rolling std of returns)
        if len(self._vol_window) >= 2:
            obs[ws + 9] = float(np.std(list(self._vol_window)))

        # Step fraction
        obs[ws + 10] = self.engine.current_step / max(self.engine.config.n_steps, 1)

        return obs
