"""Gymnasium environment for AMM fee setting.

Observation: ~15-dim vector of market state.
Action: (bid_fee, ask_fee) continuous.
Reward: per-step change in agent AMM edge.
"""

from __future__ import annotations

from collections import deque
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

        # Observation: 15-dim vector
        # [0:window_size] recent log-returns
        # [ws] reserve_x (normalized)
        # [ws+1] reserve_y (normalized)
        # [ws+2] inventory imbalance [-1, 1]
        # [ws+3] edge so far (normalized)
        # [ws+4] recent retail volume EMA (normalized)
        # [ws+5] recent retail count EMA
        # [ws+6] recent buy ratio [0, 1]
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

        # EMA state for trade stats
        self._ema_volume = 0.0
        self._ema_count = 0.0
        self._ema_buy_ratio = 0.5
        self._ema_alpha = 0.1

        self._prev_edge = 0.0
        self._initial_value = 0.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        # Use Gymnasium's seeding
        env_seed = seed if seed is not None else self.config.seed
        self.engine = SimulationEngine(self.config)
        self.engine.reset(seed=env_seed)

        self._initial_value = (
            self.config.initial_x * self.config.initial_price
            + self.config.initial_y
        )

        self._price_history.clear()
        self._return_history.clear()
        self._vol_window.clear()
        self._price_history.append(self.config.initial_price)

        self._ema_volume = 0.0
        self._ema_count = 0.0
        self._ema_buy_ratio = 0.5
        self._prev_edge = 0.0

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

        # Update trade EMA stats
        alpha = self._ema_alpha
        agent_vol = result.retail_volume_y.get("submission", 0.0)
        self._ema_volume = (1 - alpha) * self._ema_volume + alpha * agent_vol
        self._ema_count = (1 - alpha) * self._ema_count + alpha * result.n_retail_orders

        # Buy ratio from routed trades (approximate from orders)
        if result.n_retail_orders > 0:
            # We don't have per-order side info here, use a proxy:
            # if AMM's spot is near fair, roughly 50/50
            self._ema_buy_ratio = (1 - alpha) * self._ema_buy_ratio + alpha * 0.5

        # Reward = change in agent edge
        current_edge = result.edges.get("submission", 0.0)
        reward = current_edge - self._prev_edge
        self._prev_edge = current_edge

        terminated = self.engine.done
        truncated = False

        obs = self._get_obs()
        info = {
            "edge": current_edge,
            "edge_normalizer": result.edges.get("normalizer", 0.0),
            "pnl": result.pnls.get("submission", 0.0),
            "pnl_normalizer": result.pnls.get("normalizer", 0.0),
            "fair_price": result.fair_price,
            "spot_price": result.spot_prices.get("submission", 0.0),
            "step": result.timestamp,
            "bid_fee": bid_fee,
            "ask_fee": ask_fee,
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
        fair = self.engine.current_fair_price
        init_val = max(self._initial_value, 1.0)

        # Reserves (normalized by initial value)
        obs[ws] = (amm.reserve_x * fair) / init_val
        obs[ws + 1] = amm.reserve_y / init_val

        # Inventory imbalance: (value_x - value_y) / (value_x + value_y)
        val_x = amm.reserve_x * fair
        val_y = amm.reserve_y
        total = val_x + val_y
        obs[ws + 2] = (val_x - val_y) / total if total > 0 else 0.0

        # Edge so far (normalized)
        obs[ws + 3] = self._prev_edge / init_val

        # Trade flow stats (EMA, normalized)
        obs[ws + 4] = self._ema_volume / init_val
        obs[ws + 5] = self._ema_count / max(self.engine.config.retail_arrival_rate, 1.0)
        obs[ws + 6] = self._ema_buy_ratio

        # Current fees
        obs[ws + 7] = amm.fees.bid_fee
        obs[ws + 8] = amm.fees.ask_fee

        # Volatility estimate (rolling std of returns)
        if len(self._vol_window) >= 2:
            obs[ws + 9] = float(np.std(list(self._vol_window)))

        # Step fraction
        obs[ws + 10] = self.engine.current_step / max(self.engine.config.n_steps, 1)

        return obs
