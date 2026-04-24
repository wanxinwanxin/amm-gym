"""Gym wrapper for the exact simple AMM evaluator."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from arena_eval.core.types import TradeInfo
from arena_eval.exact_simple_amm import ExactSimpleAMMConfig
from arena_eval.exact_simple_amm.simulator import ExactSimpleAMMSimulator, FixedFeeStrategy


class _TimestampControlledStrategy:
    """Bridge strategy that applies one fee pair for an entire simulator timestamp."""

    def __init__(self, default_bid_fee: float = 0.003, default_ask_fee: float = 0.003) -> None:
        self.default_bid_fee = float(default_bid_fee)
        self.default_ask_fee = float(default_ask_fee)
        self.current_bid_fee = self.default_bid_fee
        self.current_ask_fee = self.default_ask_fee
        self.last_trade: TradeInfo | None = None

    def set_fees(self, bid_fee: float, ask_fee: float) -> None:
        self.current_bid_fee = float(min(0.1, max(0.0, bid_fee)))
        self.current_ask_fee = float(min(0.1, max(0.0, ask_fee)))

    def after_initialize(self, initial_x: float, initial_y: float) -> tuple[float, float]:
        return (self.current_bid_fee, self.current_ask_fee)

    def after_swap(self, trade: TradeInfo) -> tuple[float, float]:
        self.last_trade = trade
        return (self.current_bid_fee, self.current_ask_fee)


class ExactSimpleAMMGymEnv(gym.Env):
    """RL-facing wrapper over the challenge-faithful simulator."""

    metadata = {"render_modes": []}

    def __init__(self, config: ExactSimpleAMMConfig | None = None) -> None:
        super().__init__()
        self.base_config = config
        self.strategy = _TimestampControlledStrategy()
        self.simulator: ExactSimpleAMMSimulator | None = None
        self._last_edge = 0.0
        self._last_submission_trade = False
        self.action_space = spaces.Box(low=np.float32(0.0), high=np.float32(0.1), shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        chosen_seed = 0 if seed is None else int(seed)
        config = self.base_config or ExactSimpleAMMConfig.from_seed(chosen_seed)
        self.strategy = _TimestampControlledStrategy()
        self.simulator = ExactSimpleAMMSimulator(
            config=config,
            submission_strategy=self.strategy,
            normalizer_strategy=FixedFeeStrategy(),
            seed=chosen_seed,
        )
        self._last_edge = 0.0
        self._last_submission_trade = False
        obs = self._build_obs()
        info = {"edge": 0.0, "pnl": 0.0, "seed": chosen_seed, "submission_trade": False}
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.simulator is not None, "Call reset() first"
        clipped = np.clip(np.asarray(action, dtype=np.float32), 0.0, 0.1)
        self.strategy.set_fees(float(clipped[0]), float(clipped[1]))

        step_info = self.simulator.step_once()
        current_edge = self.simulator.edge_submission
        reward = current_edge - self._last_edge
        self._last_edge = current_edge
        self._last_submission_trade = bool(step_info["submission_trade_occurred"])

        result = self.simulator.result() if self.simulator.done else None
        info = {
            "edge": current_edge,
            "edge_normalizer": self.simulator.edge_normalizer,
            "edge_advantage": current_edge - self.simulator.edge_normalizer,
            "pnl": (result.pnl_submission if result else 0.0),
            "pnl_normalizer": (result.pnl_normalizer if result else 0.0),
            "submission_trade": self._last_submission_trade,
            "timestamp": int(step_info["timestamp"]),
        }
        return self._build_obs(), float(reward), self.simulator.done, False, info

    def _build_obs(self) -> np.ndarray:
        assert self.simulator is not None
        trade = self.strategy.last_trade
        current_step = self.simulator.current_step
        total_steps = max(self.simulator.config.n_steps, 1)
        initial_x = max(self.simulator.config.initial_x, 1.0)
        initial_y = max(self.simulator.config.initial_y, 1.0)
        fair_scale = max(self.simulator.config.initial_price, 1.0)
        if trade is None:
            trade_features = np.zeros(6, dtype=np.float32)
        else:
            trade_features = np.asarray(
                [
                    1.0 if trade.is_buy else -1.0,
                    trade.amount_x / initial_x,
                    trade.amount_y / initial_y,
                    trade.reserve_x / initial_x,
                    trade.reserve_y / initial_y,
                    trade.timestamp / total_steps,
                ],
                dtype=np.float32,
            )
        obs = np.zeros(12, dtype=np.float32)
        obs[:6] = trade_features
        obs[6] = np.float32(self.strategy.current_bid_fee)
        obs[7] = np.float32(self.strategy.current_ask_fee)
        obs[8] = np.float32(self.simulator.submission.reserve_x / initial_x)
        obs[9] = np.float32(self.simulator.submission.reserve_y / initial_y)
        obs[10] = np.float32(current_step / total_steps)
        obs[11] = np.float32(self.simulator.submission.spot_price / fair_scale)
        return obs
