"""Stateful quote-surface venue inspired by the prop-amm challenge.

The venue exposes size-dependent quotes on both sides from a constrained,
monotone surface. Parameters are refreshed once per step from a bounded action
vector, while a small internal state persists across real trades.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from amm_gym.sim.amm import TradeResult


QUOTE_SURFACE_ACTION_DIM = 14
QUOTE_SURFACE_STATE_DIM = 4


@dataclass
class SurfaceSide:
    base_price: float
    capacity_x: float
    sharpness: float


class ParametricQuoteSurfaceAMM:
    """Stateful AMM with exponentially sloped quote surfaces on both sides."""

    def __init__(
        self,
        name: str,
        reserve_x: float,
        reserve_y: float,
        state_dim: int = QUOTE_SURFACE_STATE_DIM,
    ) -> None:
        self.name = name
        self.reserve_x = float(reserve_x)
        self.reserve_y = float(reserve_y)
        self.initial_reserve_x = float(reserve_x)
        self.initial_reserve_y = float(reserve_y)
        self.reference_price = self.reserve_y / max(self.reserve_x, 1e-9)
        self.accumulated_fees_x = 0.0
        self.accumulated_fees_y = 0.0
        self.last_action = np.zeros(QUOTE_SURFACE_ACTION_DIM, dtype=np.float64)
        self.state = np.zeros(state_dim, dtype=np.float64)
        self.bid_side = SurfaceSide(self.reference_price * 0.995, self.reserve_y / self.reference_price, 1.0)
        self.ask_side = SurfaceSide(self.reference_price * 1.005, self.reserve_x, 1.0)
        self.sampled_bid_spread_bps = 50.0
        self.sampled_ask_spread_bps = 50.0

    @property
    def spot_price(self) -> float:
        return 0.5 * (self.best_bid_price + self.best_ask_price)

    @property
    def best_bid_price(self) -> float:
        return self.bid_side.base_price

    @property
    def best_ask_price(self) -> float:
        return self.ask_side.base_price

    def reserves(self) -> tuple[float, float]:
        return (self.reserve_x, self.reserve_y)

    def reset(self, reserve_x: float, reserve_y: float) -> None:
        self.reserve_x = float(reserve_x)
        self.reserve_y = float(reserve_y)
        self.reference_price = self.reserve_y / max(self.reserve_x, 1e-9)
        self.accumulated_fees_x = 0.0
        self.accumulated_fees_y = 0.0
        self.last_action.fill(0.0)
        self.state.fill(0.0)
        self._recompute_surface()

    def configure(self, reference_price: float, action: np.ndarray) -> None:
        clipped = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if clipped.shape != (QUOTE_SURFACE_ACTION_DIM,):
            raise ValueError(f"quote surface action must have shape ({QUOTE_SURFACE_ACTION_DIM},)")
        self.reference_price = max(float(reference_price), 1e-6)
        self.last_action = clipped
        self._recompute_surface()

    def current_quote_summary(self) -> dict[str, float]:
        return {
            "bid_spread_bps": float(self.sampled_bid_spread_bps),
            "ask_spread_bps": float(self.sampled_ask_spread_bps),
            "bid_capacity_x": float(self.bid_side.capacity_x),
            "ask_capacity_x": float(self.ask_side.capacity_x),
            "quote_state_inventory": float(self.state[0]),
            "quote_state_flow": float(self.state[1]),
            "quote_state_arb": float(self.state[2]),
            "quote_state_latent": float(self.state[3]),
        }

    def quote_buy_x(self, amount_x: float) -> tuple[float, float]:
        y_out = self._bid_total_y_for_x(float(amount_x))
        return (y_out, 0.0) if y_out > 0.0 else (0.0, 0.0)

    def quote_sell_x(self, amount_x: float) -> tuple[float, float]:
        total_y = self._ask_total_y_for_x(float(amount_x))
        return (total_y, 0.0) if total_y > 0.0 else (0.0, 0.0)

    def quote_x_for_y(self, amount_y: float) -> tuple[float, float]:
        x_out = self._ask_x_for_total_y(float(amount_y))
        return (x_out, 0.0) if x_out > 0.0 else (0.0, 0.0)

    def execute_buy_x(self, amount_x: float, timestamp: int) -> TradeResult | None:
        amount_x = float(amount_x)
        y_out = self._bid_total_y_for_x(amount_x)
        if y_out <= 0.0:
            return None
        self.reserve_x += amount_x
        self.reserve_y -= y_out
        return TradeResult(
            is_buy=True,
            amount_x=amount_x,
            amount_y=y_out,
            fee_amount=0.0,
            timestamp=timestamp,
        )

    def execute_sell_x(self, amount_x: float, timestamp: int) -> TradeResult | None:
        amount_x = float(amount_x)
        total_y = self._ask_total_y_for_x(amount_x)
        if total_y <= 0.0:
            return None
        self.reserve_x -= amount_x
        self.reserve_y += total_y
        return TradeResult(
            is_buy=False,
            amount_x=amount_x,
            amount_y=total_y,
            fee_amount=0.0,
            timestamp=timestamp,
        )

    def execute_buy_x_with_y(self, amount_y: float, timestamp: int) -> TradeResult | None:
        amount_y = float(amount_y)
        x_out = self._ask_x_for_total_y(amount_y)
        if x_out <= 0.0:
            return None
        self.reserve_x -= x_out
        self.reserve_y += amount_y
        return TradeResult(
            is_buy=False,
            amount_x=x_out,
            amount_y=amount_y,
            fee_amount=0.0,
            timestamp=timestamp,
        )

    def marginal_ask_price_after_y(self, amount_y: float) -> float:
        x_out = self._ask_x_for_total_y(float(amount_y))
        return self._ask_marginal_price_for_x(x_out)

    def marginal_bid_price_after_x(self, amount_x: float) -> float:
        return self._bid_marginal_price_for_x(float(amount_x))

    def total_remaining_ask_y(self) -> float:
        max_x = min(self.ask_side.capacity_x, self.reserve_x * 0.98)
        return self._ask_total_y_for_x(max_x)

    def total_remaining_bid_x(self) -> float:
        return min(self.bid_side.capacity_x, self.reserve_y / max(self.bid_side.base_price, 1e-9))

    def record_trade(
        self,
        *,
        amount_y: float,
        signed_flow_y: float,
        is_arbitrage: bool,
    ) -> None:
        value_scale = max(self.initial_reserve_y, 1.0)
        flow_signal = np.clip(signed_flow_y / value_scale, -1.0, 1.0)
        volume_signal = np.clip(amount_y / value_scale, 0.0, 1.0)
        reserve_value = self.reserve_x * self.reference_price + self.reserve_y
        inventory_signal = 0.0
        if reserve_value > 1e-9:
            inventory_signal = np.clip(
                (self.reserve_x * self.reference_price - self.reserve_y) / reserve_value,
                -1.0,
                1.0,
            )
        self.state[0] = inventory_signal
        self.state[1] = 0.88 * self.state[1] + 0.12 * flow_signal
        self.state[2] = 0.88 * self.state[2] + 0.12 * (volume_signal if is_arbitrage else 0.0)
        latent_write = 0.65 * float(self.last_action[12]) + 0.35 * float(self.last_action[13])
        self.state[3] = math.tanh(0.9 * self.state[3] + 0.15 * latent_write)
        self._recompute_surface()

    def _recompute_surface(self) -> None:
        action = self.last_action
        inventory_shift = 0.010 * float(action[10]) * self.state[0]
        flow_shift = 0.008 * float(action[11]) * (self.state[1] - 0.5 * self.state[2])
        latent_shift = 0.004 * self.state[3]
        common_shift = inventory_shift + flow_shift + latent_shift

        bid_offset = 0.012 * float(action[0])
        ask_offset = 0.012 * float(action[1])
        bid_spread = np.interp(float(action[2]), [-1.0, 1.0], [0.0005, 0.0200])
        ask_spread = np.interp(float(action[3]), [-1.0, 1.0], [0.0005, 0.0200])
        bid_level = math.exp(math.log(4.0) * float(action[4]))
        ask_level = math.exp(math.log(4.0) * float(action[5]))
        bid_curvature = np.interp(float(action[6]), [-1.0, 1.0], [0.20, 3.00])
        ask_curvature = np.interp(float(action[7]), [-1.0, 1.0], [0.20, 3.00])
        bid_tail = np.interp(float(action[8]), [-1.0, 1.0], [0.35, 2.50])
        ask_tail = np.interp(float(action[9]), [-1.0, 1.0], [0.35, 2.50])

        bid_base = self.reference_price * max(0.05, 1.0 - bid_spread - bid_offset - common_shift)
        ask_base = self.reference_price * max(1.0 + ask_spread + ask_offset - common_shift, 0.06)
        if ask_base <= bid_base:
            midpoint = 0.5 * (ask_base + bid_base)
            bid_base = max(0.05 * self.reference_price, midpoint - 0.0025 * self.reference_price)
            ask_base = midpoint + 0.0025 * self.reference_price

        bid_capacity_x = min(
            max((self.reserve_y / max(bid_base, 1e-6)) * (0.20 + 0.18 * bid_level * bid_tail), 1e-6),
            self.reserve_y / max(bid_base, 1e-6),
        )
        ask_capacity_x = min(
            max(self.reserve_x * (0.20 + 0.18 * ask_level * ask_tail), 1e-6),
            self.reserve_x,
        )

        self.bid_side = SurfaceSide(
            base_price=bid_base,
            capacity_x=max(1e-6, bid_capacity_x),
            sharpness=bid_curvature / max(bid_level, 1e-6),
        )
        self.ask_side = SurfaceSide(
            base_price=ask_base,
            capacity_x=max(1e-6, ask_capacity_x),
            sharpness=ask_curvature / max(ask_level, 1e-6),
        )
        self.sampled_bid_spread_bps = max(0.0, (self.reference_price - bid_base) / self.reference_price * 10_000.0)
        self.sampled_ask_spread_bps = max(0.0, (ask_base - self.reference_price) / self.reference_price * 10_000.0)

    def _ask_total_y_for_x(self, amount_x: float) -> float:
        amount_x = max(0.0, min(float(amount_x), self.ask_side.capacity_x, self.reserve_x * 0.98))
        if amount_x <= 0.0:
            return 0.0
        k = self.ask_side.sharpness
        cap = self.ask_side.capacity_x
        base = self.ask_side.base_price
        if k <= 1e-9:
            return base * amount_x
        return base * cap / k * (math.exp(k * amount_x / cap) - 1.0)

    def _ask_x_for_total_y(self, amount_y: float) -> float:
        amount_y = max(0.0, float(amount_y))
        if amount_y <= 0.0:
            return 0.0
        k = self.ask_side.sharpness
        cap = self.ask_side.capacity_x
        base = self.ask_side.base_price
        max_y = self._ask_total_y_for_x(min(cap, self.reserve_x * 0.98))
        clamped_y = min(amount_y, max_y)
        if k <= 1e-9:
            return min(clamped_y / max(base, 1e-9), cap, self.reserve_x * 0.98)
        inside = 1.0 + clamped_y * k / max(base * cap, 1e-9)
        return min(cap / k * math.log(max(inside, 1.0)), cap, self.reserve_x * 0.98)

    def _ask_marginal_price_for_x(self, amount_x: float) -> float:
        x = max(0.0, min(float(amount_x), self.ask_side.capacity_x))
        return self.ask_side.base_price * math.exp(self.ask_side.sharpness * x / self.ask_side.capacity_x)

    def _bid_total_y_for_x(self, amount_x: float) -> float:
        amount_x = max(0.0, min(float(amount_x), self.bid_side.capacity_x))
        if amount_x <= 0.0:
            return 0.0
        k = self.bid_side.sharpness
        cap = self.bid_side.capacity_x
        base = self.bid_side.base_price
        if k <= 1e-9:
            return min(base * amount_x, self.reserve_y * 0.98)
        y_out = base * cap / k * (1.0 - math.exp(-k * amount_x / cap))
        return min(y_out, self.reserve_y * 0.98)

    def _bid_marginal_price_for_x(self, amount_x: float) -> float:
        x = max(0.0, min(float(amount_x), self.bid_side.capacity_x))
        return self.bid_side.base_price * math.exp(-self.bid_side.sharpness * x / self.bid_side.capacity_x)
