"""Banded depth-ladder venue used for the submission strategy.

The venue posts a fixed reference-price ladder for a single step. Retail flow
and arbitrage both trade against the same posted surface. Each side uses fixed
bps bands whose depths are generated from a low-dimensional control vector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class LadderTradeResult:
    is_buy: bool  # True if venue bought X, False if venue sold X
    amount_x: float
    amount_y: float
    timestamp: int


class DepthLadderAMM:
    """Stylized dynamic-liquidity AMM with a fixed per-step ladder."""

    def __init__(
        self,
        name: str,
        reserve_x: float,
        reserve_y: float,
        band_bps: tuple[float, ...],
        base_notional_y: float,
    ) -> None:
        self.name = name
        self.reserve_x = reserve_x
        self.reserve_y = reserve_y
        self.accumulated_fees_x = 0.0
        self.accumulated_fees_y = 0.0
        self.band_bps = np.asarray(band_bps, dtype=np.float64)
        self.band_rel = self.band_bps / 10_000.0
        self._band_lower = np.concatenate(([0.0], self.band_rel[:-1]))
        self.base_notional_y = float(base_notional_y)
        self.reference_price = 1.0
        self.bid_controls = np.zeros(3, dtype=np.float64)
        self.ask_controls = np.zeros(3, dtype=np.float64)
        self.bid_depth_x = np.zeros_like(self.band_rel)
        self.ask_depth_y = np.zeros_like(self.band_rel)
        self.bid_consumed_x = np.zeros_like(self.band_rel)
        self.ask_consumed_y = np.zeros_like(self.band_rel)

    @property
    def spot_price(self) -> float:
        return 0.5 * (self.best_bid_price + self.best_ask_price)

    @property
    def best_ask_price(self) -> float:
        return self._current_ask_start_price()

    @property
    def best_bid_price(self) -> float:
        return self._current_bid_start_price()

    def reserves(self) -> tuple[float, float]:
        return (self.reserve_x, self.reserve_y)

    def reset(self, reserve_x: float, reserve_y: float) -> None:
        self.reserve_x = reserve_x
        self.reserve_y = reserve_y
        self.accumulated_fees_x = 0.0
        self.accumulated_fees_y = 0.0
        self.bid_consumed_x.fill(0.0)
        self.ask_consumed_y.fill(0.0)

    def configure(
        self,
        reference_price: float,
        bid_raw: np.ndarray,
        ask_raw: np.ndarray,
    ) -> None:
        """Generate a fresh ladder for the current step."""
        self.reference_price = max(float(reference_price), 1e-9)
        self.bid_controls = np.asarray(bid_raw, dtype=np.float64)
        self.ask_controls = np.asarray(ask_raw, dtype=np.float64)
        self.bid_consumed_x.fill(0.0)
        self.ask_consumed_y.fill(0.0)

        self.ask_depth_y = self._generate_depths(self.ask_controls)
        bid_depth_y = self._generate_depths(self.bid_controls)
        self.bid_depth_x = bid_depth_y / self.reference_price

        self._cap_ask_depths_to_inventory()
        self._cap_bid_depths_to_cash()

    def current_ladder_summary(self) -> dict[str, float]:
        return {
            "ask_near_depth_y": float(self.ask_depth_y[0]) if len(self.ask_depth_y) else 0.0,
            "ask_far_depth_y": float(self.ask_depth_y[-1]) if len(self.ask_depth_y) else 0.0,
            "bid_near_depth_y": float(self.bid_depth_x[0] * self.reference_price)
            if len(self.bid_depth_x)
            else 0.0,
            "bid_far_depth_y": float(self.bid_depth_x[-1] * self.reference_price)
            if len(self.bid_depth_x)
            else 0.0,
        }

    def total_remaining_ask_y(self) -> float:
        total = 0.0
        for depth_y, lower, upper, consumed_y in zip(
            self.ask_depth_y, self._band_lower, self.band_rel, self.ask_consumed_y
        ):
            total += max(0.0, depth_y * (upper - lower) - consumed_y)
        return float(total)

    def total_remaining_bid_x(self) -> float:
        total = 0.0
        for depth_x, lower, upper, consumed_x in zip(
            self.bid_depth_x, self._band_lower, self.band_rel, self.bid_consumed_x
        ):
            total += max(0.0, depth_x * (upper - lower) - consumed_x)
        return float(total)

    def marginal_ask_price_after_y(self, amount_y: float) -> float:
        _, end_price = self._simulate_buy_x_with_y(amount_y)
        return end_price

    def marginal_bid_price_after_x(self, amount_x: float) -> float:
        _, end_price = self._simulate_buy_x(amount_x)
        return end_price

    def quote_x_for_y(self, amount_y: float) -> tuple[float, float]:
        x_out, _ = self._simulate_buy_x_with_y(amount_y)
        return (x_out, 0.0)

    def execute_buy_x_with_y(
        self, amount_y: float, timestamp: int
    ) -> LadderTradeResult | None:
        x_out, _, consumed = self._simulate_buy_x_with_y(amount_y, return_trace=True)
        if x_out <= 0.0:
            return None

        for idx, value in consumed:
            self.ask_consumed_y[idx] += value
        self.reserve_x -= x_out
        self.reserve_y += amount_y
        return LadderTradeResult(
            is_buy=False,
            amount_x=x_out,
            amount_y=amount_y,
            timestamp=timestamp,
        )

    def quote_buy_x(self, amount_x: float) -> tuple[float, float]:
        y_out, _ = self._simulate_buy_x(amount_x)
        return (y_out, 0.0)

    def execute_buy_x(self, amount_x: float, timestamp: int) -> LadderTradeResult | None:
        y_out, _, consumed = self._simulate_buy_x(amount_x, return_trace=True)
        if y_out <= 0.0:
            return None

        for idx, value in consumed:
            self.bid_consumed_x[idx] += value
        self.reserve_x += amount_x
        self.reserve_y -= y_out
        return LadderTradeResult(
            is_buy=True,
            amount_x=amount_x,
            amount_y=y_out,
            timestamp=timestamp,
        )

    def _generate_depths(self, raw_controls: np.ndarray) -> np.ndarray:
        scale_raw, decay_raw, tilt_raw = [float(x) for x in raw_controls]
        scale = math.exp(math.log(4.0) * scale_raw)
        decay = 0.25 + 1.25 * (decay_raw + 1.0) / 2.0
        tilt = 0.35 * tilt_raw

        idx = np.arange(len(self.band_rel), dtype=np.float64)
        basis = np.linspace(1.5, -1.5, len(self.band_rel), dtype=np.float64)
        weights = np.exp(-decay * idx + tilt * basis)
        weights = np.minimum.accumulate(weights)
        return self.base_notional_y * scale * weights

    def _cap_ask_depths_to_inventory(self) -> None:
        total_x = 0.0
        for depth_y, lower, upper in zip(
            self.ask_depth_y, self._band_lower, self.band_rel
        ):
            start_p = self.reference_price * (1.0 + lower)
            end_p = self.reference_price * (1.0 + upper)
            total_x += depth_y / self.reference_price * math.log(end_p / start_p)

        max_x = max(self.reserve_x * 0.98, 1e-9)
        if total_x > max_x:
            self.ask_depth_y *= max_x / total_x

    def _cap_bid_depths_to_cash(self) -> None:
        total_y = 0.0
        for depth_x, lower, upper in zip(
            self.bid_depth_x, self._band_lower, self.band_rel
        ):
            x_cap = depth_x * (upper - lower)
            start_p = self.reference_price * (1.0 - lower)
            total_y += start_p * x_cap - 0.5 * self.reference_price / depth_x * x_cap * x_cap

        max_y = max(self.reserve_y * 0.98, 1e-9)
        if total_y > max_y:
            self.bid_depth_x *= max_y / total_y

    def _current_ask_start_price(self) -> float:
        for depth_y, lower, upper, consumed in zip(
            self.ask_depth_y, self._band_lower, self.band_rel, self.ask_consumed_y
        ):
            band_cap = depth_y * (upper - lower)
            if consumed < band_cap - 1e-12:
                return self.reference_price * (1.0 + lower + consumed / depth_y)
        return self.reference_price * (1.0 + self.band_rel[-1])

    def _current_bid_start_price(self) -> float:
        for depth_x, lower, upper, consumed in zip(
            self.bid_depth_x, self._band_lower, self.band_rel, self.bid_consumed_x
        ):
            band_cap = depth_x * (upper - lower)
            if consumed < band_cap - 1e-12:
                return self.reference_price * (1.0 - lower - consumed / depth_x)
        return self.reference_price * max(1e-6, 1.0 - self.band_rel[-1])

    def _simulate_buy_x_with_y(
        self,
        amount_y: float,
        return_trace: bool = False,
    ) -> tuple[float, float] | tuple[float, float, list[tuple[int, float]]]:
        remaining = max(float(amount_y), 0.0)
        if remaining <= 0.0:
            result = (0.0, self._current_ask_start_price())
            return (*result, []) if return_trace else result

        x_out = 0.0
        end_price = self._current_ask_start_price()
        consumed_trace: list[tuple[int, float]] = []

        for idx, (depth_y, lower, upper, consumed_y) in enumerate(
            zip(self.ask_depth_y, self._band_lower, self.band_rel, self.ask_consumed_y)
        ):
            band_cap = depth_y * (upper - lower)
            remaining_cap = max(0.0, band_cap - consumed_y)
            if remaining_cap <= 1e-12:
                continue

            band_y = min(remaining, remaining_cap)
            start_price = self.reference_price * (1.0 + lower + consumed_y / depth_y)
            end_price = start_price + self.reference_price * band_y / depth_y
            x_out += depth_y / self.reference_price * math.log(end_price / start_price)
            consumed_trace.append((idx, band_y))
            remaining -= band_y
            if remaining <= 1e-12:
                break

        result = (x_out, end_price)
        return (*result, consumed_trace) if return_trace else result

    def _simulate_buy_x(
        self,
        amount_x: float,
        return_trace: bool = False,
    ) -> tuple[float, float] | tuple[float, float, list[tuple[int, float]]]:
        remaining = max(float(amount_x), 0.0)
        if remaining <= 0.0:
            result = (0.0, self._current_bid_start_price())
            return (*result, []) if return_trace else result

        y_out = 0.0
        end_price = self._current_bid_start_price()
        consumed_trace: list[tuple[int, float]] = []

        for idx, (depth_x, lower, upper, consumed_x) in enumerate(
            zip(self.bid_depth_x, self._band_lower, self.band_rel, self.bid_consumed_x)
        ):
            band_cap = depth_x * (upper - lower)
            remaining_cap = max(0.0, band_cap - consumed_x)
            if remaining_cap <= 1e-12:
                continue

            band_x = min(remaining, remaining_cap)
            start_price = self.reference_price * (1.0 - lower - consumed_x / depth_x)
            y_out += (
                start_price * band_x
                - 0.5 * self.reference_price / depth_x * band_x * band_x
            )
            end_price = start_price - self.reference_price * band_x / depth_x
            consumed_trace.append((idx, band_x))
            remaining -= band_x
            if remaining <= 1e-12:
                break

        result = (y_out, max(end_price, 1e-9))
        return (*result, consumed_trace) if return_trace else result
