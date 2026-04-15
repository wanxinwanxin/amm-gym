"""Constant Function Market Maker (x * y = k) with fee-on-input.

Matches the math in amm_sim_rs/src/amm/cfmm.rs exactly.
Fees are collected into separate buckets (not reinvested into liquidity),
so k stays constant between trades.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class FeeQuote:
    bid_fee: float  # fee when AMM buys X (trader sells X)
    ask_fee: float  # fee when AMM sells X (trader buys X)


@dataclass
class TradeResult:
    is_buy: bool  # True if AMM bought X
    amount_x: float
    amount_y: float
    fee_amount: float
    timestamp: int


class ConstantProductAMM:
    """Constant product AMM with separate fee collection (Uniswap V3/V4 style)."""

    def __init__(
        self,
        name: str,
        reserve_x: float,
        reserve_y: float,
        bid_fee: float = 0.003,
        ask_fee: float = 0.003,
    ) -> None:
        self.name = name
        self.reserve_x = reserve_x
        self.reserve_y = reserve_y
        self.fees = FeeQuote(bid_fee, ask_fee)
        self.accumulated_fees_x = 0.0
        self.accumulated_fees_y = 0.0

    @property
    def spot_price(self) -> float:
        """Current spot price (Y per X)."""
        if self.reserve_x == 0.0:
            return 0.0
        return self.reserve_y / self.reserve_x

    @property
    def k(self) -> float:
        return self.reserve_x * self.reserve_y

    def reserves(self) -> tuple[float, float]:
        return (self.reserve_x, self.reserve_y)

    def quote_buy_x(self, amount_x: float) -> tuple[float, float]:
        """Quote for AMM buying X (trader sells X for Y).

        Returns (y_out, fee_amount) or (0, 0) if invalid.
        """
        if amount_x <= 0.0:
            return (0.0, 0.0)

        fee = self.fees.bid_fee
        gamma = max(0.0, min(1.0, 1.0 - fee))
        if gamma <= 0.0:
            return (0.0, 0.0)

        net_x = amount_x * gamma
        k = self.reserve_x * self.reserve_y
        new_rx = self.reserve_x + net_x
        new_ry = k / new_rx
        y_out = self.reserve_y - new_ry

        if y_out > 0.0:
            return (y_out, amount_x * fee)
        return (0.0, 0.0)

    def quote_sell_x(self, amount_x: float) -> tuple[float, float]:
        """Quote for AMM selling X (trader buys X with Y).

        Returns (total_y_in, fee_amount) or (0, 0) if invalid.
        """
        if amount_x <= 0.0 or amount_x >= self.reserve_x:
            return (0.0, 0.0)

        k = self.reserve_x * self.reserve_y
        fee = self.fees.ask_fee
        gamma = max(0.0, min(1.0, 1.0 - fee))
        if gamma <= 0.0:
            return (0.0, 0.0)

        new_rx = self.reserve_x - amount_x
        new_ry = k / new_rx
        net_y = new_ry - self.reserve_y

        if net_y <= 0.0:
            return (0.0, 0.0)

        total_y = net_y / gamma
        return (total_y, total_y - net_y)

    def quote_x_for_y(self, amount_y: float) -> tuple[float, float]:
        """Quote for trader paying Y to receive X.

        Returns (x_out, fee_amount) or (0, 0) if invalid.
        """
        if amount_y <= 0.0:
            return (0.0, 0.0)

        k = self.reserve_x * self.reserve_y
        fee = self.fees.ask_fee
        gamma = max(0.0, min(1.0, 1.0 - fee))
        if gamma <= 0.0:
            return (0.0, 0.0)

        net_y = amount_y * gamma
        new_ry = self.reserve_y + net_y
        new_rx = k / new_ry
        x_out = self.reserve_x - new_rx

        if x_out > 0.0:
            return (x_out, amount_y * fee)
        return (0.0, 0.0)

    def execute_buy_x(self, amount_x: float, timestamp: int) -> TradeResult | None:
        """Execute trade where AMM buys X (trader sells X for Y)."""
        y_out, fee_amount = self.quote_buy_x(amount_x)
        if y_out <= 0.0:
            return None

        net_x = amount_x - fee_amount
        self.reserve_x += net_x
        self.accumulated_fees_x += fee_amount
        self.reserve_y -= y_out

        return TradeResult(
            is_buy=True,
            amount_x=amount_x,
            amount_y=y_out,
            fee_amount=fee_amount,
            timestamp=timestamp,
        )

    def execute_sell_x(self, amount_x: float, timestamp: int) -> TradeResult | None:
        """Execute trade where AMM sells X (trader buys X with Y)."""
        total_y, fee_amount = self.quote_sell_x(amount_x)
        if total_y <= 0.0:
            return None

        net_y = total_y - fee_amount
        self.reserve_x -= amount_x
        self.reserve_y += net_y
        self.accumulated_fees_y += fee_amount

        return TradeResult(
            is_buy=False,
            amount_x=amount_x,
            amount_y=total_y,
            fee_amount=fee_amount,
            timestamp=timestamp,
        )

    def execute_buy_x_with_y(self, amount_y: float, timestamp: int) -> TradeResult | None:
        """Execute trade where trader pays Y to receive X."""
        x_out, fee_amount = self.quote_x_for_y(amount_y)
        if x_out <= 0.0:
            return None

        net_y = amount_y - fee_amount
        self.reserve_x -= x_out
        self.reserve_y += net_y
        self.accumulated_fees_y += fee_amount

        return TradeResult(
            is_buy=False,
            amount_x=x_out,
            amount_y=amount_y,
            fee_amount=fee_amount,
            timestamp=timestamp,
        )

    def reset(self, reserve_x: float, reserve_y: float) -> None:
        self.reserve_x = reserve_x
        self.reserve_y = reserve_y
        self.accumulated_fees_x = 0.0
        self.accumulated_fees_y = 0.0

    def marginal_ask_price_after_y(self, amount_y: float) -> float:
        """Marginal ask price (Y per X) after paying `amount_y` of Y."""
        fee = self.fees.ask_fee
        gamma = max(0.0, min(1.0, 1.0 - fee))
        if gamma <= 0.0:
            return math.inf

        net_y = max(0.0, amount_y) * gamma
        new_ry = self.reserve_y + net_y
        new_rx = self.k / new_ry if new_ry > 0.0 else 0.0
        if new_rx <= 0.0:
            return math.inf
        return (new_ry / new_rx) / gamma

    def marginal_bid_price_after_x(self, amount_x: float) -> float:
        """Marginal bid payout price (Y per X) after selling `amount_x` of X."""
        fee = self.fees.bid_fee
        gamma = max(0.0, min(1.0, 1.0 - fee))
        if gamma <= 0.0:
            return 0.0

        net_x = max(0.0, amount_x) * gamma
        new_rx = self.reserve_x + net_x
        new_ry = self.k / new_rx if new_rx > 0.0 else 0.0
        return gamma * (new_ry / new_rx) if new_rx > 0.0 else 0.0
