"""Market actors: arbitrageur, retail traders, and order router.

Matches the math in amm_sim_rs/src/market/ exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from amm_gym.sim.amm import ConstantProductAMM, TradeResult
from amm_gym.sim.ladder import DepthLadderAMM


# ---------------------------------------------------------------------------
# Arbitrageur
# ---------------------------------------------------------------------------

@dataclass
class ArbResult:
    amm_name: str
    profit: float
    side: str  # "buy" or "sell" (from AMM perspective)
    amount_x: float
    amount_y: float


class Arbitrageur:
    """Rational arbitrageur that trades AMMs back to fair price.

    Closed-form optimal trade sizes for constant-product AMMs with
    fee-on-input (matches amm_sim_rs/src/market/arbitrageur.rs).
    """

    def execute_arb(
        self, amm: ConstantProductAMM | DepthLadderAMM, fair_price: float, timestamp: int
    ) -> ArbResult | None:
        if isinstance(amm, DepthLadderAMM):
            return self._execute_ladder_arb(amm, fair_price, timestamp)
        spot = amm.spot_price
        if spot < fair_price:
            return self._buy_arb(amm, fair_price, timestamp)
        elif spot > fair_price:
            return self._sell_arb(amm, fair_price, timestamp)
        return None

    def _execute_ladder_arb(
        self, amm: DepthLadderAMM, fair_price: float, timestamp: int
    ) -> ArbResult | None:
        best_ask = amm.best_ask_price
        if fair_price > best_ask:
            max_y = amm.total_remaining_ask_y()
            if max_y <= 0.0:
                return None
            end_price = amm.marginal_ask_price_after_y(max_y)
            if fair_price >= end_price:
                amount_y = max_y
            else:
                amount_y = self._bisect_increasing(
                    lambda y: amm.marginal_ask_price_after_y(y),
                    target=fair_price,
                    low=0.0,
                    high=max_y,
                )
            trade = amm.execute_buy_x_with_y(amount_y, timestamp)
            if trade is None:
                return None
            profit = trade.amount_x * fair_price - amount_y
            if profit <= 0.0:
                return None
            return ArbResult(
                amm_name=amm.name,
                profit=profit,
                side="sell",
                amount_x=trade.amount_x,
                amount_y=amount_y,
            )

        best_bid = amm.best_bid_price
        if fair_price < best_bid:
            max_x = amm.total_remaining_bid_x()
            if max_x <= 0.0:
                return None
            end_price = amm.marginal_bid_price_after_x(max_x)
            if fair_price <= end_price:
                amount_x = max_x
            else:
                amount_x = self._bisect_decreasing(
                    lambda x: amm.marginal_bid_price_after_x(x),
                    target=fair_price,
                    low=0.0,
                    high=max_x,
                )
            trade = amm.execute_buy_x(amount_x, timestamp)
            if trade is None:
                return None
            profit = trade.amount_y - trade.amount_x * fair_price
            if profit <= 0.0:
                return None
            return ArbResult(
                amm_name=amm.name,
                profit=profit,
                side="buy",
                amount_x=amount_x,
                amount_y=trade.amount_y,
            )

        return None

    def _bisect_increasing(self, fn, target: float, low: float, high: float) -> float:
        lo = low
        hi = high
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if fn(mid) < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _bisect_decreasing(self, fn, target: float, low: float, high: float) -> float:
        lo = low
        hi = high
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if fn(mid) > target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _buy_arb(
        self, amm: ConstantProductAMM, fair_price: float, timestamp: int
    ) -> ArbResult | None:
        """AMM underprices X (spot < fair) -> buy X from AMM (AMM sells X).

        Optimal: amount_x_out = rx - sqrt(k / (gamma * fair_price))
        """
        rx, ry = amm.reserve_x, amm.reserve_y
        k = rx * ry
        fee = amm.fees.ask_fee
        gamma = 1.0 - fee

        if gamma <= 0.0 or fair_price <= 0.0:
            return None

        new_x = math.sqrt(k / (gamma * fair_price))
        amount_x = rx - new_x

        if amount_x <= 0.0:
            return None

        # Cap at 99% of reserves
        amount_x = min(amount_x, rx * 0.99)

        total_y, _ = amm.quote_sell_x(amount_x)
        if total_y <= 0.0:
            return None

        profit = amount_x * fair_price - total_y
        if profit <= 0.0:
            return None

        amm.execute_sell_x(amount_x, timestamp)

        return ArbResult(
            amm_name=amm.name,
            profit=profit,
            side="sell",  # AMM sells X
            amount_x=amount_x,
            amount_y=total_y,
        )

    def _sell_arb(
        self, amm: ConstantProductAMM, fair_price: float, timestamp: int
    ) -> ArbResult | None:
        """AMM overprices X (spot > fair) -> sell X to AMM (AMM buys X).

        Optimal: amount_x_in = (sqrt(k * gamma / fair_price) - rx) / gamma
        """
        rx, ry = amm.reserve_x, amm.reserve_y
        k = rx * ry
        fee = amm.fees.bid_fee
        gamma = 1.0 - fee

        if gamma <= 0.0 or fair_price <= 0.0:
            return None

        x_virtual = math.sqrt(k * gamma / fair_price)
        net_x = x_virtual - rx
        amount_x = net_x / gamma

        if amount_x <= 0.0:
            return None

        y_out, _ = amm.quote_buy_x(amount_x)
        if y_out <= 0.0:
            return None

        profit = y_out - amount_x * fair_price
        if profit <= 0.0:
            return None

        amm.execute_buy_x(amount_x, timestamp)

        return ArbResult(
            amm_name=amm.name,
            profit=profit,
            side="buy",  # AMM buys X
            amount_x=amount_x,
            amount_y=y_out,
        )


# ---------------------------------------------------------------------------
# Retail trader
# ---------------------------------------------------------------------------

@dataclass
class RetailOrder:
    side: str  # "buy" or "sell" (from trader perspective, re: X)
    size: float  # in Y terms


class RetailTrader:
    """Generates retail flow: Poisson arrivals, log-normal sizes.

    Matches amm_sim_rs/src/market/retail.rs.
    """

    def __init__(
        self,
        arrival_rate: float,
        mean_size: float,
        size_sigma: float,
        buy_prob: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self.arrival_rate = arrival_rate
        self.mean_size = mean_size
        self.size_sigma = max(size_sigma, 0.01)
        self.buy_prob = buy_prob
        self.rng = np.random.default_rng(seed)

        # Log-normal params: E[X] = mean_size => mu = ln(mean) - 0.5*sigma^2
        mean = max(mean_size, 0.01)
        self._ln_mu = math.log(mean) - 0.5 * self.size_sigma ** 2
        self._ln_sigma = self.size_sigma

    def generate_orders(self) -> list[RetailOrder]:
        if self.arrival_rate <= 0.0:
            return []

        n = self.rng.poisson(self.arrival_rate)
        if n == 0:
            return []

        sizes = self.rng.lognormal(self._ln_mu, self._ln_sigma, size=n)
        sides = self.rng.random(size=n)

        return [
            RetailOrder(
                side="buy" if s < self.buy_prob else "sell",
                size=float(sz),
            )
            for s, sz in zip(sides, sizes)
        ]

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Order router
# ---------------------------------------------------------------------------

@dataclass
class RoutedTrade:
    amm_name: str
    amount_y: float
    amount_x: float
    amm_buys_x: bool


MIN_AMOUNT = 0.0001


class OrderRouter:
    """Routes retail orders optimally across two AMMs.

    Uses closed-form optimal splitting that equalizes marginal prices
    (matches amm_sim_rs/src/market/router.rs).
    """

    def split_buy_two_amms(
        self,
        amm1: ConstantProductAMM,
        amm2: ConstantProductAMM,
        total_y: float,
    ) -> tuple[float, float]:
        """Optimal Y split for buying X across two AMMs.

        A_i = sqrt(x_i * gamma_i * y_i), r = A1/A2
        dy1 = (r * (y2 + gamma2 * Y) - y1) / (gamma1 + r * gamma2)
        """
        x1, y1 = amm1.reserve_x, amm1.reserve_y
        x2, y2 = amm2.reserve_x, amm2.reserve_y
        gamma1 = 1.0 - amm1.fees.ask_fee
        gamma2 = 1.0 - amm2.fees.ask_fee

        a1 = math.sqrt(x1 * gamma1 * y1)
        a2 = math.sqrt(x2 * gamma2 * y2)

        if a2 == 0.0:
            return (total_y, 0.0)

        r = a1 / a2
        numerator = r * (y2 + gamma2 * total_y) - y1
        denominator = gamma1 + r * gamma2

        if denominator == 0.0:
            y1_amount = total_y / 2.0
        else:
            y1_amount = numerator / denominator

        y1_amount = max(0.0, min(total_y, y1_amount))
        return (y1_amount, total_y - y1_amount)

    def split_sell_two_amms(
        self,
        amm1: ConstantProductAMM,
        amm2: ConstantProductAMM,
        total_x: float,
    ) -> tuple[float, float]:
        """Optimal X split for selling X across two AMMs.

        B_i = sqrt(y_i * gamma_i * x_i), r = B1/B2
        dx1 = (r * (x2 + gamma2 * X) - x1) / (gamma1 + r * gamma2)
        """
        x1, y1 = amm1.reserve_x, amm1.reserve_y
        x2, y2 = amm2.reserve_x, amm2.reserve_y
        gamma1 = 1.0 - amm1.fees.bid_fee
        gamma2 = 1.0 - amm2.fees.bid_fee

        b1 = math.sqrt(y1 * gamma1 * x1)
        b2 = math.sqrt(y2 * gamma2 * x2)

        if b2 == 0.0:
            return (total_x, 0.0)

        r = b1 / b2
        numerator = r * (x2 + gamma2 * total_x) - x1
        denominator = gamma1 + r * gamma2

        if denominator == 0.0:
            x1_amount = total_x / 2.0
        else:
            x1_amount = numerator / denominator

        x1_amount = max(0.0, min(total_x, x1_amount))
        return (x1_amount, total_x - x1_amount)

    def route_order(
        self,
        order: RetailOrder,
        amm_agent: ConstantProductAMM | DepthLadderAMM,
        amm_norm: ConstantProductAMM,
        fair_price: float,
        timestamp: int,
    ) -> list[RoutedTrade]:
        """Route a single retail order across agent and normalizer AMMs."""
        trades: list[RoutedTrade] = []

        if order.side == "buy":
            # Trader wants to buy X, spending Y
            if isinstance(amm_agent, ConstantProductAMM):
                y1, y2 = self.split_buy_two_amms(amm_agent, amm_norm, order.size)
            else:
                y1, y2 = self.solve_buy_split(amm_agent, amm_norm, order.size)

            if y1 > MIN_AMOUNT:
                result = amm_agent.execute_buy_x_with_y(y1, timestamp)
                if result is not None:
                    trades.append(RoutedTrade(
                        amm_name=amm_agent.name,
                        amount_y=y1,
                        amount_x=result.amount_x,
                        amm_buys_x=False,
                    ))

            if y2 > MIN_AMOUNT:
                result = amm_norm.execute_buy_x_with_y(y2, timestamp)
                if result is not None:
                    trades.append(RoutedTrade(
                        amm_name=amm_norm.name,
                        amount_y=y2,
                        amount_x=result.amount_x,
                        amm_buys_x=False,
                    ))
        else:
            # Trader wants to sell X, receiving Y
            total_x = order.size / fair_price
            if isinstance(amm_agent, ConstantProductAMM):
                x1, x2 = self.split_sell_two_amms(amm_agent, amm_norm, total_x)
            else:
                x1, x2 = self.solve_sell_split(amm_agent, amm_norm, total_x)

            if x1 > MIN_AMOUNT:
                result = amm_agent.execute_buy_x(x1, timestamp)
                if result is not None:
                    trades.append(RoutedTrade(
                        amm_name=amm_agent.name,
                        amount_y=result.amount_y,
                        amount_x=x1,
                        amm_buys_x=True,
                    ))

            if x2 > MIN_AMOUNT:
                result = amm_norm.execute_buy_x(x2, timestamp)
                if result is not None:
                    trades.append(RoutedTrade(
                        amm_name=amm_norm.name,
                        amount_y=result.amount_y,
                        amount_x=x2,
                        amm_buys_x=True,
                    ))

        return trades

    def route_orders(
        self,
        orders: list[RetailOrder],
        amm_agent: ConstantProductAMM | DepthLadderAMM,
        amm_norm: ConstantProductAMM,
        fair_price: float,
        timestamp: int,
    ) -> list[RoutedTrade]:
        all_trades: list[RoutedTrade] = []
        for order in orders:
            all_trades.extend(
                self.route_order(order, amm_agent, amm_norm, fair_price, timestamp)
            )
        return all_trades

    def solve_buy_split(
        self,
        amm_agent: DepthLadderAMM,
        amm_norm: ConstantProductAMM,
        total_y: float,
    ) -> tuple[float, float]:
        def diff(amount_y_agent: float) -> float:
            return (
                amm_agent.marginal_ask_price_after_y(amount_y_agent)
                - amm_norm.marginal_ask_price_after_y(total_y - amount_y_agent)
            )

        g0 = diff(0.0)
        g1 = diff(total_y)
        if g0 >= 0.0:
            return (0.0, total_y)
        if g1 <= 0.0:
            return (total_y, 0.0)

        lo = 0.0
        hi = total_y
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if diff(mid) < 0.0:
                lo = mid
            else:
                hi = mid
        agent_y = 0.5 * (lo + hi)
        return (agent_y, total_y - agent_y)

    def solve_sell_split(
        self,
        amm_agent: DepthLadderAMM,
        amm_norm: ConstantProductAMM,
        total_x: float,
    ) -> tuple[float, float]:
        def diff(amount_x_agent: float) -> float:
            return (
                amm_agent.marginal_bid_price_after_x(amount_x_agent)
                - amm_norm.marginal_bid_price_after_x(total_x - amount_x_agent)
            )

        g0 = diff(0.0)
        g1 = diff(total_x)
        if g0 <= 0.0:
            return (0.0, total_x)
        if g1 >= 0.0:
            return (total_x, 0.0)

        lo = 0.0
        hi = total_x
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if diff(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        agent_x = 0.5 * (lo + hi)
        return (agent_x, total_x - agent_x)
