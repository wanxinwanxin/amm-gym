"""Tests for arbitrageur, retail trader, and order router.

Verifies against the closed-form formulas in the Rust reference.
"""

import math
import numpy as np
import pytest

from amm_gym.sim.amm import ConstantProductAMM
from amm_gym.sim.actors import (
    Arbitrageur,
    RetailTrader,
    RetailOrder,
    OrderRouter,
)


def make_amm(name="test", rx=1000.0, ry=1000.0, bid_fee=0.003, ask_fee=0.003):
    return ConstantProductAMM(name, rx, ry, bid_fee, ask_fee)


# ---------------------------------------------------------------------------
# Arbitrageur tests
# ---------------------------------------------------------------------------

class TestArbitrageur:
    def test_no_arb_at_fair_price(self):
        amm = make_amm()
        arb = Arbitrageur()
        # spot = 1.0, fair = 1.0 -> no arb
        result = arb.execute_arb(amm, 1.0, 0)
        assert result is None

    def test_buy_arb_when_underpriced(self):
        """spot < fair -> buy X from AMM (AMM sells X)."""
        amm = make_amm()  # spot = 1.0
        arb = Arbitrageur()
        result = arb.execute_arb(amm, 1.1, 0)  # fair > spot
        assert result is not None
        assert result.side == "sell"  # AMM sells X
        assert result.profit > 0

    def test_sell_arb_when_overpriced(self):
        """spot > fair -> sell X to AMM (AMM buys X)."""
        amm = make_amm()  # spot = 1.0
        arb = Arbitrageur()
        result = arb.execute_arb(amm, 0.9, 0)  # fair < spot
        assert result is not None
        assert result.side == "buy"  # AMM buys X
        assert result.profit > 0

    def test_arb_moves_spot_into_no_arb_band(self):
        """After arb, spot should be within fee band of fair price."""
        amm = make_amm(bid_fee=0.05, ask_fee=0.05)
        arb = Arbitrageur()
        gamma = 0.95

        # Underpriced: buy X
        result = arb.execute_arb(amm, 1.2, 0)
        assert result is not None
        # Post-arb spot should be >= fair * gamma (approximately)
        assert amm.spot_price >= 1.2 * gamma - 0.01

    def test_arb_size_maximizes_profit(self):
        """Optimal size should give higher profit than nearby sizes."""
        rx, ry = 1000.0, 1000.0
        fee = 0.05
        gamma = 1.0 - fee
        k = rx * ry
        fair = 1.2

        # Optimal amount from closed form
        x_opt = rx - math.sqrt(k / (gamma * fair))

        def profit_at(amount_x):
            new_rx = rx - amount_x
            new_ry = k / new_rx
            net_y = new_ry - ry
            total_y = net_y / gamma
            return amount_x * fair - total_y

        p_opt = profit_at(x_opt)
        p_lo = profit_at(x_opt * 0.99)
        p_hi = profit_at(x_opt * 1.01)

        assert p_opt >= p_lo - 1e-9
        assert p_opt >= p_hi - 1e-9

    def test_k_preserved_after_arb(self):
        amm = make_amm(bid_fee=0.05, ask_fee=0.05)
        k_before = amm.k
        arb = Arbitrageur()
        arb.execute_arb(amm, 1.5, 0)
        assert amm.k == pytest.approx(k_before, rel=1e-9)


# ---------------------------------------------------------------------------
# Retail trader tests
# ---------------------------------------------------------------------------

class TestRetailTrader:
    def test_deterministic_with_seed(self):
        t1 = RetailTrader(5.0, 2.0, 0.5, 0.5, seed=42)
        t2 = RetailTrader(5.0, 2.0, 0.5, 0.5, seed=42)
        for _ in range(20):
            o1 = t1.generate_orders()
            o2 = t2.generate_orders()
            assert len(o1) == len(o2)
            for a, b in zip(o1, o2):
                assert a.side == b.side
                assert a.size == b.size

    def test_positive_sizes(self):
        trader = RetailTrader(5.0, 2.0, 0.5, 0.5, seed=42)
        for _ in range(100):
            for order in trader.generate_orders():
                assert order.size > 0

    def test_mean_arrival_rate(self):
        trader = RetailTrader(5.0, 2.0, 0.5, 0.5, seed=42)
        counts = [len(trader.generate_orders()) for _ in range(10_000)]
        assert np.mean(counts) == pytest.approx(5.0, rel=0.05)

    def test_mean_size(self):
        trader = RetailTrader(10.0, 2.0, 0.7, 0.5, seed=42)
        sizes = []
        for _ in range(5000):
            for order in trader.generate_orders():
                sizes.append(order.size)
        # E[X] = mean_size for log-normal with our parameterization
        assert np.mean(sizes) == pytest.approx(2.0, rel=0.05)

    def test_zero_arrival_rate_disables_retail_flow(self):
        trader = RetailTrader(0.0, 2.0, 0.7, 0.5, seed=42)
        for _ in range(100):
            assert trader.generate_orders() == []


# ---------------------------------------------------------------------------
# Router tests
# ---------------------------------------------------------------------------

class TestOrderRouter:
    def test_equal_amms_equal_split(self):
        """With identical AMMs and fees, split should be ~50/50."""
        router = OrderRouter()
        amm1 = make_amm("a", 1000, 1000, 0.003, 0.003)
        amm2 = make_amm("b", 1000, 1000, 0.003, 0.003)

        y1, y2 = router.split_buy_two_amms(amm1, amm2, 100.0)
        assert y1 == pytest.approx(50.0, rel=0.02)
        assert y2 == pytest.approx(50.0, rel=0.02)

    def test_lower_fee_gets_more_flow(self):
        """AMM with lower fee should receive more flow."""
        router = OrderRouter()
        amm_low = make_amm("low", 1000, 1000, 0.001, 0.001)
        amm_high = make_amm("high", 1000, 1000, 0.01, 0.01)

        y_low, y_high = router.split_buy_two_amms(amm_low, amm_high, 100.0)
        assert y_low > y_high

    def test_sell_split_symmetry(self):
        """Sell split should also be ~50/50 for identical AMMs."""
        router = OrderRouter()
        amm1 = make_amm("a", 1000, 1000, 0.003, 0.003)
        amm2 = make_amm("b", 1000, 1000, 0.003, 0.003)

        x1, x2 = router.split_sell_two_amms(amm1, amm2, 100.0)
        assert x1 == pytest.approx(50.0, rel=0.02)

    def test_route_buy_order(self):
        router = OrderRouter()
        amm1 = make_amm("agent", 1000, 1000)
        amm2 = make_amm("norm", 1000, 1000)
        order = RetailOrder(side="buy", size=10.0)

        trades = router.route_order(order, amm1, amm2, 1.0, 0)
        assert len(trades) == 2
        total_y = sum(t.amount_y for t in trades)
        assert total_y == pytest.approx(10.0, rel=0.01)

    def test_route_sell_order(self):
        router = OrderRouter()
        amm1 = make_amm("agent", 1000, 1000)
        amm2 = make_amm("norm", 1000, 1000)
        order = RetailOrder(side="sell", size=10.0)

        trades = router.route_order(order, amm1, amm2, 1.0, 0)
        assert len(trades) == 2
        for t in trades:
            assert t.amm_buys_x is True
