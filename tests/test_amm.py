"""Tests for constant-product AMM math.

Verifies swap formulas, fee collection, and k invariant against
the Rust reference implementation in amm_sim_rs/src/amm/cfmm.rs.
"""

import math
import pytest
from amm_gym.sim.amm import ConstantProductAMM


def make_amm(rx=1000.0, ry=1000.0, bid_fee=0.003, ask_fee=0.003):
    return ConstantProductAMM("test", rx, ry, bid_fee, ask_fee)


class TestQuoteBuyX:
    """AMM buys X (trader sells X for Y). Fee-on-input with bid_fee."""

    def test_basic_swap(self):
        amm = make_amm()
        y_out, fee = amm.quote_buy_x(10.0)
        # Manual: gamma=0.997, net_x=9.97, new_rx=1009.97
        # new_ry = 1e6/1009.97 ≈ 990.13, y_out ≈ 9.87
        assert y_out > 0
        assert fee == pytest.approx(10.0 * 0.003)

    def test_manual_calculation(self):
        amm = make_amm(rx=1000.0, ry=1000.0, bid_fee=0.0025)
        amount_x = 10.0
        fee = 0.0025
        gamma = 1.0 - fee
        net_x = amount_x * gamma
        k = 1000.0 * 1000.0
        new_rx = 1000.0 + net_x
        expected_y = 1000.0 - k / new_rx

        y_out, fee_amt = amm.quote_buy_x(amount_x)
        assert y_out == pytest.approx(expected_y, rel=1e-10)
        assert fee_amt == pytest.approx(amount_x * fee, rel=1e-10)

    def test_zero_amount(self):
        amm = make_amm()
        assert amm.quote_buy_x(0.0) == (0.0, 0.0)
        assert amm.quote_buy_x(-1.0) == (0.0, 0.0)


class TestQuoteSellX:
    """AMM sells X (trader buys X with Y). Fee-on-input with ask_fee."""

    def test_basic_swap(self):
        amm = make_amm()
        total_y, fee = amm.quote_sell_x(10.0)
        assert total_y > 0
        assert fee > 0

    def test_manual_calculation(self):
        amm = make_amm(rx=1000.0, ry=1000.0, ask_fee=0.0025)
        amount_x = 10.0
        fee = 0.0025
        gamma = 1.0 - fee
        k = 1000.0 * 1000.0
        new_rx = 1000.0 - amount_x
        new_ry = k / new_rx
        net_y = new_ry - 1000.0
        expected_total = net_y / gamma

        total_y, fee_amt = amm.quote_sell_x(amount_x)
        assert total_y == pytest.approx(expected_total, rel=1e-10)
        assert fee_amt == pytest.approx(expected_total - net_y, rel=1e-10)

    def test_cant_sell_all_reserves(self):
        amm = make_amm()
        assert amm.quote_sell_x(1000.0) == (0.0, 0.0)
        assert amm.quote_sell_x(1001.0) == (0.0, 0.0)


class TestQuoteXForY:
    """Trader pays Y to receive X."""

    def test_basic_swap(self):
        amm = make_amm()
        x_out, fee = amm.quote_x_for_y(10.0)
        assert x_out > 0
        assert fee == pytest.approx(10.0 * 0.003)

    def test_manual_calculation(self):
        amm = make_amm(rx=1000.0, ry=1000.0, ask_fee=0.005)
        amount_y = 50.0
        fee = 0.005
        gamma = 1.0 - fee
        net_y = amount_y * gamma
        k = 1e6
        new_ry = 1000.0 + net_y
        expected_x = 1000.0 - k / new_ry

        x_out, fee_amt = amm.quote_x_for_y(amount_y)
        assert x_out == pytest.approx(expected_x, rel=1e-10)
        assert fee_amt == pytest.approx(amount_y * fee, rel=1e-10)


class TestExecuteTrades:
    """Test trade execution updates reserves and fees correctly."""

    def test_buy_x_reserves(self):
        amm = make_amm(bid_fee=0.003)
        k_before = amm.k
        result = amm.execute_buy_x(10.0, timestamp=0)
        assert result is not None
        assert result.is_buy is True
        # k should be preserved (fees collected separately)
        assert amm.k == pytest.approx(k_before, rel=1e-9)
        # Fees collected in X
        assert amm.accumulated_fees_x == pytest.approx(10.0 * 0.003, rel=1e-10)
        assert amm.accumulated_fees_y == 0.0

    def test_sell_x_reserves(self):
        amm = make_amm(ask_fee=0.003)
        k_before = amm.k
        result = amm.execute_sell_x(10.0, timestamp=0)
        assert result is not None
        assert result.is_buy is False
        # k preserved
        assert amm.k == pytest.approx(k_before, rel=1e-9)
        # Fees collected in Y
        assert amm.accumulated_fees_x == 0.0
        assert amm.accumulated_fees_y > 0

    def test_buy_x_with_y_reserves(self):
        amm = make_amm(ask_fee=0.003)
        k_before = amm.k
        result = amm.execute_buy_x_with_y(50.0, timestamp=0)
        assert result is not None
        assert result.is_buy is False
        assert amm.k == pytest.approx(k_before, rel=1e-9)
        assert amm.accumulated_fees_y == pytest.approx(50.0 * 0.003, rel=1e-10)

    def test_spot_price_moves_correctly(self):
        amm = make_amm()
        initial_spot = amm.spot_price
        # Buying X from AMM -> X reserves drop -> price (Y/X) increases
        amm.execute_sell_x(50.0, timestamp=0)
        assert amm.spot_price > initial_spot

        amm2 = make_amm()
        # Selling X to AMM -> X reserves rise -> price (Y/X) decreases
        amm2.execute_buy_x(50.0, timestamp=0)
        assert amm2.spot_price < initial_spot


class TestKInvariant:
    """Verify k stays constant through multiple trades."""

    def test_many_trades(self):
        amm = make_amm(rx=1000.0, ry=1000.0, bid_fee=0.05, ask_fee=0.05)
        k_initial = amm.k

        for i in range(100):
            if i % 2 == 0:
                amm.execute_buy_x(5.0, timestamp=i)
            else:
                amm.execute_sell_x(min(5.0, amm.reserve_x * 0.9), timestamp=i)

        assert amm.k == pytest.approx(k_initial, rel=1e-6)
