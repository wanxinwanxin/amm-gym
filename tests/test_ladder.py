"""Tests for the depth ladder venue."""

import numpy as np
import pytest

from amm_gym.sim.ladder import DepthLadderAMM


def make_ladder() -> DepthLadderAMM:
    amm = DepthLadderAMM(
        name="submission",
        reserve_x=100.0,
        reserve_y=10_000.0,
        band_bps=(2.0, 4.0, 8.0, 16.0),
        base_notional_y=1_000.0,
    )
    amm.configure(
        reference_price=100.0,
        bid_raw=np.array([0.0, 0.0, 0.0]),
        ask_raw=np.array([0.0, 0.0, 0.0]),
    )
    return amm


class TestDepthLadderAMM:
    def test_small_buy_consumes_near_band_only(self):
        amm = make_ladder()
        near_capacity = amm.ask_depth_y[0] * (amm.band_rel[0] - 0.0)
        trade = amm.execute_buy_x_with_y(0.5 * near_capacity, timestamp=0)
        assert trade is not None
        assert amm.ask_consumed_y[0] > 0.0
        assert np.allclose(amm.ask_consumed_y[1:], 0.0)

    def test_large_buy_sweeps_multiple_bands(self):
        amm = make_ladder()
        near = amm.ask_depth_y[0] * (amm.band_rel[0] - 0.0)
        mid = amm.ask_depth_y[1] * (amm.band_rel[1] - amm.band_rel[0])
        trade = amm.execute_buy_x_with_y(near + 0.5 * mid, timestamp=0)
        assert trade is not None
        assert amm.ask_consumed_y[0] > 0.0
        assert amm.ask_consumed_y[1] > 0.0

    def test_marginal_ask_price_is_monotone(self):
        amm = make_ladder()
        p1 = amm.marginal_ask_price_after_y(10.0)
        p2 = amm.marginal_ask_price_after_y(20.0)
        p3 = amm.marginal_ask_price_after_y(40.0)
        assert p1 <= p2 <= p3

    def test_marginal_bid_price_is_monotone(self):
        amm = make_ladder()
        p1 = amm.marginal_bid_price_after_x(0.05)
        p2 = amm.marginal_bid_price_after_x(0.10)
        p3 = amm.marginal_bid_price_after_x(0.20)
        assert p1 >= p2 >= p3

    def test_depths_are_positive_after_configuration(self):
        amm = make_ladder()
        assert np.all(amm.ask_depth_y > 0.0)
        assert np.all(amm.bid_depth_x > 0.0)
