"""Tests for the Nezlobin intra-block directional dynamic-fee strategy."""

from __future__ import annotations

import math

from arena_eval.core.types import IncomingSwap
from arena_eval.exact_simple_amm.nezlobin_dynamic_fee import NezlobinDynamicFeeStrategy


def _swap(s, block, spot, is_buy, size=100.0, x=10_000.0):
    return s.before_swap(IncomingSwap(is_buy=is_buy, size=size, reserve_x=x, reserve_y=x * spot, block=block))


def test_resting_fee_is_symmetric_45bps_at_open():
    s = NezlobinDynamicFeeStrategy()
    s.after_initialize(10_000.0, 1_000_000.0)  # mid 100
    bid, ask = _swap(s, 0, 100.0, True)         # block open, spot == p0 -> no surcharge
    assert math.isclose(bid, 4.5e-4, abs_tol=1e-9)
    assert math.isclose(ask, 4.5e-4, abs_tol=1e-9)


def test_surcharge_taxes_the_reversing_side_by_half_the_move():
    s = NezlobinDynamicFeeStrategy()
    s.after_initialize(10_000.0, 1_000_000.0)
    _swap(s, 0, 100.0, True)                    # p0 = 100
    bid, ask = _swap(s, 0, 101.0, False)        # +1% intra-block -> tax the SELL (bid) side
    assert math.isclose(bid, 4.5e-4 + 0.5 * 0.01, abs_tol=1e-6)   # 4.5bp + 50bp
    assert math.isclose(ask, 4.5e-4, abs_tol=1e-9)               # ask untouched
    bid2, ask2 = _swap(s, 0, 99.0, True)        # -1% intra-block -> tax the BUY (ask) side
    assert math.isclose(ask2, 4.5e-4 + 0.5 * 0.01, abs_tol=1e-6)
    assert math.isclose(bid2, 4.5e-4, abs_tol=1e-9)


def test_skew_up_when_open_above_previous_range():
    s = NezlobinDynamicFeeStrategy()
    s.after_initialize(10_000.0, 1_000_000.0)   # prev_open = 100, range ±4.5bp
    _swap(s, 0, 100.0, True)                     # block 0 establishes prev range around 100
    bid, ask = _swap(s, 1, 101.0, True)          # block 1 opens at 101 (> prev ask) -> big UP
    assert math.isclose(bid, 3e-4, abs_tol=1e-9)   # bid light
    assert math.isclose(ask, 6e-4, abs_tol=1e-9)   # ask heavy


def test_skew_down_when_open_below_previous_range():
    s = NezlobinDynamicFeeStrategy()
    s.after_initialize(10_000.0, 1_000_000.0)
    _swap(s, 0, 100.0, True)
    bid, ask = _swap(s, 1, 99.0, True)           # opens at 99 (< prev bid) -> big DOWN
    assert math.isclose(bid, 6e-4, abs_tol=1e-9)   # bid heavy
    assert math.isclose(ask, 3e-4, abs_tol=1e-9)   # ask light


def test_no_skew_for_small_move_within_range():
    s = NezlobinDynamicFeeStrategy()
    s.after_initialize(10_000.0, 1_000_000.0)
    _swap(s, 0, 100.0, True)
    bid, ask = _swap(s, 1, 100.02, True)         # within ±4.5bp band -> stays symmetric
    assert math.isclose(bid, 4.5e-4, abs_tol=1e-9)
    assert math.isclose(ask, 4.5e-4, abs_tol=1e-9)
