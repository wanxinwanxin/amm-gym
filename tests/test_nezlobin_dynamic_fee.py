"""Tests for the Nezlobin / Volatile-Hook dynamic-fee strategy (mainnet doc spec)."""

from __future__ import annotations

import math

from arena_eval.core.types import IncomingSwap
from arena_eval.exact_simple_amm.nezlobin_dynamic_fee import NezlobinDynamicFeeStrategy


def _swap(s, block, spot, is_buy=True, size=100.0, x=10_000.0):
    return s.before_swap(IncomingSwap(is_buy=is_buy, size=size, reserve_x=x, reserve_y=x * spot, block=block))


def test_resting_symmetric_at_open():
    s = NezlobinDynamicFeeStrategy()
    s.after_initialize(10_000.0, 1_000_000.0)        # mid 100
    bid, ask = _swap(s, 0, 100.0)                     # first swap of block -> resting, no surcharge
    assert math.isclose(bid, 4.5e-4, abs_tol=1e-9)
    assert math.isclose(ask, 4.5e-4, abs_tol=1e-9)


def test_intra_block_surcharge_rose():
    s = NezlobinDynamicFeeStrategy()
    s.after_initialize(10_000.0, 1_000_000.0)
    _swap(s, 0, 100.0)                                # p0 = 100
    bid, ask = _swap(s, 0, 101.0)                     # +1% intra-block, n>1
    m = 1.0 / 101.0
    assert math.isclose(bid, 4.5e-4 + 0.5 * m, abs_tol=1e-7)            # reverting (bid) += half the move
    assert math.isclose(ask, 4.5e-4 + min(0.25 * m, 2e-4), abs_tol=1e-7)  # continuation (ask) += min(alpha*m, d)


def test_intra_block_surcharge_fell_is_mirror():
    s = NezlobinDynamicFeeStrategy()
    s.after_initialize(10_000.0, 1_000_000.0)
    _swap(s, 0, 100.0)
    bid, ask = _swap(s, 0, 99.0)                      # -1% intra-block
    m = 1.0 / 99.0
    assert math.isclose(ask, 4.5e-4 + 0.5 * m, abs_tol=1e-7)           # reverting is now the ask
    assert math.isclose(bid, 4.5e-4 + min(0.25 * m, 2e-4), abs_tol=1e-7)


def test_continuation_side_is_capped_at_d():
    s = NezlobinDynamicFeeStrategy()
    s.after_initialize(10_000.0, 1_000_000.0)
    _swap(s, 0, 100.0)
    bid, ask = _swap(s, 0, 102.0)                     # +2% move: alpha*m >> d -> continuation capped at d=2bps
    assert math.isclose(ask, 4.5e-4 + 2e-4, abs_tol=1e-7)             # capped at d, not alpha*m
    assert bid > 4.5e-4 + 0.5 * 0.019                                 # reverting still gets ~half the move


def test_first_swap_of_block_is_resting():
    s = NezlobinDynamicFeeStrategy()
    s.after_initialize(10_000.0, 1_000_000.0)
    _swap(s, 0, 100.0)
    _swap(s, 0, 101.0)                                # surcharge active within block 0
    bid, ask = _swap(s, 1, 100.02)                    # new block, small move within range -> resting again
    assert bid < 9e-4 and ask < 9e-4                  # not surcharged
    assert math.isclose(bid + ask, 9e-4, abs_tol=1e-6)  # sum = TS


def test_ema_skew_leans_toward_recent_up_impact():
    s = NezlobinDynamicFeeStrategy()
    s.after_initialize(10_000.0, 1_000_000.0)
    _swap(s, 0, 100.0)
    bid, ask = _swap(s, 1, 100.04)                    # +4bps block move (< cutoff) -> skew ask up, bid down
    assert ask > 4.5e-4 > bid
    assert math.isclose(bid + ask, 9e-4, abs_tol=1e-9)  # still sums to TS


def test_big_pi_exception_widens_then_bypasses():
    s = NezlobinDynamicFeeStrategy(beta=0.0)          # beta=0 isolates the exception from the skew
    s.after_initialize(10_000.0, 1_000_000.0)
    _swap(s, 0, 100.0)
    bid1, ask1 = _swap(s, 1, 102.0)                   # +2% (200bps) > 10bps cutoff -> widen
    pi = 0.02
    assert math.isclose(bid1 + ask1, 9e-4 + 0.5 * pi, abs_tol=1e-6)   # sum = TS + 1/2 PI
    assert bid1 >= 0.5 * pi - 1e-9                                    # reverting (bid) carries >= 1/2 PI
    bid2, ask2 = _swap(s, 2, 104.0)                   # another big move, but bypassed this block
    assert math.isclose(bid2 + ask2, 9e-4, abs_tol=1e-6)             # back to TS
