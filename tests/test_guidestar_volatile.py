"""Tests for the Guidestar volatile-pair dynamic-fee port.

Checks the invariants stated in the spec (guidestar-hooks docs/VolatilePair.md
"Algorithm invariants") plus the qualitative dynamics validated against the
contract: rest fee = feeInit, permanent floor, hard cap, decay-with-time, and the
counter-intuitive "the opposite side of the book spikes after a directional move."
"""

from __future__ import annotations

import math

from arena_eval.core.types import IncomingSwap
from arena_eval.exact_simple_amm.guidestar_volatile import GuidestarVolatileStrategy

_M = 1_000_000.0


def _make(**kw) -> GuidestarVolatileStrategy:
    s = GuidestarVolatileStrategy(**kw)
    s.after_initialize(1.0, 100.0)  # initial spot = 100
    return s


def _quote(s: GuidestarVolatileStrategy, block: int, spot: float, is_buy: bool) -> tuple[float, float]:
    return s.before_swap(IncomingSwap(is_buy=is_buy, size=None, reserve_x=1.0, reserve_y=spot, block=block))


def test_rest_fee_equals_feeinit():
    s = _make(fee_init_bps=3.0)
    bid, ask = _quote(s, 0, 100.0, True)  # no price move
    assert math.isclose(bid, 0.0003, abs_tol=1e-9)
    assert math.isclose(ask, 0.0003, abs_tol=1e-9)


def test_returns_are_fractions_in_range():
    s = _make(fee_init_bps=5.0)
    spot = 100.0
    moves = [0.0, 0.02, -0.03, 0.015, 0.0, -0.05, 0.04]
    for blk, m in enumerate(moves):
        spot *= (1.0 + m)
        bid, ask = _quote(s, blk, spot, m >= 0)
        assert 0.0 <= bid <= 0.99 and 0.0 <= ask <= 0.99


def test_permanent_floor_and_cap_hold_over_a_path():
    s = _make(fee_init_bps=3.0, alpha=20000.0)  # aggressive alpha to stress the cap
    floor = 2.0 * s._feeInit
    spot = 100.0
    moves = [0.04, 0.03, -0.08, 0.0, 0.06, -0.02, 0.10, 0.0, 0.0, -0.07, 0.05]
    for blk, m in enumerate(moves):
        spot *= (1.0 + m)
        _quote(s, blk, spot, m >= 0)
        assert s._buyPerm + s._sellPerm >= floor - 1e-6           # permanent floor
        assert s._buy_fee() <= 990_000.0 + 1e-6                   # hard cap (millionths)
        assert s._sell_fee() <= 990_000.0 + 1e-6


def test_opposite_side_spikes_after_a_buy():
    s = _make(fee_init_bps=5.0, alpha=8000.0)
    _quote(s, 0, 100.0, True)                  # settle at the floor
    bid, ask = _quote(s, 1, 102.0, True)       # +2% buy shock
    # After a buy (price up), the SELL fee (bid) is held high to stop the reversal,
    # while the BUY fee (ask) only nudges up via the permanent skew.
    assert bid > ask
    assert bid > 0.0050                         # the bid spike is large (>50 bps)


def test_opposite_side_spikes_after_a_sell():
    s = _make(fee_init_bps=5.0, alpha=8000.0)
    _quote(s, 0, 100.0, False)
    bid, ask = _quote(s, 1, 97.5, False)       # -2.5% sell shock
    assert ask > bid                            # mirror: the ASK spikes after a sell
    assert ask > 0.0050


def test_transitory_decays_toward_floor_in_quiet_blocks():
    s = _make(fee_init_bps=3.0, alpha=8000.0, trans_decline=0.5, k_perm=0.7)
    _quote(s, 0, 100.0, True)
    _quote(s, 1, 103.0, True)                  # shock at block 1 (top-of-block now 103)
    floor = 2.0 * s._feeInit
    excess = []
    for blk in range(2, 18):
        _quote(s, blk, 103.0, True)            # quiet: spot unchanged
        excess.append((s._buy_fee() + s._sell_fee()) - floor)
    assert excess[0] > excess[-1]              # monotone-ish decline
    assert excess[-1] < 0.25 * excess[0]       # decays substantially toward the floor
    assert excess[-1] >= -1e-6                  # never below floor


def test_no_swap_blocks_do_not_update_state():
    # Two calls in the SAME block: the 2nd must not re-run the heavy update.
    s = _make(fee_init_bps=5.0, alpha=8000.0)
    _quote(s, 5, 100.0, True)
    _quote(s, 6, 104.0, True)                  # heavy update at block 6
    snap = (s._buyPerm, s._buyTrans, s._sellPerm, s._sellTrans, s._top)
    _quote(s, 6, 106.0, True)                  # same block -> sandwich branch only
    assert (s._buyPerm, s._buyTrans, s._sellPerm, s._sellTrans, s._top) == snap
