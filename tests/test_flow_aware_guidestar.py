"""Smoke + behavior tests for the §11 flow-aware Guidestar prototypes.

These keep Guidestar's volatility machinery and blend its quote with a quiet flat
floor by a controller weight; the tests check the interface (valid fee fractions)
and the qualitative controller behavior the §11 design relies on.
"""

from __future__ import annotations

from arena_eval.core.types import IncomingSwap
from arena_eval.exact_simple_amm.flow_aware_guidestar import (
    FlowAwareGuidestarStrategy,
    SizeAwareGuidestarStrategy,
    UnifiedGuidestarStrategy,
)

_STRATS = [FlowAwareGuidestarStrategy, SizeAwareGuidestarStrategy, UnifiedGuidestarStrategy]


def _quote(s, block, spot, is_buy, size):
    return s.before_swap(IncomingSwap(is_buy=is_buy, size=size, reserve_x=1.0, reserve_y=spot, block=block))


def test_interface_returns_valid_fractions():
    for cls in _STRATS:
        s = cls()
        bid, ask = s.after_initialize(1.0, 100.0)
        assert 0.0 <= bid <= 0.99 and 0.0 <= ask <= 0.99
        spot = 100.0
        for blk, mv in enumerate([0.0, 0.02, -0.03, 0.0, 0.05]):
            spot *= 1.0 + mv
            for size in (None, 50.0, 50_000.0):
                bid, ask = _quote(s, blk, spot, mv >= 0, size)
                assert 0.0 <= bid <= 0.99 and 0.0 <= ask <= 0.99, (cls.__name__, size)


def test_sizeaware_small_order_is_near_floor_even_after_a_shock():
    # A small order is quoted ~the flat floor regardless of a fresh price spike,
    # while the arb leg (size None) gets the full defensive quote.
    s = SizeAwareGuidestarStrategy(cheap_fee_bps=3.5)
    s.after_initialize(1.0, 100.0)
    _quote(s, 0, 100.0, True, None)
    _quote(s, 1, 103.0, True, None)               # +3% shock on the arb leg (defended)
    bid_small, ask_small = _quote(s, 1, 103.0, True, 100.0)   # tiny order, same block
    assert max(bid_small, ask_small) < 0.0010      # < 10 bps: spike not applied to small order
    bid_big, ask_big = _quote(s, 1, 103.0, True, 100_000.0)   # large order, same block
    assert max(bid_big, ask_big) > max(bid_small, ask_small)  # large order is defended more


def test_unified_arb_defense_is_flow_gated():
    # Busy regime -> arb defense relaxed; sparse regime -> arb defended.
    busy = UnifiedGuidestarStrategy()
    busy.after_initialize(1.0, 100.0)
    for b in range(150):
        busy.before_swap(IncomingSwap(is_buy=True, size=None, reserve_x=1.0, reserve_y=100.0, block=b))
        for _ in range(8):
            busy.before_swap(IncomingSwap(is_buy=True, size=80.0, reserve_x=1.0, reserve_y=100.0, block=b))
    sparse = UnifiedGuidestarStrategy()
    sparse.after_initialize(1.0, 100.0)
    for b in range(150):
        sparse.before_swap(IncomingSwap(is_buy=True, size=None, reserve_x=1.0, reserve_y=100.0, block=b))
    assert busy._arb_d < 0.3        # heavy retail -> let the arb realign cheaply
    assert sparse._arb_d > 0.9      # sparse -> defend the arb


def test_flowaware_defensiveness_tracks_regime():
    s = FlowAwareGuidestarStrategy()
    s.after_initialize(1.0, 100.0)
    # frequent + small -> low defensiveness
    for b in range(200):
        s.before_swap(IncomingSwap(is_buy=True, size=None, reserve_x=1.0, reserve_y=100.0, block=b))
        for _ in range(3):
            s.before_swap(IncomingSwap(is_buy=True, size=50.0, reserve_x=1.0, reserve_y=100.0, block=b))
    assert s.defensiveness() < 0.2
