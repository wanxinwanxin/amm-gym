from __future__ import annotations

from arena_eval.core.types import TradeInfo
from arena_policies import PiecewiseControllerParams, PiecewiseControllerStrategy


def test_piecewise_controller_widens_after_fast_reversal():
    strategy = PiecewiseControllerStrategy(
        PiecewiseControllerParams(
            continuation_medium=0.003,
            reversal_medium=0.02,
            continuation_to_same_side=1.0,
            toxicity_to_side=0.05,
        )
    )
    initial = strategy.after_initialize(100.0, 10000.0)
    after_bid = strategy.after_swap(
        TradeInfo(
            is_buy=True,
            amount_x=1.0,
            amount_y=60.0,
            timestamp=1,
            reserve_x=101.0,
            reserve_y=9940.0,
        )
    )
    after_reversal = strategy.after_swap(
        TradeInfo(
            is_buy=False,
            amount_x=0.6,
            amount_y=61.0,
            timestamp=2,
            reserve_x=100.4,
            reserve_y=10001.0,
        )
    )
    assert after_bid != initial
    assert after_reversal[0] > after_bid[0]
