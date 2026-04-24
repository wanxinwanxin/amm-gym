from __future__ import annotations

from arena_eval.core.types import TradeInfo
from arena_policies import InventoryToxicityParams, InventoryToxicityStrategy


def test_inventory_toxicity_widens_after_quick_reversal():
    strategy = InventoryToxicityStrategy(
        InventoryToxicityParams(
            base_fee=0.003,
            reverse_reaction_weight=0.15,
            toxicity_to_side=0.08,
            same_side_toxicity_weight=0.0,
        )
    )
    initial_bid, initial_ask = strategy.after_initialize(100.0, 10_000.0)
    post_buy_bid, post_buy_ask = strategy.after_swap(
        TradeInfo(
            is_buy=True,
            amount_x=1.0,
            amount_y=120.0,
            timestamp=1,
            reserve_x=99.0,
            reserve_y=10_120.0,
        )
    )
    post_reversal_bid, post_reversal_ask = strategy.after_swap(
        TradeInfo(
            is_buy=False,
            amount_x=1.0,
            amount_y=121.0,
            timestamp=2,
            reserve_x=100.0,
            reserve_y=9_999.0,
        )
    )
    assert post_buy_ask >= initial_ask
    assert post_reversal_ask > initial_ask
    assert post_reversal_bid >= initial_bid
