from __future__ import annotations

import json
from pathlib import Path

from arena_eval.core.types import TradeInfo
from arena_policies import ReactiveControllerParams, ReactiveControllerStrategy
from scripts.export_strategy import render_strategy_source


def test_reactive_controller_changes_fees_after_trade():
    strategy = ReactiveControllerStrategy(
        ReactiveControllerParams(
            base_fee=0.003,
            flow_to_spread=0.04,
            toxicity_to_side=0.05,
            buy_toxicity_weight=1.5,
        )
    )
    initial = strategy.after_initialize(100.0, 10000.0)
    updated = strategy.after_swap(
        TradeInfo(
            is_buy=True,
            amount_x=1.0,
            amount_y=140.0,
            timestamp=3,
            reserve_x=101.0,
            reserve_y=9860.0,
        )
    )
    assert updated != initial
    assert updated[0] >= initial[0]


def test_export_strategy_renders_constants():
    params = ReactiveControllerParams(base_fee=0.004, buy_toxicity_weight=1.25)
    source = render_strategy_source(params)
    assert "__BASE_FEE__" not in source
    assert "contract Strategy is AMMStrategyBase" in source
    assert str(params.to_wad_dict()["base_fee"]) in source
    assert str(params.to_wad_dict()["buy_toxicity_weight"]) in source
