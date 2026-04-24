"""Strategy helpers for the exact simple AMM evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from arena_eval.core.types import TradeInfo


class ExactSimpleAMMStrategy(Protocol):
    """Strategy interface mirroring the challenge contract callbacks."""

    def after_initialize(self, initial_x: float, initial_y: float) -> tuple[float, float]:
        ...

    def after_swap(self, trade: TradeInfo) -> tuple[float, float]:
        ...


@dataclass
class FixedFeeStrategy:
    """Simple fixed-fee strategy useful for baselines and tests."""

    bid_fee: float = 0.003
    ask_fee: float = 0.003

    def after_initialize(self, initial_x: float, initial_y: float) -> tuple[float, float]:
        return (self.bid_fee, self.ask_fee)

    def after_swap(self, trade: TradeInfo) -> tuple[float, float]:
        return (self.bid_fee, self.ask_fee)
