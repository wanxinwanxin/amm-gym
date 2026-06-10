"""Strategy helpers for the exact simple AMM evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from arena_eval.core.types import IncomingSwap, TradeInfo


@runtime_checkable
class ExactSimpleAMMStrategy(Protocol):
    """Strategy interface mirroring the challenge contract callbacks.

    ``after_initialize`` and ``after_swap`` are the original *post*-event hooks.
    ``before_swap`` is the *pre*-swap hook (the v4 ``beforeSwap`` analog): the
    simulator calls it right before each swap — the arb and each routed retail
    order — so a strategy can price the swap from the pool's current state and
    the incoming swap's direction/size before the fee feeds the routing split and
    execution. Strategies that only define the post hooks keep working unchanged;
    the simulator no-ops ``before_swap`` when a strategy does not implement it.
    """

    def after_initialize(self, initial_x: float, initial_y: float) -> tuple[float, float]:
        ...

    def after_swap(self, trade: TradeInfo) -> tuple[float, float]:
        ...

    def before_swap(self, incoming: IncomingSwap) -> tuple[float, float]:
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

    def before_swap(self, incoming: IncomingSwap) -> tuple[float, float]:
        return (self.bid_fee, self.ask_fee)
