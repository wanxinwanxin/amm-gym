"""Piecewise side-specific fee controller for the exact simple-AMM challenge."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from arena_eval.core.types import TradeInfo


MAX_FEE = 0.1


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


@dataclass(frozen=True)
class PiecewiseControllerParams:
    """Compact piecewise controller with size buckets and reversal defense."""

    base_fee: float = 0.003
    base_spread: float = 0.0
    signal_decay: float = 0.82
    toxicity_decay: float = 0.9
    small_trade_threshold: float = 0.003
    large_trade_threshold: float = 0.012
    continuation_small: float = 0.002
    continuation_medium: float = 0.004
    continuation_large: float = 0.001
    reversal_small: float = 0.006
    reversal_medium: float = 0.012
    reversal_large: float = 0.02
    continuation_to_same_side: float = 1.0
    continuation_to_cross_side: float = 0.3
    toxicity_to_mid: float = 0.015
    toxicity_to_side: float = 0.03

    def normalized(self) -> "PiecewiseControllerParams":
        small = _clamp(self.small_trade_threshold, 1e-4, 0.05)
        large = _clamp(self.large_trade_threshold, small + 1e-4, 0.1)
        return PiecewiseControllerParams(
            base_fee=_clamp(self.base_fee, 0.0, MAX_FEE),
            base_spread=_clamp(self.base_spread, 0.0, MAX_FEE),
            signal_decay=_clamp(self.signal_decay, 0.0, 0.999),
            toxicity_decay=_clamp(self.toxicity_decay, 0.0, 0.999),
            small_trade_threshold=small,
            large_trade_threshold=large,
            continuation_small=_clamp(self.continuation_small, -0.05, 0.05),
            continuation_medium=_clamp(self.continuation_medium, -0.05, 0.05),
            continuation_large=_clamp(self.continuation_large, -0.05, 0.05),
            reversal_small=_clamp(self.reversal_small, 0.0, 0.08),
            reversal_medium=_clamp(self.reversal_medium, 0.0, 0.08),
            reversal_large=_clamp(self.reversal_large, 0.0, 0.08),
            continuation_to_same_side=_clamp(self.continuation_to_same_side, -4.0, 4.0),
            continuation_to_cross_side=_clamp(self.continuation_to_cross_side, -4.0, 4.0),
            toxicity_to_mid=_clamp(self.toxicity_to_mid, 0.0, 0.5),
            toxicity_to_side=_clamp(self.toxicity_to_side, 0.0, 0.5),
        )

    def to_dict(self) -> dict[str, float]:
        return dict(asdict(self.normalized()))


@dataclass
class PiecewiseControllerState:
    last_timestamp: int = 0
    last_side: int = 0
    bid_signal: float = 0.0
    ask_signal: float = 0.0
    bid_toxicity: float = 0.0
    ask_toxicity: float = 0.0
    initialized: bool = False


class PiecewiseControllerStrategy:
    """React to trade-size buckets and short-horizon reversals."""

    def __init__(self, params: PiecewiseControllerParams | None = None) -> None:
        self.params = (params or PiecewiseControllerParams()).normalized()
        self.state = PiecewiseControllerState()

    def after_initialize(self, initial_x: float, initial_y: float) -> tuple[float, float]:
        self.state = PiecewiseControllerState(initialized=True)
        return self._fees()

    def after_swap(self, trade: TradeInfo) -> tuple[float, float]:
        params = self.params
        state = self.state
        dt = 1 if not state.initialized else max(1, int(trade.timestamp - state.last_timestamp))
        size_ratio = float(trade.amount_y) / max(trade.reserve_y, 1e-9)
        continuation_weight, reversal_weight = self._bucket_weights(size_ratio)
        reversal_scale = 1.0 / float(dt)
        current_side = 1 if trade.is_buy else -1

        state.bid_signal *= params.signal_decay
        state.ask_signal *= params.signal_decay
        state.bid_toxicity *= params.toxicity_decay
        state.ask_toxicity *= params.toxicity_decay

        if current_side == state.last_side:
            if trade.is_buy:
                state.bid_signal += continuation_weight
            else:
                state.ask_signal += continuation_weight
        elif state.last_side != 0:
            if trade.is_buy:
                state.ask_toxicity += reversal_weight * reversal_scale
                state.bid_signal += 0.5 * continuation_weight
            else:
                state.bid_toxicity += reversal_weight * reversal_scale
                state.ask_signal += 0.5 * continuation_weight
        else:
            if trade.is_buy:
                state.bid_signal += 0.5 * continuation_weight
            else:
                state.ask_signal += 0.5 * continuation_weight

        state.last_side = current_side
        state.last_timestamp = int(trade.timestamp)
        state.initialized = True
        return self._fees()

    def _bucket_weights(self, size_ratio: float) -> tuple[float, float]:
        params = self.params
        if size_ratio < params.small_trade_threshold:
            return (params.continuation_small, params.reversal_small)
        if size_ratio < params.large_trade_threshold:
            return (params.continuation_medium, params.reversal_medium)
        return (params.continuation_large, params.reversal_large)

    def _fees(self) -> tuple[float, float]:
        params = self.params
        state = self.state
        toxicity_total = state.bid_toxicity + state.ask_toxicity
        base = params.base_fee + 0.5 * params.base_spread + params.toxicity_to_mid * toxicity_total
        bid_fee = _clamp(
            base
            + params.toxicity_to_side * state.bid_toxicity
            - params.continuation_to_same_side * state.bid_signal
            + params.continuation_to_cross_side * state.ask_signal,
            0.0,
            MAX_FEE,
        )
        ask_fee = _clamp(
            base
            + params.toxicity_to_side * state.ask_toxicity
            - params.continuation_to_same_side * state.ask_signal
            + params.continuation_to_cross_side * state.bid_signal,
            0.0,
            MAX_FEE,
        )
        return (bid_fee, ask_fee)
