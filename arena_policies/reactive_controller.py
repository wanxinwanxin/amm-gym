"""Reactive fee controller designed to map cleanly into Solidity."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from arena_eval.core.types import TradeInfo


MAX_FEE = 0.1
WAD = 10**18


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


@dataclass(frozen=True)
class ReactiveControllerParams:
    """Compact reactive controller parameterization.

    All parameters are continuous floats so they can be optimized locally and
    then frozen into constants for the submission contract.
    """

    base_fee: float = 0.003
    base_spread: float = 0.0
    flow_decay: float = 0.85
    size_decay: float = 0.85
    gap_decay: float = 0.75
    toxicity_decay: float = 0.9
    size_weight: float = 0.04
    gap_weight: float = 0.002
    gap_target: float = 1.5
    flow_to_mid: float = 0.01
    flow_to_spread: float = 0.02
    flow_to_skew: float = 0.02
    toxicity_to_mid: float = 0.02
    toxicity_to_side: float = 0.03
    buy_toxicity_weight: float = 1.0
    sell_toxicity_weight: float = 1.0

    def normalized(self) -> "ReactiveControllerParams":
        return ReactiveControllerParams(
            base_fee=_clamp(self.base_fee, 0.0, MAX_FEE),
            base_spread=_clamp(self.base_spread, 0.0, MAX_FEE),
            flow_decay=_clamp(self.flow_decay, 0.0, 0.999),
            size_decay=_clamp(self.size_decay, 0.0, 0.999),
            gap_decay=_clamp(self.gap_decay, 0.0, 0.999),
            toxicity_decay=_clamp(self.toxicity_decay, 0.0, 0.999),
            size_weight=_clamp(self.size_weight, -1.0, 1.0),
            gap_weight=_clamp(self.gap_weight, -0.05, 0.05),
            gap_target=_clamp(self.gap_target, 0.0, 20.0),
            flow_to_mid=_clamp(self.flow_to_mid, -0.5, 0.5),
            flow_to_spread=_clamp(self.flow_to_spread, 0.0, 0.5),
            flow_to_skew=_clamp(self.flow_to_skew, -0.5, 0.5),
            toxicity_to_mid=_clamp(self.toxicity_to_mid, 0.0, 0.5),
            toxicity_to_side=_clamp(self.toxicity_to_side, 0.0, 0.5),
            buy_toxicity_weight=_clamp(self.buy_toxicity_weight, 0.0, 5.0),
            sell_toxicity_weight=_clamp(self.sell_toxicity_weight, 0.0, 5.0),
        )

    def to_dict(self) -> dict[str, float]:
        return dict(asdict(self.normalized()))

    def to_wad_dict(self) -> dict[str, int]:
        values: dict[str, int] = {}
        for key, value in self.to_dict().items():
            if key == "gap_target":
                values[key] = int(round(value * WAD))
            else:
                values[key] = int(round(value * WAD))
        return values


@dataclass
class ReactiveControllerState:
    """Minimal event-driven state that can be represented in Solidity slots."""

    last_timestamp: int = 0
    buy_flow: float = 0.0
    sell_flow: float = 0.0
    size_ema: float = 0.0
    gap_ema: float = 1.0
    toxicity_bid: float = 0.0
    toxicity_ask: float = 0.0
    initialized: bool = False


class ReactiveControllerStrategy:
    """Trade-reactive policy that updates only after executed fills."""

    def __init__(self, params: ReactiveControllerParams | None = None) -> None:
        self.params = (params or ReactiveControllerParams()).normalized()
        self.state = ReactiveControllerState()

    def after_initialize(self, initial_x: float, initial_y: float) -> tuple[float, float]:
        self.state = ReactiveControllerState(initialized=True)
        return self._fees()

    def after_swap(self, trade: TradeInfo) -> tuple[float, float]:
        params = self.params
        state = self.state
        dt = 1 if not state.initialized else max(1, int(trade.timestamp - state.last_timestamp))
        reserve_y = max(trade.reserve_y, 1e-9)
        size_ratio = float(trade.amount_y) / reserve_y
        state.buy_flow *= params.flow_decay
        state.sell_flow *= params.flow_decay
        if trade.is_buy:
            state.buy_flow += (1.0 - params.flow_decay) * size_ratio
        else:
            state.sell_flow += (1.0 - params.flow_decay) * size_ratio
        state.size_ema = params.size_decay * state.size_ema + (1.0 - params.size_decay) * size_ratio
        state.gap_ema = params.gap_decay * state.gap_ema + (1.0 - params.gap_decay) * float(dt)
        state.toxicity_bid *= params.toxicity_decay
        state.toxicity_ask *= params.toxicity_decay
        if trade.is_buy:
            state.toxicity_bid += params.buy_toxicity_weight * size_ratio
        else:
            state.toxicity_ask += params.sell_toxicity_weight * size_ratio
        state.last_timestamp = int(trade.timestamp)
        state.initialized = True
        return self._fees()

    def _fees(self) -> tuple[float, float]:
        params = self.params
        state = self.state
        gap_pressure = max(0.0, params.gap_target - state.gap_ema)
        flow_total = state.buy_flow + state.sell_flow
        flow_skew = state.buy_flow - state.sell_flow
        mid = (
            params.base_fee
            + params.size_weight * state.size_ema
            + params.gap_weight * gap_pressure
            + params.flow_to_mid * flow_total
            + params.toxicity_to_mid * (state.toxicity_bid + state.toxicity_ask)
        )
        spread = (
            params.base_spread
            + params.flow_to_spread * flow_total
            + params.toxicity_to_side * (state.toxicity_bid + state.toxicity_ask)
        )
        skew = params.flow_to_skew * flow_skew
        bid_fee = _clamp(mid + 0.5 * spread + state.toxicity_bid * params.toxicity_to_side - skew, 0.0, MAX_FEE)
        ask_fee = _clamp(mid + 0.5 * spread + state.toxicity_ask * params.toxicity_to_side + skew, 0.0, MAX_FEE)
        return (bid_fee, ask_fee)
