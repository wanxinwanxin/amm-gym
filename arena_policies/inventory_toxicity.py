"""Inventory-aware fee controller with explicit reverse-arb defense."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from arena_eval.core.types import TradeInfo


MAX_FEE = 0.1


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


@dataclass(frozen=True)
class InventoryToxicityParams:
    """Compact controller that balances inventory skew and toxic reversal risk."""

    base_fee: float = 0.003
    base_spread: float = 0.001
    inventory_weight: float = 0.02
    inventory_target: float = 0.0
    flow_decay: float = 0.9
    flow_skew_weight: float = 0.01
    toxicity_decay: float = 0.9
    same_side_toxicity_weight: float = 0.03
    reverse_reaction_weight: float = 0.08
    reversal_time_scale: float = 3.0
    toxicity_to_mid: float = 0.01
    toxicity_to_spread: float = 0.015
    toxicity_to_side: float = 0.03
    min_fee: float = 0.0001
    max_fee: float = 0.02

    def normalized(self) -> "InventoryToxicityParams":
        min_fee = _clamp(self.min_fee, 0.0, MAX_FEE)
        max_fee = _clamp(self.max_fee, min_fee, MAX_FEE)
        return InventoryToxicityParams(
            base_fee=_clamp(self.base_fee, min_fee, max_fee),
            base_spread=_clamp(self.base_spread, 0.0, max_fee),
            inventory_weight=_clamp(self.inventory_weight, -0.5, 0.5),
            inventory_target=_clamp(self.inventory_target, -1.0, 1.0),
            flow_decay=_clamp(self.flow_decay, 0.0, 0.999),
            flow_skew_weight=_clamp(self.flow_skew_weight, -0.2, 0.2),
            toxicity_decay=_clamp(self.toxicity_decay, 0.0, 0.999),
            same_side_toxicity_weight=_clamp(self.same_side_toxicity_weight, 0.0, 0.3),
            reverse_reaction_weight=_clamp(self.reverse_reaction_weight, 0.0, 0.5),
            reversal_time_scale=_clamp(self.reversal_time_scale, 1.0, 20.0),
            toxicity_to_mid=_clamp(self.toxicity_to_mid, 0.0, 0.3),
            toxicity_to_spread=_clamp(self.toxicity_to_spread, 0.0, 0.3),
            toxicity_to_side=_clamp(self.toxicity_to_side, 0.0, 0.5),
            min_fee=min_fee,
            max_fee=max_fee,
        )

    def to_dict(self) -> dict[str, float]:
        return dict(asdict(self.normalized()))


@dataclass
class InventoryToxicityState:
    initial_x: float = 0.0
    initial_y: float = 0.0
    buy_flow: float = 0.0
    sell_flow: float = 0.0
    bid_toxicity: float = 0.0
    ask_toxicity: float = 0.0
    last_timestamp: int = 0
    last_is_buy: bool | None = None
    last_size_ratio: float = 0.0
    initialized: bool = False


class InventoryToxicityStrategy:
    """Quote controller that skews for inventory and widens after toxic reversals."""

    def __init__(self, params: InventoryToxicityParams | None = None) -> None:
        self.params = (params or InventoryToxicityParams()).normalized()
        self.state = InventoryToxicityState()

    def after_initialize(self, initial_x: float, initial_y: float) -> tuple[float, float]:
        self.state = InventoryToxicityState(
            initial_x=float(initial_x),
            initial_y=float(initial_y),
            initialized=True,
        )
        return self._fees(initial_x=initial_x, initial_y=initial_y)

    def after_swap(self, trade: TradeInfo) -> tuple[float, float]:
        params = self.params
        state = self.state
        dt = 1 if not state.initialized else max(1, int(trade.timestamp - state.last_timestamp))
        reserve_y = max(float(trade.reserve_y), 1e-9)
        size_ratio = float(trade.amount_y) / reserve_y
        state.buy_flow *= params.flow_decay
        state.sell_flow *= params.flow_decay
        state.bid_toxicity *= params.toxicity_decay
        state.ask_toxicity *= params.toxicity_decay

        if trade.is_buy:
            state.buy_flow += (1.0 - params.flow_decay) * size_ratio
            state.ask_toxicity += params.same_side_toxicity_weight * size_ratio
        else:
            state.sell_flow += (1.0 - params.flow_decay) * size_ratio
            state.bid_toxicity += params.same_side_toxicity_weight * size_ratio

        if state.last_is_buy is not None and trade.is_buy != state.last_is_buy:
            decay = math.exp(-(dt - 1.0) / params.reversal_time_scale)
            reversal_signal = decay * max(state.last_size_ratio, size_ratio)
            if state.last_is_buy:
                state.ask_toxicity += params.reverse_reaction_weight * reversal_signal
            else:
                state.bid_toxicity += params.reverse_reaction_weight * reversal_signal

        state.last_timestamp = int(trade.timestamp)
        state.last_is_buy = bool(trade.is_buy)
        state.last_size_ratio = size_ratio
        state.initialized = True
        return self._fees(initial_x=trade.reserve_x, initial_y=trade.reserve_y)

    def _fees(self, *, initial_x: float, initial_y: float) -> tuple[float, float]:
        params = self.params
        state = self.state
        inventory_skew = self._inventory_skew(
            reserve_x=float(initial_x),
            reserve_y=float(initial_y),
        )
        toxicity_total = state.bid_toxicity + state.ask_toxicity
        flow_skew = state.buy_flow - state.sell_flow
        inventory_term = params.inventory_weight * (inventory_skew - params.inventory_target)
        flow_term = params.flow_skew_weight * flow_skew
        mid = params.base_fee + params.toxicity_to_mid * toxicity_total
        spread = params.base_spread + params.toxicity_to_spread * toxicity_total
        bid_fee = _clamp(
            mid + 0.5 * spread + inventory_term - flow_term + params.toxicity_to_side * state.bid_toxicity,
            params.min_fee,
            params.max_fee,
        )
        ask_fee = _clamp(
            mid + 0.5 * spread - inventory_term + flow_term + params.toxicity_to_side * state.ask_toxicity,
            params.min_fee,
            params.max_fee,
        )
        return (bid_fee, ask_fee)

    def _inventory_skew(self, *, reserve_x: float, reserve_y: float) -> float:
        state = self.state
        if state.initial_x <= 0.0 or state.initial_y <= 0.0:
            return 0.0
        x_ratio = reserve_x / state.initial_x
        y_ratio = reserve_y / state.initial_y
        return _clamp(x_ratio - y_ratio, -2.0, 2.0)
