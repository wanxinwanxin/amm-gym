"""Belief-state fee controller for the exact simple-AMM challenge replica."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from arena_eval.core.types import TradeInfo


MAX_FEE = 0.1
WAD = 10**18


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


@dataclass(frozen=True)
class BeliefStateControllerParams:
    """Compact latent-state controller that only uses executed-fill history."""

    base_fee: float = 0.003
    min_fee: float = 0.0001
    max_fee: float = 0.02
    fair_decay: float = 0.92
    fair_size_weight: float = 2.5
    flow_decay: float = 0.85
    opportunity_decay: float = 0.9
    opportunity_scale: float = 5.0
    opportunity_weight: float = 0.002
    competition_decay: float = 0.9
    competition_weight: float = 0.003
    toxicity_decay: float = 0.92
    reverse_toxicity_weight: float = 1.5
    toxicity_weight: float = 0.004
    flow_skew_weight: float = 0.01
    fair_gap_weight: float = 0.015

    def normalized(self) -> "BeliefStateControllerParams":
        min_fee = _clamp(self.min_fee, 0.0, MAX_FEE)
        max_fee = _clamp(self.max_fee, min_fee, MAX_FEE)
        return BeliefStateControllerParams(
            base_fee=_clamp(self.base_fee, min_fee, max_fee),
            min_fee=min_fee,
            max_fee=max_fee,
            fair_decay=_clamp(self.fair_decay, 0.0, 0.999),
            fair_size_weight=_clamp(self.fair_size_weight, 0.0, 10.0),
            flow_decay=_clamp(self.flow_decay, 0.0, 0.999),
            opportunity_decay=_clamp(self.opportunity_decay, 0.0, 0.999),
            opportunity_scale=_clamp(self.opportunity_scale, 0.1, 50.0),
            opportunity_weight=_clamp(self.opportunity_weight, -0.02, 0.05),
            competition_decay=_clamp(self.competition_decay, 0.0, 0.999),
            competition_weight=_clamp(self.competition_weight, 0.0, 0.05),
            toxicity_decay=_clamp(self.toxicity_decay, 0.0, 0.999),
            reverse_toxicity_weight=_clamp(self.reverse_toxicity_weight, 0.0, 5.0),
            toxicity_weight=_clamp(self.toxicity_weight, 0.0, 0.05),
            flow_skew_weight=_clamp(self.flow_skew_weight, -0.1, 0.1),
            fair_gap_weight=_clamp(self.fair_gap_weight, 0.0, 0.1),
        )

    def to_dict(self) -> dict[str, float]:
        return dict(asdict(self.normalized()))

    def to_wad_dict(self) -> dict[str, int]:
        return {key: int(round(value * WAD)) for key, value in self.to_dict().items()}


@dataclass
class BeliefStateControllerState:
    last_timestamp: int = 0
    estimated_fair: float = 0.0
    buy_flow: float = 0.0
    sell_flow: float = 0.0
    opportunity: float = 0.0
    competitor_bid: float = 0.0
    competitor_ask: float = 0.0
    toxicity_bid: float = 0.0
    toxicity_ask: float = 0.0
    last_side: int = 0
    last_size_ratio: float = 0.0
    last_bid_fee: float = 0.003
    last_ask_fee: float = 0.003
    initialized: bool = False


class BeliefStateControllerStrategy:
    """Controller that estimates latent fair/opportunity/toxicity from fills."""

    def __init__(self, params: BeliefStateControllerParams | None = None) -> None:
        self.params = (params or BeliefStateControllerParams()).normalized()
        self.state = BeliefStateControllerState()

    def after_initialize(self, initial_x: float, initial_y: float) -> tuple[float, float]:
        initial_fair = float(initial_y) / max(float(initial_x), 1e-9)
        base = self.params.base_fee
        self.state = BeliefStateControllerState(
            estimated_fair=initial_fair,
            last_bid_fee=base,
            last_ask_fee=base,
            initialized=True,
        )
        return self._fees(current_spot=initial_fair)

    def after_swap(self, trade: TradeInfo) -> tuple[float, float]:
        params = self.params
        state = self.state
        dt = 1 if not state.initialized else max(1, int(trade.timestamp - state.last_timestamp))
        current_spot = float(trade.reserve_y) / max(float(trade.reserve_x), 1e-9)
        reserve_y = max(float(trade.reserve_y), 1e-9)
        size_ratio = float(trade.amount_y) / reserve_y
        side = 1 if trade.is_buy else -1

        state.buy_flow *= params.flow_decay
        state.sell_flow *= params.flow_decay
        if trade.is_buy:
            state.buy_flow += (1.0 - params.flow_decay) * size_ratio
        else:
            state.sell_flow += (1.0 - params.flow_decay) * size_ratio

        fill_intensity = 1.0 / float(dt)
        state.opportunity = (
            params.opportunity_decay * state.opportunity
            + (1.0 - params.opportunity_decay) * fill_intensity
        )

        if trade.is_buy:
            state.competitor_bid = (
                params.competition_decay * state.competitor_bid
                + (1.0 - params.competition_decay) * max(state.last_bid_fee - params.base_fee, 0.0) * fill_intensity
            )
            state.competitor_ask *= params.competition_decay
        else:
            state.competitor_ask = (
                params.competition_decay * state.competitor_ask
                + (1.0 - params.competition_decay) * max(state.last_ask_fee - params.base_fee, 0.0) * fill_intensity
            )
            state.competitor_bid *= params.competition_decay

        state.toxicity_bid *= params.toxicity_decay
        state.toxicity_ask *= params.toxicity_decay
        if state.last_side != 0 and side != state.last_side:
            reversal_signal = state.last_size_ratio * math.exp(-0.5 * float(dt))
            if state.last_side > 0:
                state.toxicity_bid += params.reverse_toxicity_weight * reversal_signal
            else:
                state.toxicity_ask += params.reverse_toxicity_weight * reversal_signal

        fair_from_trade = current_spot * math.exp(side * params.fair_size_weight * size_ratio)
        if state.estimated_fair <= 0.0:
            state.estimated_fair = fair_from_trade
        else:
            state.estimated_fair = (
                params.fair_decay * state.estimated_fair
                + (1.0 - params.fair_decay) * fair_from_trade
            )

        state.last_timestamp = int(trade.timestamp)
        state.last_side = side
        state.last_size_ratio = size_ratio
        state.initialized = True
        return self._fees(current_spot=current_spot)

    def _fees(self, *, current_spot: float) -> tuple[float, float]:
        params = self.params
        state = self.state
        fair_gap = 0.0
        if current_spot > 0.0 and state.estimated_fair > 0.0:
            fair_gap = math.log(max(state.estimated_fair, 1e-9) / max(current_spot, 1e-9))
        flow_skew = state.buy_flow - state.sell_flow
        opportunity_edge = math.tanh(state.opportunity * params.opportunity_scale)
        bid_fee = params.base_fee
        ask_fee = params.base_fee

        bid_fee -= params.opportunity_weight * opportunity_edge
        ask_fee -= params.opportunity_weight * opportunity_edge
        bid_fee += params.competition_weight * state.competitor_bid
        ask_fee += params.competition_weight * state.competitor_ask
        bid_fee += params.toxicity_weight * state.toxicity_bid
        ask_fee += params.toxicity_weight * state.toxicity_ask
        bid_fee -= params.flow_skew_weight * flow_skew
        ask_fee += params.flow_skew_weight * flow_skew
        bid_fee += params.fair_gap_weight * fair_gap
        ask_fee -= params.fair_gap_weight * fair_gap

        bid_fee = _clamp(bid_fee, params.min_fee, params.max_fee)
        ask_fee = _clamp(ask_fee, params.min_fee, params.max_fee)
        state.last_bid_fee = bid_fee
        state.last_ask_fee = ask_fee
        return (bid_fee, ask_fee)
