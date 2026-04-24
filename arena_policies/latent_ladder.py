"""Progressively richer latent-state controllers for the exact simple-AMM challenge."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from arena_eval.core.types import TradeInfo


MAX_FEE = 0.1


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _current_spot(trade: TradeInfo) -> float:
    return float(trade.reserve_y) / max(float(trade.reserve_x), 1e-9)


def _size_ratio(trade: TradeInfo) -> float:
    return float(trade.amount_y) / max(float(trade.reserve_y), 1e-9)


@dataclass
class LatentControllerState:
    initial_x: float = 0.0
    initial_y: float = 0.0
    last_timestamp: int = 0
    last_side: int = 0
    last_size_ratio: float = 0.0
    last_bid_fee: float = 0.003
    last_ask_fee: float = 0.003
    estimated_fair: float = 0.0
    buy_flow: float = 0.0
    sell_flow: float = 0.0
    opportunity: float = 0.0
    toxicity_bid: float = 0.0
    toxicity_ask: float = 0.0
    competitor_bid: float = 0.0
    competitor_ask: float = 0.0
    fill_pressure: float = 0.0
    initialized: bool = False


class _LatentControllerBase:
    def __init__(self) -> None:
        self.state = LatentControllerState()

    def _initialize_state(self, initial_x: float, initial_y: float, base_fee: float) -> float:
        initial_fair = float(initial_y) / max(float(initial_x), 1e-9)
        self.state = LatentControllerState(
            initial_x=float(initial_x),
            initial_y=float(initial_y),
            estimated_fair=initial_fair,
            last_bid_fee=base_fee,
            last_ask_fee=base_fee,
            initialized=True,
        )
        return initial_fair

    def _decay_flow(self, decay: float) -> None:
        self.state.buy_flow *= decay
        self.state.sell_flow *= decay

    def _update_flow(self, trade: TradeInfo, decay: float) -> None:
        self._decay_flow(decay)
        size_ratio = _size_ratio(trade)
        if trade.is_buy:
            self.state.buy_flow += (1.0 - decay) * size_ratio
        else:
            self.state.sell_flow += (1.0 - decay) * size_ratio

    def _update_opportunity(self, dt: int, decay: float) -> None:
        fill_intensity = 1.0 / float(dt)
        self.state.opportunity = decay * self.state.opportunity + (1.0 - decay) * fill_intensity
        self.state.fill_pressure = decay * self.state.fill_pressure + (1.0 - decay) * min(fill_intensity, 1.0)

    def _update_fair(self, trade: TradeInfo, *, decay: float, size_weight: float) -> float:
        spot = _current_spot(trade)
        size_ratio = _size_ratio(trade)
        side = 1 if trade.is_buy else -1
        fair_from_trade = spot * math.exp(side * size_weight * size_ratio)
        self.state.estimated_fair = (
            fair_from_trade
            if self.state.estimated_fair <= 0.0
            else decay * self.state.estimated_fair + (1.0 - decay) * fair_from_trade
        )
        return spot

    def _update_toxicity(self, trade: TradeInfo, *, decay: float, reverse_weight: float, dt: int) -> None:
        self.state.toxicity_bid *= decay
        self.state.toxicity_ask *= decay
        side = 1 if trade.is_buy else -1
        same_side_weight = 0.25 * reverse_weight
        if side > 0:
            self.state.toxicity_bid += same_side_weight * _size_ratio(trade)
        else:
            self.state.toxicity_ask += same_side_weight * _size_ratio(trade)
        if self.state.last_side != 0 and side != self.state.last_side:
            reversal_signal = self.state.last_size_ratio * math.exp(-0.5 * float(dt))
            if self.state.last_side > 0:
                self.state.toxicity_bid += reverse_weight * reversal_signal
            else:
                self.state.toxicity_ask += reverse_weight * reversal_signal

    def _update_competition(self, trade: TradeInfo, *, decay: float, base_fee: float, dt: int) -> None:
        fill_intensity = 1.0 / float(dt)
        if trade.is_buy:
            self.state.competitor_bid = (
                decay * self.state.competitor_bid
                + (1.0 - decay) * max(self.state.last_bid_fee - base_fee, 0.0) * fill_intensity
            )
            self.state.competitor_ask *= decay
        else:
            self.state.competitor_ask = (
                decay * self.state.competitor_ask
                + (1.0 - decay) * max(self.state.last_ask_fee - base_fee, 0.0) * fill_intensity
            )
            self.state.competitor_bid *= decay

    def _inventory_skew(self, reserve_x: float, reserve_y: float) -> float:
        state = self.state
        if state.initial_x <= 0.0 or state.initial_y <= 0.0:
            return 0.0
        x_ratio = reserve_x / state.initial_x
        y_ratio = reserve_y / state.initial_y
        return _clamp(x_ratio - y_ratio, -2.0, 2.0)

    def _finalize_trade(self, trade: TradeInfo, bid_fee: float, ask_fee: float) -> tuple[float, float]:
        self.state.last_timestamp = int(trade.timestamp)
        self.state.last_side = 1 if trade.is_buy else -1
        self.state.last_size_ratio = _size_ratio(trade)
        self.state.last_bid_fee = bid_fee
        self.state.last_ask_fee = ask_fee
        self.state.initialized = True
        return (bid_fee, ask_fee)

    def _flow_total(self) -> float:
        return self.state.buy_flow + self.state.sell_flow

    def _flow_skew(self) -> float:
        return self.state.buy_flow - self.state.sell_flow

    def _opportunity_signal(self, scale: float) -> float:
        return math.tanh(self.state.opportunity * scale)

    def _fair_gap(self, spot: float) -> float:
        if spot <= 0.0 or self.state.estimated_fair <= 0.0:
            return 0.0
        return math.log(max(self.state.estimated_fair, 1e-9) / max(spot, 1e-9))

    def _competition_total(self) -> float:
        return self.state.competitor_bid + self.state.competitor_ask

    def _competition_skew(self) -> float:
        return self.state.competitor_bid - self.state.competitor_ask


@dataclass(frozen=True)
class LatentFlowParams:
    base_fee: float = 0.003
    min_fee: float = 0.0001
    max_fee: float = 0.02
    flow_decay: float = 0.9
    opportunity_decay: float = 0.9
    opportunity_scale: float = 4.0
    opportunity_weight: float = 0.003
    flow_skew_weight: float = 0.01

    def normalized(self) -> "LatentFlowParams":
        min_fee = _clamp(self.min_fee, 0.0, MAX_FEE)
        max_fee = _clamp(self.max_fee, min_fee, MAX_FEE)
        return LatentFlowParams(
            base_fee=_clamp(self.base_fee, min_fee, max_fee),
            min_fee=min_fee,
            max_fee=max_fee,
            flow_decay=_clamp(self.flow_decay, 0.0, 0.999),
            opportunity_decay=_clamp(self.opportunity_decay, 0.0, 0.999),
            opportunity_scale=_clamp(self.opportunity_scale, 0.1, 20.0),
            opportunity_weight=_clamp(self.opportunity_weight, -0.02, 0.05),
            flow_skew_weight=_clamp(self.flow_skew_weight, -0.1, 0.1),
        )

    def to_dict(self) -> dict[str, float]:
        return dict(asdict(self.normalized()))


class LatentFlowStrategy(_LatentControllerBase):
    def __init__(self, params: LatentFlowParams | None = None) -> None:
        super().__init__()
        self.params = (params or LatentFlowParams()).normalized()

    def after_initialize(self, initial_x: float, initial_y: float) -> tuple[float, float]:
        self._initialize_state(initial_x, initial_y, self.params.base_fee)
        return (self.params.base_fee, self.params.base_fee)

    def after_swap(self, trade: TradeInfo) -> tuple[float, float]:
        params = self.params
        dt = max(1, int(trade.timestamp - self.state.last_timestamp))
        self._update_flow(trade, params.flow_decay)
        self._update_opportunity(dt, params.opportunity_decay)
        flow_total = self._flow_total()
        flow_skew = self._flow_skew()
        opportunity_edge = self._opportunity_signal(params.opportunity_scale)
        mid = (
            params.base_fee
            - params.opportunity_weight * opportunity_edge
            + 0.25 * abs(params.flow_skew_weight) * flow_total
        )
        skew = params.flow_skew_weight * (0.5 + opportunity_edge) * flow_skew
        bid_fee = _clamp(
            mid - skew,
            params.min_fee,
            params.max_fee,
        )
        ask_fee = _clamp(
            mid + skew,
            params.min_fee,
            params.max_fee,
        )
        return self._finalize_trade(trade, bid_fee, ask_fee)


@dataclass(frozen=True)
class LatentFairParams:
    base_fee: float = 0.003
    min_fee: float = 0.0001
    max_fee: float = 0.02
    flow_decay: float = 0.9
    opportunity_decay: float = 0.9
    opportunity_scale: float = 4.0
    opportunity_weight: float = 0.003
    flow_skew_weight: float = 0.01
    fair_decay: float = 0.92
    fair_size_weight: float = 2.0
    fair_gap_weight: float = 0.012

    def normalized(self) -> "LatentFairParams":
        min_fee = _clamp(self.min_fee, 0.0, MAX_FEE)
        max_fee = _clamp(self.max_fee, min_fee, MAX_FEE)
        return LatentFairParams(
            base_fee=_clamp(self.base_fee, min_fee, max_fee),
            min_fee=min_fee,
            max_fee=max_fee,
            flow_decay=_clamp(self.flow_decay, 0.0, 0.999),
            opportunity_decay=_clamp(self.opportunity_decay, 0.0, 0.999),
            opportunity_scale=_clamp(self.opportunity_scale, 0.1, 20.0),
            opportunity_weight=_clamp(self.opportunity_weight, -0.02, 0.05),
            flow_skew_weight=_clamp(self.flow_skew_weight, -0.1, 0.1),
            fair_decay=_clamp(self.fair_decay, 0.0, 0.999),
            fair_size_weight=_clamp(self.fair_size_weight, 0.0, 10.0),
            fair_gap_weight=_clamp(self.fair_gap_weight, 0.0, 0.1),
        )

    def to_dict(self) -> dict[str, float]:
        return dict(asdict(self.normalized()))


class LatentFairStrategy(_LatentControllerBase):
    def __init__(self, params: LatentFairParams | None = None) -> None:
        super().__init__()
        self.params = (params or LatentFairParams()).normalized()

    def after_initialize(self, initial_x: float, initial_y: float) -> tuple[float, float]:
        self._initialize_state(initial_x, initial_y, self.params.base_fee)
        return (self.params.base_fee, self.params.base_fee)

    def after_swap(self, trade: TradeInfo) -> tuple[float, float]:
        params = self.params
        dt = max(1, int(trade.timestamp - self.state.last_timestamp))
        self._update_flow(trade, params.flow_decay)
        self._update_opportunity(dt, params.opportunity_decay)
        spot = self._update_fair(trade, decay=params.fair_decay, size_weight=params.fair_size_weight)
        fair_gap = self._fair_gap(spot)
        flow_total = self._flow_total()
        flow_skew = self._flow_skew()
        opportunity_edge = self._opportunity_signal(params.opportunity_scale)
        mid = (
            params.base_fee
            - params.opportunity_weight * opportunity_edge
            + 0.25 * abs(params.flow_skew_weight) * flow_total
        )
        skew = params.flow_skew_weight * (0.5 + opportunity_edge) * flow_skew
        fair_term = params.fair_gap_weight * fair_gap
        bid_fee = _clamp(
            mid - skew + fair_term,
            params.min_fee,
            params.max_fee,
        )
        ask_fee = _clamp(
            mid + skew - fair_term,
            params.min_fee,
            params.max_fee,
        )
        return self._finalize_trade(trade, bid_fee, ask_fee)


@dataclass(frozen=True)
class LatentToxicityParams:
    base_fee: float = 0.003
    min_fee: float = 0.0001
    max_fee: float = 0.02
    flow_decay: float = 0.9
    opportunity_decay: float = 0.9
    opportunity_scale: float = 4.0
    opportunity_weight: float = 0.003
    flow_skew_weight: float = 0.01
    fair_decay: float = 0.92
    fair_size_weight: float = 2.0
    fair_gap_weight: float = 0.012
    toxicity_decay: float = 0.92
    reverse_toxicity_weight: float = 1.0
    toxicity_weight: float = 0.005

    def normalized(self) -> "LatentToxicityParams":
        min_fee = _clamp(self.min_fee, 0.0, MAX_FEE)
        max_fee = _clamp(self.max_fee, min_fee, MAX_FEE)
        return LatentToxicityParams(
            base_fee=_clamp(self.base_fee, min_fee, max_fee),
            min_fee=min_fee,
            max_fee=max_fee,
            flow_decay=_clamp(self.flow_decay, 0.0, 0.999),
            opportunity_decay=_clamp(self.opportunity_decay, 0.0, 0.999),
            opportunity_scale=_clamp(self.opportunity_scale, 0.1, 20.0),
            opportunity_weight=_clamp(self.opportunity_weight, -0.02, 0.05),
            flow_skew_weight=_clamp(self.flow_skew_weight, -0.1, 0.1),
            fair_decay=_clamp(self.fair_decay, 0.0, 0.999),
            fair_size_weight=_clamp(self.fair_size_weight, 0.0, 10.0),
            fair_gap_weight=_clamp(self.fair_gap_weight, 0.0, 0.1),
            toxicity_decay=_clamp(self.toxicity_decay, 0.0, 0.999),
            reverse_toxicity_weight=_clamp(self.reverse_toxicity_weight, 0.0, 5.0),
            toxicity_weight=_clamp(self.toxicity_weight, 0.0, 0.05),
        )

    def to_dict(self) -> dict[str, float]:
        return dict(asdict(self.normalized()))


class LatentToxicityStrategy(_LatentControllerBase):
    def __init__(self, params: LatentToxicityParams | None = None) -> None:
        super().__init__()
        self.params = (params or LatentToxicityParams()).normalized()

    def after_initialize(self, initial_x: float, initial_y: float) -> tuple[float, float]:
        self._initialize_state(initial_x, initial_y, self.params.base_fee)
        return (self.params.base_fee, self.params.base_fee)

    def after_swap(self, trade: TradeInfo) -> tuple[float, float]:
        params = self.params
        dt = max(1, int(trade.timestamp - self.state.last_timestamp))
        self._update_flow(trade, params.flow_decay)
        self._update_opportunity(dt, params.opportunity_decay)
        spot = self._update_fair(trade, decay=params.fair_decay, size_weight=params.fair_size_weight)
        self._update_toxicity(trade, decay=params.toxicity_decay, reverse_weight=params.reverse_toxicity_weight, dt=dt)
        fair_gap = self._fair_gap(spot)
        flow_total = self._flow_total()
        flow_skew = self._flow_skew()
        opportunity_edge = self._opportunity_signal(params.opportunity_scale)
        toxicity_total = self.state.toxicity_bid + self.state.toxicity_ask
        mid = (
            params.base_fee
            - params.opportunity_weight * opportunity_edge
            + 0.25 * abs(params.flow_skew_weight) * flow_total
            + 0.35 * params.toxicity_weight * toxicity_total
        )
        skew = params.flow_skew_weight * (0.5 + opportunity_edge) * flow_skew
        fair_term = params.fair_gap_weight * fair_gap
        bid_fee = _clamp(
            mid - skew + fair_term + params.toxicity_weight * self.state.toxicity_bid,
            params.min_fee,
            params.max_fee,
        )
        ask_fee = _clamp(
            mid + skew - fair_term + params.toxicity_weight * self.state.toxicity_ask,
            params.min_fee,
            params.max_fee,
        )
        return self._finalize_trade(trade, bid_fee, ask_fee)


@dataclass(frozen=True)
class LatentCompetitionParams:
    base_fee: float = 0.003
    min_fee: float = 0.0001
    max_fee: float = 0.02
    flow_decay: float = 0.9
    opportunity_decay: float = 0.9
    opportunity_scale: float = 4.0
    opportunity_weight: float = 0.003
    flow_skew_weight: float = 0.01
    fair_decay: float = 0.92
    fair_size_weight: float = 2.0
    fair_gap_weight: float = 0.012
    toxicity_decay: float = 0.92
    reverse_toxicity_weight: float = 1.0
    toxicity_weight: float = 0.005
    competition_decay: float = 0.9
    competition_weight: float = 0.004

    def normalized(self) -> "LatentCompetitionParams":
        min_fee = _clamp(self.min_fee, 0.0, MAX_FEE)
        max_fee = _clamp(self.max_fee, min_fee, MAX_FEE)
        return LatentCompetitionParams(
            base_fee=_clamp(self.base_fee, min_fee, max_fee),
            min_fee=min_fee,
            max_fee=max_fee,
            flow_decay=_clamp(self.flow_decay, 0.0, 0.999),
            opportunity_decay=_clamp(self.opportunity_decay, 0.0, 0.999),
            opportunity_scale=_clamp(self.opportunity_scale, 0.1, 20.0),
            opportunity_weight=_clamp(self.opportunity_weight, -0.02, 0.05),
            flow_skew_weight=_clamp(self.flow_skew_weight, -0.1, 0.1),
            fair_decay=_clamp(self.fair_decay, 0.0, 0.999),
            fair_size_weight=_clamp(self.fair_size_weight, 0.0, 10.0),
            fair_gap_weight=_clamp(self.fair_gap_weight, 0.0, 0.1),
            toxicity_decay=_clamp(self.toxicity_decay, 0.0, 0.999),
            reverse_toxicity_weight=_clamp(self.reverse_toxicity_weight, 0.0, 5.0),
            toxicity_weight=_clamp(self.toxicity_weight, 0.0, 0.05),
            competition_decay=_clamp(self.competition_decay, 0.0, 0.999),
            competition_weight=_clamp(self.competition_weight, 0.0, 0.05),
        )

    def to_dict(self) -> dict[str, float]:
        return dict(asdict(self.normalized()))


class LatentCompetitionStrategy(_LatentControllerBase):
    def __init__(self, params: LatentCompetitionParams | None = None) -> None:
        super().__init__()
        self.params = (params or LatentCompetitionParams()).normalized()

    def after_initialize(self, initial_x: float, initial_y: float) -> tuple[float, float]:
        self._initialize_state(initial_x, initial_y, self.params.base_fee)
        return (self.params.base_fee, self.params.base_fee)

    def after_swap(self, trade: TradeInfo) -> tuple[float, float]:
        params = self.params
        dt = max(1, int(trade.timestamp - self.state.last_timestamp))
        self._update_flow(trade, params.flow_decay)
        self._update_opportunity(dt, params.opportunity_decay)
        spot = self._update_fair(trade, decay=params.fair_decay, size_weight=params.fair_size_weight)
        self._update_toxicity(trade, decay=params.toxicity_decay, reverse_weight=params.reverse_toxicity_weight, dt=dt)
        self._update_competition(trade, decay=params.competition_decay, base_fee=params.base_fee, dt=dt)
        fair_gap = self._fair_gap(spot)
        flow_total = self._flow_total()
        flow_skew = self._flow_skew()
        opportunity_edge = self._opportunity_signal(params.opportunity_scale)
        toxicity_total = self.state.toxicity_bid + self.state.toxicity_ask
        competition_total = self._competition_total()
        competition_skew = self._competition_skew()
        mid = (
            params.base_fee
            - params.opportunity_weight * opportunity_edge
            + 0.25 * abs(params.flow_skew_weight) * flow_total
            + 0.30 * params.toxicity_weight * toxicity_total
            + 0.25 * params.competition_weight * competition_total
        )
        skew = (
            params.flow_skew_weight * (0.5 + opportunity_edge) * flow_skew
            - 0.5 * params.competition_weight * competition_skew
        )
        fair_term = params.fair_gap_weight * fair_gap
        bid_fee = _clamp(
            mid
            - skew
            + fair_term
            + params.toxicity_weight * self.state.toxicity_bid
            + params.competition_weight * self.state.competitor_bid,
            params.min_fee,
            params.max_fee,
        )
        ask_fee = _clamp(
            mid
            + skew
            - fair_term
            + params.toxicity_weight * self.state.toxicity_ask
            + params.competition_weight * self.state.competitor_ask,
            params.min_fee,
            params.max_fee,
        )
        return self._finalize_trade(trade, bid_fee, ask_fee)


@dataclass(frozen=True)
class LatentFullParams:
    base_fee: float = 0.003
    min_fee: float = 0.0001
    max_fee: float = 0.02
    flow_decay: float = 0.9
    opportunity_decay: float = 0.9
    opportunity_scale: float = 4.0
    opportunity_weight: float = 0.003
    flow_skew_weight: float = 0.01
    fair_decay: float = 0.92
    fair_size_weight: float = 2.0
    fair_gap_weight: float = 0.012
    toxicity_decay: float = 0.92
    reverse_toxicity_weight: float = 1.0
    toxicity_weight: float = 0.005
    competition_decay: float = 0.9
    competition_weight: float = 0.004
    inventory_weight: float = 0.01
    inventory_target: float = 0.0

    def normalized(self) -> "LatentFullParams":
        min_fee = _clamp(self.min_fee, 0.0, MAX_FEE)
        max_fee = _clamp(self.max_fee, min_fee, MAX_FEE)
        return LatentFullParams(
            base_fee=_clamp(self.base_fee, min_fee, max_fee),
            min_fee=min_fee,
            max_fee=max_fee,
            flow_decay=_clamp(self.flow_decay, 0.0, 0.999),
            opportunity_decay=_clamp(self.opportunity_decay, 0.0, 0.999),
            opportunity_scale=_clamp(self.opportunity_scale, 0.1, 20.0),
            opportunity_weight=_clamp(self.opportunity_weight, -0.02, 0.05),
            flow_skew_weight=_clamp(self.flow_skew_weight, -0.1, 0.1),
            fair_decay=_clamp(self.fair_decay, 0.0, 0.999),
            fair_size_weight=_clamp(self.fair_size_weight, 0.0, 10.0),
            fair_gap_weight=_clamp(self.fair_gap_weight, 0.0, 0.1),
            toxicity_decay=_clamp(self.toxicity_decay, 0.0, 0.999),
            reverse_toxicity_weight=_clamp(self.reverse_toxicity_weight, 0.0, 5.0),
            toxicity_weight=_clamp(self.toxicity_weight, 0.0, 0.05),
            competition_decay=_clamp(self.competition_decay, 0.0, 0.999),
            competition_weight=_clamp(self.competition_weight, 0.0, 0.05),
            inventory_weight=_clamp(self.inventory_weight, -0.1, 0.1),
            inventory_target=_clamp(self.inventory_target, -1.0, 1.0),
        )

    def to_dict(self) -> dict[str, float]:
        return dict(asdict(self.normalized()))


class LatentFullStrategy(_LatentControllerBase):
    def __init__(self, params: LatentFullParams | None = None) -> None:
        super().__init__()
        self.params = (params or LatentFullParams()).normalized()

    def after_initialize(self, initial_x: float, initial_y: float) -> tuple[float, float]:
        self._initialize_state(initial_x, initial_y, self.params.base_fee)
        return (self.params.base_fee, self.params.base_fee)

    def after_swap(self, trade: TradeInfo) -> tuple[float, float]:
        params = self.params
        dt = max(1, int(trade.timestamp - self.state.last_timestamp))
        self._update_flow(trade, params.flow_decay)
        self._update_opportunity(dt, params.opportunity_decay)
        spot = self._update_fair(trade, decay=params.fair_decay, size_weight=params.fair_size_weight)
        self._update_toxicity(trade, decay=params.toxicity_decay, reverse_weight=params.reverse_toxicity_weight, dt=dt)
        self._update_competition(trade, decay=params.competition_decay, base_fee=params.base_fee, dt=dt)
        fair_gap = self._fair_gap(spot)
        flow_total = self._flow_total()
        flow_skew = self._flow_skew()
        opportunity_edge = self._opportunity_signal(params.opportunity_scale)
        toxicity_total = self.state.toxicity_bid + self.state.toxicity_ask
        competition_total = self._competition_total()
        competition_skew = self._competition_skew()
        inventory_term = params.inventory_weight * (
            self._inventory_skew(float(trade.reserve_x), float(trade.reserve_y)) - params.inventory_target
        )
        mid = (
            params.base_fee
            - params.opportunity_weight * opportunity_edge
            + 0.25 * abs(params.flow_skew_weight) * flow_total
            + 0.30 * params.toxicity_weight * toxicity_total
            + 0.20 * params.competition_weight * competition_total
        )
        skew = (
            params.flow_skew_weight * (0.5 + opportunity_edge) * flow_skew
            - 0.5 * params.competition_weight * competition_skew
        )
        fair_term = params.fair_gap_weight * fair_gap
        bid_fee = _clamp(
            mid
            - skew
            + fair_term
            + params.toxicity_weight * self.state.toxicity_bid
            + params.competition_weight * self.state.competitor_bid
            + inventory_term,
            params.min_fee,
            params.max_fee,
        )
        ask_fee = _clamp(
            mid
            + skew
            - fair_term
            + params.toxicity_weight * self.state.toxicity_ask
            + params.competition_weight * self.state.competitor_ask
            - inventory_term,
            params.min_fee,
            params.max_fee,
        )
        return self._finalize_trade(trade, bid_fee, ask_fee)
