"""Functional policy contracts for the differentiable simple-AMM stack."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from arena_eval.diff_simple_amm.types import PolicyOutput, PolicyState, TradeEvent
from arena_policies.piecewise_controller import PiecewiseControllerParams
from arena_policies.submission_safe import SubmissionCompactParams


class DiffSimpleAMMPolicy(Protocol):
    """Functional policy interface compatible with scan-based rollouts."""

    def initialize(self, initial_x: float, initial_y: float) -> PolicyOutput:
        ...

    def after_event(self, state: PolicyState, trade: TradeEvent) -> PolicyOutput:
        ...


@dataclass(frozen=True)
class FixedFeeDiffPolicy:
    """Fixed-fee functional policy for exact-path parity tests."""

    bid_fee: float = 0.003
    ask_fee: float = 0.003

    def initialize(self, initial_x: float, initial_y: float) -> PolicyOutput:
        del initial_x, initial_y
        return PolicyOutput(bid_fee=self.bid_fee, ask_fee=self.ask_fee, state=PolicyState())

    def after_event(self, state: PolicyState, trade: TradeEvent) -> PolicyOutput:
        del trade
        return PolicyOutput(bid_fee=self.bid_fee, ask_fee=self.ask_fee, state=state)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _spot_from_trade(trade: TradeEvent) -> float:
    return float(trade.reserve_y) / max(float(trade.reserve_x), 1e-9)


def _size_ratio(trade: TradeEvent) -> float:
    return float(trade.amount_y) / max(float(trade.reserve_y), 1e-9)


@dataclass(frozen=True)
class _SubmissionCompactState:
    initial_x: float = 0.0
    initial_y: float = 0.0
    last_timestamp: int = 0
    last_side: int = 0
    last_size_ratio: float = 0.0
    streak_len: float = 0.0
    buy_flow_fast: float = 0.0
    sell_flow_fast: float = 0.0
    buy_flow_slow: float = 0.0
    sell_flow_slow: float = 0.0
    size_fast: float = 0.0
    size_slow: float = 0.0
    gap_fast: float = 1.0
    gap_slow: float = 1.0
    tox_bid: float = 0.0
    tox_ask: float = 0.0
    fair_fast: float = 0.0
    fair_slow: float = 0.0
    initialized: float = 1.0

    def to_policy_state(self) -> PolicyState:
        return PolicyState(
            (
                self.initial_x,
                self.initial_y,
                float(self.last_timestamp),
                float(self.last_side),
                self.last_size_ratio,
                self.streak_len,
                self.buy_flow_fast,
                self.sell_flow_fast,
                self.buy_flow_slow,
                self.sell_flow_slow,
                self.size_fast,
                self.size_slow,
                self.gap_fast,
                self.gap_slow,
                self.tox_bid,
                self.tox_ask,
                self.fair_fast,
                self.fair_slow,
                self.initialized,
            )
        )

    @classmethod
    def from_policy_state(cls, state: PolicyState) -> "_SubmissionCompactState":
        if not state.values:
            return cls()
        values = state.values
        return cls(
            initial_x=float(values[0]),
            initial_y=float(values[1]),
            last_timestamp=int(values[2]),
            last_side=int(values[3]),
            last_size_ratio=float(values[4]),
            streak_len=float(values[5]),
            buy_flow_fast=float(values[6]),
            sell_flow_fast=float(values[7]),
            buy_flow_slow=float(values[8]),
            sell_flow_slow=float(values[9]),
            size_fast=float(values[10]),
            size_slow=float(values[11]),
            gap_fast=float(values[12]),
            gap_slow=float(values[13]),
            tox_bid=float(values[14]),
            tox_ask=float(values[15]),
            fair_fast=float(values[16]),
            fair_slow=float(values[17]),
            initialized=float(values[18]),
        )


@dataclass(frozen=True)
class _PiecewiseState:
    last_timestamp: int = 0
    last_side: int = 0
    bid_signal: float = 0.0
    ask_signal: float = 0.0
    bid_toxicity: float = 0.0
    ask_toxicity: float = 0.0
    initialized: float = 0.0

    def to_policy_state(self) -> PolicyState:
        return PolicyState(
            (
                float(self.last_timestamp),
                float(self.last_side),
                self.bid_signal,
                self.ask_signal,
                self.bid_toxicity,
                self.ask_toxicity,
                self.initialized,
            )
        )

    @classmethod
    def from_policy_state(cls, state: PolicyState) -> "_PiecewiseState":
        if not state.values:
            return cls()
        values = state.values
        return cls(
            last_timestamp=int(values[0]),
            last_side=int(values[1]),
            bid_signal=float(values[2]),
            ask_signal=float(values[3]),
            bid_toxicity=float(values[4]),
            ask_toxicity=float(values[5]),
            initialized=float(values[6]),
        )


@dataclass(frozen=True)
class SubmissionCompactDiffPolicy:
    """Functional port of `SubmissionCompactStrategy` for exact-path parity."""

    params: SubmissionCompactParams = SubmissionCompactParams()

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", self.params.normalized())

    def initialize(self, initial_x: float, initial_y: float) -> PolicyOutput:
        initial_spot = float(initial_y) / max(float(initial_x), 1e-9)
        state = _SubmissionCompactState(
            initial_x=float(initial_x),
            initial_y=float(initial_y),
            fair_fast=initial_spot,
            fair_slow=initial_spot,
            initialized=1.0,
        )
        return PolicyOutput(
            bid_fee=self.params.base_fee,
            ask_fee=self.params.base_fee,
            state=state.to_policy_state(),
        )

    def after_event(self, state: PolicyState, trade: TradeEvent) -> PolicyOutput:
        p = self.params
        s = _SubmissionCompactState.from_policy_state(state)
        next_state, size_ratio = self._update_core(
            s,
            trade,
            fast_decay=p.flow_fast_decay,
            slow_decay=p.flow_slow_decay,
            size_fast_decay=p.size_fast_decay,
            size_slow_decay=p.size_slow_decay,
            gap_fast_decay=p.gap_fast_decay,
            gap_slow_decay=p.gap_slow_decay,
            toxicity_decay=p.toxicity_decay,
            toxicity_weight=p.toxicity_weight,
            fair_fast_decay=0.9,
            fair_slow_decay=0.98,
            fair_size_weight=0.0,
        )
        flow_total = next_state.buy_flow_fast + next_state.sell_flow_fast
        flow_skew = (
            (next_state.buy_flow_fast - next_state.sell_flow_fast)
            + 0.5 * (next_state.buy_flow_slow - next_state.sell_flow_slow)
        )
        hot_gate = 1.0 if next_state.gap_fast < p.hot_gap_threshold else 0.0
        big_gate = 1.0 if size_ratio > p.big_trade_threshold else 0.0
        mid = (
            p.base_fee
            + p.flow_mid_weight * flow_total
            + p.size_mid_weight * max(next_state.size_fast - 0.5 * next_state.size_slow, 0.0)
            + p.gap_mid_weight * max(p.hot_gap_threshold - next_state.gap_fast, 0.0)
            + p.hot_fee_bump * max(hot_gate, big_gate)
        )
        spread = p.base_spread + 0.5 * p.hot_fee_bump * hot_gate
        skew = p.skew_weight * flow_skew
        bid = _clamp(mid + 0.5 * spread - skew + p.toxicity_side_weight * next_state.tox_bid, p.min_fee, p.max_fee)
        ask = _clamp(mid + 0.5 * spread + skew + p.toxicity_side_weight * next_state.tox_ask, p.min_fee, p.max_fee)
        return PolicyOutput(bid_fee=bid, ask_fee=ask, state=next_state.to_policy_state())

    @staticmethod
    def _update_core(
        state: _SubmissionCompactState,
        trade: TradeEvent,
        *,
        fast_decay: float,
        slow_decay: float,
        size_fast_decay: float,
        size_slow_decay: float,
        gap_fast_decay: float,
        gap_slow_decay: float,
        toxicity_decay: float,
        toxicity_weight: float,
        fair_fast_decay: float,
        fair_slow_decay: float,
        fair_size_weight: float,
    ) -> tuple[_SubmissionCompactState, float]:
        dt = max(1, int(trade.timestamp - state.last_timestamp))
        size_ratio = _size_ratio(trade)
        side = 1 if trade.is_buy else -1
        spot = _spot_from_trade(trade)

        buy_flow_fast = state.buy_flow_fast * fast_decay
        sell_flow_fast = state.sell_flow_fast * fast_decay
        buy_flow_slow = state.buy_flow_slow * slow_decay
        sell_flow_slow = state.sell_flow_slow * slow_decay
        if side > 0:
            buy_flow_fast += (1.0 - fast_decay) * size_ratio
            buy_flow_slow += (1.0 - slow_decay) * size_ratio
        else:
            sell_flow_fast += (1.0 - fast_decay) * size_ratio
            sell_flow_slow += (1.0 - slow_decay) * size_ratio

        size_fast = size_fast_decay * state.size_fast + (1.0 - size_fast_decay) * size_ratio
        size_slow = size_slow_decay * state.size_slow + (1.0 - size_slow_decay) * size_ratio
        gap_fast = gap_fast_decay * state.gap_fast + (1.0 - gap_fast_decay) * float(dt)
        gap_slow = gap_slow_decay * state.gap_slow + (1.0 - gap_slow_decay) * float(dt)

        tox_bid = state.tox_bid * toxicity_decay
        tox_ask = state.tox_ask * toxicity_decay
        same_side = 0.4 * toxicity_weight * size_ratio
        if side > 0:
            tox_bid += same_side
        else:
            tox_ask += same_side
        if state.last_side != 0 and side != state.last_side:
            reversal_scale = max(state.last_size_ratio, size_ratio) / float(dt)
            if state.last_side > 0:
                tox_bid += toxicity_weight * reversal_scale
            else:
                tox_ask += toxicity_weight * reversal_scale

        fair_obs = spot * math.exp(side * fair_size_weight * size_ratio)
        fair_fast = fair_fast_decay * state.fair_fast + (1.0 - fair_fast_decay) * fair_obs
        fair_slow = fair_slow_decay * state.fair_slow + (1.0 - fair_slow_decay) * fair_obs

        streak_len = min(state.streak_len + 1.0, 8.0) if side == state.last_side else 1.0
        next_state = _SubmissionCompactState(
            initial_x=state.initial_x,
            initial_y=state.initial_y,
            last_timestamp=int(trade.timestamp),
            last_side=side,
            last_size_ratio=size_ratio,
            streak_len=streak_len,
            buy_flow_fast=buy_flow_fast,
            sell_flow_fast=sell_flow_fast,
            buy_flow_slow=buy_flow_slow,
            sell_flow_slow=sell_flow_slow,
            size_fast=size_fast,
            size_slow=size_slow,
            gap_fast=gap_fast,
            gap_slow=gap_slow,
            tox_bid=tox_bid,
            tox_ask=tox_ask,
            fair_fast=fair_fast,
            fair_slow=fair_slow,
            initialized=1.0,
        )
        return (next_state, size_ratio)


@dataclass(frozen=True)
class PiecewiseDiffPolicy:
    """Functional port of `PiecewiseControllerStrategy` for exact-path parity."""

    params: PiecewiseControllerParams = PiecewiseControllerParams()

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", self.params.normalized())

    def initialize(self, initial_x: float, initial_y: float) -> PolicyOutput:
        del initial_x, initial_y
        state = _PiecewiseState(initialized=1.0)
        bid_fee, ask_fee = self._fees(state)
        return PolicyOutput(bid_fee=bid_fee, ask_fee=ask_fee, state=state.to_policy_state())

    def after_event(self, state: PolicyState, trade: TradeEvent) -> PolicyOutput:
        params = self.params
        current = _PiecewiseState.from_policy_state(state)
        dt = 1 if current.initialized < 0.5 else max(1, int(trade.timestamp - current.last_timestamp))
        size_ratio = _size_ratio(trade)
        continuation_weight, reversal_weight = self._bucket_weights(size_ratio)
        reversal_scale = 1.0 / float(dt)
        side = 1 if trade.is_buy else -1

        bid_signal = current.bid_signal * params.signal_decay
        ask_signal = current.ask_signal * params.signal_decay
        bid_toxicity = current.bid_toxicity * params.toxicity_decay
        ask_toxicity = current.ask_toxicity * params.toxicity_decay

        if side == current.last_side:
            if trade.is_buy:
                bid_signal += continuation_weight
            else:
                ask_signal += continuation_weight
        elif current.last_side != 0:
            if trade.is_buy:
                ask_toxicity += reversal_weight * reversal_scale
                bid_signal += 0.5 * continuation_weight
            else:
                bid_toxicity += reversal_weight * reversal_scale
                ask_signal += 0.5 * continuation_weight
        else:
            if trade.is_buy:
                bid_signal += 0.5 * continuation_weight
            else:
                ask_signal += 0.5 * continuation_weight

        next_state = _PiecewiseState(
            last_timestamp=int(trade.timestamp),
            last_side=side,
            bid_signal=bid_signal,
            ask_signal=ask_signal,
            bid_toxicity=bid_toxicity,
            ask_toxicity=ask_toxicity,
            initialized=1.0,
        )
        bid_fee, ask_fee = self._fees(next_state)
        return PolicyOutput(bid_fee=bid_fee, ask_fee=ask_fee, state=next_state.to_policy_state())

    def _bucket_weights(self, size_ratio: float) -> tuple[float, float]:
        params = self.params
        if size_ratio < params.small_trade_threshold:
            return (params.continuation_small, params.reversal_small)
        if size_ratio < params.large_trade_threshold:
            return (params.continuation_medium, params.reversal_medium)
        return (params.continuation_large, params.reversal_large)

    def _fees(self, state: _PiecewiseState) -> tuple[float, float]:
        params = self.params
        toxicity_total = state.bid_toxicity + state.ask_toxicity
        base = params.base_fee + 0.5 * params.base_spread + params.toxicity_to_mid * toxicity_total
        bid_fee = _clamp(
            base
            + params.toxicity_to_side * state.bid_toxicity
            - params.continuation_to_same_side * state.bid_signal
            + params.continuation_to_cross_side * state.ask_signal,
            0.0,
            0.1,
        )
        ask_fee = _clamp(
            base
            + params.toxicity_to_side * state.ask_toxicity
            - params.continuation_to_same_side * state.ask_signal
            + params.continuation_to_cross_side * state.bid_signal,
            0.0,
            0.1,
        )
        return (bid_fee, ask_fee)
