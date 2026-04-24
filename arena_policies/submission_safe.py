"""Submission-safe high-DOF controller families for the simple AMM challenge.

These strategies are intentionally designed around the challenge callback surface:
they update only on executed submission fills, use a small fixed set of state
variables, and rely on piecewise/gated arithmetic rather than large learned
models. The formulations aim to stay exportable to Solidity under the official
storage/gas constraints while being materially more expressive than the existing
reactive controllers.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from arena_eval.core.types import TradeInfo


MAX_FEE = 0.1


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _spot_from_trade(trade: TradeInfo) -> float:
    return float(trade.reserve_y) / max(float(trade.reserve_x), 1e-9)


def _size_ratio(trade: TradeInfo) -> float:
    return float(trade.amount_y) / max(float(trade.reserve_y), 1e-9)


def _hinge(value: float, threshold: float) -> float:
    return max(0.0, float(value) - float(threshold))


@dataclass
class SubmissionSafeState:
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
    initialized: bool = False


class _SubmissionSafeBase:
    def __init__(self) -> None:
        self.state = SubmissionSafeState()

    def _initialize_state(self, initial_x: float, initial_y: float) -> float:
        initial_spot = float(initial_y) / max(float(initial_x), 1e-9)
        self.state = SubmissionSafeState(
            initial_x=float(initial_x),
            initial_y=float(initial_y),
            fair_fast=initial_spot,
            fair_slow=initial_spot,
            initialized=True,
        )
        return initial_spot

    def _update_core(
        self,
        trade: TradeInfo,
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
    ) -> tuple[float, float]:
        state = self.state
        dt = max(1, int(trade.timestamp - state.last_timestamp))
        size_ratio = _size_ratio(trade)
        side = 1 if trade.is_buy else -1
        spot = _spot_from_trade(trade)

        state.buy_flow_fast *= fast_decay
        state.sell_flow_fast *= fast_decay
        state.buy_flow_slow *= slow_decay
        state.sell_flow_slow *= slow_decay
        if side > 0:
            state.buy_flow_fast += (1.0 - fast_decay) * size_ratio
            state.buy_flow_slow += (1.0 - slow_decay) * size_ratio
        else:
            state.sell_flow_fast += (1.0 - fast_decay) * size_ratio
            state.sell_flow_slow += (1.0 - slow_decay) * size_ratio

        state.size_fast = size_fast_decay * state.size_fast + (1.0 - size_fast_decay) * size_ratio
        state.size_slow = size_slow_decay * state.size_slow + (1.0 - size_slow_decay) * size_ratio
        state.gap_fast = gap_fast_decay * state.gap_fast + (1.0 - gap_fast_decay) * float(dt)
        state.gap_slow = gap_slow_decay * state.gap_slow + (1.0 - gap_slow_decay) * float(dt)

        state.tox_bid *= toxicity_decay
        state.tox_ask *= toxicity_decay
        same_side = 0.4 * toxicity_weight * size_ratio
        if side > 0:
            state.tox_bid += same_side
        else:
            state.tox_ask += same_side
        if state.last_side != 0 and side != state.last_side:
            reversal_scale = max(state.last_size_ratio, size_ratio) / float(dt)
            if state.last_side > 0:
                state.tox_bid += toxicity_weight * reversal_scale
            else:
                state.tox_ask += toxicity_weight * reversal_scale

        fair_obs = spot * math.exp(side * fair_size_weight * size_ratio)
        state.fair_fast = fair_fast_decay * state.fair_fast + (1.0 - fair_fast_decay) * fair_obs
        state.fair_slow = fair_slow_decay * state.fair_slow + (1.0 - fair_slow_decay) * fair_obs

        state.streak_len = min(state.streak_len + 1.0, 8.0) if side == state.last_side else 1.0
        state.last_timestamp = int(trade.timestamp)
        state.last_side = side
        state.last_size_ratio = size_ratio
        state.initialized = True
        return (spot, size_ratio)

    def _inventory_skew(self, reserve_x: float, reserve_y: float) -> float:
        state = self.state
        if state.initial_x <= 0.0 or state.initial_y <= 0.0:
            return 0.0
        x_ratio = reserve_x / state.initial_x
        y_ratio = reserve_y / state.initial_y
        return _clamp(x_ratio - y_ratio, -2.0, 2.0)


@dataclass(frozen=True)
class SubmissionCompactParams:
    base_fee: float = 0.003
    min_fee: float = 0.0001
    max_fee: float = 0.02
    flow_fast_decay: float = 0.55
    flow_slow_decay: float = 0.92
    size_fast_decay: float = 0.45
    size_slow_decay: float = 0.92
    gap_fast_decay: float = 0.55
    gap_slow_decay: float = 0.96
    toxicity_decay: float = 0.8
    toxicity_weight: float = 1.2
    base_spread: float = 0.002
    flow_mid_weight: float = 0.02
    size_mid_weight: float = 0.05
    gap_mid_weight: float = 0.002
    skew_weight: float = 0.06
    toxicity_side_weight: float = 0.04
    hot_gap_threshold: float = 1.3
    big_trade_threshold: float = 0.01
    hot_fee_bump: float = 0.004

    def normalized(self) -> "SubmissionCompactParams":
        min_fee = _clamp(self.min_fee, 0.0, MAX_FEE)
        max_fee = _clamp(self.max_fee, min_fee, MAX_FEE)
        return SubmissionCompactParams(
            base_fee=_clamp(self.base_fee, min_fee, max_fee),
            min_fee=min_fee,
            max_fee=max_fee,
            flow_fast_decay=_clamp(self.flow_fast_decay, 0.0, 0.999),
            flow_slow_decay=_clamp(self.flow_slow_decay, 0.0, 0.999),
            size_fast_decay=_clamp(self.size_fast_decay, 0.0, 0.999),
            size_slow_decay=_clamp(self.size_slow_decay, 0.0, 0.999),
            gap_fast_decay=_clamp(self.gap_fast_decay, 0.0, 0.999),
            gap_slow_decay=_clamp(self.gap_slow_decay, 0.0, 0.999),
            toxicity_decay=_clamp(self.toxicity_decay, 0.0, 0.999),
            toxicity_weight=_clamp(self.toxicity_weight, 0.0, 5.0),
            base_spread=_clamp(self.base_spread, 0.0, 0.03),
            flow_mid_weight=_clamp(self.flow_mid_weight, -0.1, 0.1),
            size_mid_weight=_clamp(self.size_mid_weight, -0.1, 0.2),
            gap_mid_weight=_clamp(self.gap_mid_weight, -0.02, 0.02),
            skew_weight=_clamp(self.skew_weight, -0.2, 0.2),
            toxicity_side_weight=_clamp(self.toxicity_side_weight, 0.0, 0.2),
            hot_gap_threshold=_clamp(self.hot_gap_threshold, 0.0, 8.0),
            big_trade_threshold=_clamp(self.big_trade_threshold, 0.0001, 0.05),
            hot_fee_bump=_clamp(self.hot_fee_bump, 0.0, 0.03),
        )

    def to_dict(self) -> dict[str, float]:
        return dict(asdict(self.normalized()))


class SubmissionCompactStrategy(_SubmissionSafeBase):
    def __init__(self, params: SubmissionCompactParams | None = None) -> None:
        super().__init__()
        self.params = (params or SubmissionCompactParams()).normalized()

    def after_initialize(self, initial_x: float, initial_y: float) -> tuple[float, float]:
        self._initialize_state(initial_x, initial_y)
        return (self.params.base_fee, self.params.base_fee)

    def after_swap(self, trade: TradeInfo) -> tuple[float, float]:
        p = self.params
        _, size_ratio = self._update_core(
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
        s = self.state
        flow_total = s.buy_flow_fast + s.sell_flow_fast
        flow_skew = (s.buy_flow_fast - s.sell_flow_fast) + 0.5 * (s.buy_flow_slow - s.sell_flow_slow)
        hot_gate = 1.0 if s.gap_fast < p.hot_gap_threshold else 0.0
        big_gate = 1.0 if size_ratio > p.big_trade_threshold else 0.0
        mid = (
            p.base_fee
            + p.flow_mid_weight * flow_total
            + p.size_mid_weight * max(s.size_fast - 0.5 * s.size_slow, 0.0)
            + p.gap_mid_weight * max(p.hot_gap_threshold - s.gap_fast, 0.0)
            + p.hot_fee_bump * max(hot_gate, big_gate)
        )
        spread = p.base_spread + 0.5 * p.hot_fee_bump * hot_gate
        skew = p.skew_weight * flow_skew
        bid = _clamp(mid + 0.5 * spread - skew + p.toxicity_side_weight * s.tox_bid, p.min_fee, p.max_fee)
        ask = _clamp(mid + 0.5 * spread + skew + p.toxicity_side_weight * s.tox_ask, p.min_fee, p.max_fee)
        return (bid, ask)


@dataclass(frozen=True)
class SubmissionRegimeParams:
    base_fee: float = 0.003
    min_fee: float = 0.0001
    max_fee: float = 0.025
    flow_fast_decay: float = 0.5
    flow_slow_decay: float = 0.95
    size_fast_decay: float = 0.45
    size_slow_decay: float = 0.94
    gap_fast_decay: float = 0.5
    gap_slow_decay: float = 0.97
    toxicity_decay: float = 0.82
    toxicity_weight: float = 1.5
    fair_fast_decay: float = 0.72
    fair_slow_decay: float = 0.96
    fair_size_weight: float = 2.0
    base_spread: float = 0.001
    calm_mid_shift: float = -0.001
    active_mid_shift: float = 0.001
    panic_mid_shift: float = 0.005
    continuation_spread: float = 0.001
    reversal_spread: float = 0.005
    fair_weight: float = 0.02
    inventory_weight: float = 0.015
    skew_weight: float = 0.05
    toxicity_side_weight: float = 0.05
    active_gap_threshold: float = 1.6
    panic_gap_threshold: float = 1.05

    def normalized(self) -> "SubmissionRegimeParams":
        min_fee = _clamp(self.min_fee, 0.0, MAX_FEE)
        max_fee = _clamp(self.max_fee, min_fee, MAX_FEE)
        return SubmissionRegimeParams(
            base_fee=_clamp(self.base_fee, min_fee, max_fee),
            min_fee=min_fee,
            max_fee=max_fee,
            flow_fast_decay=_clamp(self.flow_fast_decay, 0.0, 0.999),
            flow_slow_decay=_clamp(self.flow_slow_decay, 0.0, 0.999),
            size_fast_decay=_clamp(self.size_fast_decay, 0.0, 0.999),
            size_slow_decay=_clamp(self.size_slow_decay, 0.0, 0.999),
            gap_fast_decay=_clamp(self.gap_fast_decay, 0.0, 0.999),
            gap_slow_decay=_clamp(self.gap_slow_decay, 0.0, 0.999),
            toxicity_decay=_clamp(self.toxicity_decay, 0.0, 0.999),
            toxicity_weight=_clamp(self.toxicity_weight, 0.0, 5.0),
            fair_fast_decay=_clamp(self.fair_fast_decay, 0.0, 0.999),
            fair_slow_decay=_clamp(self.fair_slow_decay, 0.0, 0.999),
            fair_size_weight=_clamp(self.fair_size_weight, 0.0, 8.0),
            base_spread=_clamp(self.base_spread, 0.0, 0.03),
            calm_mid_shift=_clamp(self.calm_mid_shift, -0.02, 0.02),
            active_mid_shift=_clamp(self.active_mid_shift, -0.02, 0.02),
            panic_mid_shift=_clamp(self.panic_mid_shift, 0.0, 0.04),
            continuation_spread=_clamp(self.continuation_spread, 0.0, 0.02),
            reversal_spread=_clamp(self.reversal_spread, 0.0, 0.04),
            fair_weight=_clamp(self.fair_weight, 0.0, 0.08),
            inventory_weight=_clamp(self.inventory_weight, -0.08, 0.08),
            skew_weight=_clamp(self.skew_weight, -0.15, 0.15),
            toxicity_side_weight=_clamp(self.toxicity_side_weight, 0.0, 0.2),
            active_gap_threshold=_clamp(self.active_gap_threshold, 0.5, 5.0),
            panic_gap_threshold=_clamp(self.panic_gap_threshold, 0.5, 5.0),
        )

    def to_dict(self) -> dict[str, float]:
        return dict(asdict(self.normalized()))


class SubmissionRegimeStrategy(_SubmissionSafeBase):
    def __init__(self, params: SubmissionRegimeParams | None = None) -> None:
        super().__init__()
        self.params = (params or SubmissionRegimeParams()).normalized()

    def after_initialize(self, initial_x: float, initial_y: float) -> tuple[float, float]:
        self._initialize_state(initial_x, initial_y)
        return (self.params.base_fee, self.params.base_fee)

    def after_swap(self, trade: TradeInfo) -> tuple[float, float]:
        p = self.params
        spot, _ = self._update_core(
            trade,
            fast_decay=p.flow_fast_decay,
            slow_decay=p.flow_slow_decay,
            size_fast_decay=p.size_fast_decay,
            size_slow_decay=p.size_slow_decay,
            gap_fast_decay=p.gap_fast_decay,
            gap_slow_decay=p.gap_slow_decay,
            toxicity_decay=p.toxicity_decay,
            toxicity_weight=p.toxicity_weight,
            fair_fast_decay=p.fair_fast_decay,
            fair_slow_decay=p.fair_slow_decay,
            fair_size_weight=p.fair_size_weight,
        )
        s = self.state
        flow_skew = (s.buy_flow_fast - s.sell_flow_fast) + 0.35 * (s.buy_flow_slow - s.sell_flow_slow)
        fair_gap = math.log(max(s.fair_fast, 1e-9) / max(spot, 1e-9)) - math.log(max(s.fair_slow, 1e-9) / max(spot, 1e-9))
        inventory = self._inventory_skew(float(trade.reserve_x), float(trade.reserve_y))
        is_panic = s.gap_fast <= p.panic_gap_threshold
        is_active = s.gap_fast <= p.active_gap_threshold
        is_reversal = s.last_side != 0 and (1 if trade.is_buy else -1) != s.last_side
        mid = p.base_fee + p.calm_mid_shift
        if is_active:
            mid += p.active_mid_shift
        if is_panic:
            mid += p.panic_mid_shift
        mid += p.fair_weight * abs(fair_gap)
        mid += 0.4 * p.inventory_weight * abs(inventory)
        spread = p.base_spread + (p.reversal_spread if is_reversal else p.continuation_spread)
        skew = p.skew_weight * flow_skew + p.inventory_weight * inventory
        bid = _clamp(mid + 0.5 * spread - skew + p.toxicity_side_weight * s.tox_bid + p.fair_weight * fair_gap, p.min_fee, p.max_fee)
        ask = _clamp(mid + 0.5 * spread + skew + p.toxicity_side_weight * s.tox_ask - p.fair_weight * fair_gap, p.min_fee, p.max_fee)
        return (bid, ask)


@dataclass(frozen=True)
class SubmissionBasisParams:
    base_fee: float = 0.003
    min_fee: float = 0.0001
    max_fee: float = 0.03
    flow_fast_decay: float = 0.45
    flow_slow_decay: float = 0.96
    size_fast_decay: float = 0.35
    size_slow_decay: float = 0.94
    gap_fast_decay: float = 0.45
    gap_slow_decay: float = 0.98
    toxicity_decay: float = 0.84
    toxicity_weight: float = 1.7
    fair_fast_decay: float = 0.68
    fair_slow_decay: float = 0.97
    fair_size_weight: float = 2.4
    spread_base: float = 0.001
    size_hinge_low: float = 0.004
    size_hinge_high: float = 0.012
    flow_hinge: float = 0.006
    tox_hinge: float = 0.01
    fair_hinge: float = 0.001
    gap_hinge: float = 1.4
    basis_size_low_w: float = 0.03
    basis_size_high_w: float = 0.08
    basis_flow_w: float = 0.05
    basis_gap_w: float = 0.004
    basis_tox_w: float = 0.06
    basis_fair_w: float = 0.03
    basis_streak_w: float = 0.003
    skew_flow_w: float = 0.09
    skew_inventory_w: float = 0.03
    skew_fair_w: float = 0.04
    side_tox_w: float = 0.07

    def normalized(self) -> "SubmissionBasisParams":
        min_fee = _clamp(self.min_fee, 0.0, MAX_FEE)
        max_fee = _clamp(self.max_fee, min_fee, MAX_FEE)
        low = _clamp(self.size_hinge_low, 0.0001, 0.04)
        high = _clamp(self.size_hinge_high, low + 0.0001, 0.08)
        return SubmissionBasisParams(
            base_fee=_clamp(self.base_fee, min_fee, max_fee),
            min_fee=min_fee,
            max_fee=max_fee,
            flow_fast_decay=_clamp(self.flow_fast_decay, 0.0, 0.999),
            flow_slow_decay=_clamp(self.flow_slow_decay, 0.0, 0.999),
            size_fast_decay=_clamp(self.size_fast_decay, 0.0, 0.999),
            size_slow_decay=_clamp(self.size_slow_decay, 0.0, 0.999),
            gap_fast_decay=_clamp(self.gap_fast_decay, 0.0, 0.999),
            gap_slow_decay=_clamp(self.gap_slow_decay, 0.0, 0.999),
            toxicity_decay=_clamp(self.toxicity_decay, 0.0, 0.999),
            toxicity_weight=_clamp(self.toxicity_weight, 0.0, 5.0),
            fair_fast_decay=_clamp(self.fair_fast_decay, 0.0, 0.999),
            fair_slow_decay=_clamp(self.fair_slow_decay, 0.0, 0.999),
            fair_size_weight=_clamp(self.fair_size_weight, 0.0, 8.0),
            spread_base=_clamp(self.spread_base, 0.0, 0.03),
            size_hinge_low=low,
            size_hinge_high=high,
            flow_hinge=_clamp(self.flow_hinge, 0.0, 0.05),
            tox_hinge=_clamp(self.tox_hinge, 0.0, 0.08),
            fair_hinge=_clamp(self.fair_hinge, 0.0, 0.02),
            gap_hinge=_clamp(self.gap_hinge, 0.5, 5.0),
            basis_size_low_w=_clamp(self.basis_size_low_w, 0.0, 0.2),
            basis_size_high_w=_clamp(self.basis_size_high_w, 0.0, 0.2),
            basis_flow_w=_clamp(self.basis_flow_w, 0.0, 0.2),
            basis_gap_w=_clamp(self.basis_gap_w, 0.0, 0.03),
            basis_tox_w=_clamp(self.basis_tox_w, 0.0, 0.2),
            basis_fair_w=_clamp(self.basis_fair_w, 0.0, 0.1),
            basis_streak_w=_clamp(self.basis_streak_w, 0.0, 0.03),
            skew_flow_w=_clamp(self.skew_flow_w, -0.2, 0.2),
            skew_inventory_w=_clamp(self.skew_inventory_w, -0.1, 0.1),
            skew_fair_w=_clamp(self.skew_fair_w, -0.1, 0.1),
            side_tox_w=_clamp(self.side_tox_w, 0.0, 0.2),
        )

    def to_dict(self) -> dict[str, float]:
        return dict(asdict(self.normalized()))


class SubmissionBasisStrategy(_SubmissionSafeBase):
    def __init__(self, params: SubmissionBasisParams | None = None) -> None:
        super().__init__()
        self.params = (params or SubmissionBasisParams()).normalized()

    def after_initialize(self, initial_x: float, initial_y: float) -> tuple[float, float]:
        self._initialize_state(initial_x, initial_y)
        return (self.params.base_fee, self.params.base_fee)

    def after_swap(self, trade: TradeInfo) -> tuple[float, float]:
        p = self.params
        spot, size_ratio = self._update_core(
            trade,
            fast_decay=p.flow_fast_decay,
            slow_decay=p.flow_slow_decay,
            size_fast_decay=p.size_fast_decay,
            size_slow_decay=p.size_slow_decay,
            gap_fast_decay=p.gap_fast_decay,
            gap_slow_decay=p.gap_slow_decay,
            toxicity_decay=p.toxicity_decay,
            toxicity_weight=p.toxicity_weight,
            fair_fast_decay=p.fair_fast_decay,
            fair_slow_decay=p.fair_slow_decay,
            fair_size_weight=p.fair_size_weight,
        )
        s = self.state
        flow_total = s.buy_flow_fast + s.sell_flow_fast
        flow_skew = (s.buy_flow_fast - s.sell_flow_fast) + 0.5 * (s.buy_flow_slow - s.sell_flow_slow)
        tox_total = s.tox_bid + s.tox_ask
        fair_gap = math.log(max(s.fair_fast, 1e-9) / max(spot, 1e-9)) - math.log(max(s.fair_slow, 1e-9) / max(spot, 1e-9))
        inventory = self._inventory_skew(float(trade.reserve_x), float(trade.reserve_y))
        size_basis = (
            p.basis_size_low_w * _hinge(size_ratio, p.size_hinge_low)
            + p.basis_size_high_w * _hinge(size_ratio, p.size_hinge_high)
        )
        flow_basis = p.basis_flow_w * _hinge(flow_total, p.flow_hinge)
        gap_basis = p.basis_gap_w * _hinge(p.gap_hinge - s.gap_fast, 0.0)
        tox_basis = p.basis_tox_w * _hinge(tox_total, p.tox_hinge)
        fair_basis = p.basis_fair_w * _hinge(abs(fair_gap), p.fair_hinge)
        streak_basis = p.basis_streak_w * max(s.streak_len - 1.0, 0.0)
        mid = p.base_fee + size_basis + flow_basis + gap_basis + tox_basis + fair_basis + streak_basis
        spread = p.spread_base + 0.5 * size_basis + 0.5 * tox_basis + gap_basis
        skew = p.skew_flow_w * flow_skew + p.skew_inventory_w * inventory + p.skew_fair_w * fair_gap
        bid = _clamp(mid + 0.5 * spread - skew + p.side_tox_w * s.tox_bid, p.min_fee, p.max_fee)
        ask = _clamp(mid + 0.5 * spread + skew + p.side_tox_w * s.tox_ask, p.min_fee, p.max_fee)
        return (bid, ask)

