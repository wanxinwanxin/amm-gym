"""Challenge-tape simulator with clairvoyant controller hooks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Iterable, Protocol

import numpy as np

from arena_eval.core.types import BatchResult, SimulationResult
from arena_eval.diff_simple_amm.challenge_dynamics import build_challenge_tape
from arena_eval.diff_simple_amm.orders import decode_challenge_orders
from arena_eval.diff_simple_amm.router import MIN_AMOUNT, split_buy_two_amms, split_sell_two_amms
from arena_eval.diff_simple_amm.types import AMMState, ChallengeTape, RetailOrder
from arena_eval.exact_simple_amm.config import ExactSimpleAMMConfig

DEFAULT_BASELINE_FEE = 0.003
MAX_FEE = 0.1


def _clamp_fee(value: float) -> float:
    return float(min(MAX_FEE, max(0.0, value)))


def _set_fees(state: AMMState, *, bid_fee: float | None = None, ask_fee: float | None = None) -> AMMState:
    return AMMState(
        reserve_x=state.reserve_x,
        reserve_y=state.reserve_y,
        bid_fee=state.bid_fee if bid_fee is None else _clamp_fee(bid_fee),
        ask_fee=state.ask_fee if ask_fee is None else _clamp_fee(ask_fee),
        accumulated_fees_x=state.accumulated_fees_x,
        accumulated_fees_y=state.accumulated_fees_y,
    )


def _spot_price(state: AMMState) -> float:
    if state.reserve_x <= 0.0:
        return 0.0
    return state.reserve_y / state.reserve_x


def _invariant(state: AMMState) -> float:
    return state.reserve_x * state.reserve_y


def _quote_buy_x(state: AMMState, amount_x: float) -> tuple[float, float]:
    if amount_x <= 0.0:
        return (0.0, 0.0)
    gamma = 1.0 - state.bid_fee
    if gamma <= 0.0:
        return (0.0, 0.0)
    net_x = amount_x * gamma
    new_rx = state.reserve_x + net_x
    new_ry = _invariant(state) / new_rx
    y_out = state.reserve_y - new_ry
    return (y_out, amount_x * state.bid_fee) if y_out > 0.0 else (0.0, 0.0)


def _quote_sell_x(state: AMMState, amount_x: float) -> tuple[float, float]:
    if amount_x <= 0.0 or amount_x >= state.reserve_x:
        return (0.0, 0.0)
    gamma = 1.0 - state.ask_fee
    if gamma <= 0.0:
        return (0.0, 0.0)
    new_rx = state.reserve_x - amount_x
    new_ry = _invariant(state) / new_rx
    net_y = new_ry - state.reserve_y
    if net_y <= 0.0:
        return (0.0, 0.0)
    total_y = net_y / gamma
    return (total_y, total_y - net_y)


def _quote_x_for_y(state: AMMState, amount_y: float) -> tuple[float, float]:
    if amount_y <= 0.0:
        return (0.0, 0.0)
    gamma = 1.0 - state.ask_fee
    if gamma <= 0.0:
        return (0.0, 0.0)
    net_y = amount_y * gamma
    new_ry = state.reserve_y + net_y
    new_rx = _invariant(state) / new_ry
    x_out = state.reserve_x - new_rx
    return (x_out, amount_y * state.ask_fee) if x_out > 0.0 else (0.0, 0.0)


def _execute_buy_x(state: AMMState, amount_x: float) -> tuple[AMMState, float]:
    amount_y, fee_amount = _quote_buy_x(state, amount_x)
    if amount_y <= 0.0:
        return (state, 0.0)
    net_x = amount_x - fee_amount
    return (
        AMMState(
            reserve_x=state.reserve_x + net_x,
            reserve_y=state.reserve_y - amount_y,
            bid_fee=state.bid_fee,
            ask_fee=state.ask_fee,
            accumulated_fees_x=state.accumulated_fees_x + fee_amount,
            accumulated_fees_y=state.accumulated_fees_y,
        ),
        amount_y,
    )


def _execute_sell_x(state: AMMState, amount_x: float) -> tuple[AMMState, float]:
    amount_y, fee_amount = _quote_sell_x(state, amount_x)
    if amount_y <= 0.0:
        return (state, 0.0)
    net_y = amount_y - fee_amount
    return (
        AMMState(
            reserve_x=state.reserve_x - amount_x,
            reserve_y=state.reserve_y + net_y,
            bid_fee=state.bid_fee,
            ask_fee=state.ask_fee,
            accumulated_fees_x=state.accumulated_fees_x,
            accumulated_fees_y=state.accumulated_fees_y + fee_amount,
        ),
        amount_y,
    )


def _execute_buy_x_with_y(state: AMMState, amount_y: float) -> tuple[AMMState, float]:
    amount_x, fee_amount = _quote_x_for_y(state, amount_y)
    if amount_x <= 0.0:
        return (state, 0.0)
    net_y = amount_y - fee_amount
    return (
        AMMState(
            reserve_x=state.reserve_x - amount_x,
            reserve_y=state.reserve_y + net_y,
            bid_fee=state.bid_fee,
            ask_fee=state.ask_fee,
            accumulated_fees_x=state.accumulated_fees_x,
            accumulated_fees_y=state.accumulated_fees_y + fee_amount,
        ),
        amount_x,
    )


def _execute_arb(state: AMMState, fair_price: float) -> tuple[AMMState, float, float]:
    spot = _spot_price(state)
    if fair_price <= 0.0 or spot <= 0.0:
        return (state, 0.0, 0.0)
    if spot < fair_price:
        gamma = 1.0 - state.ask_fee
        if gamma <= 0.0:
            return (state, 0.0, 0.0)
        amount_x = state.reserve_x - math.sqrt(_invariant(state) / (gamma * fair_price))
        if amount_x <= 0.0:
            return (state, 0.0, 0.0)
        amount_x = min(amount_x, state.reserve_x * 0.99)
        total_y, _ = _quote_sell_x(state, amount_x)
        if total_y <= 0.0:
            return (state, 0.0, 0.0)
        profit = amount_x * fair_price - total_y
        if profit <= 0.0:
            return (state, 0.0, 0.0)
        next_state, amount_y = _execute_sell_x(state, amount_x)
        return (next_state, profit, amount_y)
    gamma = 1.0 - state.bid_fee
    if gamma <= 0.0:
        return (state, 0.0, 0.0)
    amount_x = (math.sqrt(_invariant(state) * gamma / fair_price) - state.reserve_x) / gamma
    if amount_x <= 0.0:
        return (state, 0.0, 0.0)
    amount_y, _ = _quote_buy_x(state, amount_x)
    if amount_y <= 0.0:
        return (state, 0.0, 0.0)
    profit = amount_y - amount_x * fair_price
    if profit <= 0.0:
        return (state, 0.0, 0.0)
    next_state, trade_amount_y = _execute_buy_x(state, amount_x)
    return (next_state, profit, trade_amount_y)


def _route_one_order(
    submission: AMMState,
    normalizer: AMMState,
    fair_price: float,
    order: RetailOrder,
) -> tuple[AMMState, AMMState, float, float, float, float]:
    submission_edge = 0.0
    normalizer_edge = 0.0
    submission_volume_y = 0.0
    normalizer_volume_y = 0.0

    if order.side == "buy":
        y_submission, y_normalizer = split_buy_two_amms(submission, normalizer, order.size)
        if y_submission > MIN_AMOUNT:
            submission, amount_x = _execute_buy_x_with_y(submission, y_submission)
            if amount_x > 0.0:
                submission_edge += y_submission - amount_x * fair_price
                submission_volume_y += y_submission
        if y_normalizer > MIN_AMOUNT:
            normalizer, amount_x = _execute_buy_x_with_y(normalizer, y_normalizer)
            if amount_x > 0.0:
                normalizer_edge += y_normalizer - amount_x * fair_price
                normalizer_volume_y += y_normalizer
        return (
            submission,
            normalizer,
            submission_edge,
            normalizer_edge,
            submission_volume_y,
            normalizer_volume_y,
        )

    total_x = order.size / max(fair_price, 1e-12)
    x_submission, x_normalizer = split_sell_two_amms(submission, normalizer, total_x)
    if x_submission > MIN_AMOUNT:
        submission, amount_y = _execute_buy_x(submission, x_submission)
        if amount_y > 0.0:
            submission_edge += x_submission * fair_price - amount_y
            submission_volume_y += amount_y
    if x_normalizer > MIN_AMOUNT:
        normalizer, amount_y = _execute_buy_x(normalizer, x_normalizer)
        if amount_y > 0.0:
            normalizer_edge += x_normalizer * fair_price - amount_y
            normalizer_volume_y += amount_y
    return (
        submission,
        normalizer,
        submission_edge,
        normalizer_edge,
        submission_volume_y,
        normalizer_volume_y,
    )


def _build_challenge_price_path(config: ExactSimpleAMMConfig, tape: ChallengeTape) -> tuple[float, ...]:
    drift = (config.gbm_mu - 0.5 * config.gbm_sigma * config.gbm_sigma) * config.gbm_dt
    vol = config.gbm_sigma * math.sqrt(config.gbm_dt)
    price = float(config.initial_price)
    path: list[float] = []
    for normal in tape.gbm_normals:
        price *= math.exp(drift + vol * float(normal))
        path.append(price)
    return tuple(path)


@dataclass(frozen=True)
class OracleContext:
    event: str
    step: int
    fair_price: float
    submission: AMMState
    normalizer: AMMState
    tape: ChallengeTape
    config: ExactSimpleAMMConfig
    current_order_index: int = -1
    current_order: RetailOrder | None = None
    upcoming_orders: tuple[RetailOrder, ...] = ()


class ClairvoyantController(Protocol):
    def initialize(self, *, config: ExactSimpleAMMConfig, tape: ChallengeTape) -> tuple[float, float]:
        ...

    def choose_fees(self, context: OracleContext) -> tuple[float, float]:
        ...


@dataclass(frozen=True)
class FixedFeeClairvoyantController:
    bid_fee: float = DEFAULT_BASELINE_FEE
    ask_fee: float = DEFAULT_BASELINE_FEE

    def initialize(self, *, config: ExactSimpleAMMConfig, tape: ChallengeTape) -> tuple[float, float]:
        del config, tape
        return (self.bid_fee, self.ask_fee)

    def choose_fees(self, context: OracleContext) -> tuple[float, float]:
        del context
        return (self.bid_fee, self.ask_fee)


@dataclass(frozen=True)
class GreedyStepOracleController:
    """Strong clairvoyant benchmark with same-step lookahead only."""

    fee_grid_size: int = 41
    fallback_fee: float = DEFAULT_BASELINE_FEE

    def initialize(self, *, config: ExactSimpleAMMConfig, tape: ChallengeTape) -> tuple[float, float]:
        del config, tape
        return (self.fallback_fee, self.fallback_fee)

    def choose_fees(self, context: OracleContext) -> tuple[float, float]:
        if context.event == "pre_step":
            return self._choose_pre_step_fees(context)
        if context.event == "pre_order" and context.current_order is not None:
            return self._choose_pre_order_fees(context)
        return (context.submission.bid_fee, context.submission.ask_fee)

    def _fee_grid(self) -> tuple[float, ...]:
        points = max(int(self.fee_grid_size), 2)
        return tuple(MAX_FEE * index / float(points - 1) for index in range(points))

    def _choose_pre_step_fees(self, context: OracleContext) -> tuple[float, float]:
        spot = _spot_price(context.submission)
        if context.fair_price <= 0.0 or spot <= 0.0 or math.isclose(spot, context.fair_price):
            return (context.submission.bid_fee, context.submission.ask_fee)
        if spot < context.fair_price:
            return self._search_step_side(context, side="ask")
        return self._search_step_side(context, side="bid")

    def _search_step_side(self, context: OracleContext, *, side: str) -> tuple[float, float]:
        best_score = float("-inf")
        best_fees = (context.submission.bid_fee, context.submission.ask_fee)
        current_side_fee = context.submission.bid_fee if side == "bid" else context.submission.ask_fee
        for fee in self._fee_grid():
            candidate = _set_fees(
                context.submission,
                bid_fee=fee if side == "bid" else context.submission.bid_fee,
                ask_fee=fee if side == "ask" else context.submission.ask_fee,
            )
            candidate, arb_profit, _ = _execute_arb(candidate, context.fair_price)
            normalizer_after_arb, _normalizer_arb_profit, _normalizer_arb_y = _execute_arb(
                context.normalizer,
                context.fair_price,
            )
            tail_edge = self._simulate_greedy_tail(
                submission=candidate,
                normalizer=normalizer_after_arb,
                fair_price=context.fair_price,
                orders=context.upcoming_orders,
            )
            score = -arb_profit + tail_edge
            best_side_fee = best_fees[0] if side == "bid" else best_fees[1]
            if score > best_score or (
                math.isclose(score, best_score, rel_tol=1e-12, abs_tol=1e-12)
                and abs(fee - current_side_fee) < abs(best_side_fee - current_side_fee)
            ):
                best_score = score
                if side == "bid":
                    best_fees = (fee, context.submission.ask_fee)
                else:
                    best_fees = (context.submission.bid_fee, fee)
        return best_fees

    def _simulate_greedy_tail(
        self,
        *,
        submission: AMMState,
        normalizer: AMMState,
        fair_price: float,
        orders: tuple[RetailOrder, ...],
    ) -> float:
        score = 0.0
        current_submission = submission
        current_normalizer = normalizer
        for order in orders:
            best_fees = self._best_order_fees(
                submission=current_submission,
                normalizer=current_normalizer,
                fair_price=fair_price,
                order=order,
            )
            current_submission = _set_fees(
                current_submission,
                bid_fee=best_fees[0],
                ask_fee=best_fees[1],
            )
            (
                current_submission,
                current_normalizer,
                submission_edge,
                _normalizer_edge,
                _submission_volume_y,
                _normalizer_volume_y,
            ) = _route_one_order(current_submission, current_normalizer, fair_price, order)
            score += submission_edge
        return score

    def _best_order_fees(
        self,
        *,
        submission: AMMState,
        normalizer: AMMState,
        fair_price: float,
        order: RetailOrder,
    ) -> tuple[float, float]:
        best_score = float("-inf")
        best_fees = (submission.bid_fee, submission.ask_fee)
        current_side_fee = submission.bid_fee if order.side == "sell" else submission.ask_fee
        for fee in self._fee_grid():
            candidate = _set_fees(
                submission,
                bid_fee=fee if order.side == "sell" else submission.bid_fee,
                ask_fee=fee if order.side == "buy" else submission.ask_fee,
            )
            _sub, _nor, submission_edge, _normalizer_edge, _sub_vol, _nor_vol = _route_one_order(
                candidate,
                normalizer,
                fair_price,
                order,
            )
            best_side_fee = best_fees[0] if order.side == "sell" else best_fees[1]
            if submission_edge > best_score or (
                math.isclose(submission_edge, best_score, rel_tol=1e-12, abs_tol=1e-12)
                and abs(fee - current_side_fee) < abs(best_side_fee - current_side_fee)
            ):
                best_score = submission_edge
                if order.side == "sell":
                    best_fees = (fee, submission.ask_fee)
                else:
                    best_fees = (submission.bid_fee, fee)
        return best_fees

    def _choose_pre_order_fees(self, context: OracleContext) -> tuple[float, float]:
        if context.current_order is None:
            return (context.submission.bid_fee, context.submission.ask_fee)
        return self._best_order_fees(
            submission=context.submission,
            normalizer=context.normalizer,
            fair_price=context.fair_price,
            order=context.current_order,
        )


@dataclass(frozen=True)
class StructuredRetailOracleController:
    """Closed-form arb suppression plus 1D retail-side search."""

    fee_grid_size: int = 1001
    fallback_fee: float = DEFAULT_BASELINE_FEE

    def __post_init__(self) -> None:
        points = max(int(self.fee_grid_size), 2)
        object.__setattr__(self, "_fee_grid_values", np.linspace(0.0, MAX_FEE, num=points, dtype=float))

    def initialize(self, *, config: ExactSimpleAMMConfig, tape: ChallengeTape) -> tuple[float, float]:
        del config, tape
        return (self.fallback_fee, self.fallback_fee)

    def choose_fees(self, context: OracleContext) -> tuple[float, float]:
        if context.event == "pre_step":
            return self._choose_pre_step_fees(context)
        if context.event == "pre_order" and context.current_order is not None:
            return self._choose_pre_order_fees(context)
        return (context.submission.bid_fee, context.submission.ask_fee)

    def _choose_pre_step_fees(self, context: OracleContext) -> tuple[float, float]:
        bid_fee = context.submission.bid_fee
        ask_fee = context.submission.ask_fee
        spot = _spot_price(context.submission)
        fair_price = context.fair_price
        if fair_price <= 0.0 or spot <= 0.0:
            return (bid_fee, ask_fee)
        if spot < fair_price:
            ask_fee = _clamp_fee(1.0 - spot / fair_price)
        elif spot > fair_price:
            bid_fee = _clamp_fee(1.0 - fair_price / spot)
        return (bid_fee, ask_fee)

    def _choose_pre_order_fees(self, context: OracleContext) -> tuple[float, float]:
        order = context.current_order
        if order is None:
            return (context.submission.bid_fee, context.submission.ask_fee)
        if order.side == "buy":
            best_ask = self._best_buy_fee(
                submission=context.submission,
                normalizer=context.normalizer,
                fair_price=context.fair_price,
                total_y=order.size,
            )
            return (context.submission.bid_fee, best_ask)
        best_bid = self._best_sell_fee(
            submission=context.submission,
            normalizer=context.normalizer,
            fair_price=context.fair_price,
            total_y_notional=order.size,
        )
        return (best_bid, context.submission.ask_fee)

    def _best_buy_fee(
        self,
        *,
        submission: AMMState,
        normalizer: AMMState,
        fair_price: float,
        total_y: float,
    ) -> float:
        if total_y <= 0.0 or fair_price <= 0.0:
            return submission.ask_fee
        fees = self._fee_grid_values
        gamma_s = 1.0 - fees
        gamma_n = 1.0 - normalizer.ask_fee
        if gamma_n <= 0.0:
            y_submission = np.full_like(fees, float(total_y))
        else:
            a1 = np.sqrt(submission.reserve_x * submission.reserve_y * gamma_s)
            a2 = math.sqrt(normalizer.reserve_x * normalizer.reserve_y * gamma_n)
            r = a1 / max(a2, 1e-12)
            numerator = r * (normalizer.reserve_y + gamma_n * total_y) - submission.reserve_y
            denominator = gamma_s + r * gamma_n
            with np.errstate(divide="ignore", invalid="ignore"):
                y_submission = np.where(denominator == 0.0, total_y / 2.0, numerator / denominator)
            y_submission = np.clip(y_submission, 0.0, total_y)
        net_y = gamma_s * y_submission
        new_ry = submission.reserve_y + net_y
        k = submission.reserve_x * submission.reserve_y
        with np.errstate(divide="ignore", invalid="ignore"):
            x_out = submission.reserve_x - k / new_ry
        edge = y_submission - fair_price * x_out
        invalid = (gamma_s <= 0.0) | (y_submission <= MIN_AMOUNT) | (new_ry <= 0.0) | ~np.isfinite(edge)
        edge = np.where(invalid, 0.0, edge)
        return self._best_fee_from_scores(fees=fees, scores=edge, current_fee=submission.ask_fee)

    def _best_sell_fee(
        self,
        *,
        submission: AMMState,
        normalizer: AMMState,
        fair_price: float,
        total_y_notional: float,
    ) -> float:
        if total_y_notional <= 0.0 or fair_price <= 0.0:
            return submission.bid_fee
        total_x = total_y_notional / max(fair_price, 1e-12)
        fees = self._fee_grid_values
        gamma_s = 1.0 - fees
        gamma_n = 1.0 - normalizer.bid_fee
        if gamma_n <= 0.0:
            x_submission = np.full_like(fees, float(total_x))
        else:
            b1 = np.sqrt(submission.reserve_x * submission.reserve_y * gamma_s)
            b2 = math.sqrt(normalizer.reserve_x * normalizer.reserve_y * gamma_n)
            r = b1 / max(b2, 1e-12)
            numerator = r * (normalizer.reserve_x + gamma_n * total_x) - submission.reserve_x
            denominator = gamma_s + r * gamma_n
            with np.errstate(divide="ignore", invalid="ignore"):
                x_submission = np.where(denominator == 0.0, total_x / 2.0, numerator / denominator)
            x_submission = np.clip(x_submission, 0.0, total_x)
        net_x = gamma_s * x_submission
        new_rx = submission.reserve_x + net_x
        k = submission.reserve_x * submission.reserve_y
        with np.errstate(divide="ignore", invalid="ignore"):
            y_out = submission.reserve_y - k / new_rx
        edge = x_submission * fair_price - y_out
        invalid = (gamma_s <= 0.0) | (x_submission <= MIN_AMOUNT) | (new_rx <= 0.0) | ~np.isfinite(edge)
        edge = np.where(invalid, 0.0, edge)
        return self._best_fee_from_scores(fees=fees, scores=edge, current_fee=submission.bid_fee)

    @staticmethod
    def _best_fee_from_scores(*, fees: np.ndarray, scores: np.ndarray, current_fee: float) -> float:
        best_score = float(np.max(scores))
        best_mask = np.isclose(scores, best_score, rtol=1e-12, atol=1e-12)
        candidate_fees = fees[best_mask]
        if candidate_fees.size == 0:
            return float(current_fee)
        distance = np.abs(candidate_fees - current_fee)
        return float(candidate_fees[int(np.argmin(distance))])


def run_clairvoyant_seed(
    controller: ClairvoyantController,
    seed: int,
    *,
    config: ExactSimpleAMMConfig | None = None,
    normalizer_fee: float = DEFAULT_BASELINE_FEE,
) -> SimulationResult:
    exact_config = config or ExactSimpleAMMConfig.from_seed(seed)
    if exact_config.evaluator_kind != "challenge":
        raise ValueError("clairvoyant oracle currently supports challenge-mode tapes only")
    tape = build_challenge_tape(config=exact_config, seed=seed)
    price_path = _build_challenge_price_path(exact_config, tape)
    bid_fee, ask_fee = controller.initialize(config=exact_config, tape=tape)
    submission = AMMState(
        reserve_x=exact_config.submission_initial_x,
        reserve_y=exact_config.submission_initial_y,
        bid_fee=_clamp_fee(bid_fee),
        ask_fee=_clamp_fee(ask_fee),
    )
    normalizer = AMMState(
        reserve_x=exact_config.normalizer_initial_x,
        reserve_y=exact_config.normalizer_initial_y,
        bid_fee=_clamp_fee(normalizer_fee),
        ask_fee=_clamp_fee(normalizer_fee),
    )

    edge_submission = 0.0
    edge_normalizer = 0.0
    retail_edge_submission = 0.0
    retail_edge_normalizer = 0.0
    arb_loss_submission = 0.0
    arb_loss_normalizer = 0.0
    retail_volume_submission_y = 0.0
    retail_volume_normalizer_y = 0.0
    arb_volume_submission_y = 0.0
    arb_volume_normalizer_y = 0.0
    bid_fee_submission_sum = 0.0
    ask_fee_submission_sum = 0.0
    bid_fee_normalizer_sum = 0.0
    ask_fee_normalizer_sum = 0.0

    for step, fair_price in enumerate(price_path):
        orders = decode_challenge_orders(config=exact_config, tape=tape, step=step)
        pre_step_fees = controller.choose_fees(
            OracleContext(
                event="pre_step",
                step=step,
                fair_price=fair_price,
                submission=submission,
                normalizer=normalizer,
                tape=tape,
                config=exact_config,
                upcoming_orders=orders,
            )
        )
        submission = _set_fees(submission, bid_fee=pre_step_fees[0], ask_fee=pre_step_fees[1])

        submission, arb_profit, submission_arb_y = _execute_arb(submission, fair_price)
        arb_loss_submission += arb_profit
        edge_submission -= arb_profit
        arb_volume_submission_y += submission_arb_y

        normalizer, normalizer_arb_profit, normalizer_arb_y = _execute_arb(normalizer, fair_price)
        arb_loss_normalizer += normalizer_arb_profit
        edge_normalizer -= normalizer_arb_profit
        arb_volume_normalizer_y += normalizer_arb_y

        for order_index, order in enumerate(orders):
            bid_fee, ask_fee = controller.choose_fees(
                OracleContext(
                    event="pre_order",
                    step=step,
                    fair_price=fair_price,
                    submission=submission,
                    normalizer=normalizer,
                    tape=tape,
                    config=exact_config,
                    current_order_index=order_index,
                    current_order=order,
                    upcoming_orders=orders[order_index:],
                )
            )
            submission = _set_fees(submission, bid_fee=bid_fee, ask_fee=ask_fee)
            (
                submission,
                normalizer,
                submission_edge,
                normalizer_edge,
                submission_volume_y,
                normalizer_volume_y,
            ) = _route_one_order(submission, normalizer, fair_price, order)
            retail_edge_submission += submission_edge
            retail_edge_normalizer += normalizer_edge
            edge_submission += submission_edge
            edge_normalizer += normalizer_edge
            retail_volume_submission_y += submission_volume_y
            retail_volume_normalizer_y += normalizer_volume_y

        bid_fee_submission_sum += submission.bid_fee
        ask_fee_submission_sum += submission.ask_fee
        bid_fee_normalizer_sum += normalizer.bid_fee
        ask_fee_normalizer_sum += normalizer.ask_fee

    submission_initial_value = exact_config.submission_initial_value
    normalizer_initial_value = exact_config.normalizer_initial_value
    episode_seconds = float(exact_config.n_steps) * float(exact_config.step_seconds)
    terminal_fair = price_path[-1] if price_path else exact_config.initial_price
    pnl_submission = _mark_to_market(submission, terminal_fair) - submission_initial_value
    pnl_normalizer = _mark_to_market(normalizer, terminal_fair) - normalizer_initial_value
    steps = max(exact_config.n_steps, 1)
    return SimulationResult(
        seed=seed,
        edge_submission=edge_submission,
        edge_normalizer=edge_normalizer,
        pnl_submission=pnl_submission,
        pnl_normalizer=pnl_normalizer,
        score=edge_submission,
        retail_volume_submission_y=retail_volume_submission_y,
        retail_volume_normalizer_y=retail_volume_normalizer_y,
        arb_volume_submission_y=arb_volume_submission_y,
        arb_volume_normalizer_y=arb_volume_normalizer_y,
        average_bid_fee_submission=bid_fee_submission_sum / steps,
        average_ask_fee_submission=ask_fee_submission_sum / steps,
        average_bid_fee_normalizer=bid_fee_normalizer_sum / steps,
        average_ask_fee_normalizer=ask_fee_normalizer_sum / steps,
        retail_edge_submission=retail_edge_submission,
        retail_edge_normalizer=retail_edge_normalizer,
        arb_loss_submission=arb_loss_submission,
        arb_loss_normalizer=arb_loss_normalizer,
        initial_value=submission_initial_value,
        initial_value_normalizer=normalizer_initial_value,
        episode_seconds=episode_seconds,
    )


def run_clairvoyant_batch(
    controller_factory,
    seeds: Iterable[int],
    *,
    normalizer_fee: float = DEFAULT_BASELINE_FEE,
    config_factory=None,
) -> BatchResult:
    seed_tuple = tuple(int(seed) for seed in seeds)
    simulations = tuple(
        run_clairvoyant_seed(
            controller_factory(),
            seed,
            config=config_factory(seed) if config_factory is not None else None,
            normalizer_fee=normalizer_fee,
        )
        for seed in seed_tuple
    )
    if not simulations:
        raise ValueError("at least one seed is required")
    return BatchResult(
        seeds=seed_tuple,
        simulations=simulations,
        score=mean(sim.score for sim in simulations),
        edge_mean_submission=mean(sim.edge_submission for sim in simulations),
        edge_mean_normalizer=mean(sim.edge_normalizer for sim in simulations),
        edge_advantage_mean=mean(sim.edge_advantage for sim in simulations),
        pnl_mean_submission=mean(sim.pnl_submission for sim in simulations),
        pnl_mean_normalizer=mean(sim.pnl_normalizer for sim in simulations),
        pnl_advantage_mean=mean(sim.pnl_advantage for sim in simulations),
        retail_edge_mean_submission=mean(sim.retail_edge_submission for sim in simulations),
        retail_edge_mean_normalizer=mean(sim.retail_edge_normalizer for sim in simulations),
        arb_loss_mean_submission=mean(sim.arb_loss_submission for sim in simulations),
        arb_loss_mean_normalizer=mean(sim.arb_loss_normalizer for sim in simulations),
        retail_volume_mean_submission_y=mean(sim.retail_volume_submission_y for sim in simulations),
        retail_volume_mean_normalizer_y=mean(sim.retail_volume_normalizer_y for sim in simulations),
        arb_volume_mean_submission_y=mean(sim.arb_volume_submission_y for sim in simulations),
        arb_volume_mean_normalizer_y=mean(sim.arb_volume_normalizer_y for sim in simulations),
        initial_value_mean=mean(sim.initial_value for sim in simulations),
        initial_value_mean_normalizer=mean(sim.initial_value_normalizer for sim in simulations),
        episode_seconds_mean=mean(sim.episode_seconds for sim in simulations),
        metadata={
            "normalizer_fee": normalizer_fee,
            "controller": controller_factory().__class__.__name__,
            "submission_liquidity_fraction": mean(sim.initial_value / sim.initial_value_normalizer for sim in simulations),
        },
    )


def _mark_to_market(state: AMMState, fair_price: float) -> float:
    reserve_value = state.reserve_x * fair_price + state.reserve_y
    fee_value = state.accumulated_fees_x * fair_price + state.accumulated_fees_y
    return reserve_value + fee_value
