"""Retail routing helpers for the diff simple-AMM rewrite."""

from __future__ import annotations

import math

from . import amm
from arena_eval.diff_simple_amm.policies import DiffSimpleAMMPolicy
from arena_eval.diff_simple_amm.types import AMMState, PolicyState, RetailOrder, TradeEvent


MIN_AMOUNT = 0.0001


def split_buy_two_amms(amm1: AMMState, amm2: AMMState, total_y: float) -> tuple[float, float]:
    """Exact challenge retail split for buy orders."""

    gamma1 = 1.0 - amm1.ask_fee
    gamma2 = 1.0 - amm2.ask_fee
    a1 = math.sqrt(amm1.reserve_x * gamma1 * amm1.reserve_y)
    a2 = math.sqrt(amm2.reserve_x * gamma2 * amm2.reserve_y)
    if a2 == 0.0:
        return (total_y, 0.0)
    r = a1 / a2
    numerator = r * (amm2.reserve_y + gamma2 * total_y) - amm1.reserve_y
    denominator = gamma1 + r * gamma2
    y1_amount = total_y / 2.0 if denominator == 0.0 else numerator / denominator
    y1_amount = max(0.0, min(total_y, y1_amount))
    return (y1_amount, total_y - y1_amount)


def split_sell_two_amms(amm1: AMMState, amm2: AMMState, total_x: float) -> tuple[float, float]:
    """Exact challenge retail split for sell orders."""

    gamma1 = 1.0 - amm1.bid_fee
    gamma2 = 1.0 - amm2.bid_fee
    b1 = math.sqrt(amm1.reserve_y * gamma1 * amm1.reserve_x)
    b2 = math.sqrt(amm2.reserve_y * gamma2 * amm2.reserve_x)
    if b2 == 0.0:
        return (total_x, 0.0)
    r = b1 / b2
    numerator = r * (amm2.reserve_x + gamma2 * total_x) - amm1.reserve_x
    denominator = gamma1 + r * gamma2
    x1_amount = total_x / 2.0 if denominator == 0.0 else numerator / denominator
    x1_amount = max(0.0, min(total_x, x1_amount))
    return (x1_amount, total_x - x1_amount)


def route_orders(
    *,
    orders: tuple[RetailOrder, ...],
    submission: AMMState,
    submission_policy: DiffSimpleAMMPolicy,
    submission_policy_state: PolicyState,
    normalizer: AMMState,
    normalizer_policy: DiffSimpleAMMPolicy,
    normalizer_policy_state: PolicyState,
    fair_price: float,
    timestamp: int,
) -> tuple[AMMState, PolicyState, AMMState, PolicyState, tuple[TradeEvent, ...]]:
    """Route retail orders across the two AMMs using exact challenge formulas."""

    trades: list[TradeEvent] = []
    current_submission = submission
    current_submission_policy_state = submission_policy_state
    current_normalizer = normalizer
    current_normalizer_policy_state = normalizer_policy_state

    for order in orders:
        if order.side == "buy":
            y_submission, y_normalizer = split_buy_two_amms(current_submission, current_normalizer, order.size)
            if y_submission > MIN_AMOUNT:
                current_submission, current_submission_policy_state, trade = amm.execute_buy_x_with_y(
                    amm_name="submission",
                    state=current_submission,
                    policy=submission_policy,
                    policy_state=current_submission_policy_state,
                    amount_y=y_submission,
                    timestamp=timestamp,
                    source="retail",
                )
                if trade is not None:
                    trades.append(trade)
            if y_normalizer > MIN_AMOUNT:
                current_normalizer, current_normalizer_policy_state, trade = amm.execute_buy_x_with_y(
                    amm_name="normalizer",
                    state=current_normalizer,
                    policy=normalizer_policy,
                    policy_state=current_normalizer_policy_state,
                    amount_y=y_normalizer,
                    timestamp=timestamp,
                    source="retail",
                )
                if trade is not None:
                    trades.append(trade)
            continue

        total_x = order.size / fair_price
        x_submission, x_normalizer = split_sell_two_amms(current_submission, current_normalizer, total_x)
        if x_submission > MIN_AMOUNT:
            current_submission, current_submission_policy_state, trade = amm.execute_buy_x(
                amm_name="submission",
                state=current_submission,
                policy=submission_policy,
                policy_state=current_submission_policy_state,
                amount_x=x_submission,
                timestamp=timestamp,
                source="retail",
            )
            if trade is not None:
                trades.append(trade)
        if x_normalizer > MIN_AMOUNT:
            current_normalizer, current_normalizer_policy_state, trade = amm.execute_buy_x(
                amm_name="normalizer",
                state=current_normalizer,
                policy=normalizer_policy,
                policy_state=current_normalizer_policy_state,
                amount_x=x_normalizer,
                timestamp=timestamp,
                source="retail",
            )
            if trade is not None:
                trades.append(trade)

    return (
        current_submission,
        current_submission_policy_state,
        current_normalizer,
        current_normalizer_policy_state,
        tuple(trades),
    )
