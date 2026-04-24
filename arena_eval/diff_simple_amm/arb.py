"""Arbitrage helpers for the diff simple-AMM rewrite."""

from __future__ import annotations

import math

from . import amm
from arena_eval.diff_simple_amm.policies import DiffSimpleAMMPolicy
from arena_eval.diff_simple_amm.types import AMMState, PolicyState, TradeEvent


def execute_arb(
    *,
    amm_name: str,
    state: AMMState,
    policy: DiffSimpleAMMPolicy,
    policy_state: PolicyState,
    fair_price: float,
    timestamp: int,
) -> tuple[AMMState, PolicyState, TradeEvent | None, float]:
    """Execute exact challenge arbitrage against one AMM."""

    spot = amm.spot_price(state)
    if spot < fair_price:
        return _buy_arb(
            amm_name=amm_name,
            state=state,
            policy=policy,
            policy_state=policy_state,
            fair_price=fair_price,
            timestamp=timestamp,
        )
    if spot > fair_price:
        return _sell_arb(
            amm_name=amm_name,
            state=state,
            policy=policy,
            policy_state=policy_state,
            fair_price=fair_price,
            timestamp=timestamp,
        )
    return (state, policy_state, None, 0.0)


def _buy_arb(
    *,
    amm_name: str,
    state: AMMState,
    policy: DiffSimpleAMMPolicy,
    policy_state: PolicyState,
    fair_price: float,
    timestamp: int,
) -> tuple[AMMState, PolicyState, TradeEvent | None, float]:
    gamma = 1.0 - state.ask_fee
    if gamma <= 0.0 or fair_price <= 0.0:
        return (state, policy_state, None, 0.0)
    amount_x = state.reserve_x - math.sqrt((state.reserve_x * state.reserve_y) / (gamma * fair_price))
    if amount_x <= 0.0:
        return (state, policy_state, None, 0.0)
    amount_x = min(amount_x, state.reserve_x * 0.99)
    total_y, _ = amm.quote_sell_x(state, amount_x)
    if total_y <= 0.0:
        return (state, policy_state, None, 0.0)
    profit = amount_x * fair_price - total_y
    if profit <= 0.0:
        return (state, policy_state, None, 0.0)
    next_state, next_policy_state, trade = amm.execute_sell_x(
        amm_name=amm_name,
        state=state,
        policy=policy,
        policy_state=policy_state,
        amount_x=amount_x,
        timestamp=timestamp,
        source="arb",
    )
    if trade is None:
        return (state, policy_state, None, 0.0)
    return (next_state, next_policy_state, trade, profit)


def _sell_arb(
    *,
    amm_name: str,
    state: AMMState,
    policy: DiffSimpleAMMPolicy,
    policy_state: PolicyState,
    fair_price: float,
    timestamp: int,
) -> tuple[AMMState, PolicyState, TradeEvent | None, float]:
    gamma = 1.0 - state.bid_fee
    if gamma <= 0.0 or fair_price <= 0.0:
        return (state, policy_state, None, 0.0)
    amount_x = (math.sqrt((state.reserve_x * state.reserve_y) * gamma / fair_price) - state.reserve_x) / gamma
    if amount_x <= 0.0:
        return (state, policy_state, None, 0.0)
    amount_y, _ = amm.quote_buy_x(state, amount_x)
    if amount_y <= 0.0:
        return (state, policy_state, None, 0.0)
    profit = amount_y - amount_x * fair_price
    if profit <= 0.0:
        return (state, policy_state, None, 0.0)
    next_state, next_policy_state, trade = amm.execute_buy_x(
        amm_name=amm_name,
        state=state,
        policy=policy,
        policy_state=policy_state,
        amount_x=amount_x,
        timestamp=timestamp,
        source="arb",
    )
    if trade is None:
        return (state, policy_state, None, 0.0)
    return (next_state, next_policy_state, trade, profit)
