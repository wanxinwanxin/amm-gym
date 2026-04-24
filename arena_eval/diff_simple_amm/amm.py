"""Pure AMM math helpers for the diff simple-AMM rewrite."""

from __future__ import annotations

from arena_eval.diff_simple_amm.policies import DiffSimpleAMMPolicy
from arena_eval.diff_simple_amm.types import AMMState, PolicyOutput, PolicyState, TradeEvent


DEFAULT_BASELINE_FEE = 0.003
MAX_FEE = 0.1


def clamp_fee(value: float) -> float:
    """Clamp fees to the challenge bounds."""

    return float(min(MAX_FEE, max(0.0, value)))


def spot_price(state: AMMState) -> float:
    """Return the current spot price of the AMM."""

    if state.reserve_x == 0.0:
        return 0.0
    return state.reserve_y / state.reserve_x


def invariant(state: AMMState) -> float:
    """Return the constant-product invariant."""

    return state.reserve_x * state.reserve_y


def initialize_amm(
    *,
    reserve_x: float,
    reserve_y: float,
    policy: DiffSimpleAMMPolicy,
) -> tuple[AMMState, PolicyState]:
    """Initialize an AMM state and fees from a functional policy."""

    try:
        output = policy.initialize(reserve_x, reserve_y)
    except Exception:
        output = PolicyOutput(DEFAULT_BASELINE_FEE, DEFAULT_BASELINE_FEE, PolicyState())
    state = AMMState(
        reserve_x=float(reserve_x),
        reserve_y=float(reserve_y),
        bid_fee=clamp_fee(output.bid_fee),
        ask_fee=clamp_fee(output.ask_fee),
    )
    return (state, output.state)


def quote_buy_x(state: AMMState, amount_x: float) -> tuple[float, float]:
    """Quote a trade where the AMM buys X and pays out Y."""

    if amount_x <= 0.0:
        return (0.0, 0.0)
    gamma = max(0.0, min(1.0, 1.0 - state.bid_fee))
    if gamma <= 0.0:
        return (0.0, 0.0)
    net_x = amount_x * gamma
    new_rx = state.reserve_x + net_x
    new_ry = invariant(state) / new_rx
    y_out = state.reserve_y - new_ry
    if y_out > 0.0:
        return (y_out, amount_x * state.bid_fee)
    return (0.0, 0.0)


def quote_sell_x(state: AMMState, amount_x: float) -> tuple[float, float]:
    """Quote a trade where the AMM sells X and receives Y."""

    if amount_x <= 0.0 or amount_x >= state.reserve_x:
        return (0.0, 0.0)
    gamma = max(0.0, min(1.0, 1.0 - state.ask_fee))
    if gamma <= 0.0:
        return (0.0, 0.0)
    new_rx = state.reserve_x - amount_x
    new_ry = invariant(state) / new_rx
    net_y = new_ry - state.reserve_y
    if net_y <= 0.0:
        return (0.0, 0.0)
    total_y = net_y / gamma
    return (total_y, total_y - net_y)


def quote_x_for_y(state: AMMState, amount_y: float) -> tuple[float, float]:
    """Quote a trade where the AMM sells X against an input of Y."""

    if amount_y <= 0.0:
        return (0.0, 0.0)
    gamma = max(0.0, min(1.0, 1.0 - state.ask_fee))
    if gamma <= 0.0:
        return (0.0, 0.0)
    net_y = amount_y * gamma
    new_ry = state.reserve_y + net_y
    new_rx = invariant(state) / new_ry
    x_out = state.reserve_x - new_rx
    if x_out > 0.0:
        return (x_out, amount_y * state.ask_fee)
    return (0.0, 0.0)


def execute_buy_x(
    *,
    amm_name: str,
    state: AMMState,
    policy: DiffSimpleAMMPolicy,
    policy_state: PolicyState,
    amount_x: float,
    timestamp: int,
    source: str,
) -> tuple[AMMState, PolicyState, TradeEvent | None]:
    """Execute a trade where the AMM buys X and pays out Y."""

    amount_y, fee_amount = quote_buy_x(state, amount_x)
    if amount_y <= 0.0:
        return (state, policy_state, None)
    net_x = amount_x - fee_amount
    next_state = AMMState(
        reserve_x=state.reserve_x + net_x,
        reserve_y=state.reserve_y - amount_y,
        bid_fee=state.bid_fee,
        ask_fee=state.ask_fee,
        accumulated_fees_x=state.accumulated_fees_x + fee_amount,
        accumulated_fees_y=state.accumulated_fees_y,
    )
    trade = TradeEvent(
        venue=amm_name,
        source=source,
        is_buy=True,
        amount_x=float(amount_x),
        amount_y=float(amount_y),
        timestamp=int(timestamp),
        reserve_x=next_state.reserve_x,
        reserve_y=next_state.reserve_y,
    )
    final_state, next_policy_state = _after_event(
        state=next_state,
        policy=policy,
        policy_state=policy_state,
        trade=trade,
    )
    return (final_state, next_policy_state, trade)


def execute_sell_x(
    *,
    amm_name: str,
    state: AMMState,
    policy: DiffSimpleAMMPolicy,
    policy_state: PolicyState,
    amount_x: float,
    timestamp: int,
    source: str,
) -> tuple[AMMState, PolicyState, TradeEvent | None]:
    """Execute a trade where the AMM sells X and receives Y."""

    amount_y, fee_amount = quote_sell_x(state, amount_x)
    if amount_y <= 0.0:
        return (state, policy_state, None)
    net_y = amount_y - fee_amount
    next_state = AMMState(
        reserve_x=state.reserve_x - amount_x,
        reserve_y=state.reserve_y + net_y,
        bid_fee=state.bid_fee,
        ask_fee=state.ask_fee,
        accumulated_fees_x=state.accumulated_fees_x,
        accumulated_fees_y=state.accumulated_fees_y + fee_amount,
    )
    trade = TradeEvent(
        venue=amm_name,
        source=source,
        is_buy=False,
        amount_x=float(amount_x),
        amount_y=float(amount_y),
        timestamp=int(timestamp),
        reserve_x=next_state.reserve_x,
        reserve_y=next_state.reserve_y,
    )
    final_state, next_policy_state = _after_event(
        state=next_state,
        policy=policy,
        policy_state=policy_state,
        trade=trade,
    )
    return (final_state, next_policy_state, trade)


def execute_buy_x_with_y(
    *,
    amm_name: str,
    state: AMMState,
    policy: DiffSimpleAMMPolicy,
    policy_state: PolicyState,
    amount_y: float,
    timestamp: int,
    source: str,
) -> tuple[AMMState, PolicyState, TradeEvent | None]:
    """Execute a trade where the AMM sells X for an incoming amount of Y."""

    amount_x, fee_amount = quote_x_for_y(state, amount_y)
    if amount_x <= 0.0:
        return (state, policy_state, None)
    net_y = amount_y - fee_amount
    next_state = AMMState(
        reserve_x=state.reserve_x - amount_x,
        reserve_y=state.reserve_y + net_y,
        bid_fee=state.bid_fee,
        ask_fee=state.ask_fee,
        accumulated_fees_x=state.accumulated_fees_x,
        accumulated_fees_y=state.accumulated_fees_y + fee_amount,
    )
    trade = TradeEvent(
        venue=amm_name,
        source=source,
        is_buy=False,
        amount_x=float(amount_x),
        amount_y=float(amount_y),
        timestamp=int(timestamp),
        reserve_x=next_state.reserve_x,
        reserve_y=next_state.reserve_y,
    )
    final_state, next_policy_state = _after_event(
        state=next_state,
        policy=policy,
        policy_state=policy_state,
        trade=trade,
    )
    return (final_state, next_policy_state, trade)


def _after_event(
    *,
    state: AMMState,
    policy: DiffSimpleAMMPolicy,
    policy_state: PolicyState,
    trade: TradeEvent,
) -> tuple[AMMState, PolicyState]:
    try:
        output = policy.after_event(policy_state, trade)
    except Exception:
        return (state, policy_state)
    return (
        AMMState(
            reserve_x=state.reserve_x,
            reserve_y=state.reserve_y,
            bid_fee=clamp_fee(output.bid_fee),
            ask_fee=clamp_fee(output.ask_fee),
            accumulated_fees_x=state.accumulated_fees_x,
            accumulated_fees_y=state.accumulated_fees_y,
        ),
        output.state,
    )
