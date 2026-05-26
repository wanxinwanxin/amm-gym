"""Tape-faithful differentiable simulator (no sharpness / no relaxation).

Replays the *exact* tape's discrete realizations of arrivals, sizes, sides
(challenge) or impact logs (realistic), and price draws. Substitutes the
measure-zero hard branches of the exact simulator with naturally-continuous
formulations:

- arb side selection -> ``max(0, Δx_buy) + max(0, Δx_sell)`` (only one is
  ever positive; the dead zone makes both zero by construction).
- arb profit gate    -> redundant once both sides use ``max(0, ·)`` on
  the optimal-size formula; profit is non-negative whenever the size is.
- fee clamp          -> ``jnp.clip(fee, 0, MAX_FEE)``.
- AMM quote guards   -> trust the math; valid reserves stay positive.

The result is a JAX function differentiable in policy parameters (fees in
this MVP) with no temperature parameter and bit-close agreement with the
exact simulator on the same seed. It is *not* differentiable in environment
parameters (arrival rate, buy probability, regime transition matrix) because
those are baked into the tape — by design, since the search code does not
optimize them.

Currently supports the FixedFee policy on both submission and normalizer
venues, in both challenge and realistic modes.
"""

from __future__ import annotations

from dataclasses import replace

try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - exercised only when jax is absent
    jax = None
    jnp = None

import numpy as np

from arena_eval.exact_simple_amm.config import ExactSimpleAMMConfig

from .types import ChallengeTape, DiffSimulationResult, RealisticTape


DEFAULT_BASELINE_FEE = 0.003
MAX_FEE = 0.1
MIN_DENOM = 1e-12
MIN_AMOUNT = 0.0001  # router-level threshold; mirrors exact_simple_amm.simulator.MIN_AMOUNT


# ----- adaptive-policy state vector layouts -----
# Compact policy: 19 fields, mirrors arena_policies/submission_safe.py SubmissionSafeState
_S_INITIAL_X = 0
_S_INITIAL_Y = 1
_S_LAST_TIMESTAMP = 2
_S_LAST_SIDE = 3
_S_LAST_SIZE_RATIO = 4
_S_STREAK_LEN = 5
_S_BUY_FLOW_FAST = 6
_S_SELL_FLOW_FAST = 7
_S_BUY_FLOW_SLOW = 8
_S_SELL_FLOW_SLOW = 9
_S_SIZE_FAST = 10
_S_SIZE_SLOW = 11
_S_GAP_FAST = 12
_S_GAP_SLOW = 13
_S_TOX_BID = 14
_S_TOX_ASK = 15
_S_FAIR_FAST = 16
_S_FAIR_SLOW = 17
_S_INITIALIZED = 18

# Compact param indices (matches objectives.SUBMISSION_COMPACT_PARAM_NAMES)
_C_BASE_FEE = 0
_C_MIN_FEE = 1
_C_MAX_FEE = 2
_C_FLOW_FAST_DECAY = 3
_C_FLOW_SLOW_DECAY = 4
_C_SIZE_FAST_DECAY = 5
_C_SIZE_SLOW_DECAY = 6
_C_GAP_FAST_DECAY = 7
_C_GAP_SLOW_DECAY = 8
_C_TOXICITY_DECAY = 9
_C_TOXICITY_WEIGHT = 10
_C_BASE_SPREAD = 11
_C_FLOW_MID_WEIGHT = 12
_C_SIZE_MID_WEIGHT = 13
_C_GAP_MID_WEIGHT = 14
_C_SKEW_WEIGHT = 15
_C_TOXICITY_SIDE_WEIGHT = 16
_C_HOT_GAP_THRESHOLD = 17
_C_BIG_TRADE_THRESHOLD = 18
_C_HOT_FEE_BUMP = 19

# Piecewise policy: 7 fields, mirrors arena_policies/piecewise_controller.PiecewiseControllerState
_P_LAST_TIMESTAMP = 0
_P_LAST_SIDE = 1
_P_BID_SIGNAL = 2
_P_ASK_SIGNAL = 3
_P_BID_TOXICITY = 4
_P_ASK_TOXICITY = 5
_P_INITIALIZED = 6

# Piecewise param indices (matches objectives.PIECEWISE_PARAM_NAMES)
_PW_BASE_FEE = 0
_PW_BASE_SPREAD = 1
_PW_SIGNAL_DECAY = 2
_PW_TOXICITY_DECAY = 3
_PW_SMALL_THRESH = 4
_PW_LARGE_THRESH = 5
_PW_CONT_SMALL = 6
_PW_CONT_MED = 7
_PW_CONT_LARGE = 8
_PW_REV_SMALL = 9
_PW_REV_MED = 10
_PW_REV_LARGE = 11
_PW_CONT_TO_SAME = 12
_PW_CONT_TO_CROSS = 13
_PW_TOX_TO_MID = 14
_PW_TOX_TO_SIDE = 15


def _require_jax() -> None:
    if jax is None or jnp is None:
        raise RuntimeError("jax is required for tape-faithful smooth simulation")


def _amm_state(*, rx, ry, bid_fee, ask_fee):
    return {
        "rx": jnp.asarray(rx, dtype=jnp.float64),
        "ry": jnp.asarray(ry, dtype=jnp.float64),
        "bid_fee": jnp.asarray(bid_fee, dtype=jnp.float64),
        "ask_fee": jnp.asarray(ask_fee, dtype=jnp.float64),
        "fee_x": jnp.zeros((), dtype=jnp.float64),
        "fee_y": jnp.zeros((), dtype=jnp.float64),
    }


def _execute_buy_x(state, amount_x):
    """AMM buys X (trader sells X), pays Y. amount_x clipped at >=0."""
    amount_x = jnp.maximum(amount_x, 0.0)
    gamma = jnp.clip(1.0 - state["bid_fee"], 0.0, 1.0)
    net_x = amount_x * gamma
    new_rx = state["rx"] + net_x
    new_ry = state["rx"] * state["ry"] / jnp.maximum(new_rx, MIN_DENOM)
    amount_y = jnp.maximum(state["ry"] - new_ry, 0.0)
    fee_x = amount_x * state["bid_fee"]
    next_state = {
        **state,
        "rx": new_rx,
        "ry": state["ry"] - amount_y,
        "fee_x": state["fee_x"] + fee_x,
    }
    return next_state, amount_x, amount_y


def _execute_sell_x(state, amount_x):
    """AMM sells X (trader buys X), receives Y. amount_x capped at 0.99 * rx."""
    capped = jnp.clip(amount_x, 0.0, 0.99 * state["rx"])
    amount_x_eff = jnp.maximum(capped, 0.0)
    gamma = jnp.clip(1.0 - state["ask_fee"], 0.0, 1.0)
    new_rx = jnp.maximum(state["rx"] - amount_x_eff, MIN_DENOM)
    new_ry = state["rx"] * state["ry"] / new_rx
    net_y = jnp.maximum(new_ry - state["ry"], 0.0)
    total_y = net_y / jnp.maximum(gamma, MIN_DENOM)
    fee_y = total_y - net_y
    next_state = {
        **state,
        "rx": new_rx,
        "ry": state["ry"] + net_y,
        "fee_y": state["fee_y"] + fee_y,
    }
    return next_state, amount_x_eff, total_y


def _execute_buy_x_with_y(state, amount_y):
    """AMM sells X for incoming Y. amount_y clipped at >=0."""
    amount_y = jnp.maximum(amount_y, 0.0)
    gamma = jnp.clip(1.0 - state["ask_fee"], 0.0, 1.0)
    net_y = amount_y * gamma
    new_ry = state["ry"] + net_y
    new_rx = state["rx"] * state["ry"] / jnp.maximum(new_ry, MIN_DENOM)
    amount_x = jnp.maximum(state["rx"] - new_rx, 0.0)
    fee_y = amount_y * state["ask_fee"]
    next_state = {
        **state,
        "rx": state["rx"] - amount_x,
        "ry": new_ry,
        "fee_y": state["fee_y"] + fee_y,
    }
    return next_state, amount_x, amount_y


def _split_buy(submission, normalizer, total_y):
    g1 = 1.0 - submission["ask_fee"]
    g2 = 1.0 - normalizer["ask_fee"]
    a1 = jnp.sqrt(jnp.maximum(submission["rx"] * g1 * submission["ry"], 0.0))
    a2 = jnp.sqrt(jnp.maximum(normalizer["rx"] * g2 * normalizer["ry"], 0.0))
    r = a1 / jnp.maximum(a2, MIN_DENOM)
    den = g1 + r * g2
    raw = (r * (normalizer["ry"] + g2 * total_y) - submission["ry"]) / jnp.maximum(den, MIN_DENOM)
    y_sub = jnp.clip(raw, 0.0, total_y)
    return y_sub, total_y - y_sub


def _split_sell(submission, normalizer, total_x):
    g1 = 1.0 - submission["bid_fee"]
    g2 = 1.0 - normalizer["bid_fee"]
    b1 = jnp.sqrt(jnp.maximum(submission["ry"] * g1 * submission["rx"], 0.0))
    b2 = jnp.sqrt(jnp.maximum(normalizer["ry"] * g2 * normalizer["rx"], 0.0))
    r = b1 / jnp.maximum(b2, MIN_DENOM)
    den = g1 + r * g2
    raw = (r * (normalizer["rx"] + g2 * total_x) - submission["rx"]) / jnp.maximum(den, MIN_DENOM)
    x_sub = jnp.clip(raw, 0.0, total_x)
    return x_sub, total_x - x_sub


def _execute_arb(state, fair_price):
    """Naturally-continuous arb: signed Δx via max(0, ·) per direction.

    Both `buy_amount_x` and `sell_amount_x` are zero in the dead zone (and
    one is zero outside it), so executing both sequentially is a no-op on
    the inactive side.
    """
    gamma_ask = jnp.clip(1.0 - state["ask_fee"], 0.0, 1.0)
    gamma_bid = jnp.clip(1.0 - state["bid_fee"], 0.0, 1.0)

    buy_raw = state["rx"] - jnp.sqrt(
        jnp.maximum(
            state["rx"] * state["ry"] / jnp.maximum(gamma_ask * fair_price, MIN_DENOM),
            0.0,
        )
    )
    buy_amount_x = jnp.maximum(buy_raw, 0.0)
    state_after_buy, buy_ax, buy_ay = _execute_sell_x(state, buy_amount_x)
    buy_profit = jnp.maximum(buy_ax * fair_price - buy_ay, 0.0)

    sell_raw = (
        jnp.sqrt(
            jnp.maximum(
                state["rx"] * state["ry"] * gamma_bid / jnp.maximum(fair_price, MIN_DENOM),
                0.0,
            )
        )
        - state["rx"]
    ) / jnp.maximum(gamma_bid, MIN_DENOM)
    sell_amount_x = jnp.maximum(sell_raw, 0.0)
    state_after_sell, sell_ax, sell_ay = _execute_buy_x(state_after_buy, sell_amount_x)
    sell_profit = jnp.maximum(sell_ay - sell_ax * fair_price, 0.0)

    arb_y = buy_ay + sell_ay
    arb_profit = buy_profit + sell_profit
    return state_after_sell, arb_y, arb_profit


# ----- tape -> fixed-shape JAX arrays -----
def _pad_step_lists(values: list[list[float]], width: int, fill: float = 0.0):
    return np.asarray(
        [row + [fill] * (width - len(row)) for row in values],
        dtype=np.float64,
    )


def challenge_tape_to_arrays(tape: ChallengeTape) -> dict[str, object]:
    _require_jax()
    width = max(int(tape.max_orders_per_step), 1)
    n_steps = len(tape.gbm_normals)
    sizes = [list(row) for row in tape.order_sizes]
    sides = [list(row) for row in tape.order_side_uniforms]
    masks = [[1.0] * len(row) for row in tape.order_sizes]
    return {
        "gbm_normals": jnp.asarray(tape.gbm_normals, dtype=jnp.float64),
        "sizes": jnp.asarray(_pad_step_lists(sizes, width), dtype=jnp.float64),
        "sides": jnp.asarray(_pad_step_lists(sides, width, fill=1.0), dtype=jnp.float64),
        "mask": jnp.asarray(_pad_step_lists(masks, width), dtype=jnp.float64),
        "n_steps": n_steps,
        "width": width,
    }


def realistic_tape_to_arrays(tape: RealisticTape) -> dict[str, object]:
    _require_jax()
    width = max(int(tape.max_orders_per_step), 1)
    n_steps = len(tape.log_returns)
    impacts = [list(row) for row in tape.impact_logs]
    masks = [[1.0] * len(row) for row in tape.impact_logs]
    return {
        "log_returns": jnp.asarray(tape.log_returns, dtype=jnp.float64),
        "impact_logs": jnp.asarray(_pad_step_lists(impacts, width), dtype=jnp.float64),
        "mask": jnp.asarray(_pad_step_lists(masks, width), dtype=jnp.float64),
        "n_steps": n_steps,
        "width": width,
    }


# ----- rollouts -----
def _initial_carry(config: ExactSimpleAMMConfig, fees):
    sub_bid, sub_ask, norm_bid, norm_ask = fees
    return {
        "submission": _amm_state(
            rx=config.submission_initial_x,
            ry=config.submission_initial_y,
            bid_fee=jnp.clip(sub_bid, 0.0, MAX_FEE),
            ask_fee=jnp.clip(sub_ask, 0.0, MAX_FEE),
        ),
        "normalizer": _amm_state(
            rx=config.normalizer_initial_x,
            ry=config.normalizer_initial_y,
            bid_fee=jnp.clip(norm_bid, 0.0, MAX_FEE),
            ask_fee=jnp.clip(norm_ask, 0.0, MAX_FEE),
        ),
        "fair_price": jnp.asarray(config.initial_price, dtype=jnp.float64),
        "edge_submission": jnp.zeros((), dtype=jnp.float64),
        "edge_normalizer": jnp.zeros((), dtype=jnp.float64),
        "retail_volume_submission_y": jnp.zeros((), dtype=jnp.float64),
        "retail_volume_normalizer_y": jnp.zeros((), dtype=jnp.float64),
        "arb_volume_submission_y": jnp.zeros((), dtype=jnp.float64),
        "arb_volume_normalizer_y": jnp.zeros((), dtype=jnp.float64),
        "bid_fee_submission_sum": jnp.zeros((), dtype=jnp.float64),
        "ask_fee_submission_sum": jnp.zeros((), dtype=jnp.float64),
        "bid_fee_normalizer_sum": jnp.zeros((), dtype=jnp.float64),
        "ask_fee_normalizer_sum": jnp.zeros((), dtype=jnp.float64),
    }


def _route_one_order(carry, *, buy_y, sell_y_notional, fair_price):
    """Route a single retail order described by (buy_y, sell_y_notional).

    Both can be zero — masked-out orders are handled by passing zeros.
    """
    y_sub, y_norm = _split_buy(carry["submission"], carry["normalizer"], buy_y)
    sub_after_buy, ax_sub_buy, ay_sub_buy = _execute_buy_x_with_y(carry["submission"], y_sub)
    norm_after_buy, ax_norm_buy, ay_norm_buy = _execute_buy_x_with_y(carry["normalizer"], y_norm)

    sell_total_x = sell_y_notional / jnp.maximum(fair_price, MIN_DENOM)
    x_sub, x_norm = _split_sell(sub_after_buy, norm_after_buy, sell_total_x)
    sub_final, ax_sub_sell, ay_sub_sell = _execute_buy_x(sub_after_buy, x_sub)
    norm_final, ax_norm_sell, ay_norm_sell = _execute_buy_x(norm_after_buy, x_norm)

    edge_sub_delta = (ay_sub_buy - ax_sub_buy * fair_price) + (ax_sub_sell * fair_price - ay_sub_sell)
    edge_norm_delta = (ay_norm_buy - ax_norm_buy * fair_price) + (ax_norm_sell * fair_price - ay_norm_sell)
    retail_sub_delta = ay_sub_buy + ay_sub_sell
    retail_norm_delta = ay_norm_buy + ay_norm_sell
    return (sub_final, norm_final, edge_sub_delta, edge_norm_delta, retail_sub_delta, retail_norm_delta)


def fixed_fee_metrics_challenge(
    config: ExactSimpleAMMConfig,
    tape: ChallengeTape,
    *,
    submission_bid_fee: float = DEFAULT_BASELINE_FEE,
    submission_ask_fee: float = DEFAULT_BASELINE_FEE,
    normalizer_bid_fee: float = DEFAULT_BASELINE_FEE,
    normalizer_ask_fee: float = DEFAULT_BASELINE_FEE,
) -> dict[str, object]:
    _require_jax()
    arrays = challenge_tape_to_arrays(tape)
    carry = _initial_carry(
        config, (submission_bid_fee, submission_ask_fee, normalizer_bid_fee, normalizer_ask_fee)
    )
    drift = (config.gbm_mu - 0.5 * config.gbm_sigma ** 2) * config.gbm_dt
    vol = config.gbm_sigma * jnp.sqrt(jnp.asarray(config.gbm_dt, dtype=jnp.float64))
    buy_prob = jnp.asarray(config.retail_buy_prob, dtype=jnp.float64)

    def step_fn(carry, xs):
        z, sizes_t, sides_t, mask_t = xs
        fair_price = carry["fair_price"] * jnp.exp(drift + vol * z)

        sub_state, sub_arb_y, sub_arb_profit = _execute_arb(carry["submission"], fair_price)
        norm_state, norm_arb_y, norm_arb_profit = _execute_arb(carry["normalizer"], fair_price)

        order_carry = {
            "submission": sub_state,
            "normalizer": norm_state,
            "edge_submission": carry["edge_submission"] - sub_arb_profit,
            "edge_normalizer": carry["edge_normalizer"] - norm_arb_profit,
            "retail_volume_submission_y": carry["retail_volume_submission_y"],
            "retail_volume_normalizer_y": carry["retail_volume_normalizer_y"],
        }

        def order_fn(oc, order_inputs):
            size, side_u, mask = order_inputs
            is_buy = jnp.where(side_u < buy_prob, 1.0, 0.0)
            buy_y = mask * is_buy * size
            sell_y_notional = mask * (1.0 - is_buy) * size
            sub_final, norm_final, ed_sub, ed_norm, rv_sub, rv_norm = _route_one_order(
                oc, buy_y=buy_y, sell_y_notional=sell_y_notional, fair_price=fair_price
            )
            return (
                {
                    "submission": sub_final,
                    "normalizer": norm_final,
                    "edge_submission": oc["edge_submission"] + ed_sub,
                    "edge_normalizer": oc["edge_normalizer"] + ed_norm,
                    "retail_volume_submission_y": oc["retail_volume_submission_y"] + rv_sub,
                    "retail_volume_normalizer_y": oc["retail_volume_normalizer_y"] + rv_norm,
                },
                None,
            )

        order_carry, _ = jax.lax.scan(order_fn, order_carry, (sizes_t, sides_t, mask_t))

        next_carry = {
            "submission": order_carry["submission"],
            "normalizer": order_carry["normalizer"],
            "fair_price": fair_price,
            "edge_submission": order_carry["edge_submission"],
            "edge_normalizer": order_carry["edge_normalizer"],
            "retail_volume_submission_y": order_carry["retail_volume_submission_y"],
            "retail_volume_normalizer_y": order_carry["retail_volume_normalizer_y"],
            "arb_volume_submission_y": carry["arb_volume_submission_y"] + sub_arb_y,
            "arb_volume_normalizer_y": carry["arb_volume_normalizer_y"] + norm_arb_y,
            "bid_fee_submission_sum": carry["bid_fee_submission_sum"]
            + order_carry["submission"]["bid_fee"],
            "ask_fee_submission_sum": carry["ask_fee_submission_sum"]
            + order_carry["submission"]["ask_fee"],
            "bid_fee_normalizer_sum": carry["bid_fee_normalizer_sum"]
            + order_carry["normalizer"]["bid_fee"],
            "ask_fee_normalizer_sum": carry["ask_fee_normalizer_sum"]
            + order_carry["normalizer"]["ask_fee"],
        }
        return next_carry, None

    final_carry, _ = jax.lax.scan(
        step_fn,
        carry,
        (arrays["gbm_normals"], arrays["sizes"], arrays["sides"], arrays["mask"]),
    )
    return _build_metrics(final_carry, config)


def fixed_fee_metrics_realistic(
    config: ExactSimpleAMMConfig,
    tape: RealisticTape,
    *,
    submission_bid_fee: float = DEFAULT_BASELINE_FEE,
    submission_ask_fee: float = DEFAULT_BASELINE_FEE,
    normalizer_bid_fee: float = DEFAULT_BASELINE_FEE,
    normalizer_ask_fee: float = DEFAULT_BASELINE_FEE,
) -> dict[str, object]:
    _require_jax()
    arrays = realistic_tape_to_arrays(tape)
    carry = _initial_carry(
        config, (submission_bid_fee, submission_ask_fee, normalizer_bid_fee, normalizer_ask_fee)
    )
    use_initial_state = config.retail_impact_scale_mode == "initial_state"
    use_submission_ref = config.retail_impact_reference_venue == "submission"

    def step_fn(carry, xs):
        log_return, impacts_t, mask_t = xs
        fair_price = carry["fair_price"] * jnp.exp(log_return)

        sub_state, sub_arb_y, sub_arb_profit = _execute_arb(carry["submission"], fair_price)
        norm_state, norm_arb_y, norm_arb_profit = _execute_arb(carry["normalizer"], fair_price)

        # Snapshot reference state at start of step (post-arb, pre-orders).
        if use_initial_state:
            ref_rx = jnp.asarray(
                config.submission_initial_x if use_submission_ref else config.normalizer_initial_x,
                dtype=jnp.float64,
            )
            ref_ry = jnp.asarray(
                config.submission_initial_y if use_submission_ref else config.normalizer_initial_y,
                dtype=jnp.float64,
            )
            ref_bid = jnp.asarray(0.003, dtype=jnp.float64)
            ref_ask = jnp.asarray(0.003, dtype=jnp.float64)
        else:
            ref_state = sub_state if use_submission_ref else norm_state
            ref_rx = ref_state["rx"]
            ref_ry = ref_state["ry"]
            ref_bid = ref_state["bid_fee"]
            ref_ask = ref_state["ask_fee"]

        order_carry = {
            "submission": sub_state,
            "normalizer": norm_state,
            "edge_submission": carry["edge_submission"] - sub_arb_profit,
            "edge_normalizer": carry["edge_normalizer"] - norm_arb_profit,
            "retail_volume_submission_y": carry["retail_volume_submission_y"],
            "retail_volume_normalizer_y": carry["retail_volume_normalizer_y"],
        }

        gamma_ask = jnp.maximum(1.0 - ref_ask, MIN_DENOM)
        gamma_bid = jnp.maximum(1.0 - ref_bid, MIN_DENOM)

        def order_fn(oc, order_inputs):
            impact_log, mask = order_inputs
            magnitude = jnp.abs(impact_log)
            is_buy = jnp.where(impact_log > 0.0, 1.0, 0.0)
            exp_half = jnp.exp(0.5 * magnitude) - 1.0
            buy_size_y = ref_ry * exp_half / gamma_ask
            sell_size_x = ref_rx * exp_half / gamma_bid
            sell_size_y_notional = sell_size_x * jnp.maximum(fair_price, MIN_DENOM)

            buy_y = mask * is_buy * buy_size_y
            sell_y_notional = mask * (1.0 - is_buy) * sell_size_y_notional

            sub_final, norm_final, ed_sub, ed_norm, rv_sub, rv_norm = _route_one_order(
                oc, buy_y=buy_y, sell_y_notional=sell_y_notional, fair_price=fair_price
            )
            return (
                {
                    "submission": sub_final,
                    "normalizer": norm_final,
                    "edge_submission": oc["edge_submission"] + ed_sub,
                    "edge_normalizer": oc["edge_normalizer"] + ed_norm,
                    "retail_volume_submission_y": oc["retail_volume_submission_y"] + rv_sub,
                    "retail_volume_normalizer_y": oc["retail_volume_normalizer_y"] + rv_norm,
                },
                None,
            )

        order_carry, _ = jax.lax.scan(order_fn, order_carry, (impacts_t, mask_t))

        next_carry = {
            "submission": order_carry["submission"],
            "normalizer": order_carry["normalizer"],
            "fair_price": fair_price,
            "edge_submission": order_carry["edge_submission"],
            "edge_normalizer": order_carry["edge_normalizer"],
            "retail_volume_submission_y": order_carry["retail_volume_submission_y"],
            "retail_volume_normalizer_y": order_carry["retail_volume_normalizer_y"],
            "arb_volume_submission_y": carry["arb_volume_submission_y"] + sub_arb_y,
            "arb_volume_normalizer_y": carry["arb_volume_normalizer_y"] + norm_arb_y,
            "bid_fee_submission_sum": carry["bid_fee_submission_sum"]
            + order_carry["submission"]["bid_fee"],
            "ask_fee_submission_sum": carry["ask_fee_submission_sum"]
            + order_carry["submission"]["ask_fee"],
            "bid_fee_normalizer_sum": carry["bid_fee_normalizer_sum"]
            + order_carry["normalizer"]["bid_fee"],
            "ask_fee_normalizer_sum": carry["ask_fee_normalizer_sum"]
            + order_carry["normalizer"]["ask_fee"],
        }
        return next_carry, None

    final_carry, _ = jax.lax.scan(
        step_fn,
        carry,
        (arrays["log_returns"], arrays["impact_logs"], arrays["mask"]),
    )
    return _build_metrics(final_carry, config)


def _build_metrics(final_carry, config: ExactSimpleAMMConfig) -> dict[str, object]:
    sub_state = final_carry["submission"]
    norm_state = final_carry["normalizer"]
    sub_value = (
        sub_state["rx"] * final_carry["fair_price"]
        + sub_state["ry"]
        + sub_state["fee_x"] * final_carry["fair_price"]
        + sub_state["fee_y"]
    )
    norm_value = (
        norm_state["rx"] * final_carry["fair_price"]
        + norm_state["ry"]
        + norm_state["fee_x"] * final_carry["fair_price"]
        + norm_state["fee_y"]
    )
    submission_initial_value = config.submission_initial_value
    normalizer_initial_value = config.normalizer_initial_value
    steps = max(int(config.n_steps), 1)
    return {
        "edge_submission": final_carry["edge_submission"],
        "edge_normalizer": final_carry["edge_normalizer"],
        "pnl_submission": sub_value - submission_initial_value,
        "pnl_normalizer": norm_value - normalizer_initial_value,
        "retail_volume_submission_y": final_carry["retail_volume_submission_y"],
        "retail_volume_normalizer_y": final_carry["retail_volume_normalizer_y"],
        "arb_volume_submission_y": final_carry["arb_volume_submission_y"],
        "arb_volume_normalizer_y": final_carry["arb_volume_normalizer_y"],
        "average_bid_fee_submission": final_carry["bid_fee_submission_sum"] / steps,
        "average_ask_fee_submission": final_carry["ask_fee_submission_sum"] / steps,
        "average_bid_fee_normalizer": final_carry["bid_fee_normalizer_sum"] / steps,
        "average_ask_fee_normalizer": final_carry["ask_fee_normalizer_sum"] / steps,
    }


def fixed_fee_result(
    config: ExactSimpleAMMConfig,
    tape: ChallengeTape | RealisticTape,
    *,
    submission_bid_fee: float = DEFAULT_BASELINE_FEE,
    submission_ask_fee: float = DEFAULT_BASELINE_FEE,
    normalizer_bid_fee: float = DEFAULT_BASELINE_FEE,
    normalizer_ask_fee: float = DEFAULT_BASELINE_FEE,
    seed: int = 0,
) -> DiffSimulationResult:
    """Convenience wrapper returning a `DiffSimulationResult`."""

    if isinstance(tape, ChallengeTape):
        metrics = fixed_fee_metrics_challenge(
            config,
            tape,
            submission_bid_fee=submission_bid_fee,
            submission_ask_fee=submission_ask_fee,
            normalizer_bid_fee=normalizer_bid_fee,
            normalizer_ask_fee=normalizer_ask_fee,
        )
    elif isinstance(tape, RealisticTape):
        metrics = fixed_fee_metrics_realistic(
            config,
            tape,
            submission_bid_fee=submission_bid_fee,
            submission_ask_fee=submission_ask_fee,
            normalizer_bid_fee=normalizer_bid_fee,
            normalizer_ask_fee=normalizer_ask_fee,
        )
    else:
        raise TypeError(f"Unsupported tape type: {type(tape)!r}")
    return _build_result(metrics, seed)


def _build_result(metrics: dict, seed: int) -> DiffSimulationResult:
    return DiffSimulationResult(
        seed=seed,
        edge_submission=float(metrics["edge_submission"]),
        edge_normalizer=float(metrics["edge_normalizer"]),
        pnl_submission=float(metrics["pnl_submission"]),
        pnl_normalizer=float(metrics["pnl_normalizer"]),
        score=float(metrics["edge_submission"]),
        retail_volume_submission_y=float(metrics["retail_volume_submission_y"]),
        retail_volume_normalizer_y=float(metrics["retail_volume_normalizer_y"]),
        arb_volume_submission_y=float(metrics["arb_volume_submission_y"]),
        arb_volume_normalizer_y=float(metrics["arb_volume_normalizer_y"]),
        average_bid_fee_submission=float(metrics["average_bid_fee_submission"]),
        average_ask_fee_submission=float(metrics["average_ask_fee_submission"]),
        average_bid_fee_normalizer=float(metrics["average_bid_fee_normalizer"]),
        average_ask_fee_normalizer=float(metrics["average_ask_fee_normalizer"]),
        metadata={"mode": "smooth_tape"},
    )


# ============================================================================
# Adaptive policies: SubmissionCompact and Piecewise.
# ============================================================================
#
# These are functional re-derivations of the exact policies in
# arena_policies/submission_safe.py and arena_policies/piecewise_controller.py.
# Discrete branches (side ∈ {-1,+1}, last_side ∈ {-1,0,+1}, size-bucket
# selectors) are encoded with jnp.where on the actual discrete realizations,
# which is exact (no smoothing). Continuous threshold gates (gap_fast vs.
# hot_gap_threshold, size_ratio vs. big_trade_threshold) are also encoded
# with jnp.where; their boundaries are measure-zero so this matches the exact
# behavior almost surely. Fee clamps are jnp.clip.
#
# A policy update fires only when the corresponding trade actually fires
# (matching exact MIN_AMOUNT and profit-positivity gates) — this is enforced
# at the call site by `_maybe_update`.


def _maybe_update(fired, old_state, old_bid, old_ask, new_state, new_bid, new_ask):
    """Select between old and new (state, bid, ask) based on a boolean `fired`."""
    fired_f = fired.astype(jnp.float64)
    state = jax.tree_util.tree_map(lambda o, n: jnp.where(fired, n, o), old_state, new_state)
    bid = old_bid * (1.0 - fired_f) + new_bid * fired_f
    ask = old_ask * (1.0 - fired_f) + new_ask * fired_f
    return state, bid, ask


# ----- compact policy primitives -----
def compact_initial_policy_state(initial_x: float, initial_y: float):
    _require_jax()
    initial_spot = float(initial_y) / max(float(initial_x), 1e-9)
    state = jnp.zeros(19, dtype=jnp.float64)
    state = state.at[_S_INITIAL_X].set(float(initial_x))
    state = state.at[_S_INITIAL_Y].set(float(initial_y))
    state = state.at[_S_GAP_FAST].set(1.0)
    state = state.at[_S_GAP_SLOW].set(1.0)
    state = state.at[_S_FAIR_FAST].set(initial_spot)
    state = state.at[_S_FAIR_SLOW].set(initial_spot)
    state = state.at[_S_INITIALIZED].set(1.0)
    return state


def compact_initial_fees(params):
    return params[_C_BASE_FEE], params[_C_BASE_FEE]


def compact_after_event(params, state, trade):
    """Functional version of SubmissionCompactStrategy.after_swap."""
    flow_fast_decay = params[_C_FLOW_FAST_DECAY]
    flow_slow_decay = params[_C_FLOW_SLOW_DECAY]
    size_fast_decay = params[_C_SIZE_FAST_DECAY]
    size_slow_decay = params[_C_SIZE_SLOW_DECAY]
    gap_fast_decay = params[_C_GAP_FAST_DECAY]
    gap_slow_decay = params[_C_GAP_SLOW_DECAY]
    toxicity_decay = params[_C_TOXICITY_DECAY]
    toxicity_weight = params[_C_TOXICITY_WEIGHT]

    size_ratio = trade["amount_y"] / jnp.maximum(trade["reserve_y"], 1e-9)
    side = jnp.where(trade["is_buy"] > 0.5, 1.0, -1.0)
    spot = trade["reserve_y"] / jnp.maximum(trade["reserve_x"], 1e-9)
    dt = jnp.maximum(1.0, trade["timestamp"] - state[_S_LAST_TIMESTAMP])

    is_buy_mask = jnp.where(side > 0.0, 1.0, 0.0)
    is_sell_mask = 1.0 - is_buy_mask

    buy_flow_fast = state[_S_BUY_FLOW_FAST] * flow_fast_decay + (1.0 - flow_fast_decay) * size_ratio * is_buy_mask
    sell_flow_fast = state[_S_SELL_FLOW_FAST] * flow_fast_decay + (1.0 - flow_fast_decay) * size_ratio * is_sell_mask
    buy_flow_slow = state[_S_BUY_FLOW_SLOW] * flow_slow_decay + (1.0 - flow_slow_decay) * size_ratio * is_buy_mask
    sell_flow_slow = state[_S_SELL_FLOW_SLOW] * flow_slow_decay + (1.0 - flow_slow_decay) * size_ratio * is_sell_mask

    size_fast = size_fast_decay * state[_S_SIZE_FAST] + (1.0 - size_fast_decay) * size_ratio
    size_slow = size_slow_decay * state[_S_SIZE_SLOW] + (1.0 - size_slow_decay) * size_ratio
    gap_fast = gap_fast_decay * state[_S_GAP_FAST] + (1.0 - gap_fast_decay) * dt
    gap_slow = gap_slow_decay * state[_S_GAP_SLOW] + (1.0 - gap_slow_decay) * dt

    tox_bid = state[_S_TOX_BID] * toxicity_decay
    tox_ask = state[_S_TOX_ASK] * toxicity_decay
    same_side_tox = 0.4 * toxicity_weight * size_ratio
    tox_bid = tox_bid + same_side_tox * is_buy_mask
    tox_ask = tox_ask + same_side_tox * is_sell_mask

    last_side_present = jnp.where(state[_S_LAST_SIDE] != 0.0, 1.0, 0.0)
    side_changed = jnp.where(side != state[_S_LAST_SIDE], 1.0, 0.0)
    reversal_active = last_side_present * side_changed
    reversal_scale = jnp.maximum(state[_S_LAST_SIZE_RATIO], size_ratio) / dt
    last_side_buy = jnp.where(state[_S_LAST_SIDE] > 0.0, 1.0, 0.0)
    last_side_sell = jnp.where(state[_S_LAST_SIDE] < 0.0, 1.0, 0.0)
    tox_bid = tox_bid + toxicity_weight * reversal_scale * reversal_active * last_side_buy
    tox_ask = tox_ask + toxicity_weight * reversal_scale * reversal_active * last_side_sell

    # Compact uses fair_size_weight=0.0 (see SubmissionCompactStrategy.after_swap),
    # so fair_obs == spot. Decays are hard-coded 0.9 / 0.98 in the exact stack.
    fair_fast = 0.9 * state[_S_FAIR_FAST] + 0.1 * spot
    fair_slow = 0.98 * state[_S_FAIR_SLOW] + 0.02 * spot

    same_side_continuation = jnp.where(side == state[_S_LAST_SIDE], 1.0, 0.0)
    new_streak = jnp.where(
        same_side_continuation > 0.5,
        jnp.minimum(state[_S_STREAK_LEN] + 1.0, 8.0),
        1.0,
    )

    new_state = state.at[_S_LAST_TIMESTAMP].set(trade["timestamp"])
    new_state = new_state.at[_S_LAST_SIDE].set(side)
    new_state = new_state.at[_S_LAST_SIZE_RATIO].set(size_ratio)
    new_state = new_state.at[_S_STREAK_LEN].set(new_streak)
    new_state = new_state.at[_S_BUY_FLOW_FAST].set(buy_flow_fast)
    new_state = new_state.at[_S_SELL_FLOW_FAST].set(sell_flow_fast)
    new_state = new_state.at[_S_BUY_FLOW_SLOW].set(buy_flow_slow)
    new_state = new_state.at[_S_SELL_FLOW_SLOW].set(sell_flow_slow)
    new_state = new_state.at[_S_SIZE_FAST].set(size_fast)
    new_state = new_state.at[_S_SIZE_SLOW].set(size_slow)
    new_state = new_state.at[_S_GAP_FAST].set(gap_fast)
    new_state = new_state.at[_S_GAP_SLOW].set(gap_slow)
    new_state = new_state.at[_S_TOX_BID].set(tox_bid)
    new_state = new_state.at[_S_TOX_ASK].set(tox_ask)
    new_state = new_state.at[_S_FAIR_FAST].set(fair_fast)
    new_state = new_state.at[_S_FAIR_SLOW].set(fair_slow)
    new_state = new_state.at[_S_INITIALIZED].set(1.0)

    flow_total = buy_flow_fast + sell_flow_fast
    flow_skew = (buy_flow_fast - sell_flow_fast) + 0.5 * (buy_flow_slow - sell_flow_slow)
    hot_gate = jnp.where(gap_fast < params[_C_HOT_GAP_THRESHOLD], 1.0, 0.0)
    big_gate = jnp.where(size_ratio > params[_C_BIG_TRADE_THRESHOLD], 1.0, 0.0)
    hot_or_big = jnp.maximum(hot_gate, big_gate)

    mid = (
        params[_C_BASE_FEE]
        + params[_C_FLOW_MID_WEIGHT] * flow_total
        + params[_C_SIZE_MID_WEIGHT] * jnp.maximum(size_fast - 0.5 * size_slow, 0.0)
        + params[_C_GAP_MID_WEIGHT] * jnp.maximum(params[_C_HOT_GAP_THRESHOLD] - gap_fast, 0.0)
        + params[_C_HOT_FEE_BUMP] * hot_or_big
    )
    spread = params[_C_BASE_SPREAD] + 0.5 * params[_C_HOT_FEE_BUMP] * hot_gate
    skew = params[_C_SKEW_WEIGHT] * flow_skew
    bid = jnp.clip(
        mid + 0.5 * spread - skew + params[_C_TOXICITY_SIDE_WEIGHT] * tox_bid,
        params[_C_MIN_FEE],
        params[_C_MAX_FEE],
    )
    ask = jnp.clip(
        mid + 0.5 * spread + skew + params[_C_TOXICITY_SIDE_WEIGHT] * tox_ask,
        params[_C_MIN_FEE],
        params[_C_MAX_FEE],
    )
    return new_state, bid, ask


# ----- piecewise policy primitives -----
def piecewise_initial_policy_state():
    _require_jax()
    return jnp.zeros(7, dtype=jnp.float64)


def piecewise_initial_fees(params):
    base = params[_PW_BASE_FEE] + 0.5 * params[_PW_BASE_SPREAD]
    return jnp.clip(base, 0.0, MAX_FEE), jnp.clip(base, 0.0, MAX_FEE)


def piecewise_fees(params, state):
    toxicity_total = state[_P_BID_TOXICITY] + state[_P_ASK_TOXICITY]
    base = params[_PW_BASE_FEE] + 0.5 * params[_PW_BASE_SPREAD] + params[_PW_TOX_TO_MID] * toxicity_total
    bid = jnp.clip(
        base
        + params[_PW_TOX_TO_SIDE] * state[_P_BID_TOXICITY]
        - params[_PW_CONT_TO_SAME] * state[_P_BID_SIGNAL]
        + params[_PW_CONT_TO_CROSS] * state[_P_ASK_SIGNAL],
        0.0,
        MAX_FEE,
    )
    ask = jnp.clip(
        base
        + params[_PW_TOX_TO_SIDE] * state[_P_ASK_TOXICITY]
        - params[_PW_CONT_TO_SAME] * state[_P_ASK_SIGNAL]
        + params[_PW_CONT_TO_CROSS] * state[_P_BID_SIGNAL],
        0.0,
        MAX_FEE,
    )
    return bid, ask


def piecewise_after_event(params, state, trade):
    """Functional version of PiecewiseControllerStrategy.after_swap."""
    initialized = state[_P_INITIALIZED]
    dt_raw = trade["timestamp"] - state[_P_LAST_TIMESTAMP]
    dt = jnp.where(initialized > 0.5, jnp.maximum(1.0, dt_raw), 1.0)

    size_ratio = trade["amount_y"] / jnp.maximum(trade["reserve_y"], 1e-9)
    is_small = jnp.where(size_ratio < params[_PW_SMALL_THRESH], 1.0, 0.0)
    is_large = jnp.where(size_ratio >= params[_PW_LARGE_THRESH], 1.0, 0.0)
    is_medium = 1.0 - is_small - is_large
    continuation_weight = (
        params[_PW_CONT_SMALL] * is_small
        + params[_PW_CONT_MED] * is_medium
        + params[_PW_CONT_LARGE] * is_large
    )
    reversal_weight = (
        params[_PW_REV_SMALL] * is_small
        + params[_PW_REV_MED] * is_medium
        + params[_PW_REV_LARGE] * is_large
    )
    reversal_scale = 1.0 / dt

    current_side = jnp.where(trade["is_buy"] > 0.5, 1.0, -1.0)
    is_buy = jnp.where(trade["is_buy"] > 0.5, 1.0, 0.0)
    is_sell = 1.0 - is_buy

    bid_signal = state[_P_BID_SIGNAL] * params[_PW_SIGNAL_DECAY]
    ask_signal = state[_P_ASK_SIGNAL] * params[_PW_SIGNAL_DECAY]
    bid_toxicity = state[_P_BID_TOXICITY] * params[_PW_TOXICITY_DECAY]
    ask_toxicity = state[_P_ASK_TOXICITY] * params[_PW_TOXICITY_DECAY]

    same_side = jnp.where(current_side == state[_P_LAST_SIDE], 1.0, 0.0)
    last_side_present = jnp.where(state[_P_LAST_SIDE] != 0.0, 1.0, 0.0)
    is_continuation = same_side
    is_reversal = last_side_present * (1.0 - same_side)
    is_initial = 1.0 - last_side_present

    cont_factor = is_continuation * 1.0 + is_reversal * 0.5 + is_initial * 0.5
    bid_signal = bid_signal + continuation_weight * cont_factor * is_buy
    ask_signal = ask_signal + continuation_weight * cont_factor * is_sell

    bid_toxicity = bid_toxicity + reversal_weight * reversal_scale * is_reversal * is_sell
    ask_toxicity = ask_toxicity + reversal_weight * reversal_scale * is_reversal * is_buy

    new_state = state.at[_P_LAST_TIMESTAMP].set(trade["timestamp"])
    new_state = new_state.at[_P_LAST_SIDE].set(current_side)
    new_state = new_state.at[_P_BID_SIGNAL].set(bid_signal)
    new_state = new_state.at[_P_ASK_SIGNAL].set(ask_signal)
    new_state = new_state.at[_P_BID_TOXICITY].set(bid_toxicity)
    new_state = new_state.at[_P_ASK_TOXICITY].set(ask_toxicity)
    new_state = new_state.at[_P_INITIALIZED].set(1.0)

    bid_fee, ask_fee = piecewise_fees(params, new_state)
    return new_state, bid_fee, ask_fee


# ----- adaptive rollouts -----
def _adaptive_initial_carry(config, *, initial_policy_state, initial_bid, initial_ask):
    return {
        "submission": _amm_state(
            rx=config.submission_initial_x,
            ry=config.submission_initial_y,
            bid_fee=jnp.clip(initial_bid, 0.0, MAX_FEE),
            ask_fee=jnp.clip(initial_ask, 0.0, MAX_FEE),
        ),
        "normalizer": _amm_state(
            rx=config.normalizer_initial_x,
            ry=config.normalizer_initial_y,
            bid_fee=jnp.clip(DEFAULT_BASELINE_FEE, 0.0, MAX_FEE),
            ask_fee=jnp.clip(DEFAULT_BASELINE_FEE, 0.0, MAX_FEE),
        ),
        "submission_policy_state": initial_policy_state,
        "fair_price": jnp.asarray(config.initial_price, dtype=jnp.float64),
        "edge_submission": jnp.zeros((), dtype=jnp.float64),
        "edge_normalizer": jnp.zeros((), dtype=jnp.float64),
        "retail_volume_submission_y": jnp.zeros((), dtype=jnp.float64),
        "retail_volume_normalizer_y": jnp.zeros((), dtype=jnp.float64),
        "arb_volume_submission_y": jnp.zeros((), dtype=jnp.float64),
        "arb_volume_normalizer_y": jnp.zeros((), dtype=jnp.float64),
        "bid_fee_submission_sum": jnp.zeros((), dtype=jnp.float64),
        "ask_fee_submission_sum": jnp.zeros((), dtype=jnp.float64),
        "bid_fee_normalizer_sum": jnp.zeros((), dtype=jnp.float64),
        "ask_fee_normalizer_sum": jnp.zeros((), dtype=jnp.float64),
    }


def _apply_submission_after_event(carry, *, after_event, params, trade, fired):
    """If `fired`, run the policy's after_event and update submission fees + state."""
    sub = carry["submission"]
    new_state, new_bid, new_ask = after_event(params, carry["submission_policy_state"], trade)
    selected_state, selected_bid, selected_ask = _maybe_update(
        fired,
        carry["submission_policy_state"],
        sub["bid_fee"],
        sub["ask_fee"],
        new_state,
        new_bid,
        new_ask,
    )
    new_sub = {**sub, "bid_fee": selected_bid, "ask_fee": selected_ask}
    return {**carry, "submission": new_sub, "submission_policy_state": selected_state}


def _arb_with_policy(carry, fair_price, timestamp, *, after_event, params):
    """Submission-side smooth arb that fires after_event when arb actually executed.

    For naturally-continuous arb, exactly one of buy/sell sides is positive
    (or both zero in the dead zone). The exact stack runs only the active
    branch, then runs after_event with that trade. We replicate that by:
    executing both branches sequentially (the inactive one is a no-op),
    then constructing a single trade dict from whichever branch fired.
    """
    sub = carry["submission"]
    gamma_ask = jnp.clip(1.0 - sub["ask_fee"], 0.0, 1.0)
    gamma_bid = jnp.clip(1.0 - sub["bid_fee"], 0.0, 1.0)

    # Buy-arb branch (AMM sells X, trader is the arb buyer)
    buy_raw = sub["rx"] - jnp.sqrt(
        jnp.maximum(sub["rx"] * sub["ry"] / jnp.maximum(gamma_ask * fair_price, MIN_DENOM), 0.0)
    )
    buy_amount_x = jnp.maximum(buy_raw, 0.0)
    sub_after_buy, buy_ax, buy_ay = _execute_sell_x(sub, buy_amount_x)
    buy_profit = jnp.maximum(buy_ax * fair_price - buy_ay, 0.0)
    buy_fired = buy_amount_x > 0.0

    buy_trade = {
        "amount_x": buy_ax,
        "amount_y": buy_ay,
        "is_buy": jnp.asarray(0.0, dtype=jnp.float64),  # AMM sold X
        "timestamp": jnp.asarray(timestamp, dtype=jnp.float64),
        "reserve_x": sub_after_buy["rx"],
        "reserve_y": sub_after_buy["ry"],
    }
    carry_after_buy_arb = _apply_submission_after_event(
        {**carry, "submission": sub_after_buy},
        after_event=after_event,
        params=params,
        trade=buy_trade,
        fired=buy_fired,
    )

    # Sell-arb branch (AMM buys X, trader is the arb seller).
    # IMPORTANT: build sub_after_sell from carry_after_buy_arb["submission"],
    # which carries any fee update from the buy-arb's after_event. Otherwise
    # overwriting carry["submission"] = sub_after_sell would clobber the update.
    fresh_sub = carry_after_buy_arb["submission"]
    sell_raw = (
        jnp.sqrt(
            jnp.maximum(
                sub["rx"] * sub["ry"] * gamma_bid / jnp.maximum(fair_price, MIN_DENOM),
                0.0,
            )
        )
        - sub["rx"]
    ) / jnp.maximum(gamma_bid, MIN_DENOM)
    sell_amount_x = jnp.maximum(sell_raw, 0.0)
    sub_after_sell, sell_ax, sell_ay = _execute_buy_x(fresh_sub, sell_amount_x)
    sell_profit = jnp.maximum(sell_ay - sell_ax * fair_price, 0.0)
    sell_fired = sell_amount_x > 0.0

    sell_trade = {
        "amount_x": sell_ax,
        "amount_y": sell_ay,
        "is_buy": jnp.asarray(1.0, dtype=jnp.float64),  # AMM bought X
        "timestamp": jnp.asarray(timestamp, dtype=jnp.float64),
        "reserve_x": sub_after_sell["rx"],
        "reserve_y": sub_after_sell["ry"],
    }
    carry_after_sell_arb = _apply_submission_after_event(
        {**carry_after_buy_arb, "submission": sub_after_sell},
        after_event=after_event,
        params=params,
        trade=sell_trade,
        fired=sell_fired,
    )

    arb_y = buy_ay + sell_ay
    profit = buy_profit + sell_profit
    return carry_after_sell_arb, arb_y, profit


def _route_one_order_with_policy(
    oc,
    *,
    buy_y,
    sell_y_notional,
    fair_price,
    timestamp,
    after_event,
    params,
):
    """Adaptive version of _route_one_order: applies after_event on submission
    after each fired sub-trade (using MIN_AMOUNT gates that mirror exact)."""
    # Buy route
    y_sub, y_norm = _split_buy(oc["submission"], oc["normalizer"], buy_y)
    sub_after_buy, ax_sub_buy, ay_sub_buy = _execute_buy_x_with_y(oc["submission"], y_sub)
    buy_fired_sub = y_sub > MIN_AMOUNT
    buy_trade_sub = {
        "amount_x": ax_sub_buy,
        "amount_y": ay_sub_buy,
        "is_buy": jnp.asarray(0.0, dtype=jnp.float64),
        "timestamp": jnp.asarray(timestamp, dtype=jnp.float64),
        "reserve_x": sub_after_buy["rx"],
        "reserve_y": sub_after_buy["ry"],
    }
    oc1 = _apply_submission_after_event(
        {**oc, "submission": sub_after_buy},
        after_event=after_event,
        params=params,
        trade=buy_trade_sub,
        fired=buy_fired_sub,
    )
    norm_after_buy, ax_norm_buy, ay_norm_buy = _execute_buy_x_with_y(oc["normalizer"], y_norm)
    oc1 = {**oc1, "normalizer": norm_after_buy}

    # Sell route: split based on POST-buy state of both venues (exact does same)
    sell_total_x = sell_y_notional / jnp.maximum(fair_price, MIN_DENOM)
    x_sub, x_norm = _split_sell(oc1["submission"], oc1["normalizer"], sell_total_x)
    sub_final, ax_sub_sell, ay_sub_sell = _execute_buy_x(oc1["submission"], x_sub)
    sell_fired_sub = x_sub > MIN_AMOUNT
    sell_trade_sub = {
        "amount_x": ax_sub_sell,
        "amount_y": ay_sub_sell,
        "is_buy": jnp.asarray(1.0, dtype=jnp.float64),
        "timestamp": jnp.asarray(timestamp, dtype=jnp.float64),
        "reserve_x": sub_final["rx"],
        "reserve_y": sub_final["ry"],
    }
    oc2 = _apply_submission_after_event(
        {**oc1, "submission": sub_final},
        after_event=after_event,
        params=params,
        trade=sell_trade_sub,
        fired=sell_fired_sub,
    )
    norm_final, ax_norm_sell, ay_norm_sell = _execute_buy_x(oc1["normalizer"], x_norm)
    oc2 = {**oc2, "normalizer": norm_final}

    edge_sub_delta = (ay_sub_buy - ax_sub_buy * fair_price) + (ax_sub_sell * fair_price - ay_sub_sell)
    edge_norm_delta = (ay_norm_buy - ax_norm_buy * fair_price) + (ax_norm_sell * fair_price - ay_norm_sell)
    retail_sub_delta = ay_sub_buy + ay_sub_sell
    retail_norm_delta = ay_norm_buy + ay_norm_sell

    oc2 = {
        **oc2,
        "edge_submission": oc2["edge_submission"] + edge_sub_delta,
        "edge_normalizer": oc2["edge_normalizer"] + edge_norm_delta,
        "retail_volume_submission_y": oc2["retail_volume_submission_y"] + retail_sub_delta,
        "retail_volume_normalizer_y": oc2["retail_volume_normalizer_y"] + retail_norm_delta,
    }
    return oc2


def _adaptive_metrics_challenge_from_arrays(
    config: ExactSimpleAMMConfig,
    arrays: dict,
    *,
    initial_policy_state,
    initial_bid,
    initial_ask,
    after_event,
    params,
) -> dict:
    """Array-driven challenge-mode rollout (no tape parsing inside).

    Mirror of `_adaptive_metrics_realistic_from_arrays`: the only seed-dependent
    inputs are arrays, so this is the function we vmap across seeds.
    """
    carry = _adaptive_initial_carry(
        config,
        initial_policy_state=initial_policy_state,
        initial_bid=initial_bid,
        initial_ask=initial_ask,
    )
    drift = (config.gbm_mu - 0.5 * config.gbm_sigma ** 2) * config.gbm_dt
    vol = config.gbm_sigma * jnp.sqrt(jnp.asarray(config.gbm_dt, dtype=jnp.float64))
    buy_prob = jnp.asarray(config.retail_buy_prob, dtype=jnp.float64)

    def step_fn(carry, xs):
        step_index, z, sizes_t, sides_t, mask_t = xs
        fair_price = carry["fair_price"] * jnp.exp(drift + vol * z)
        timestamp_f = step_index

        carry_arb, sub_arb_y, sub_arb_profit = _arb_with_policy(
            carry, fair_price, timestamp_f, after_event=after_event, params=params
        )
        # normalizer arb (no policy update; fixed-fee normalizer)
        norm_state, norm_arb_y, norm_arb_profit = _execute_arb(carry_arb["normalizer"], fair_price)
        carry_arb = {
            **carry_arb,
            "normalizer": norm_state,
            "edge_submission": carry_arb["edge_submission"] - sub_arb_profit,
            "edge_normalizer": carry_arb["edge_normalizer"] - norm_arb_profit,
            "fair_price": fair_price,
        }

        def order_fn(oc, order_inputs):
            size, side_u, mask = order_inputs
            is_buy = jnp.where(side_u < buy_prob, 1.0, 0.0)
            buy_y = mask * is_buy * size
            sell_y_notional = mask * (1.0 - is_buy) * size
            new_oc = _route_one_order_with_policy(
                oc,
                buy_y=buy_y,
                sell_y_notional=sell_y_notional,
                fair_price=fair_price,
                timestamp=timestamp_f,
                after_event=after_event,
                params=params,
            )
            return new_oc, None

        carry_after_orders, _ = jax.lax.scan(order_fn, carry_arb, (sizes_t, sides_t, mask_t))

        next_carry = {
            **carry_after_orders,
            "arb_volume_submission_y": carry["arb_volume_submission_y"] + sub_arb_y,
            "arb_volume_normalizer_y": carry["arb_volume_normalizer_y"] + norm_arb_y,
            "bid_fee_submission_sum": carry["bid_fee_submission_sum"]
            + carry_after_orders["submission"]["bid_fee"],
            "ask_fee_submission_sum": carry["ask_fee_submission_sum"]
            + carry_after_orders["submission"]["ask_fee"],
            "bid_fee_normalizer_sum": carry["bid_fee_normalizer_sum"]
            + carry_after_orders["normalizer"]["bid_fee"],
            "ask_fee_normalizer_sum": carry["ask_fee_normalizer_sum"]
            + carry_after_orders["normalizer"]["ask_fee"],
        }
        return next_carry, None

    gbm_normals = arrays["gbm_normals"]
    n_steps = gbm_normals.shape[0]
    step_indices = jnp.arange(n_steps, dtype=jnp.float64)
    final_carry, _ = jax.lax.scan(
        step_fn,
        carry,
        (step_indices, gbm_normals, arrays["sizes"], arrays["sides"], arrays["mask"]),
    )
    return _build_metrics(final_carry, config)


def _adaptive_metrics_challenge(
    config: ExactSimpleAMMConfig,
    tape: ChallengeTape,
    *,
    initial_policy_state,
    initial_bid,
    initial_ask,
    after_event,
    params,
) -> dict:
    arrays = challenge_tape_to_arrays(tape)
    # `challenge_tape_to_arrays` returns the static `n_steps` as a Python int —
    # drop it so the dict is JAX-clean (only array leaves).
    arrays = {k: v for k, v in arrays.items() if k != "n_steps"}
    return _adaptive_metrics_challenge_from_arrays(
        config,
        arrays,
        initial_policy_state=initial_policy_state,
        initial_bid=initial_bid,
        initial_ask=initial_ask,
        after_event=after_event,
        params=params,
    )


def _adaptive_metrics_realistic_from_arrays(
    config: ExactSimpleAMMConfig,
    arrays: dict,
    *,
    initial_policy_state,
    initial_bid,
    initial_ask,
    after_event,
    params,
) -> dict:
    """Array-driven realistic rollout (no tape parsing inside).

    `arrays` is the dict produced by `realistic_tape_to_arrays`. Splitting this
    out lets us call the rollout under `vmap` with a stacked-over-seeds arrays
    dict (each leaf has a leading batch dim), since the only seed-dependent
    inputs are the arrays themselves.
    """
    carry = _adaptive_initial_carry(
        config,
        initial_policy_state=initial_policy_state,
        initial_bid=initial_bid,
        initial_ask=initial_ask,
    )
    use_initial_state = config.retail_impact_scale_mode == "initial_state"
    use_submission_ref = config.retail_impact_reference_venue == "submission"

    def step_fn(carry, xs):
        step_index, log_return, impacts_t, mask_t = xs
        fair_price = carry["fair_price"] * jnp.exp(log_return)
        timestamp_f = step_index

        carry_arb, sub_arb_y, sub_arb_profit = _arb_with_policy(
            carry, fair_price, timestamp_f, after_event=after_event, params=params
        )
        norm_state, norm_arb_y, norm_arb_profit = _execute_arb(carry_arb["normalizer"], fair_price)
        carry_arb = {
            **carry_arb,
            "normalizer": norm_state,
            "edge_submission": carry_arb["edge_submission"] - sub_arb_profit,
            "edge_normalizer": carry_arb["edge_normalizer"] - norm_arb_profit,
            "fair_price": fair_price,
        }

        if use_initial_state:
            ref_rx = jnp.asarray(
                config.submission_initial_x if use_submission_ref else config.normalizer_initial_x,
                dtype=jnp.float64,
            )
            ref_ry = jnp.asarray(
                config.submission_initial_y if use_submission_ref else config.normalizer_initial_y,
                dtype=jnp.float64,
            )
            ref_bid = jnp.asarray(0.003, dtype=jnp.float64)
            ref_ask = jnp.asarray(0.003, dtype=jnp.float64)
        else:
            ref_state = carry_arb["submission"] if use_submission_ref else carry_arb["normalizer"]
            ref_rx = ref_state["rx"]
            ref_ry = ref_state["ry"]
            ref_bid = ref_state["bid_fee"]
            ref_ask = ref_state["ask_fee"]
        gamma_ask_ref = jnp.maximum(1.0 - ref_ask, MIN_DENOM)
        gamma_bid_ref = jnp.maximum(1.0 - ref_bid, MIN_DENOM)

        def order_fn(oc, order_inputs):
            impact_log, mask = order_inputs
            magnitude = jnp.abs(impact_log)
            is_buy = jnp.where(impact_log > 0.0, 1.0, 0.0)
            exp_half = jnp.exp(0.5 * magnitude) - 1.0
            buy_size_y = ref_ry * exp_half / gamma_ask_ref
            sell_size_x = ref_rx * exp_half / gamma_bid_ref
            sell_size_y_notional = sell_size_x * jnp.maximum(fair_price, MIN_DENOM)

            buy_y = mask * is_buy * buy_size_y
            sell_y_notional = mask * (1.0 - is_buy) * sell_size_y_notional
            new_oc = _route_one_order_with_policy(
                oc,
                buy_y=buy_y,
                sell_y_notional=sell_y_notional,
                fair_price=fair_price,
                timestamp=timestamp_f,
                after_event=after_event,
                params=params,
            )
            return new_oc, None

        carry_after_orders, _ = jax.lax.scan(order_fn, carry_arb, (impacts_t, mask_t))

        next_carry = {
            **carry_after_orders,
            "arb_volume_submission_y": carry["arb_volume_submission_y"] + sub_arb_y,
            "arb_volume_normalizer_y": carry["arb_volume_normalizer_y"] + norm_arb_y,
            "bid_fee_submission_sum": carry["bid_fee_submission_sum"]
            + carry_after_orders["submission"]["bid_fee"],
            "ask_fee_submission_sum": carry["ask_fee_submission_sum"]
            + carry_after_orders["submission"]["ask_fee"],
            "bid_fee_normalizer_sum": carry["bid_fee_normalizer_sum"]
            + carry_after_orders["normalizer"]["bid_fee"],
            "ask_fee_normalizer_sum": carry["ask_fee_normalizer_sum"]
            + carry_after_orders["normalizer"]["ask_fee"],
        }
        return next_carry, None

    log_returns = arrays["log_returns"]
    impact_logs = arrays["impact_logs"]
    mask = arrays["mask"]
    n_steps = log_returns.shape[0]
    step_indices = jnp.arange(n_steps, dtype=jnp.float64)
    final_carry, _ = jax.lax.scan(
        step_fn,
        carry,
        (step_indices, log_returns, impact_logs, mask),
    )
    return _build_metrics(final_carry, config)


def _adaptive_metrics_realistic(
    config: ExactSimpleAMMConfig,
    tape: RealisticTape,
    *,
    initial_policy_state,
    initial_bid,
    initial_ask,
    after_event,
    params,
) -> dict:
    arrays = realistic_tape_to_arrays(tape)
    return _adaptive_metrics_realistic_from_arrays(
        config,
        arrays,
        initial_policy_state=initial_policy_state,
        initial_bid=initial_bid,
        initial_ask=initial_ask,
        after_event=after_event,
        params=params,
    )


# ----- public per-policy entry points -----
def compact_metrics(config, tape, params) -> dict:
    """Run the SubmissionCompact tape-faithful rollout. Auto-detects mode."""
    init_state = compact_initial_policy_state(config.submission_initial_x, config.submission_initial_y)
    init_bid, init_ask = compact_initial_fees(params)
    if isinstance(tape, ChallengeTape):
        return _adaptive_metrics_challenge(
            config, tape,
            initial_policy_state=init_state,
            initial_bid=init_bid,
            initial_ask=init_ask,
            after_event=compact_after_event,
            params=params,
        )
    if isinstance(tape, RealisticTape):
        return _adaptive_metrics_realistic(
            config, tape,
            initial_policy_state=init_state,
            initial_bid=init_bid,
            initial_ask=init_ask,
            after_event=compact_after_event,
            params=params,
        )
    raise TypeError(f"Unsupported tape type: {type(tape)!r}")


def piecewise_metrics(config, tape, params) -> dict:
    """Run the Piecewise tape-faithful rollout. Auto-detects mode."""
    init_state = piecewise_initial_policy_state()
    init_bid, init_ask = piecewise_initial_fees(params)
    if isinstance(tape, ChallengeTape):
        return _adaptive_metrics_challenge(
            config, tape,
            initial_policy_state=init_state,
            initial_bid=init_bid,
            initial_ask=init_ask,
            after_event=piecewise_after_event,
            params=params,
        )
    if isinstance(tape, RealisticTape):
        return _adaptive_metrics_realistic(
            config, tape,
            initial_policy_state=init_state,
            initial_bid=init_bid,
            initial_ask=init_ask,
            after_event=piecewise_after_event,
            params=params,
        )
    raise TypeError(f"Unsupported tape type: {type(tape)!r}")


def compact_result(config, tape, params, *, seed: int = 0) -> DiffSimulationResult:
    return _build_result(compact_metrics(config, tape, params), seed)


def piecewise_result(config, tape, params, *, seed: int = 0) -> DiffSimulationResult:
    return _build_result(piecewise_metrics(config, tape, params), seed)


# ----- batched (vmap-friendly) realistic-mode entrypoints -----
def _pad_across_seeds(per_seed_arrays: list[dict]) -> dict:
    """Stack per-seed arrays into batched arrays, padding `impact_logs` / `mask`
    on the width axis so all seeds share a common width.

    Each input dict is the output of `realistic_tape_to_arrays`. All seeds must
    share the same `n_steps`.
    """
    _require_jax()
    n_steps_set = {int(a["log_returns"].shape[0]) for a in per_seed_arrays}
    if len(n_steps_set) != 1:
        raise ValueError(
            f"Batched rollout requires identical n_steps across seeds, got {n_steps_set}"
        )
    max_width = max(int(a["impact_logs"].shape[1]) for a in per_seed_arrays)

    def pad_width(a, width):
        cur = int(a.shape[1])
        if cur == width:
            return a
        pad = jnp.zeros((a.shape[0], width - cur), dtype=a.dtype)
        return jnp.concatenate([a, pad], axis=1)

    log_returns = jnp.stack([a["log_returns"] for a in per_seed_arrays], axis=0)
    impact_logs = jnp.stack(
        [pad_width(a["impact_logs"], max_width) for a in per_seed_arrays], axis=0
    )
    masks = jnp.stack(
        [pad_width(a["mask"], max_width) for a in per_seed_arrays], axis=0
    )
    return {
        "log_returns": log_returns,
        "impact_logs": impact_logs,
        "mask": masks,
        "n_steps": int(log_returns.shape[1]),
        "width": int(max_width),
    }


def realistic_tapes_to_batched_arrays(tapes: list[RealisticTape]) -> dict:
    """Convert a list of RealisticTape into a single batched arrays dict."""
    _require_jax()
    per_seed = [realistic_tape_to_arrays(t) for t in tapes]
    return _pad_across_seeds(per_seed)


def compact_metrics_realistic_batched(config, batched_arrays, params) -> dict:
    """Run the SubmissionCompact realistic rollout for K seeds in one vmap.

    `batched_arrays` has shape (K, n_steps) for log_returns and (K, n_steps, W)
    for impact_logs/mask (W = max width across seeds; padding rows have mask=0).
    All seeds share the same static `config`.
    """
    _require_jax()
    init_state = compact_initial_policy_state(
        config.submission_initial_x, config.submission_initial_y
    )
    init_bid, init_ask = compact_initial_fees(params)

    def single_seed(arr_one):
        return _adaptive_metrics_realistic_from_arrays(
            config,
            arr_one,
            initial_policy_state=init_state,
            initial_bid=init_bid,
            initial_ask=init_ask,
            after_event=compact_after_event,
            params=params,
        )

    return jax.vmap(single_seed)(
        {
            "log_returns": batched_arrays["log_returns"],
            "impact_logs": batched_arrays["impact_logs"],
            "mask": batched_arrays["mask"],
        }
    )


def _pad_across_seeds_challenge(per_seed_arrays: list[dict]) -> dict:
    """Stack per-seed challenge arrays into batched arrays, padding the
    per-step order width axis so all seeds share a common width.
    """
    _require_jax()
    n_steps_set = {int(a["gbm_normals"].shape[0]) for a in per_seed_arrays}
    if len(n_steps_set) != 1:
        raise ValueError(
            f"Batched rollout requires identical n_steps across seeds, got {n_steps_set}"
        )
    max_width = max(int(a["sizes"].shape[1]) for a in per_seed_arrays)

    def pad_width(a, width, fill=0.0):
        cur = int(a.shape[1])
        if cur == width:
            return a
        pad = jnp.full((a.shape[0], width - cur), fill, dtype=a.dtype)
        return jnp.concatenate([a, pad], axis=1)

    gbm_normals = jnp.stack([a["gbm_normals"] for a in per_seed_arrays], axis=0)
    sizes = jnp.stack(
        [pad_width(a["sizes"], max_width) for a in per_seed_arrays], axis=0
    )
    sides = jnp.stack(
        [pad_width(a["sides"], max_width, fill=1.0) for a in per_seed_arrays], axis=0
    )
    masks = jnp.stack(
        [pad_width(a["mask"], max_width) for a in per_seed_arrays], axis=0
    )
    return {
        "gbm_normals": gbm_normals,
        "sizes": sizes,
        "sides": sides,
        "mask": masks,
        "n_steps": int(gbm_normals.shape[1]),
        "width": int(max_width),
    }


def challenge_tapes_to_batched_arrays(tapes: list[ChallengeTape]) -> dict:
    """Convert a list of ChallengeTape into a single batched arrays dict."""
    _require_jax()
    per_seed = []
    for t in tapes:
        a = challenge_tape_to_arrays(t)
        per_seed.append({k: v for k, v in a.items() if k != "n_steps"})
    return _pad_across_seeds_challenge(per_seed)


def metrics_challenge_batched(
    config,
    batched_arrays,
    *,
    after_event,
    params,
    initial_policy_state,
    initial_bid,
    initial_ask,
) -> dict:
    """Run a generic-policy challenge-mode rollout for K seeds in one vmap.

    `batched_arrays` has shape (K, n_steps) for gbm_normals and (K, n_steps, W)
    for sizes/sides/mask. The policy is fully specified by `after_event`,
    `params`, `initial_policy_state`, and (`initial_bid`, `initial_ask`) — none
    of which depend on seed.
    """
    _require_jax()

    def single_seed(arr_one):
        return _adaptive_metrics_challenge_from_arrays(
            config,
            arr_one,
            initial_policy_state=initial_policy_state,
            initial_bid=initial_bid,
            initial_ask=initial_ask,
            after_event=after_event,
            params=params,
        )

    return jax.vmap(single_seed)(
        {
            "gbm_normals": batched_arrays["gbm_normals"],
            "sizes": batched_arrays["sizes"],
            "sides": batched_arrays["sides"],
            "mask": batched_arrays["mask"],
        }
    )


def metrics_realistic_batched(
    config,
    batched_arrays,
    *,
    after_event,
    params,
    initial_policy_state,
    initial_bid,
    initial_ask,
) -> dict:
    """Generic-policy mirror of `compact_metrics_realistic_batched`."""
    _require_jax()

    def single_seed(arr_one):
        return _adaptive_metrics_realistic_from_arrays(
            config,
            arr_one,
            initial_policy_state=initial_policy_state,
            initial_bid=initial_bid,
            initial_ask=initial_ask,
            after_event=after_event,
            params=params,
        )

    return jax.vmap(single_seed)(
        {
            "log_returns": batched_arrays["log_returns"],
            "impact_logs": batched_arrays["impact_logs"],
            "mask": batched_arrays["mask"],
        }
    )


def piecewise_metrics_realistic_batched(config, batched_arrays, params) -> dict:
    """Run the Piecewise realistic rollout for K seeds in one vmap."""
    _require_jax()
    init_state = piecewise_initial_policy_state()
    init_bid, init_ask = piecewise_initial_fees(params)

    def single_seed(arr_one):
        return _adaptive_metrics_realistic_from_arrays(
            config,
            arr_one,
            initial_policy_state=init_state,
            initial_bid=init_bid,
            initial_ask=init_ask,
            after_event=piecewise_after_event,
            params=params,
        )

    return jax.vmap(single_seed)(
        {
            "log_returns": batched_arrays["log_returns"],
            "impact_logs": batched_arrays["impact_logs"],
            "mask": batched_arrays["mask"],
        }
    )
