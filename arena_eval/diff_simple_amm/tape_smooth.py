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
