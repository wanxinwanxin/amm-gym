"""Retail order helpers for the diff simple-AMM rewrite."""

from __future__ import annotations

import math

try:
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - exercised only when jax is absent
    jnp = None

from arena_eval.exact_simple_amm.config import ExactSimpleAMMConfig
from arena_eval.diff_simple_amm.types import AMMState, ChallengeTape, RealisticTape, RetailOrder


def decode_challenge_orders(
    *,
    config: ExactSimpleAMMConfig,
    tape: ChallengeTape,
    step: int,
) -> tuple[RetailOrder, ...]:
    """Decode one challenge-mode step worth of retail orders from an explicit tape."""

    count = tape.order_counts[step]
    sizes = tape.order_sizes[step]
    sides = tape.order_side_uniforms[step]
    if count == 0:
        return ()
    return tuple(
        RetailOrder(
            side="buy" if side < config.retail_buy_prob else "sell",
            size=float(size),
        )
        for size, side in zip(sizes, sides)
    )


def decode_realistic_orders(
    *,
    config: ExactSimpleAMMConfig,
    tape: RealisticTape,
    step: int,
    fair_price: float,
    reference_state: AMMState | None,
) -> tuple[RetailOrder, ...]:
    """Decode one realistic-mode step worth of retail orders from an explicit tape."""

    if step >= len(tape.impact_logs):
        return ()

    def reference_tuple() -> tuple[float, float, float, float]:
        if reference_state is None or config.retail_impact_scale_mode == "initial_state":
            return (config.initial_x, config.initial_y, 0.003, 0.003)
        return (
            float(reference_state.reserve_x),
            float(reference_state.reserve_y),
            float(reference_state.bid_fee),
            float(reference_state.ask_fee),
        )

    reserve_x, reserve_y, bid_fee, ask_fee = reference_tuple()
    orders: list[RetailOrder] = []
    for impact_log in tape.impact_logs[step]:
        if abs(float(impact_log)) <= 1e-12:
            continue
        magnitude = abs(float(impact_log))
        exp_half = math.exp(0.5 * magnitude) - 1.0
        if impact_log > 0.0:
            gamma = max(1e-12, 1.0 - ask_fee)
            gross_y = reserve_y * exp_half / gamma
            orders.append(RetailOrder(side="buy", size=float(gross_y)))
            continue
        gamma = max(1e-12, 1.0 - bid_fee)
        gross_x = reserve_x * exp_half / gamma
        gross_y_notional = gross_x * max(float(fair_price), 1e-12)
        orders.append(RetailOrder(side="sell", size=float(gross_y_notional)))
    return tuple(orders)


def _pad_rows(rows: tuple[tuple[float, ...], ...], *, width: int, fill: float):
    padded = [tuple(row) + (fill,) * (width - len(row)) for row in rows]
    return jnp.asarray(padded, dtype=jnp.float32)


def challenge_tape_to_smooth_arrays(tape: ChallengeTape) -> dict[str, object]:
    """Convert the challenge tape into fixed-shape arrays for smooth training."""

    if jnp is None:
        raise RuntimeError("jax is required for smooth training arrays")
    width = max(int(tape.max_orders_per_step), 1)

    arrival_uniforms = tape.smooth_arrival_uniforms or tuple(((0.5,) * width) for _ in tape.gbm_normals)
    size_normals = tape.smooth_size_normals or tuple(((0.0,) * width) for _ in tape.gbm_normals)
    side_uniforms = tape.smooth_side_uniforms or tuple(((0.5,) * width) for _ in tape.gbm_normals)

    return {
        "gbm_normals": jnp.asarray(tape.gbm_normals, dtype=jnp.float32),
        "order_counts": jnp.asarray(tape.order_counts, dtype=jnp.int32),
        "order_sizes": _pad_rows(tape.order_sizes, width=width, fill=0.0),
        "order_side_uniforms": _pad_rows(tape.order_side_uniforms, width=width, fill=1.0),
        "arrival_uniforms": _pad_rows(arrival_uniforms, width=width, fill=0.5),
        "size_normals": _pad_rows(size_normals, width=width, fill=0.0),
        "side_uniforms": _pad_rows(side_uniforms, width=width, fill=0.5),
        "width": width,
    }


def realistic_tape_to_smooth_arrays(tape: RealisticTape) -> dict[str, object]:
    """Convert the realistic tape into fixed-shape arrays for smooth training."""

    if jnp is None:
        raise RuntimeError("jax is required for smooth training arrays")
    width = max(int(tape.max_orders_per_step), 1)
    arrival_uniforms = tape.smooth_arrival_uniforms or tuple(((0.5,) * width) for _ in tape.log_returns)
    impact_percentiles = tape.smooth_impact_percentiles or tuple(((50.0,) * width) for _ in tape.log_returns)
    return {
        "log_returns": jnp.asarray(tape.log_returns, dtype=jnp.float32),
        "regimes": jnp.asarray(tape.regimes, dtype=jnp.int32),
        "return_percentiles": jnp.asarray(tape.return_percentiles, dtype=jnp.float32),
        "order_counts": jnp.asarray(tape.order_counts, dtype=jnp.int32),
        "impact_logs": _pad_rows(tape.impact_logs, width=width, fill=0.0),
        "arrival_uniforms": _pad_rows(arrival_uniforms, width=width, fill=0.5),
        "impact_percentiles": _pad_rows(impact_percentiles, width=width, fill=50.0),
        "width": width,
    }
