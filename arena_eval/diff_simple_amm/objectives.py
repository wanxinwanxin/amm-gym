"""Differentiable smooth-training objectives for the simple-AMM rewrite."""

from __future__ import annotations

import math

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - exercised only when jax is absent
    jax = None
    jnp = None

from arena_eval.exact_simple_amm.config import ExactSimpleAMMConfig
from arena_policies.piecewise_controller import PiecewiseControllerParams
from arena_policies.submission_safe import SubmissionCompactParams

from .orders import challenge_tape_to_smooth_arrays, realistic_tape_to_smooth_arrays
from .realistic_dynamics import load_realistic_artifacts
from .relaxations import smooth_clip, smooth_gate, smooth_or, smooth_positive, smooth_trade_amount
from .types import ChallengeTape, DiffBatchResult, DiffSimulationResult, RealisticTape, SmoothRelaxationConfig


SUBMISSION_COMPACT_PARAM_NAMES = (
    "base_fee",
    "min_fee",
    "max_fee",
    "flow_fast_decay",
    "flow_slow_decay",
    "size_fast_decay",
    "size_slow_decay",
    "gap_fast_decay",
    "gap_slow_decay",
    "toxicity_decay",
    "toxicity_weight",
    "base_spread",
    "flow_mid_weight",
    "size_mid_weight",
    "gap_mid_weight",
    "skew_weight",
    "toxicity_side_weight",
    "hot_gap_threshold",
    "big_trade_threshold",
    "hot_fee_bump",
)

PIECEWISE_PARAM_NAMES = (
    "base_fee",
    "base_spread",
    "signal_decay",
    "toxicity_decay",
    "small_trade_threshold",
    "large_trade_threshold",
    "continuation_small",
    "continuation_medium",
    "continuation_large",
    "reversal_small",
    "reversal_medium",
    "reversal_large",
    "continuation_to_same_side",
    "continuation_to_cross_side",
    "toxicity_to_mid",
    "toxicity_to_side",
)

_S0_INITIAL_X = 0
_S1_INITIAL_Y = 1
_S2_LAST_TIMESTAMP = 2
_S3_LAST_SIDE = 3
_S4_LAST_SIZE_RATIO = 4
_S5_STREAK_LEN = 5
_S6_BUY_FLOW_FAST = 6
_S7_SELL_FLOW_FAST = 7
_S8_BUY_FLOW_SLOW = 8
_S9_SELL_FLOW_SLOW = 9
_S10_SIZE_FAST = 10
_S11_SIZE_SLOW = 11
_S12_GAP_FAST = 12
_S13_GAP_SLOW = 13
_S14_TOX_BID = 14
_S15_TOX_ASK = 15
_S16_FAIR_FAST = 16
_S17_FAIR_SLOW = 17
_S18_INITIALIZED = 18

_P0_LAST_TIMESTAMP = 0
_P1_LAST_SIDE = 1
_P2_BID_SIGNAL = 2
_P3_ASK_SIGNAL = 3
_P4_BID_TOXICITY = 4
_P5_ASK_TOXICITY = 5
_P6_INITIALIZED = 6


def _require_jax() -> None:
    if jax is None or jnp is None:
        raise RuntimeError("jax is required for smooth differentiable objectives")


def submission_compact_param_vector(params: SubmissionCompactParams | None = None):
    """Convert compact policy parameters into a JAX vector."""

    _require_jax()
    p = (params or SubmissionCompactParams()).normalized()
    return jnp.asarray([getattr(p, name) for name in SUBMISSION_COMPACT_PARAM_NAMES], dtype=jnp.float32)


def piecewise_param_vector(params: PiecewiseControllerParams | None = None):
    """Convert piecewise policy parameters into a JAX vector."""

    _require_jax()
    p = (params or PiecewiseControllerParams()).normalized()
    return jnp.asarray([getattr(p, name) for name in PIECEWISE_PARAM_NAMES], dtype=jnp.float32)


def challenge_env_vector(config: ExactSimpleAMMConfig):
    """Extract differentiable challenge-mode environment parameters."""

    _require_jax()
    return jnp.asarray(
        [
            float(config.gbm_sigma),
            float(config.retail_arrival_rate),
            float(config.retail_mean_size),
            float(config.retail_buy_prob),
        ],
        dtype=jnp.float32,
    )


def realistic_env_vector(config: ExactSimpleAMMConfig):
    """Extract differentiable realistic-mode environment parameters."""

    _require_jax()
    return jnp.asarray([float(config.retail_arrival_rate)], dtype=jnp.float32)


def submission_compact_bounds():
    """Return lower and upper bounds for the differentiable compact vector."""

    _require_jax()
    lower = submission_compact_param_vector(
        SubmissionCompactParams(
            base_fee=0.0,
            min_fee=0.0,
            max_fee=0.0,
            flow_fast_decay=0.0,
            flow_slow_decay=0.0,
            size_fast_decay=0.0,
            size_slow_decay=0.0,
            gap_fast_decay=0.0,
            gap_slow_decay=0.0,
            toxicity_decay=0.0,
            toxicity_weight=0.0,
            base_spread=0.0,
            flow_mid_weight=-0.1,
            size_mid_weight=-0.1,
            gap_mid_weight=-0.02,
            skew_weight=-0.2,
            toxicity_side_weight=0.0,
            hot_gap_threshold=0.0,
            big_trade_threshold=0.0001,
            hot_fee_bump=0.0,
        )
    )
    upper = submission_compact_param_vector(
        SubmissionCompactParams(
            base_fee=0.1,
            min_fee=0.1,
            max_fee=0.1,
            flow_fast_decay=0.999,
            flow_slow_decay=0.999,
            size_fast_decay=0.999,
            size_slow_decay=0.999,
            gap_fast_decay=0.999,
            gap_slow_decay=0.999,
            toxicity_decay=0.999,
            toxicity_weight=5.0,
            base_spread=0.03,
            flow_mid_weight=0.1,
            size_mid_weight=0.2,
            gap_mid_weight=0.02,
            skew_weight=0.2,
            toxicity_side_weight=0.2,
            hot_gap_threshold=8.0,
            big_trade_threshold=0.05,
            hot_fee_bump=0.03,
        )
    )
    return (lower, upper)


def piecewise_bounds():
    """Return lower and upper bounds for the differentiable piecewise vector."""

    _require_jax()
    lower = piecewise_param_vector(
        PiecewiseControllerParams(
            base_fee=0.0,
            base_spread=0.0,
            signal_decay=0.0,
            toxicity_decay=0.0,
            small_trade_threshold=0.0001,
            large_trade_threshold=0.0002,
            continuation_small=-0.05,
            continuation_medium=-0.05,
            continuation_large=-0.05,
            reversal_small=0.0,
            reversal_medium=0.0,
            reversal_large=0.0,
            continuation_to_same_side=-4.0,
            continuation_to_cross_side=-4.0,
            toxicity_to_mid=0.0,
            toxicity_to_side=0.0,
        )
    )
    upper = piecewise_param_vector(
        PiecewiseControllerParams(
            base_fee=0.1,
            base_spread=0.1,
            signal_decay=0.999,
            toxicity_decay=0.999,
            small_trade_threshold=0.05,
            large_trade_threshold=0.1,
            continuation_small=0.05,
            continuation_medium=0.05,
            continuation_large=0.05,
            reversal_small=0.08,
            reversal_medium=0.08,
            reversal_large=0.08,
            continuation_to_same_side=4.0,
            continuation_to_cross_side=4.0,
            toxicity_to_mid=0.5,
            toxicity_to_side=0.5,
        )
    )
    return (lower, upper)


def expected_submission_edge(
    param_vector,
    *,
    config: ExactSimpleAMMConfig,
    tape: ChallengeTape | RealisticTape,
    env_vector=None,
    relaxation: SmoothRelaxationConfig = SmoothRelaxationConfig(),
):
    """Expected submission edge under the smooth differentiable surrogate."""

    metrics = smooth_submission_compact_metrics(
        param_vector,
        config=config,
        tape=tape,
        env_vector=env_vector,
        relaxation=relaxation,
    )
    return metrics["edge_submission"]


def expected_piecewise_edge(
    param_vector,
    *,
    config: ExactSimpleAMMConfig,
    tape: ChallengeTape | RealisticTape,
    env_vector=None,
    relaxation: SmoothRelaxationConfig = SmoothRelaxationConfig(),
):
    """Expected submission edge for the piecewise surrogate."""

    metrics = smooth_piecewise_metrics(
        param_vector,
        config=config,
        tape=tape,
        env_vector=env_vector,
        relaxation=relaxation,
    )
    return metrics["edge_submission"]


def expected_policy_edge(
    param_vector,
    *,
    policy_family: str,
    config: ExactSimpleAMMConfig,
    tape: ChallengeTape | RealisticTape,
    env_vector=None,
    relaxation: SmoothRelaxationConfig = SmoothRelaxationConfig(),
):
    """Dispatch the smooth objective by policy family."""

    if policy_family == "submission_compact":
        return expected_submission_edge(
            param_vector,
            config=config,
            tape=tape,
            env_vector=env_vector,
            relaxation=relaxation,
        )
    if policy_family == "piecewise":
        return expected_piecewise_edge(
            param_vector,
            config=config,
            tape=tape,
            env_vector=env_vector,
            relaxation=relaxation,
        )
    raise ValueError(f"Unsupported diff policy_family: {policy_family}")


def expected_submission_edge_batch(
    param_vector,
    *,
    config: ExactSimpleAMMConfig,
    tapes: tuple[ChallengeTape | RealisticTape, ...],
    env_vector=None,
    relaxation: SmoothRelaxationConfig = SmoothRelaxationConfig(),
):
    """Mean submission edge across multiple tapes."""

    _require_jax()
    values = jnp.stack(
        [
            expected_submission_edge(
                param_vector,
                config=config,
                tape=tape,
                env_vector=env_vector,
                relaxation=relaxation,
            )
            for tape in tapes
        ]
    )
    return jnp.mean(values)


def smooth_submission_compact_metrics(
    param_vector,
    *,
    config: ExactSimpleAMMConfig,
    tape: ChallengeTape | RealisticTape,
    env_vector=None,
    relaxation: SmoothRelaxationConfig = SmoothRelaxationConfig(),
) -> dict[str, object]:
    """Run the smooth differentiable surrogate for the compact policy."""

    return smooth_policy_metrics(
        param_vector,
        policy_family="submission_compact",
        config=config,
        tape=tape,
        env_vector=env_vector,
        relaxation=relaxation,
    )


def smooth_piecewise_metrics(
    param_vector,
    *,
    config: ExactSimpleAMMConfig,
    tape: ChallengeTape | RealisticTape,
    env_vector=None,
    relaxation: SmoothRelaxationConfig = SmoothRelaxationConfig(),
) -> dict[str, object]:
    """Run the smooth differentiable surrogate for the piecewise policy."""

    return smooth_policy_metrics(
        param_vector,
        policy_family="piecewise",
        config=config,
        tape=tape,
        env_vector=env_vector,
        relaxation=relaxation,
    )


def smooth_policy_metrics(
    param_vector,
    *,
    policy_family: str,
    config: ExactSimpleAMMConfig,
    tape: ChallengeTape | RealisticTape,
    env_vector=None,
    relaxation: SmoothRelaxationConfig = SmoothRelaxationConfig(),
) -> dict[str, object]:
    """Run the smooth differentiable surrogate for a supported policy family."""

    _require_jax()
    params = _normalize_policy_vector(policy_family, jnp.asarray(param_vector, dtype=jnp.float32))
    if isinstance(tape, ChallengeTape):
        smooth_tape = challenge_tape_to_smooth_arrays(tape)
        env = _normalize_env_vector(
            jnp.asarray(env_vector, dtype=jnp.float32) if env_vector is not None else challenge_env_vector(config)
        )
        return _smooth_rollout_challenge(
            policy_family=policy_family,
            params=params,
            env=env,
            config=config,
            smooth_tape=smooth_tape,
            relaxation=relaxation,
        )
    if isinstance(tape, RealisticTape):
        smooth_tape = realistic_tape_to_smooth_arrays(tape)
        env = _normalize_realistic_env_vector(
            jnp.asarray(env_vector, dtype=jnp.float32) if env_vector is not None else realistic_env_vector(config)
        )
        return _smooth_rollout_realistic(
            policy_family=policy_family,
            params=params,
            env=env,
            config=config,
            smooth_tape=smooth_tape,
            relaxation=relaxation,
        )
    raise TypeError(f"Unsupported tape type: {type(tape)!r}")


def smooth_submission_compact_result(
    param_vector,
    *,
    config: ExactSimpleAMMConfig,
    tape: ChallengeTape | RealisticTape,
    env_vector=None,
    relaxation: SmoothRelaxationConfig = SmoothRelaxationConfig(),
    seed: int = 0,
) -> DiffSimulationResult:
    """Materialize a `DiffSimulationResult` from the smooth surrogate."""

    metrics = smooth_submission_compact_metrics(
        param_vector,
        config=config,
        tape=tape,
        env_vector=env_vector,
        relaxation=relaxation,
    )
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
        metadata={"mode": "smooth_train"},
    )


def smooth_piecewise_result(
    param_vector,
    *,
    config: ExactSimpleAMMConfig,
    tape: ChallengeTape | RealisticTape,
    env_vector=None,
    relaxation: SmoothRelaxationConfig = SmoothRelaxationConfig(),
    seed: int = 0,
) -> DiffSimulationResult:
    """Materialize a `DiffSimulationResult` for the piecewise surrogate."""

    metrics = smooth_piecewise_metrics(
        param_vector,
        config=config,
        tape=tape,
        env_vector=env_vector,
        relaxation=relaxation,
    )
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
        metadata={"mode": "smooth_train"},
    )


def smooth_submission_compact_batch_result(
    param_vector,
    *,
    config: ExactSimpleAMMConfig,
    tapes: tuple[ChallengeTape | RealisticTape, ...],
    env_vector=None,
    relaxation: SmoothRelaxationConfig = SmoothRelaxationConfig(),
) -> DiffBatchResult:
    """Aggregate the smooth surrogate over multiple tapes."""

    metrics = [
        smooth_submission_compact_metrics(
            param_vector,
            config=config,
            tape=tape,
            env_vector=env_vector,
            relaxation=relaxation,
        )
        for tape in tapes
    ]
    edge_submission = jnp.mean(jnp.stack([item["edge_submission"] for item in metrics]))
    edge_normalizer = jnp.mean(jnp.stack([item["edge_normalizer"] for item in metrics]))
    return DiffBatchResult(
        score=float(edge_submission),
        edge_mean_submission=float(edge_submission),
        edge_mean_normalizer=float(edge_normalizer),
        edge_advantage_mean=float(edge_submission - edge_normalizer),
        metadata={"n_tapes": len(tapes), "mode": "smooth_train"},
    )


def smooth_piecewise_batch_result(
    param_vector,
    *,
    config: ExactSimpleAMMConfig,
    tapes: tuple[ChallengeTape | RealisticTape, ...],
    env_vector=None,
    relaxation: SmoothRelaxationConfig = SmoothRelaxationConfig(),
) -> DiffBatchResult:
    """Aggregate the piecewise smooth surrogate over multiple tapes."""

    metrics = [
        smooth_piecewise_metrics(
            param_vector,
            config=config,
            tape=tape,
            env_vector=env_vector,
            relaxation=relaxation,
        )
        for tape in tapes
    ]
    edge_submission = jnp.mean(jnp.stack([item["edge_submission"] for item in metrics]))
    edge_normalizer = jnp.mean(jnp.stack([item["edge_normalizer"] for item in metrics]))
    return DiffBatchResult(
        score=float(edge_submission),
        edge_mean_submission=float(edge_submission),
        edge_mean_normalizer=float(edge_normalizer),
        edge_advantage_mean=float(edge_submission - edge_normalizer),
        metadata={"n_tapes": len(tapes), "mode": "smooth_train"},
    )


def _normalize_submission_compact_vector(vector):
    lower, upper = submission_compact_bounds()
    bounded = jnp.clip(vector, lower, upper)
    min_fee = bounded[1]
    max_fee = jnp.clip(bounded[2], min_fee, 0.1)
    base_fee = jnp.clip(bounded[0], min_fee, max_fee)
    return bounded.at[0].set(base_fee).at[1].set(min_fee).at[2].set(max_fee)


def _normalize_piecewise_vector(vector):
    lower, upper = piecewise_bounds()
    bounded = jnp.clip(vector, lower, upper)
    small = bounded[4]
    large = jnp.clip(bounded[5], small + 1e-4, 0.1)
    return bounded.at[4].set(small).at[5].set(large)


def _normalize_policy_vector(policy_family: str, vector):
    if policy_family == "submission_compact":
        return _normalize_submission_compact_vector(vector)
    if policy_family == "piecewise":
        return _normalize_piecewise_vector(vector)
    raise ValueError(f"Unsupported diff policy_family: {policy_family}")


def _normalize_env_vector(vector):
    sigma = jnp.clip(vector[0], 1e-6, 0.05)
    arrival_rate = jnp.clip(vector[1], 0.0, 5.0)
    mean_size = jnp.clip(vector[2], 1e-3, 500.0)
    buy_prob = jnp.clip(vector[3], 1e-4, 1.0 - 1e-4)
    return jnp.asarray([sigma, arrival_rate, mean_size, buy_prob], dtype=jnp.float32)


def _normalize_realistic_env_vector(vector):
    arrival_rate = jnp.clip(vector[0], 0.0, 5.0)
    return jnp.asarray([arrival_rate], dtype=jnp.float32)


def _smooth_rollout_challenge(*, policy_family: str, params, env, config, smooth_tape, relaxation):
    initial_submission = _amm_state(
        reserve_x=config.initial_x,
        reserve_y=config.initial_y,
        bid_fee=_policy_initial_bid_fee(policy_family, params),
        ask_fee=_policy_initial_ask_fee(policy_family, params),
    )
    initial_normalizer = _amm_state(
        reserve_x=config.initial_x,
        reserve_y=config.initial_y,
        bid_fee=0.003,
        ask_fee=0.003,
    )
    initial_policy_state = _initial_policy_state(policy_family, config.initial_x, config.initial_y)
    initial_carry = {
        "submission": initial_submission,
        "normalizer": initial_normalizer,
        "policy_state": initial_policy_state,
        "fair_price": jnp.asarray(config.initial_price, dtype=jnp.float32),
        "edge_submission": jnp.asarray(0.0, dtype=jnp.float32),
        "edge_normalizer": jnp.asarray(0.0, dtype=jnp.float32),
        "retail_volume_submission_y": jnp.asarray(0.0, dtype=jnp.float32),
        "retail_volume_normalizer_y": jnp.asarray(0.0, dtype=jnp.float32),
        "arb_volume_submission_y": jnp.asarray(0.0, dtype=jnp.float32),
        "arb_volume_normalizer_y": jnp.asarray(0.0, dtype=jnp.float32),
        "bid_fee_submission_sum": jnp.asarray(0.0, dtype=jnp.float32),
        "ask_fee_submission_sum": jnp.asarray(0.0, dtype=jnp.float32),
        "bid_fee_normalizer_sum": jnp.asarray(0.0, dtype=jnp.float32),
        "ask_fee_normalizer_sum": jnp.asarray(0.0, dtype=jnp.float32),
    }
    scan_inputs = (
        jnp.arange(config.n_steps, dtype=jnp.float32),
        smooth_tape["gbm_normals"],
        smooth_tape["arrival_uniforms"],
        smooth_tape["size_normals"],
        smooth_tape["side_uniforms"],
    )

    def step_fn(carry, xs):
        step_index, z_t, arrival_u_t, size_z_t, side_u_t = xs
        fair_price = _next_fair_price(
            prev_price=carry["fair_price"],
            mu=config.gbm_mu,
            sigma=env[0],
            dt=config.gbm_dt,
            z=z_t,
        )

        submission_state, policy_state, submission_arb_amount_y, submission_arb_profit = _smooth_execute_arb(
            policy_family=policy_family,
            state=carry["submission"],
            policy_state=carry["policy_state"],
            params=params,
            fair_price=fair_price,
            timestamp=step_index,
            relaxation=relaxation,
        )
        normalizer_state, _, normalizer_arb_amount_y, normalizer_arb_profit = _smooth_execute_arb(
            policy_family="fixed_fee",
            state=carry["normalizer"],
            policy_state=None,
            params=None,
            fair_price=fair_price,
            timestamp=step_index,
            relaxation=relaxation,
        )

        carry_after_arb = {
            **carry,
            "submission": submission_state,
            "normalizer": normalizer_state,
            "policy_state": policy_state,
            "fair_price": fair_price,
            "edge_submission": carry["edge_submission"] - submission_arb_profit,
            "edge_normalizer": carry["edge_normalizer"] - normalizer_arb_profit,
            "arb_volume_submission_y": carry["arb_volume_submission_y"] + submission_arb_amount_y,
            "arb_volume_normalizer_y": carry["arb_volume_normalizer_y"] + normalizer_arb_amount_y,
        }

        def slot_fn(slot_carry, slot_inputs):
            arrival_u, size_z, side_u = slot_inputs
            return _smooth_process_order_slot(
                carry=slot_carry,
                step_index=step_index,
                arrival_u=arrival_u,
                size_z=size_z,
                side_u=side_u,
                policy_family=policy_family,
                params=params,
                env=env,
                width=smooth_tape["width"],
                size_sigma=config.retail_size_sigma,
                relaxation=relaxation,
            ), None

        carry_after_slots, _ = jax.lax.scan(slot_fn, carry_after_arb, (arrival_u_t, size_z_t, side_u_t))
        return (
            {
                **carry_after_slots,
                "bid_fee_submission_sum": carry_after_slots["bid_fee_submission_sum"] + carry_after_slots["submission"]["bid_fee"],
                "ask_fee_submission_sum": carry_after_slots["ask_fee_submission_sum"] + carry_after_slots["submission"]["ask_fee"],
                "bid_fee_normalizer_sum": carry_after_slots["bid_fee_normalizer_sum"] + carry_after_slots["normalizer"]["bid_fee"],
                "ask_fee_normalizer_sum": carry_after_slots["ask_fee_normalizer_sum"] + carry_after_slots["normalizer"]["ask_fee"],
            },
            None,
        )

    final_carry, _ = jax.lax.scan(step_fn, initial_carry, scan_inputs)
    initial_value = config.initial_x * config.initial_price + config.initial_y
    pnl_submission = _mark_to_market(final_carry["submission"], final_carry["fair_price"]) - initial_value
    pnl_normalizer = _mark_to_market(final_carry["normalizer"], final_carry["fair_price"]) - initial_value
    steps = max(int(config.n_steps), 1)
    return {
        "edge_submission": final_carry["edge_submission"],
        "edge_normalizer": final_carry["edge_normalizer"],
        "pnl_submission": pnl_submission,
        "pnl_normalizer": pnl_normalizer,
        "retail_volume_submission_y": final_carry["retail_volume_submission_y"],
        "retail_volume_normalizer_y": final_carry["retail_volume_normalizer_y"],
        "arb_volume_submission_y": final_carry["arb_volume_submission_y"],
        "arb_volume_normalizer_y": final_carry["arb_volume_normalizer_y"],
        "average_bid_fee_submission": final_carry["bid_fee_submission_sum"] / steps,
        "average_ask_fee_submission": final_carry["ask_fee_submission_sum"] / steps,
        "average_bid_fee_normalizer": final_carry["bid_fee_normalizer_sum"] / steps,
        "average_ask_fee_normalizer": final_carry["ask_fee_normalizer_sum"] / steps,
    }


def _smooth_rollout_realistic(*, policy_family: str, params, env, config, smooth_tape, relaxation):
    artifacts = load_realistic_artifacts(config)
    regime_pct_grid = jnp.asarray(artifacts["regime_pct_grid"], dtype=jnp.float32)
    regime_invcdf = jnp.asarray(artifacts["regime_invcdf"], dtype=jnp.float32)
    transition_matrix = jnp.asarray(artifacts["transition_matrix"], dtype=jnp.float32)
    impact_pct_grid = jnp.asarray(artifacts["impact_pct_grid"], dtype=jnp.float32)
    impact_values = jnp.asarray(artifacts["impact_values"], dtype=jnp.float32)

    initial_submission = _amm_state(
        reserve_x=config.initial_x,
        reserve_y=config.initial_y,
        bid_fee=_policy_initial_bid_fee(policy_family, params),
        ask_fee=_policy_initial_ask_fee(policy_family, params),
    )
    initial_normalizer = _amm_state(
        reserve_x=config.initial_x,
        reserve_y=config.initial_y,
        bid_fee=0.003,
        ask_fee=0.003,
    )
    initial_policy_state = _initial_policy_state(policy_family, config.initial_x, config.initial_y)
    n_regimes = int(transition_matrix.shape[0])
    regime_index = max(0, min(int(config.regime_start) - 1, n_regimes - 1))
    initial_regime_probs = jax.nn.one_hot(regime_index, n_regimes, dtype=jnp.float32)
    initial_carry = {
        "submission": initial_submission,
        "normalizer": initial_normalizer,
        "policy_state": initial_policy_state,
        "regime_probs": initial_regime_probs,
        "fair_price": jnp.asarray(config.initial_price, dtype=jnp.float32),
        "edge_submission": jnp.asarray(0.0, dtype=jnp.float32),
        "edge_normalizer": jnp.asarray(0.0, dtype=jnp.float32),
        "retail_volume_submission_y": jnp.asarray(0.0, dtype=jnp.float32),
        "retail_volume_normalizer_y": jnp.asarray(0.0, dtype=jnp.float32),
        "arb_volume_submission_y": jnp.asarray(0.0, dtype=jnp.float32),
        "arb_volume_normalizer_y": jnp.asarray(0.0, dtype=jnp.float32),
        "bid_fee_submission_sum": jnp.asarray(0.0, dtype=jnp.float32),
        "ask_fee_submission_sum": jnp.asarray(0.0, dtype=jnp.float32),
        "bid_fee_normalizer_sum": jnp.asarray(0.0, dtype=jnp.float32),
        "ask_fee_normalizer_sum": jnp.asarray(0.0, dtype=jnp.float32),
    }
    scan_inputs = (
        jnp.arange(config.n_steps, dtype=jnp.float32),
        smooth_tape["return_percentiles"],
        smooth_tape["arrival_uniforms"],
        smooth_tape["impact_percentiles"],
    )

    def step_fn(carry, xs):
        step_index, return_pct_t, arrival_u_t, impact_pct_t = xs
        next_regime_probs = jnp.matmul(carry["regime_probs"], transition_matrix)
        regime_returns = jnp.stack(
            [
                jnp.interp(return_pct_t, regime_pct_grid, regime_invcdf[:, idx])
                for idx in range(int(regime_invcdf.shape[1]))
            ]
        )
        log_return = jnp.sum(next_regime_probs * regime_returns)
        fair_price = carry["fair_price"] * jnp.exp(log_return)

        submission_state, policy_state, submission_arb_amount_y, submission_arb_profit = _smooth_execute_arb(
            policy_family=policy_family,
            state=carry["submission"],
            policy_state=carry["policy_state"],
            params=params,
            fair_price=fair_price,
            timestamp=step_index,
            relaxation=relaxation,
        )
        normalizer_state, _, normalizer_arb_amount_y, normalizer_arb_profit = _smooth_execute_arb(
            policy_family="fixed_fee",
            state=carry["normalizer"],
            policy_state=None,
            params=None,
            fair_price=fair_price,
            timestamp=step_index,
            relaxation=relaxation,
        )
        carry_after_arb = {
            **carry,
            "submission": submission_state,
            "normalizer": normalizer_state,
            "policy_state": policy_state,
            "regime_probs": next_regime_probs,
            "fair_price": fair_price,
            "edge_submission": carry["edge_submission"] - submission_arb_profit,
            "edge_normalizer": carry["edge_normalizer"] - normalizer_arb_profit,
            "arb_volume_submission_y": carry["arb_volume_submission_y"] + submission_arb_amount_y,
            "arb_volume_normalizer_y": carry["arb_volume_normalizer_y"] + normalizer_arb_amount_y,
        }

        def slot_fn(slot_carry, slot_inputs):
            arrival_u, impact_pct = slot_inputs
            return _smooth_process_realistic_slot(
                carry=slot_carry,
                step_index=step_index,
                arrival_u=arrival_u,
                impact_pct=impact_pct,
                policy_family=policy_family,
                params=params,
                env=env,
                width=smooth_tape["width"],
                impact_pct_grid=impact_pct_grid,
                impact_values=impact_values,
                config=config,
                relaxation=relaxation,
            ), None

        carry_after_slots, _ = jax.lax.scan(slot_fn, carry_after_arb, (arrival_u_t, impact_pct_t))
        return (
            {
                **carry_after_slots,
                "bid_fee_submission_sum": carry_after_slots["bid_fee_submission_sum"] + carry_after_slots["submission"]["bid_fee"],
                "ask_fee_submission_sum": carry_after_slots["ask_fee_submission_sum"] + carry_after_slots["submission"]["ask_fee"],
                "bid_fee_normalizer_sum": carry_after_slots["bid_fee_normalizer_sum"] + carry_after_slots["normalizer"]["bid_fee"],
                "ask_fee_normalizer_sum": carry_after_slots["ask_fee_normalizer_sum"] + carry_after_slots["normalizer"]["ask_fee"],
            },
            None,
        )

    final_carry, _ = jax.lax.scan(step_fn, initial_carry, scan_inputs)
    initial_value = config.initial_x * config.initial_price + config.initial_y
    pnl_submission = _mark_to_market(final_carry["submission"], final_carry["fair_price"]) - initial_value
    pnl_normalizer = _mark_to_market(final_carry["normalizer"], final_carry["fair_price"]) - initial_value
    steps = max(int(config.n_steps), 1)
    return {
        "edge_submission": final_carry["edge_submission"],
        "edge_normalizer": final_carry["edge_normalizer"],
        "pnl_submission": pnl_submission,
        "pnl_normalizer": pnl_normalizer,
        "retail_volume_submission_y": final_carry["retail_volume_submission_y"],
        "retail_volume_normalizer_y": final_carry["retail_volume_normalizer_y"],
        "arb_volume_submission_y": final_carry["arb_volume_submission_y"],
        "arb_volume_normalizer_y": final_carry["arb_volume_normalizer_y"],
        "average_bid_fee_submission": final_carry["bid_fee_submission_sum"] / steps,
        "average_ask_fee_submission": final_carry["ask_fee_submission_sum"] / steps,
        "average_bid_fee_normalizer": final_carry["bid_fee_normalizer_sum"] / steps,
        "average_ask_fee_normalizer": final_carry["ask_fee_normalizer_sum"] / steps,
    }


def _amm_state(*, reserve_x, reserve_y, bid_fee, ask_fee, accumulated_fees_x=0.0, accumulated_fees_y=0.0):
    return {
        "reserve_x": jnp.asarray(reserve_x, dtype=jnp.float32),
        "reserve_y": jnp.asarray(reserve_y, dtype=jnp.float32),
        "bid_fee": jnp.asarray(bid_fee, dtype=jnp.float32),
        "ask_fee": jnp.asarray(ask_fee, dtype=jnp.float32),
        "accumulated_fees_x": jnp.asarray(accumulated_fees_x, dtype=jnp.float32),
        "accumulated_fees_y": jnp.asarray(accumulated_fees_y, dtype=jnp.float32),
    }


def _next_fair_price(*, prev_price, mu: float, sigma, dt: float, z):
    drift = (jnp.asarray(mu, dtype=jnp.float32) - 0.5 * sigma * sigma) * jnp.asarray(dt, dtype=jnp.float32)
    vol = sigma * jnp.sqrt(jnp.asarray(dt, dtype=jnp.float32))
    return prev_price * jnp.exp(drift + vol * z)


def _mark_to_market(state, fair_price):
    return state["reserve_x"] * fair_price + state["reserve_y"] + state["accumulated_fees_x"] * fair_price + state["accumulated_fees_y"]


def _spot_price(state):
    return state["reserve_y"] / jnp.maximum(state["reserve_x"], 1e-9)


def _invariant(state):
    return state["reserve_x"] * state["reserve_y"]


def _smooth_process_order_slot(
    *,
    carry,
    step_index,
    arrival_u,
    size_z,
    side_u,
    policy_family: str,
    params,
    env,
    width,
    size_sigma,
    relaxation,
):
    arrival_rate = env[1]
    mean_size = env[2]
    buy_prob = env[3]
    p_slot = 1.0 - jnp.exp(-arrival_rate / jnp.maximum(jnp.asarray(width, dtype=jnp.float32), 1.0))
    active_weight = smooth_gate(p_slot - arrival_u, sharpness=relaxation.arrival_sharpness)
    ln_sigma = max(float(size_sigma), 0.01)
    ln_mu = jnp.log(jnp.maximum(mean_size, 1e-6)) - 0.5 * float(ln_sigma) ** 2
    size = jnp.exp(ln_mu + float(ln_sigma) * size_z)
    buy_weight = smooth_gate(buy_prob - side_u, sharpness=relaxation.side_sharpness)
    buy_y = active_weight * buy_weight * size
    sell_y = active_weight * (1.0 - buy_weight) * size

    carry = _smooth_route_buy(
        carry=carry,
        total_y=buy_y,
        timestamp=step_index,
        policy_family=policy_family,
        params=params,
        relaxation=relaxation,
    )
    carry = _smooth_route_sell(
        carry=carry,
        total_y_notional=sell_y,
        timestamp=step_index,
        policy_family=policy_family,
        params=params,
        relaxation=relaxation,
    )
    return carry


def _smooth_process_realistic_slot(
    *,
    carry,
    step_index,
    arrival_u,
    impact_pct,
    policy_family: str,
    params,
    env,
    width,
    impact_pct_grid,
    impact_values,
    config,
    relaxation,
):
    arrival_rate = env[0]
    p_slot = 1.0 - jnp.exp(-arrival_rate / jnp.maximum(jnp.asarray(width, dtype=jnp.float32), 1.0))
    active_weight = smooth_gate(p_slot - arrival_u, sharpness=relaxation.arrival_sharpness)
    impact_log = jnp.interp(impact_pct, impact_pct_grid, impact_values)

    if config.retail_impact_scale_mode == "initial_state":
        reserve_x = jnp.asarray(config.initial_x, dtype=jnp.float32)
        reserve_y = jnp.asarray(config.initial_y, dtype=jnp.float32)
        bid_fee = jnp.asarray(0.003, dtype=jnp.float32)
        ask_fee = jnp.asarray(0.003, dtype=jnp.float32)
    else:
        reference_state = carry["submission"] if config.retail_impact_reference_venue == "submission" else carry["normalizer"]
        reserve_x = reference_state["reserve_x"]
        reserve_y = reference_state["reserve_y"]
        bid_fee = reference_state["bid_fee"]
        ask_fee = reference_state["ask_fee"]

    buy_impact = smooth_positive(impact_log, sharpness=relaxation.side_sharpness)
    sell_impact = smooth_positive(-impact_log, sharpness=relaxation.side_sharpness)
    buy_y = active_weight * reserve_y * (jnp.exp(0.5 * buy_impact) - 1.0) / jnp.maximum(1.0 - ask_fee, 1e-9)
    sell_x = active_weight * reserve_x * (jnp.exp(0.5 * sell_impact) - 1.0) / jnp.maximum(1.0 - bid_fee, 1e-9)
    sell_y_notional = sell_x * jnp.maximum(carry["fair_price"], 1e-9)

    carry = _smooth_route_buy(
        carry=carry,
        total_y=buy_y,
        timestamp=step_index,
        policy_family=policy_family,
        params=params,
        relaxation=relaxation,
    )
    carry = _smooth_route_sell(
        carry=carry,
        total_y_notional=sell_y_notional,
        timestamp=step_index,
        policy_family=policy_family,
        params=params,
        relaxation=relaxation,
    )
    return carry


def _smooth_route_buy(*, carry, total_y, timestamp, policy_family: str, params, relaxation):
    y_submission, y_normalizer = _smooth_split_buy_two_amms(
        carry["submission"],
        carry["normalizer"],
        total_y,
        sharpness=relaxation.clip_sharpness,
    )
    submission_state, policy_state, trade = _execute_buy_x_with_y_smooth(
        policy_family=policy_family,
        state=carry["submission"],
        policy_state=carry["policy_state"],
        params=params,
        amount_y=y_submission,
        timestamp=timestamp,
        relaxation=relaxation,
        update_policy=True,
    )
    normalizer_state, _, normalizer_trade = _execute_buy_x_with_y_smooth(
        policy_family="fixed_fee",
        state=carry["normalizer"],
        policy_state=None,
        params=None,
        amount_y=y_normalizer,
        timestamp=timestamp,
        relaxation=relaxation,
        update_policy=False,
    )
    edge_submission = carry["edge_submission"]
    edge_normalizer = carry["edge_normalizer"]
    retail_volume_submission_y = carry["retail_volume_submission_y"]
    retail_volume_normalizer_y = carry["retail_volume_normalizer_y"]
    if trade is not None:
        trade_edge = trade["amount_y"] - trade["amount_x"] * carry["fair_price"]
        edge_submission = edge_submission + trade_edge
        retail_volume_submission_y = retail_volume_submission_y + trade["amount_y"]
    if normalizer_trade is not None:
        trade_edge = normalizer_trade["amount_y"] - normalizer_trade["amount_x"] * carry["fair_price"]
        edge_normalizer = edge_normalizer + trade_edge
        retail_volume_normalizer_y = retail_volume_normalizer_y + normalizer_trade["amount_y"]
    return {
        **carry,
        "submission": submission_state,
        "normalizer": normalizer_state,
        "policy_state": policy_state,
        "edge_submission": edge_submission,
        "edge_normalizer": edge_normalizer,
        "retail_volume_submission_y": retail_volume_submission_y,
        "retail_volume_normalizer_y": retail_volume_normalizer_y,
    }


def _smooth_route_sell(*, carry, total_y_notional, timestamp, policy_family: str, params, relaxation):
    total_x = total_y_notional / jnp.maximum(carry["fair_price"], 1e-9)
    x_submission, x_normalizer = _smooth_split_sell_two_amms(
        carry["submission"],
        carry["normalizer"],
        total_x,
        sharpness=relaxation.clip_sharpness,
    )
    submission_state, policy_state, trade = _execute_buy_x_smooth(
        policy_family=policy_family,
        state=carry["submission"],
        policy_state=carry["policy_state"],
        params=params,
        amount_x=x_submission,
        timestamp=timestamp,
        relaxation=relaxation,
        update_policy=True,
    )
    normalizer_state, _, normalizer_trade = _execute_buy_x_smooth(
        policy_family="fixed_fee",
        state=carry["normalizer"],
        policy_state=None,
        params=None,
        amount_x=x_normalizer,
        timestamp=timestamp,
        relaxation=relaxation,
        update_policy=False,
    )
    edge_submission = carry["edge_submission"]
    edge_normalizer = carry["edge_normalizer"]
    retail_volume_submission_y = carry["retail_volume_submission_y"]
    retail_volume_normalizer_y = carry["retail_volume_normalizer_y"]
    if trade is not None:
        trade_edge = trade["amount_x"] * carry["fair_price"] - trade["amount_y"]
        edge_submission = edge_submission + trade_edge
        retail_volume_submission_y = retail_volume_submission_y + trade["amount_y"]
    if normalizer_trade is not None:
        trade_edge = normalizer_trade["amount_x"] * carry["fair_price"] - normalizer_trade["amount_y"]
        edge_normalizer = edge_normalizer + trade_edge
        retail_volume_normalizer_y = retail_volume_normalizer_y + normalizer_trade["amount_y"]
    return {
        **carry,
        "submission": submission_state,
        "normalizer": normalizer_state,
        "policy_state": policy_state,
        "edge_submission": edge_submission,
        "edge_normalizer": edge_normalizer,
        "retail_volume_submission_y": retail_volume_submission_y,
        "retail_volume_normalizer_y": retail_volume_normalizer_y,
    }


def _smooth_split_buy_two_amms(submission, normalizer, total_y, *, sharpness: float):
    gamma1 = 1.0 - submission["ask_fee"]
    gamma2 = 1.0 - normalizer["ask_fee"]
    a1 = jnp.sqrt(jnp.maximum(submission["reserve_x"] * gamma1 * submission["reserve_y"], 0.0))
    a2 = jnp.sqrt(jnp.maximum(normalizer["reserve_x"] * gamma2 * normalizer["reserve_y"], 0.0))
    denominator = gamma1 + (a1 / jnp.maximum(a2, 1e-9)) * gamma2
    raw = (
        (a1 / jnp.maximum(a2, 1e-9)) * (normalizer["reserve_y"] + gamma2 * total_y) - submission["reserve_y"]
    ) / jnp.maximum(denominator, 1e-9)
    y_submission = smooth_clip(raw, 0.0, total_y, sharpness=sharpness)
    return (y_submission, total_y - y_submission)


def _smooth_split_sell_two_amms(submission, normalizer, total_x, *, sharpness: float):
    gamma1 = 1.0 - submission["bid_fee"]
    gamma2 = 1.0 - normalizer["bid_fee"]
    b1 = jnp.sqrt(jnp.maximum(submission["reserve_y"] * gamma1 * submission["reserve_x"], 0.0))
    b2 = jnp.sqrt(jnp.maximum(normalizer["reserve_y"] * gamma2 * normalizer["reserve_x"], 0.0))
    denominator = gamma1 + (b1 / jnp.maximum(b2, 1e-9)) * gamma2
    raw = (
        (b1 / jnp.maximum(b2, 1e-9)) * (normalizer["reserve_x"] + gamma2 * total_x) - submission["reserve_x"]
    ) / jnp.maximum(denominator, 1e-9)
    x_submission = smooth_clip(raw, 0.0, total_x, sharpness=sharpness)
    return (x_submission, total_x - x_submission)


def _execute_buy_x_smooth(*, policy_family: str, state, policy_state, params, amount_x, timestamp, relaxation, update_policy: bool):
    amount_x_eff = smooth_trade_amount(amount_x, minimum=relaxation.min_trade_amount, sharpness=relaxation.gate_sharpness)
    gamma = jnp.clip(1.0 - state["bid_fee"], 0.0, 1.0)
    net_x = amount_x_eff * gamma
    new_rx = state["reserve_x"] + net_x
    new_ry = _invariant(state) / jnp.maximum(new_rx, 1e-9)
    amount_y = smooth_positive(state["reserve_y"] - new_ry, sharpness=relaxation.clip_sharpness)
    fee_x = amount_x_eff * state["bid_fee"]
    next_state = _amm_state(
        reserve_x=new_rx,
        reserve_y=state["reserve_y"] - amount_y,
        bid_fee=state["bid_fee"],
        ask_fee=state["ask_fee"],
        accumulated_fees_x=state["accumulated_fees_x"] + fee_x,
        accumulated_fees_y=state["accumulated_fees_y"],
    )
    trade = {
        "is_buy": jnp.asarray(1.0, dtype=jnp.float32),
        "amount_x": amount_x_eff,
        "amount_y": amount_y,
        "timestamp": jnp.asarray(timestamp, dtype=jnp.float32),
        "reserve_x": next_state["reserve_x"],
        "reserve_y": next_state["reserve_y"],
    }
    if not update_policy:
        return (next_state, policy_state, trade)
    next_policy_state, bid_fee, ask_fee = _policy_after_event(policy_family, params, policy_state, trade, relaxation)
    next_state = {**next_state, "bid_fee": bid_fee, "ask_fee": ask_fee}
    return (next_state, next_policy_state, trade)


def _execute_sell_x_smooth(*, policy_family: str, state, policy_state, params, amount_x, timestamp, relaxation, update_policy: bool):
    capped_amount = smooth_clip(amount_x, 0.0, 0.99 * state["reserve_x"], sharpness=relaxation.clip_sharpness)
    amount_x_eff = smooth_trade_amount(capped_amount, minimum=relaxation.min_trade_amount, sharpness=relaxation.gate_sharpness)
    gamma = jnp.clip(1.0 - state["ask_fee"], 0.0, 1.0)
    new_rx = jnp.maximum(state["reserve_x"] - amount_x_eff, 1e-9)
    new_ry = _invariant(state) / new_rx
    net_y = smooth_positive(new_ry - state["reserve_y"], sharpness=relaxation.clip_sharpness)
    total_y = net_y / jnp.maximum(gamma, 1e-9)
    fee_y = total_y - net_y
    next_state = _amm_state(
        reserve_x=new_rx,
        reserve_y=state["reserve_y"] + net_y,
        bid_fee=state["bid_fee"],
        ask_fee=state["ask_fee"],
        accumulated_fees_x=state["accumulated_fees_x"],
        accumulated_fees_y=state["accumulated_fees_y"] + fee_y,
    )
    trade = {
        "is_buy": jnp.asarray(0.0, dtype=jnp.float32),
        "amount_x": amount_x_eff,
        "amount_y": total_y,
        "timestamp": jnp.asarray(timestamp, dtype=jnp.float32),
        "reserve_x": next_state["reserve_x"],
        "reserve_y": next_state["reserve_y"],
    }
    if not update_policy:
        return (next_state, policy_state, trade)
    next_policy_state, bid_fee, ask_fee = _policy_after_event(policy_family, params, policy_state, trade, relaxation)
    next_state = {**next_state, "bid_fee": bid_fee, "ask_fee": ask_fee}
    return (next_state, next_policy_state, trade)


def _execute_buy_x_with_y_smooth(
    *,
    policy_family: str,
    state,
    policy_state,
    params,
    amount_y,
    timestamp,
    relaxation,
    update_policy: bool,
):
    amount_y_eff = smooth_trade_amount(amount_y, minimum=relaxation.min_trade_amount, sharpness=relaxation.gate_sharpness)
    gamma = jnp.clip(1.0 - state["ask_fee"], 0.0, 1.0)
    net_y = amount_y_eff * gamma
    new_ry = state["reserve_y"] + net_y
    new_rx = _invariant(state) / jnp.maximum(new_ry, 1e-9)
    amount_x = smooth_positive(state["reserve_x"] - new_rx, sharpness=relaxation.clip_sharpness)
    fee_y = amount_y_eff * state["ask_fee"]
    next_state = _amm_state(
        reserve_x=state["reserve_x"] - amount_x,
        reserve_y=new_ry,
        bid_fee=state["bid_fee"],
        ask_fee=state["ask_fee"],
        accumulated_fees_x=state["accumulated_fees_x"],
        accumulated_fees_y=state["accumulated_fees_y"] + fee_y,
    )
    trade = {
        "is_buy": jnp.asarray(0.0, dtype=jnp.float32),
        "amount_x": amount_x,
        "amount_y": amount_y_eff,
        "timestamp": jnp.asarray(timestamp, dtype=jnp.float32),
        "reserve_x": next_state["reserve_x"],
        "reserve_y": next_state["reserve_y"],
    }
    if not update_policy:
        return (next_state, policy_state, trade)
    next_policy_state, bid_fee, ask_fee = _policy_after_event(policy_family, params, policy_state, trade, relaxation)
    next_state = {**next_state, "bid_fee": bid_fee, "ask_fee": ask_fee}
    return (next_state, next_policy_state, trade)


def _smooth_execute_arb(*, policy_family: str, state, policy_state, params, fair_price, timestamp, relaxation):
    spot = _spot_price(state)
    buy_side_gate = smooth_gate(fair_price - spot, sharpness=relaxation.arb_sharpness)
    sell_side_gate = smooth_gate(spot - fair_price, sharpness=relaxation.arb_sharpness)

    gamma_ask = jnp.clip(1.0 - state["ask_fee"], 0.0, 1.0)
    buy_raw = state["reserve_x"] - jnp.sqrt(jnp.maximum((_invariant(state)) / jnp.maximum(gamma_ask * fair_price, 1e-9), 0.0))
    buy_candidate = smooth_clip(smooth_positive(buy_raw, sharpness=relaxation.clip_sharpness), 0.0, 0.99 * state["reserve_x"], sharpness=relaxation.clip_sharpness)
    buy_state, buy_policy_state, buy_trade = _execute_sell_x_smooth(
        policy_family=policy_family,
        state=state,
        policy_state=policy_state,
        params=params,
        amount_x=buy_candidate * buy_side_gate,
        timestamp=timestamp,
        relaxation=relaxation,
        update_policy=params is not None,
    )
    buy_profit = buy_trade["amount_x"] * fair_price - buy_trade["amount_y"]

    gamma_bid = jnp.clip(1.0 - state["bid_fee"], 0.0, 1.0)
    sell_raw = (jnp.sqrt(jnp.maximum(_invariant(state) * gamma_bid / jnp.maximum(fair_price, 1e-9), 0.0)) - state["reserve_x"]) / jnp.maximum(gamma_bid, 1e-9)
    sell_candidate = smooth_positive(sell_raw, sharpness=relaxation.clip_sharpness)
    sell_state, sell_policy_state, sell_trade = _execute_buy_x_smooth(
        policy_family=policy_family,
        state=state,
        policy_state=policy_state,
        params=params,
        amount_x=sell_candidate * sell_side_gate,
        timestamp=timestamp,
        relaxation=relaxation,
        update_policy=params is not None,
    )
    sell_profit = sell_trade["amount_y"] - sell_trade["amount_x"] * fair_price

    use_sell = sell_side_gate > buy_side_gate
    next_state = jax.tree_util.tree_map(lambda a, b: jnp.where(use_sell, b, a), buy_state, sell_state)
    next_policy_state = (
        jax.tree_util.tree_map(lambda a, b: jnp.where(use_sell, b, a), buy_policy_state, sell_policy_state)
        if params is not None
        else policy_state
    )
    amount_y = jnp.where(use_sell, sell_trade["amount_y"], buy_trade["amount_y"])
    profit = jnp.where(use_sell, sell_profit, buy_profit)
    return (next_state, next_policy_state, amount_y, smooth_positive(profit, sharpness=relaxation.arb_sharpness))


def _policy_initial_bid_fee(policy_family: str, params):
    if policy_family == "submission_compact":
        return params[0]
    if policy_family == "piecewise":
        return _piecewise_fees(params, _piecewise_initial_policy_state())[0]
    raise ValueError(f"Unsupported diff policy_family: {policy_family}")


def _policy_initial_ask_fee(policy_family: str, params):
    if policy_family == "submission_compact":
        return params[0]
    if policy_family == "piecewise":
        return _piecewise_fees(params, _piecewise_initial_policy_state())[1]
    raise ValueError(f"Unsupported diff policy_family: {policy_family}")


def _initial_policy_state(policy_family: str, initial_x: float, initial_y: float):
    if policy_family == "submission_compact":
        return _compact_initial_policy_state(initial_x, initial_y)
    if policy_family == "piecewise":
        del initial_x, initial_y
        return _piecewise_initial_policy_state()
    raise ValueError(f"Unsupported diff policy_family: {policy_family}")


def _policy_after_event(policy_family: str, params, state, trade, relaxation):
    if policy_family == "submission_compact":
        return _compact_after_event(params, state, trade, relaxation)
    if policy_family == "piecewise":
        return _piecewise_after_event(params, state, trade, relaxation)
    raise ValueError(f"Unsupported diff policy_family: {policy_family}")


def _compact_initial_policy_state(initial_x: float, initial_y: float):
    initial_spot = float(initial_y) / max(float(initial_x), 1e-9)
    return jnp.asarray(
        [
            float(initial_x),
            float(initial_y),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            initial_spot,
            initial_spot,
            1.0,
        ],
        dtype=jnp.float32,
    )


def _compact_after_event(params, state, trade, relaxation):
    size_ratio = trade["amount_y"] / jnp.maximum(trade["reserve_y"], 1e-9)
    side = jnp.where(trade["is_buy"] > 0.5, 1.0, -1.0)
    spot = trade["reserve_y"] / jnp.maximum(trade["reserve_x"], 1e-9)
    dt = jnp.maximum(1.0, trade["timestamp"] - state[_S2_LAST_TIMESTAMP])

    buy_flow_fast = state[_S6_BUY_FLOW_FAST] * params[3] + (1.0 - params[3]) * size_ratio * jnp.where(side > 0.0, 1.0, 0.0)
    sell_flow_fast = state[_S7_SELL_FLOW_FAST] * params[3] + (1.0 - params[3]) * size_ratio * jnp.where(side < 0.0, 1.0, 0.0)
    buy_flow_slow = state[_S8_BUY_FLOW_SLOW] * params[4] + (1.0 - params[4]) * size_ratio * jnp.where(side > 0.0, 1.0, 0.0)
    sell_flow_slow = state[_S9_SELL_FLOW_SLOW] * params[4] + (1.0 - params[4]) * size_ratio * jnp.where(side < 0.0, 1.0, 0.0)
    size_fast = params[5] * state[_S10_SIZE_FAST] + (1.0 - params[5]) * size_ratio
    size_slow = params[6] * state[_S11_SIZE_SLOW] + (1.0 - params[6]) * size_ratio
    gap_fast = params[7] * state[_S12_GAP_FAST] + (1.0 - params[7]) * dt
    gap_slow = params[8] * state[_S13_GAP_SLOW] + (1.0 - params[8]) * dt

    tox_bid = state[_S14_TOX_BID] * params[9]
    tox_ask = state[_S15_TOX_ASK] * params[9]
    same_side = 0.4 * params[10] * size_ratio
    tox_bid = tox_bid + same_side * jnp.where(side > 0.0, 1.0, 0.0)
    tox_ask = tox_ask + same_side * jnp.where(side < 0.0, 1.0, 0.0)
    reversal_gate = jnp.where(
        jnp.logical_and(state[_S3_LAST_SIDE] != 0.0, side != state[_S3_LAST_SIDE]),
        1.0,
        0.0,
    )
    reversal_scale = jnp.maximum(state[_S4_LAST_SIZE_RATIO], size_ratio) / dt
    tox_bid = tox_bid + params[10] * reversal_scale * reversal_gate * jnp.where(state[_S3_LAST_SIDE] > 0.0, 1.0, 0.0)
    tox_ask = tox_ask + params[10] * reversal_scale * reversal_gate * jnp.where(state[_S3_LAST_SIDE] < 0.0, 1.0, 0.0)

    fair_fast = 0.9 * state[_S16_FAIR_FAST] + 0.1 * spot
    fair_slow = 0.98 * state[_S17_FAIR_SLOW] + 0.02 * spot
    streak_len = jnp.where(side == state[_S3_LAST_SIDE], jnp.minimum(state[_S5_STREAK_LEN] + 1.0, 8.0), 1.0)

    next_state = state.at[_S2_LAST_TIMESTAMP].set(trade["timestamp"])
    next_state = next_state.at[_S3_LAST_SIDE].set(side)
    next_state = next_state.at[_S4_LAST_SIZE_RATIO].set(size_ratio)
    next_state = next_state.at[_S5_STREAK_LEN].set(streak_len)
    next_state = next_state.at[_S6_BUY_FLOW_FAST].set(buy_flow_fast)
    next_state = next_state.at[_S7_SELL_FLOW_FAST].set(sell_flow_fast)
    next_state = next_state.at[_S8_BUY_FLOW_SLOW].set(buy_flow_slow)
    next_state = next_state.at[_S9_SELL_FLOW_SLOW].set(sell_flow_slow)
    next_state = next_state.at[_S10_SIZE_FAST].set(size_fast)
    next_state = next_state.at[_S11_SIZE_SLOW].set(size_slow)
    next_state = next_state.at[_S12_GAP_FAST].set(gap_fast)
    next_state = next_state.at[_S13_GAP_SLOW].set(gap_slow)
    next_state = next_state.at[_S14_TOX_BID].set(tox_bid)
    next_state = next_state.at[_S15_TOX_ASK].set(tox_ask)
    next_state = next_state.at[_S16_FAIR_FAST].set(fair_fast)
    next_state = next_state.at[_S17_FAIR_SLOW].set(fair_slow)
    next_state = next_state.at[_S18_INITIALIZED].set(1.0)

    flow_total = buy_flow_fast + sell_flow_fast
    flow_skew = (buy_flow_fast - sell_flow_fast) + 0.5 * (buy_flow_slow - sell_flow_slow)
    hot_gate = smooth_gate(params[17] - gap_fast, sharpness=relaxation.gate_sharpness)
    big_gate = smooth_gate(size_ratio - params[18], sharpness=relaxation.gate_sharpness)
    hot_or_big = smooth_or(hot_gate, big_gate)
    mid = (
        params[0]
        + params[12] * flow_total
        + params[13] * smooth_positive(size_fast - 0.5 * size_slow, sharpness=relaxation.clip_sharpness)
        + params[14] * smooth_positive(params[17] - gap_fast, sharpness=relaxation.clip_sharpness)
        + params[19] * hot_or_big
    )
    spread = params[11] + 0.5 * params[19] * hot_gate
    skew = params[15] * flow_skew
    bid = jnp.clip(mid + 0.5 * spread - skew + params[16] * tox_bid, params[1], params[2])
    ask = jnp.clip(mid + 0.5 * spread + skew + params[16] * tox_ask, params[1], params[2])
    return (next_state, bid, ask)


def _piecewise_initial_policy_state():
    return jnp.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=jnp.float32)


def _piecewise_bucket_weights(params, size_ratio, relaxation):
    small_gate = smooth_gate(size_ratio - params[4], sharpness=relaxation.gate_sharpness)
    large_gate = smooth_gate(size_ratio - params[5], sharpness=relaxation.gate_sharpness)
    small_weight = 1.0 - small_gate
    medium_weight = small_gate * (1.0 - large_gate)
    large_weight = large_gate
    continuation = params[6] * small_weight + params[7] * medium_weight + params[8] * large_weight
    reversal = params[9] * small_weight + params[10] * medium_weight + params[11] * large_weight
    return (continuation, reversal)


def _piecewise_fees(params, state):
    toxicity_total = state[_P4_BID_TOXICITY] + state[_P5_ASK_TOXICITY]
    base = params[0] + 0.5 * params[1] + params[14] * toxicity_total
    bid = jnp.clip(
        base
        + params[15] * state[_P4_BID_TOXICITY]
        - params[12] * state[_P2_BID_SIGNAL]
        + params[13] * state[_P3_ASK_SIGNAL],
        0.0,
        0.1,
    )
    ask = jnp.clip(
        base
        + params[15] * state[_P5_ASK_TOXICITY]
        - params[12] * state[_P3_ASK_SIGNAL]
        + params[13] * state[_P2_BID_SIGNAL],
        0.0,
        0.1,
    )
    return (bid, ask)


def _piecewise_after_event(params, state, trade, relaxation):
    size_ratio = trade["amount_y"] / jnp.maximum(trade["reserve_y"], 1e-9)
    side = jnp.where(trade["is_buy"] > 0.5, 1.0, -1.0)
    dt = jnp.maximum(1.0, trade["timestamp"] - state[_P0_LAST_TIMESTAMP])
    continuation_weight, reversal_weight = _piecewise_bucket_weights(params, size_ratio, relaxation)
    reversal_scale = 1.0 / dt

    bid_signal = state[_P2_BID_SIGNAL] * params[2]
    ask_signal = state[_P3_ASK_SIGNAL] * params[2]
    bid_toxicity = state[_P4_BID_TOXICITY] * params[3]
    ask_toxicity = state[_P5_ASK_TOXICITY] * params[3]

    same_side = jnp.where(side == state[_P1_LAST_SIDE], 1.0, 0.0)
    last_side_present = jnp.where(state[_P1_LAST_SIDE] != 0.0, 1.0, 0.0)
    reversal_gate = last_side_present * (1.0 - same_side)

    bid_signal = bid_signal + continuation_weight * (
        same_side * jnp.where(side > 0.0, 1.0, 0.0) + (1.0 - same_side) * 0.5 * jnp.where(side > 0.0, 1.0, 0.0)
    )
    ask_signal = ask_signal + continuation_weight * (
        same_side * jnp.where(side < 0.0, 1.0, 0.0) + (1.0 - same_side) * 0.5 * jnp.where(side < 0.0, 1.0, 0.0)
    )
    bid_toxicity = bid_toxicity + reversal_weight * reversal_scale * reversal_gate * jnp.where(
        side < 0.0, 1.0, 0.0
    )
    ask_toxicity = ask_toxicity + reversal_weight * reversal_scale * reversal_gate * jnp.where(
        side > 0.0, 1.0, 0.0
    )

    next_state = state.at[_P0_LAST_TIMESTAMP].set(trade["timestamp"])
    next_state = next_state.at[_P1_LAST_SIDE].set(side)
    next_state = next_state.at[_P2_BID_SIGNAL].set(bid_signal)
    next_state = next_state.at[_P3_ASK_SIGNAL].set(ask_signal)
    next_state = next_state.at[_P4_BID_TOXICITY].set(bid_toxicity)
    next_state = next_state.at[_P5_ASK_TOXICITY].set(ask_toxicity)
    next_state = next_state.at[_P6_INITIALIZED].set(1.0)
    bid, ask = _piecewise_fees(params, next_state)
    return (next_state, bid, ask)
