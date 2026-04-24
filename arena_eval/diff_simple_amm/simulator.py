"""Top-level configuration and rollouts for the diff simple-AMM simulator."""

from __future__ import annotations

import math
from dataclasses import dataclass

from arena_eval.exact_simple_amm.config import ExactSimpleAMMConfig

from . import arb as diff_arb
from . import router as diff_router
from .amm import initialize_amm
from .objectives import smooth_submission_compact_result, submission_compact_param_vector
from .orders import decode_challenge_orders, decode_realistic_orders
from .policies import DiffSimpleAMMPolicy, FixedFeeDiffPolicy, SubmissionCompactDiffPolicy
from .types import (
    AMMState,
    ChallengeTape,
    DiffMode,
    DiffSimulationResult,
    RealisticTape,
    SimulatorState,
    SmoothRelaxationConfig,
)


@dataclass(frozen=True)
class DiffSimpleAMMSimulatorConfig:
    """Configuration shell for the new differentiable rollout stack."""

    mode: DiffMode = DiffMode.EXACT_PATH
    seed: int = 0
    exact_config: ExactSimpleAMMConfig = ExactSimpleAMMConfig()
    relaxation: SmoothRelaxationConfig = SmoothRelaxationConfig()

    @property
    def n_steps(self) -> int:
        return self.exact_config.n_steps


def run_rollout(
    *,
    config: DiffSimpleAMMSimulatorConfig,
    tape: ChallengeTape | RealisticTape,
    submission_policy: DiffSimpleAMMPolicy,
    normalizer_policy: DiffSimpleAMMPolicy | None = None,
) -> DiffSimulationResult:
    """Run a challenge or realistic rollout in exact-path or smooth-train mode."""

    if config.mode is DiffMode.SMOOTH_TRAIN:
        if not isinstance(submission_policy, SubmissionCompactDiffPolicy):
            raise ValueError("smooth_train currently supports SubmissionCompactDiffPolicy only")
        return smooth_submission_compact_result(
            submission_compact_param_vector(submission_policy.params),
            config=config.exact_config,
            tape=tape,
            relaxation=config.relaxation,
            seed=config.seed,
        )
    if config.mode is not DiffMode.EXACT_PATH:
        raise ValueError(f"Unsupported diff mode: {config.mode}")
    if isinstance(tape, ChallengeTape):
        return _run_exact_challenge_rollout(
            config=config,
            tape=tape,
            submission_policy=submission_policy,
            normalizer_policy=normalizer_policy,
        )
    if isinstance(tape, RealisticTape):
        return _run_exact_realistic_rollout(
            config=config,
            tape=tape,
            submission_policy=submission_policy,
            normalizer_policy=normalizer_policy,
        )
    raise TypeError(f"Unsupported tape type: {type(tape)!r}")


def run_challenge_rollout(
    *,
    config: DiffSimpleAMMSimulatorConfig,
    tape: ChallengeTape,
    submission_policy: DiffSimpleAMMPolicy,
    normalizer_policy: DiffSimpleAMMPolicy | None = None,
) -> DiffSimulationResult:
    """Backward-compatible wrapper for challenge rollouts."""

    return run_rollout(
        config=config,
        tape=tape,
        submission_policy=submission_policy,
        normalizer_policy=normalizer_policy,
    )


def run_realistic_rollout(
    *,
    config: DiffSimpleAMMSimulatorConfig,
    tape: RealisticTape,
    submission_policy: DiffSimpleAMMPolicy,
    normalizer_policy: DiffSimpleAMMPolicy | None = None,
) -> DiffSimulationResult:
    """Run the realistic evaluator under the diff stack."""

    return run_rollout(
        config=config,
        tape=tape,
        submission_policy=submission_policy,
        normalizer_policy=normalizer_policy,
    )


def _run_exact_challenge_rollout(
    *,
    config: DiffSimpleAMMSimulatorConfig,
    tape: ChallengeTape,
    submission_policy: DiffSimpleAMMPolicy,
    normalizer_policy: DiffSimpleAMMPolicy | None,
) -> DiffSimulationResult:
    exact_config = config.exact_config
    drift_term = (exact_config.gbm_mu - 0.5 * exact_config.gbm_sigma * exact_config.gbm_sigma) * exact_config.gbm_dt
    vol_term = exact_config.gbm_sigma * math.sqrt(exact_config.gbm_dt)

    def price_update(prev_price: float, step: int) -> float:
        return prev_price * math.exp(drift_term + vol_term * tape.gbm_normals[step])

    def decode_orders(state: SimulatorState, fair_price: float, step: int):
        del state, fair_price
        return decode_challenge_orders(config=exact_config, tape=tape, step=step)

    return _run_exact_rollout(
        config=config,
        submission_policy=submission_policy,
        normalizer_policy=normalizer_policy,
        price_update=price_update,
        decode_orders=decode_orders,
    )


def _run_exact_realistic_rollout(
    *,
    config: DiffSimpleAMMSimulatorConfig,
    tape: RealisticTape,
    submission_policy: DiffSimpleAMMPolicy,
    normalizer_policy: DiffSimpleAMMPolicy | None,
) -> DiffSimulationResult:
    exact_config = config.exact_config

    def price_update(prev_price: float, step: int) -> float:
        return prev_price * math.exp(tape.log_returns[step])

    def decode_orders(state: SimulatorState, fair_price: float, step: int):
        reference_state = (
            state.submission
            if exact_config.retail_impact_reference_venue == "submission"
            else state.normalizer
        )
        return decode_realistic_orders(
            config=exact_config,
            tape=tape,
            step=step,
            fair_price=fair_price,
            reference_state=reference_state,
        )

    return _run_exact_rollout(
        config=config,
        submission_policy=submission_policy,
        normalizer_policy=normalizer_policy,
        price_update=price_update,
        decode_orders=decode_orders,
    )


def _run_exact_rollout(
    *,
    config: DiffSimpleAMMSimulatorConfig,
    submission_policy: DiffSimpleAMMPolicy,
    normalizer_policy: DiffSimpleAMMPolicy | None,
    price_update,
    decode_orders,
) -> DiffSimulationResult:
    exact_config = config.exact_config
    normalizer_policy = normalizer_policy or FixedFeeDiffPolicy()
    submission_state, submission_policy_state = initialize_amm(
        reserve_x=exact_config.initial_x,
        reserve_y=exact_config.initial_y,
        policy=submission_policy,
    )
    normalizer_state, normalizer_policy_state = initialize_amm(
        reserve_x=exact_config.initial_x,
        reserve_y=exact_config.initial_y,
        policy=normalizer_policy,
    )
    state = SimulatorState(
        step=0,
        fair_price=exact_config.initial_price,
        submission=submission_state,
        normalizer=normalizer_state,
        submission_policy_state=submission_policy_state,
        normalizer_policy_state=normalizer_policy_state,
    )

    current_fair_price = exact_config.initial_price
    for step in range(exact_config.n_steps):
        current_fair_price = price_update(current_fair_price, step)
        submission_state, submission_policy_state, submission_arb_trade, submission_arb_profit = diff_arb.execute_arb(
            amm_name="submission",
            state=state.submission,
            policy=submission_policy,
            policy_state=state.submission_policy_state,
            fair_price=current_fair_price,
            timestamp=step,
        )
        edge_submission = state.edge_submission - submission_arb_profit
        arb_volume_submission_y = state.arb_volume_submission_y
        if submission_arb_trade is not None:
            arb_volume_submission_y += submission_arb_trade.amount_y

        normalizer_state, normalizer_policy_state, normalizer_arb_trade, normalizer_arb_profit = diff_arb.execute_arb(
            amm_name="normalizer",
            state=state.normalizer,
            policy=normalizer_policy,
            policy_state=state.normalizer_policy_state,
            fair_price=current_fair_price,
            timestamp=step,
        )
        edge_normalizer = state.edge_normalizer - normalizer_arb_profit
        arb_volume_normalizer_y = state.arb_volume_normalizer_y
        if normalizer_arb_trade is not None:
            arb_volume_normalizer_y += normalizer_arb_trade.amount_y

        step_state = SimulatorState(
            step=step,
            fair_price=current_fair_price,
            submission=submission_state,
            normalizer=normalizer_state,
            submission_policy_state=submission_policy_state,
            normalizer_policy_state=normalizer_policy_state,
            edge_submission=edge_submission,
            edge_normalizer=edge_normalizer,
            retail_volume_submission_y=state.retail_volume_submission_y,
            retail_volume_normalizer_y=state.retail_volume_normalizer_y,
            arb_volume_submission_y=arb_volume_submission_y,
            arb_volume_normalizer_y=arb_volume_normalizer_y,
            bid_fee_submission_sum=state.bid_fee_submission_sum,
            ask_fee_submission_sum=state.ask_fee_submission_sum,
            bid_fee_normalizer_sum=state.bid_fee_normalizer_sum,
            ask_fee_normalizer_sum=state.ask_fee_normalizer_sum,
        )

        orders = decode_orders(step_state, current_fair_price, step)
        (
            submission_state,
            submission_policy_state,
            normalizer_state,
            normalizer_policy_state,
            trades,
        ) = diff_router.route_orders(
            orders=orders,
            submission=submission_state,
            submission_policy=submission_policy,
            submission_policy_state=submission_policy_state,
            normalizer=normalizer_state,
            normalizer_policy=normalizer_policy,
            normalizer_policy_state=normalizer_policy_state,
            fair_price=current_fair_price,
            timestamp=step,
        )

        retail_volume_submission_y = state.retail_volume_submission_y
        retail_volume_normalizer_y = state.retail_volume_normalizer_y
        for trade in trades:
            trade_edge = (
                trade.amount_x * current_fair_price - trade.amount_y
                if trade.is_buy
                else trade.amount_y - trade.amount_x * current_fair_price
            )
            if trade.venue == "submission":
                edge_submission += trade_edge
                retail_volume_submission_y += trade.amount_y
            else:
                edge_normalizer += trade_edge
                retail_volume_normalizer_y += trade.amount_y

        state = SimulatorState(
            step=step + 1,
            fair_price=current_fair_price,
            submission=submission_state,
            normalizer=normalizer_state,
            submission_policy_state=submission_policy_state,
            normalizer_policy_state=normalizer_policy_state,
            edge_submission=edge_submission,
            edge_normalizer=edge_normalizer,
            retail_volume_submission_y=retail_volume_submission_y,
            retail_volume_normalizer_y=retail_volume_normalizer_y,
            arb_volume_submission_y=arb_volume_submission_y,
            arb_volume_normalizer_y=arb_volume_normalizer_y,
            bid_fee_submission_sum=state.bid_fee_submission_sum + submission_state.bid_fee,
            ask_fee_submission_sum=state.ask_fee_submission_sum + submission_state.ask_fee,
            bid_fee_normalizer_sum=state.bid_fee_normalizer_sum + normalizer_state.bid_fee,
            ask_fee_normalizer_sum=state.ask_fee_normalizer_sum + normalizer_state.ask_fee,
        )

    return _result(
        state=state,
        seed=config.seed,
        initial_x=exact_config.initial_x,
        initial_y=exact_config.initial_y,
        initial_price=exact_config.initial_price,
    )


def _result(
    *,
    state: SimulatorState,
    seed: int,
    initial_x: float,
    initial_y: float,
    initial_price: float,
) -> DiffSimulationResult:
    initial_value = initial_x * initial_price + initial_y
    reserve_submission = _mark_to_market(state.submission, state.fair_price)
    reserve_normalizer = _mark_to_market(state.normalizer, state.fair_price)
    steps = max(state.step, 1)
    return DiffSimulationResult(
        seed=seed,
        edge_submission=state.edge_submission,
        edge_normalizer=state.edge_normalizer,
        pnl_submission=reserve_submission - initial_value,
        pnl_normalizer=reserve_normalizer - initial_value,
        score=state.edge_submission,
        retail_volume_submission_y=state.retail_volume_submission_y,
        retail_volume_normalizer_y=state.retail_volume_normalizer_y,
        arb_volume_submission_y=state.arb_volume_submission_y,
        arb_volume_normalizer_y=state.arb_volume_normalizer_y,
        average_bid_fee_submission=state.bid_fee_submission_sum / steps,
        average_ask_fee_submission=state.ask_fee_submission_sum / steps,
        average_bid_fee_normalizer=state.bid_fee_normalizer_sum / steps,
        average_ask_fee_normalizer=state.ask_fee_normalizer_sum / steps,
        metadata={"n_steps": state.step},
    )


def _mark_to_market(state: AMMState, fair_price: float) -> float:
    reserve_value = state.reserve_x * fair_price + state.reserve_y
    fee_value = state.accumulated_fees_x * fair_price + state.accumulated_fees_y
    return reserve_value + fee_value
