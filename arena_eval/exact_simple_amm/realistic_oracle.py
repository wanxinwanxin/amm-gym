"""Realistic-tape clairvoyant benchmarks for the simple AMM evaluator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Iterable

from arena_eval.core.types import BatchResult, SimulationResult
from arena_eval.diff_simple_amm.orders import decode_realistic_orders
from arena_eval.diff_simple_amm.realistic_dynamics import build_realistic_tape
from arena_eval.diff_simple_amm.types import AMMState, RealisticTape
from arena_eval.exact_simple_amm.config import ExactSimpleAMMConfig
from arena_eval.exact_simple_amm.oracle import (
    DEFAULT_BASELINE_FEE,
    ClairvoyantController,
    OracleContext,
    _clamp_fee,
    _execute_arb,
    _mark_to_market,
    _route_one_order,
    _set_fees,
)


def _build_realistic_price_path(config: ExactSimpleAMMConfig, tape: RealisticTape) -> tuple[float, ...]:
    price = float(config.initial_price)
    path: list[float] = []
    for log_return in tape.log_returns:
        price *= math.exp(float(log_return))
        path.append(price)
    return tuple(path)


@dataclass(frozen=True)
class RealisticClairvoyantRunConfig:
    """Configuration for realized-tape realistic oracle benchmarks."""

    normalizer_fee: float = DEFAULT_BASELINE_FEE


def run_realistic_clairvoyant_seed(
    controller: ClairvoyantController,
    seed: int,
    *,
    config: ExactSimpleAMMConfig | None = None,
    normalizer_fee: float = DEFAULT_BASELINE_FEE,
) -> SimulationResult:
    """Run a clairvoyant controller on a realized realistic-mode tape."""

    exact_config = config or ExactSimpleAMMConfig.real_data_from_seed(seed)
    if exact_config.evaluator_kind != "real_data":
        raise ValueError("realistic clairvoyant oracle requires real_data evaluator config")
    tape = build_realistic_tape(config=exact_config, seed=seed)
    price_path = _build_realistic_price_path(exact_config, tape)
    bid_fee, ask_fee = controller.initialize(config=exact_config, tape=tape)  # type: ignore[arg-type]
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
        pre_step_fees = controller.choose_fees(
            OracleContext(
                event="pre_step",
                step=step,
                fair_price=fair_price,
                submission=submission,
                normalizer=normalizer,
                tape=tape,  # type: ignore[arg-type]
                config=exact_config,
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

        reference_state = submission if exact_config.retail_impact_reference_venue == "submission" else normalizer
        orders = decode_realistic_orders(
            config=exact_config,
            tape=tape,
            step=step,
            fair_price=fair_price,
            reference_state=reference_state,
        )

        for order_index, order in enumerate(orders):
            bid_fee, ask_fee = controller.choose_fees(
                OracleContext(
                    event="pre_order",
                    step=step,
                    fair_price=fair_price,
                    submission=submission,
                    normalizer=normalizer,
                    tape=tape,  # type: ignore[arg-type]
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


def run_realistic_clairvoyant_batch(
    controller_factory,
    seeds: Iterable[int],
    *,
    normalizer_fee: float = DEFAULT_BASELINE_FEE,
    config_factory=None,
) -> BatchResult:
    """Run a realistic-tape clairvoyant controller across many seeds."""

    seed_tuple = tuple(int(seed) for seed in seeds)
    simulations = tuple(
        run_realistic_clairvoyant_seed(
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
            "evaluator_kind": "real_data",
            "clairvoyance": "realized_price_and_retail_impact_tape",
            "submission_liquidity_fraction": mean(sim.initial_value / sim.initial_value_normalizer for sim in simulations),
        },
    )
