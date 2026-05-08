from __future__ import annotations

import math

import pytest

from arena_eval.exact_simple_amm import (
    ExactSimpleAMMConfig,
    FixedFeeClairvoyantController,
    FixedFeeStrategy,
    GreedyStepOracleController,
    StructuredRetailOracleController,
    run_clairvoyant_batch,
    run_clairvoyant_seed,
    run_seed,
)
from arena_eval.exact_simple_amm.oracle import OracleContext
from arena_eval.diff_simple_amm.types import AMMState, ChallengeTape


def test_fixed_fee_clairvoyant_matches_exact_simulator():
    config = ExactSimpleAMMConfig(n_steps=64)
    exact = run_seed(FixedFeeStrategy(0.003, 0.003), 7, config=config)
    oracle = run_clairvoyant_seed(FixedFeeClairvoyantController(0.003, 0.003), 7, config=config)

    assert oracle == exact


def test_greedy_step_oracle_is_deterministic():
    config = ExactSimpleAMMConfig(n_steps=32)
    first = run_clairvoyant_seed(GreedyStepOracleController(fee_grid_size=21), 11, config=config)
    second = run_clairvoyant_seed(GreedyStepOracleController(fee_grid_size=21), 11, config=config)

    assert first == second
    assert math.isfinite(first.score)


def test_clairvoyant_batch_runs():
    config = ExactSimpleAMMConfig(n_steps=24)
    batch = run_clairvoyant_batch(
        lambda: GreedyStepOracleController(fee_grid_size=11),
        range(3),
        config_factory=lambda seed: config,
    )

    assert len(batch.simulations) == 3
    assert math.isfinite(batch.score)


def test_structured_retail_oracle_is_deterministic():
    config = ExactSimpleAMMConfig(n_steps=32)
    first = run_clairvoyant_seed(StructuredRetailOracleController(fee_grid_size=101), 13, config=config)
    second = run_clairvoyant_seed(StructuredRetailOracleController(fee_grid_size=101), 13, config=config)

    assert first == second
    assert math.isfinite(first.score)


def test_structured_retail_pre_step_uses_no_arb_threshold():
    controller = StructuredRetailOracleController(fee_grid_size=11)
    context = OracleContext(
        event="pre_step",
        step=0,
        fair_price=101.0,
        submission=AMMState(
            reserve_x=100.0,
            reserve_y=10_000.0,
            bid_fee=0.003,
            ask_fee=0.003,
        ),
        normalizer=AMMState(
            reserve_x=100.0,
            reserve_y=10_000.0,
            bid_fee=0.003,
            ask_fee=0.003,
        ),
        tape=ChallengeTape((), (), (), (), 0),
        config=ExactSimpleAMMConfig(n_steps=1),
    )

    bid_fee, ask_fee = controller.choose_fees(context)

    assert bid_fee == 0.003
    assert ask_fee == pytest.approx(1.0 - 100.0 / 101.0)
