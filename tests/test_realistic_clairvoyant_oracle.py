from __future__ import annotations

from dataclasses import replace
import math

import pytest

from arena_eval.exact_simple_amm import (
    ExactSimpleAMMConfig,
    FixedFeeClairvoyantController,
    FixedFeeStrategy,
    StructuredRetailOracleController,
    run_realistic_clairvoyant_batch,
    run_realistic_clairvoyant_seed,
    run_seed,
)


def test_fixed_fee_realistic_clairvoyant_matches_exact_simulator() -> None:
    seed = 7
    config = replace(ExactSimpleAMMConfig.real_data_from_seed(seed), n_steps=64)
    exact = run_seed(
        FixedFeeStrategy(0.003, 0.003),
        seed,
        config=config,
        normalizer_strategy=FixedFeeStrategy(0.003, 0.003),
    )
    oracle = run_realistic_clairvoyant_seed(
        FixedFeeClairvoyantController(0.003, 0.003),
        seed,
        config=config,
    )

    assert oracle.edge_submission == pytest.approx(exact.edge_submission)
    assert oracle.edge_normalizer == pytest.approx(exact.edge_normalizer)
    assert oracle.pnl_submission == pytest.approx(exact.pnl_submission)
    assert oracle.pnl_normalizer == pytest.approx(exact.pnl_normalizer)
    assert oracle.retail_edge_submission == pytest.approx(exact.retail_edge_submission)
    assert oracle.retail_edge_normalizer == pytest.approx(exact.retail_edge_normalizer)
    assert oracle.arb_loss_submission == pytest.approx(exact.arb_loss_submission)
    assert oracle.arb_loss_normalizer == pytest.approx(exact.arb_loss_normalizer)


def test_structured_realistic_clairvoyant_is_deterministic() -> None:
    seed = 11
    config = replace(ExactSimpleAMMConfig.real_data_from_seed(seed), n_steps=64)
    first = run_realistic_clairvoyant_seed(
        StructuredRetailOracleController(fee_grid_size=101),
        seed,
        config=config,
    )
    second = run_realistic_clairvoyant_seed(
        StructuredRetailOracleController(fee_grid_size=101),
        seed,
        config=config,
    )

    assert first == second
    assert math.isfinite(first.score)


def test_realistic_clairvoyant_batch_runs() -> None:
    batch = run_realistic_clairvoyant_batch(
        lambda: StructuredRetailOracleController(fee_grid_size=51),
        range(3),
        config_factory=lambda seed: replace(ExactSimpleAMMConfig.real_data_from_seed(seed), n_steps=32),
    )

    assert len(batch.simulations) == 3
    assert batch.metadata["evaluator_kind"] == "real_data"
    assert math.isfinite(batch.score)
