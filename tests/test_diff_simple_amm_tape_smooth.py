"""Faithfulness tests for the tape-smooth simulator.

Replays the exact tape's discrete realizations through the naturally-continuous
(no-sharpness) simulator and asserts the score is bit-close to the exact
evaluator across many seeds, in both challenge and realistic modes.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from arena_eval.diff_simple_amm import build_challenge_tape, build_realistic_tape
from arena_eval.diff_simple_amm.tape_smooth import fixed_fee_result
from arena_eval.exact_simple_amm import ExactSimpleAMMConfig, FixedFeeStrategy, run_seed


SEEDS = (0, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43)
N_STEPS = 256


@pytest.mark.parametrize("seed", SEEDS)
def test_tape_smooth_matches_exact_fixed_fee_challenge(seed: int) -> None:
    cfg = replace(ExactSimpleAMMConfig.from_seed(seed), n_steps=N_STEPS)
    tape = build_challenge_tape(config=cfg, seed=seed)
    smooth = fixed_fee_result(cfg, tape, seed=seed)
    exact = run_seed(
        FixedFeeStrategy(),
        seed,
        config=cfg,
        normalizer_strategy=FixedFeeStrategy(),
    )
    # float32 in JAX vs float64 in numpy — relative tolerance 1e-3 is the bar.
    assert smooth.edge_submission == pytest.approx(exact.edge_submission, rel=1e-3, abs=1e-3)
    assert smooth.edge_normalizer == pytest.approx(exact.edge_normalizer, rel=1e-3, abs=1e-3)
    assert smooth.retail_volume_submission_y == pytest.approx(
        exact.retail_volume_submission_y, rel=1e-3, abs=1e-3
    )
    assert smooth.retail_volume_normalizer_y == pytest.approx(
        exact.retail_volume_normalizer_y, rel=1e-3, abs=1e-3
    )
    assert smooth.arb_volume_submission_y == pytest.approx(
        exact.arb_volume_submission_y, rel=1e-3, abs=1e-3
    )
    assert smooth.arb_volume_normalizer_y == pytest.approx(
        exact.arb_volume_normalizer_y, rel=1e-3, abs=1e-3
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_tape_smooth_matches_exact_fixed_fee_realistic(seed: int) -> None:
    cfg = replace(ExactSimpleAMMConfig.real_data_from_seed(seed), n_steps=N_STEPS)
    tape = build_realistic_tape(config=cfg, seed=seed)
    smooth = fixed_fee_result(cfg, tape, seed=seed)
    exact = run_seed(
        FixedFeeStrategy(),
        seed,
        config=cfg,
        normalizer_strategy=FixedFeeStrategy(),
    )
    assert smooth.edge_submission == pytest.approx(exact.edge_submission, rel=5e-3, abs=1e-3)
    assert smooth.edge_normalizer == pytest.approx(exact.edge_normalizer, rel=5e-3, abs=1e-3)


def test_tape_smooth_is_differentiable_in_fees_challenge() -> None:
    """The whole point: gradient wrt fee parameters is finite and non-trivial."""

    seed = 5
    cfg = replace(ExactSimpleAMMConfig.from_seed(seed), n_steps=64)
    tape = build_challenge_tape(config=cfg, seed=seed)
    from arena_eval.diff_simple_amm.tape_smooth import fixed_fee_metrics_challenge

    def edge_of_fee(fee):
        metrics = fixed_fee_metrics_challenge(
            cfg,
            tape,
            submission_bid_fee=fee,
            submission_ask_fee=fee,
        )
        return metrics["edge_submission"]

    fee = jnp.asarray(0.003, dtype=jnp.float32)
    value = edge_of_fee(fee)
    grad = jax.grad(edge_of_fee)(fee)

    assert jnp.isfinite(value)
    assert jnp.isfinite(grad)
    # No assertion on direction — just that the gradient exists and is non-zero.
    assert float(jnp.abs(grad)) > 0.0
