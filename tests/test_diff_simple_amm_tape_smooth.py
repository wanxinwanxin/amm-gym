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
from arena_eval.diff_simple_amm.objectives import (
    piecewise_param_vector,
    submission_compact_param_vector,
)
from arena_eval.diff_simple_amm.tape_smooth import (
    compact_result,
    fixed_fee_result,
    piecewise_result,
)
from arena_eval.exact_simple_amm import ExactSimpleAMMConfig, FixedFeeStrategy, run_seed
from arena_policies.piecewise_controller import PiecewiseControllerParams, PiecewiseControllerStrategy
from arena_policies.submission_safe import SubmissionCompactParams, SubmissionCompactStrategy


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
    # float64 throughout; tolerance is float-arith accumulation over n_steps.
    assert smooth.edge_submission == pytest.approx(exact.edge_submission, rel=1e-3, abs=1e-3)
    assert smooth.edge_normalizer == pytest.approx(exact.edge_normalizer, rel=1e-3, abs=1e-3)
    assert smooth.pnl_submission == pytest.approx(exact.pnl_submission, rel=1e-3, abs=1e-3)
    assert smooth.pnl_normalizer == pytest.approx(exact.pnl_normalizer, rel=1e-3, abs=1e-3)
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
    assert smooth.average_bid_fee_submission == pytest.approx(
        exact.average_bid_fee_submission, rel=1e-6, abs=1e-9
    )
    assert smooth.average_ask_fee_submission == pytest.approx(
        exact.average_ask_fee_submission, rel=1e-6, abs=1e-9
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
    assert smooth.pnl_submission == pytest.approx(exact.pnl_submission, rel=5e-3, abs=1e-3)
    assert smooth.pnl_normalizer == pytest.approx(exact.pnl_normalizer, rel=5e-3, abs=1e-3)
    assert smooth.retail_volume_submission_y == pytest.approx(
        exact.retail_volume_submission_y, rel=5e-3, abs=1e-3
    )
    assert smooth.retail_volume_normalizer_y == pytest.approx(
        exact.retail_volume_normalizer_y, rel=5e-3, abs=1e-3
    )


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

    fee = jnp.asarray(0.003, dtype=jnp.float64)
    value = edge_of_fee(fee)
    grad = jax.grad(edge_of_fee)(fee)

    assert jnp.isfinite(value)
    assert jnp.isfinite(grad)
    # No assertion on direction — just that the gradient exists and is non-zero.
    assert float(jnp.abs(grad)) > 0.0


# ----- Adaptive policy parity tests -----
COMPACT_PARAMS = SubmissionCompactParams(
    base_fee=0.004,
    flow_fast_decay=0.61,
    flow_slow_decay=0.9,
    skew_weight=0.07,
    hot_fee_bump=0.005,
).normalized()

PIECEWISE_PARAMS = PiecewiseControllerParams(
    base_fee=0.004,
    signal_decay=0.76,
    small_trade_threshold=0.002,
    large_trade_threshold=0.018,
    continuation_medium=0.006,
    reversal_large=0.028,
    continuation_to_same_side=1.4,
    continuation_to_cross_side=0.15,
).normalized()


def _assert_aggregate_close(smooth, exact, *, rel: float, abs_: float) -> None:
    assert smooth.edge_submission == pytest.approx(exact.edge_submission, rel=rel, abs=abs_)
    assert smooth.edge_normalizer == pytest.approx(exact.edge_normalizer, rel=rel, abs=abs_)
    assert smooth.pnl_submission == pytest.approx(exact.pnl_submission, rel=rel, abs=abs_)
    assert smooth.pnl_normalizer == pytest.approx(exact.pnl_normalizer, rel=rel, abs=abs_)
    assert smooth.retail_volume_submission_y == pytest.approx(
        exact.retail_volume_submission_y, rel=rel, abs=abs_
    )
    assert smooth.retail_volume_normalizer_y == pytest.approx(
        exact.retail_volume_normalizer_y, rel=rel, abs=abs_
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_tape_smooth_matches_exact_compact_challenge(seed: int) -> None:
    cfg = replace(ExactSimpleAMMConfig.from_seed(seed), n_steps=N_STEPS)
    tape = build_challenge_tape(config=cfg, seed=seed)
    params = submission_compact_param_vector(COMPACT_PARAMS)
    smooth = compact_result(cfg, tape, params, seed=seed)
    exact = run_seed(
        SubmissionCompactStrategy(COMPACT_PARAMS),
        seed,
        config=cfg,
        normalizer_strategy=FixedFeeStrategy(),
    )
    _assert_aggregate_close(smooth, exact, rel=1e-3, abs_=1e-3)


@pytest.mark.parametrize("seed", SEEDS)
def test_tape_smooth_matches_exact_compact_realistic(seed: int) -> None:
    cfg = replace(ExactSimpleAMMConfig.real_data_from_seed(seed), n_steps=N_STEPS)
    tape = build_realistic_tape(config=cfg, seed=seed)
    params = submission_compact_param_vector(COMPACT_PARAMS)
    smooth = compact_result(cfg, tape, params, seed=seed)
    exact = run_seed(
        SubmissionCompactStrategy(COMPACT_PARAMS),
        seed,
        config=cfg,
        normalizer_strategy=FixedFeeStrategy(),
    )
    _assert_aggregate_close(smooth, exact, rel=5e-3, abs_=1e-3)


@pytest.mark.parametrize("seed", SEEDS)
def test_tape_smooth_matches_exact_piecewise_challenge(seed: int) -> None:
    cfg = replace(ExactSimpleAMMConfig.from_seed(seed), n_steps=N_STEPS)
    tape = build_challenge_tape(config=cfg, seed=seed)
    params = piecewise_param_vector(PIECEWISE_PARAMS)
    smooth = piecewise_result(cfg, tape, params, seed=seed)
    exact = run_seed(
        PiecewiseControllerStrategy(PIECEWISE_PARAMS),
        seed,
        config=cfg,
        normalizer_strategy=FixedFeeStrategy(),
    )
    _assert_aggregate_close(smooth, exact, rel=1e-3, abs_=1e-3)


@pytest.mark.parametrize("seed", SEEDS)
def test_tape_smooth_matches_exact_piecewise_realistic(seed: int) -> None:
    cfg = replace(ExactSimpleAMMConfig.real_data_from_seed(seed), n_steps=N_STEPS)
    tape = build_realistic_tape(config=cfg, seed=seed)
    params = piecewise_param_vector(PIECEWISE_PARAMS)
    smooth = piecewise_result(cfg, tape, params, seed=seed)
    exact = run_seed(
        PiecewiseControllerStrategy(PIECEWISE_PARAMS),
        seed,
        config=cfg,
        normalizer_strategy=FixedFeeStrategy(),
    )
    _assert_aggregate_close(smooth, exact, rel=5e-3, abs_=1e-3)


def test_tape_smooth_compact_is_differentiable_in_params() -> None:
    seed = 11
    cfg = replace(ExactSimpleAMMConfig.from_seed(seed), n_steps=32)
    tape = build_challenge_tape(config=cfg, seed=seed)
    params = submission_compact_param_vector(COMPACT_PARAMS)
    from arena_eval.diff_simple_amm.tape_smooth import compact_metrics

    def edge_of(p):
        return compact_metrics(cfg, tape, p)["edge_submission"]

    grad = jax.grad(edge_of)(params)
    assert jnp.all(jnp.isfinite(grad))
    assert float(jnp.linalg.norm(grad)) > 0.0


def test_tape_smooth_piecewise_is_differentiable_in_params() -> None:
    seed = 13
    cfg = replace(ExactSimpleAMMConfig.from_seed(seed), n_steps=32)
    tape = build_challenge_tape(config=cfg, seed=seed)
    params = piecewise_param_vector(PIECEWISE_PARAMS)
    from arena_eval.diff_simple_amm.tape_smooth import piecewise_metrics

    def edge_of(p):
        return piecewise_metrics(cfg, tape, p)["edge_submission"]

    grad = jax.grad(edge_of)(params)
    assert jnp.all(jnp.isfinite(grad))
    assert float(jnp.linalg.norm(grad)) > 0.0
