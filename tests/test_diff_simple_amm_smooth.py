from __future__ import annotations

from dataclasses import replace

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from arena_eval.diff_simple_amm import (
    DiffMode,
    DiffSimpleAMMSimulatorConfig,
    SubmissionCompactDiffPolicy,
    build_challenge_tape,
    challenge_env_vector,
    expected_submission_edge,
    run_challenge_rollout,
    smooth_submission_compact_batch_result,
    smooth_submission_compact_result,
    submission_compact_param_vector,
)
from arena_eval.exact_simple_amm.config import ExactSimpleAMMConfig
from arena_policies.submission_safe import SubmissionCompactParams


def test_smooth_train_rollout_runs_via_simulator_mode() -> None:
    config = replace(ExactSimpleAMMConfig.from_seed(3), n_steps=8)
    tape = build_challenge_tape(config=config, seed=3)
    result = run_challenge_rollout(
        config=DiffSimpleAMMSimulatorConfig(mode=DiffMode.SMOOTH_TRAIN, seed=3, exact_config=config),
        tape=tape,
        submission_policy=SubmissionCompactDiffPolicy(),
    )

    assert isinstance(result.score, float)
    assert result.metadata["mode"] == "smooth_train"


def test_smooth_submission_compact_result_is_finite() -> None:
    config = replace(ExactSimpleAMMConfig.from_seed(5), n_steps=8)
    tape = build_challenge_tape(config=config, seed=5)
    params = submission_compact_param_vector(SubmissionCompactParams())
    env = challenge_env_vector(config)
    result = smooth_submission_compact_result(params, config=config, tape=tape, env_vector=env, seed=5)

    assert result.score == pytest.approx(result.edge_submission)
    assert jnp.isfinite(jnp.asarray(result.score))
    assert jnp.isfinite(jnp.asarray(result.pnl_submission))


def test_smooth_submission_edge_has_policy_gradient() -> None:
    config = replace(ExactSimpleAMMConfig.from_seed(7), n_steps=8)
    tape = build_challenge_tape(config=config, seed=7)
    params = submission_compact_param_vector(SubmissionCompactParams())
    env = challenge_env_vector(config)

    value = expected_submission_edge(params, config=config, tape=tape, env_vector=env)
    grad = jax.grad(lambda p: expected_submission_edge(p, config=config, tape=tape, env_vector=env))(params)

    assert jnp.isfinite(value)
    assert grad.shape == params.shape
    assert jnp.all(jnp.isfinite(grad))


def test_smooth_submission_edge_has_environment_gradient() -> None:
    config = replace(ExactSimpleAMMConfig.from_seed(11), n_steps=8)
    tape = build_challenge_tape(config=config, seed=11)
    params = submission_compact_param_vector(SubmissionCompactParams())
    env = challenge_env_vector(config)

    grad = jax.grad(lambda e: expected_submission_edge(params, config=config, tape=tape, env_vector=e))(env)

    assert grad.shape == env.shape
    assert jnp.all(jnp.isfinite(grad))


def test_smooth_batch_result_aggregates_multiple_tapes() -> None:
    config = replace(ExactSimpleAMMConfig.from_seed(13), n_steps=8)
    tapes = tuple(build_challenge_tape(config=config, seed=seed) for seed in (13, 14))
    params = submission_compact_param_vector(SubmissionCompactParams())
    batch = smooth_submission_compact_batch_result(params, config=config, tapes=tapes)

    assert isinstance(batch.score, float)
    assert batch.metadata["n_tapes"] == 2


def test_smooth_policy_gradient_matches_finite_difference_for_base_fee() -> None:
    config = replace(ExactSimpleAMMConfig.from_seed(17), n_steps=8)
    tape = build_challenge_tape(config=config, seed=17)
    params = submission_compact_param_vector(SubmissionCompactParams(base_fee=0.004, min_fee=0.001, max_fee=0.02))
    env = challenge_env_vector(config)
    epsilon = 1e-2

    def objective(vector):
        return expected_submission_edge(vector, config=config, tape=tape, env_vector=env)

    autodiff_grad = jax.grad(objective)(params)[0]
    plus = params.at[0].add(epsilon)
    minus = params.at[0].add(-epsilon)
    finite_diff = (objective(plus) - objective(minus)) / (2.0 * epsilon)

    assert float(autodiff_grad) == pytest.approx(float(finite_diff), rel=0.5, abs=0.5)
