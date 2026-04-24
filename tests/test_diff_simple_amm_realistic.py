from __future__ import annotations

from dataclasses import replace

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from arena_eval.diff_simple_amm import (
    DiffMode,
    DiffSimpleAMMSimulatorConfig,
    FixedFeeDiffPolicy,
    SubmissionCompactDiffPolicy,
    build_realistic_tape,
    expected_submission_edge,
    realistic_env_vector,
    run_realistic_rollout,
    smooth_submission_compact_result,
    submission_compact_param_vector,
)
from arena_eval.exact_simple_amm import ExactSimpleAMMConfig, FixedFeeStrategy, run_seed
from arena_policies.submission_safe import SubmissionCompactParams, SubmissionCompactStrategy


@pytest.mark.parametrize("seed", [0, 4])
def test_diff_exact_path_matches_exact_realistic_simulator_for_fixed_fee(seed: int) -> None:
    exact_config = replace(ExactSimpleAMMConfig.real_data_from_seed(seed), n_steps=64)
    tape = build_realistic_tape(config=exact_config, seed=seed)
    diff_result = run_realistic_rollout(
        config=DiffSimpleAMMSimulatorConfig(mode=DiffMode.EXACT_PATH, seed=seed, exact_config=exact_config),
        tape=tape,
        submission_policy=FixedFeeDiffPolicy(),
        normalizer_policy=FixedFeeDiffPolicy(),
    )
    exact_result = run_seed(
        FixedFeeStrategy(),
        seed,
        config=exact_config,
        normalizer_strategy=FixedFeeStrategy(),
    )

    assert diff_result.edge_submission == pytest.approx(exact_result.edge_submission)
    assert diff_result.edge_normalizer == pytest.approx(exact_result.edge_normalizer)
    assert diff_result.pnl_submission == pytest.approx(exact_result.pnl_submission)
    assert diff_result.pnl_normalizer == pytest.approx(exact_result.pnl_normalizer)
    assert diff_result.retail_volume_submission_y == pytest.approx(exact_result.retail_volume_submission_y)
    assert diff_result.retail_volume_normalizer_y == pytest.approx(exact_result.retail_volume_normalizer_y)
    assert diff_result.arb_volume_submission_y == pytest.approx(exact_result.arb_volume_submission_y)
    assert diff_result.arb_volume_normalizer_y == pytest.approx(exact_result.arb_volume_normalizer_y)


def test_diff_exact_path_matches_submission_compact_strategy_on_realistic_eval() -> None:
    seed = 7
    params = SubmissionCompactParams(
        base_fee=0.004,
        flow_fast_decay=0.61,
        flow_slow_decay=0.9,
        skew_weight=0.07,
        hot_fee_bump=0.005,
    ).normalized()
    exact_config = replace(ExactSimpleAMMConfig.real_data_from_seed(seed), n_steps=64)
    tape = build_realistic_tape(config=exact_config, seed=seed)
    diff_result = run_realistic_rollout(
        config=DiffSimpleAMMSimulatorConfig(mode=DiffMode.EXACT_PATH, seed=seed, exact_config=exact_config),
        tape=tape,
        submission_policy=SubmissionCompactDiffPolicy(params=params),
        normalizer_policy=FixedFeeDiffPolicy(),
    )
    exact_result = run_seed(
        SubmissionCompactStrategy(params),
        seed,
        config=exact_config,
        normalizer_strategy=FixedFeeStrategy(),
    )

    assert diff_result.edge_submission == pytest.approx(exact_result.edge_submission)
    assert diff_result.edge_normalizer == pytest.approx(exact_result.edge_normalizer)
    assert diff_result.pnl_submission == pytest.approx(exact_result.pnl_submission)
    assert diff_result.pnl_normalizer == pytest.approx(exact_result.pnl_normalizer)


def test_smooth_realistic_rollout_and_gradients_are_finite() -> None:
    seed = 9
    config = replace(ExactSimpleAMMConfig.real_data_from_seed(seed), n_steps=8)
    tape = build_realistic_tape(config=config, seed=seed)
    params = submission_compact_param_vector(SubmissionCompactParams())
    env = realistic_env_vector(config)

    result = smooth_submission_compact_result(params, config=config, tape=tape, env_vector=env, seed=seed)
    policy_grad = jax.grad(lambda p: expected_submission_edge(p, config=config, tape=tape, env_vector=env))(params)
    env_grad = jax.grad(lambda e: expected_submission_edge(params, config=config, tape=tape, env_vector=e))(env)

    assert jnp.isfinite(jnp.asarray(result.score))
    assert jnp.all(jnp.isfinite(policy_grad))
    assert jnp.all(jnp.isfinite(env_grad))
