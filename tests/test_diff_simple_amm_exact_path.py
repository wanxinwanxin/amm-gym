from __future__ import annotations

from dataclasses import replace

import pytest

from arena_eval.diff_simple_amm import (
    DiffMode,
    DiffSimpleAMMSimulatorConfig,
    FixedFeeDiffPolicy,
    SubmissionCompactDiffPolicy,
    build_challenge_tape,
    run_challenge_rollout,
)
from arena_eval.exact_simple_amm import ExactSimpleAMMConfig, FixedFeeStrategy, run_seed
from arena_policies.submission_safe import SubmissionCompactParams, SubmissionCompactStrategy


@pytest.mark.parametrize("seed", [0, 7, 19])
def test_diff_exact_path_matches_exact_simulator_for_fixed_fee(seed: int) -> None:
    exact_config = replace(ExactSimpleAMMConfig.from_seed(seed), n_steps=128)
    tape = build_challenge_tape(config=exact_config, seed=seed)
    diff_result = run_challenge_rollout(
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
    assert diff_result.score == pytest.approx(exact_result.score)
    assert diff_result.retail_volume_submission_y == pytest.approx(exact_result.retail_volume_submission_y)
    assert diff_result.retail_volume_normalizer_y == pytest.approx(exact_result.retail_volume_normalizer_y)
    assert diff_result.arb_volume_submission_y == pytest.approx(exact_result.arb_volume_submission_y)
    assert diff_result.arb_volume_normalizer_y == pytest.approx(exact_result.arb_volume_normalizer_y)
    assert diff_result.average_bid_fee_submission == pytest.approx(exact_result.average_bid_fee_submission)
    assert diff_result.average_ask_fee_submission == pytest.approx(exact_result.average_ask_fee_submission)
    assert diff_result.average_bid_fee_normalizer == pytest.approx(exact_result.average_bid_fee_normalizer)
    assert diff_result.average_ask_fee_normalizer == pytest.approx(exact_result.average_ask_fee_normalizer)


def test_build_challenge_tape_tracks_counts_and_max_orders() -> None:
    config = ExactSimpleAMMConfig(n_steps=32, retail_arrival_rate=2.5, retail_mean_size=10.0)
    tape = build_challenge_tape(config=config, seed=11)

    assert len(tape.gbm_normals) == config.n_steps
    assert len(tape.order_counts) == config.n_steps
    assert len(tape.order_sizes) == config.n_steps
    assert len(tape.order_side_uniforms) == config.n_steps
    assert tape.max_orders_per_step == max(tape.order_counts, default=0)
    assert all(len(sizes) == count for sizes, count in zip(tape.order_sizes, tape.order_counts))
    assert all(len(sides) == count for sides, count in zip(tape.order_side_uniforms, tape.order_counts))


def test_diff_exact_path_matches_submission_compact_strategy() -> None:
    seed = 5
    params = SubmissionCompactParams(
        base_fee=0.004,
        flow_fast_decay=0.61,
        flow_slow_decay=0.9,
        skew_weight=0.07,
        hot_fee_bump=0.005,
    ).normalized()
    exact_config = replace(ExactSimpleAMMConfig.from_seed(seed), n_steps=128)
    tape = build_challenge_tape(config=exact_config, seed=seed)
    diff_result = run_challenge_rollout(
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
    assert diff_result.score == pytest.approx(exact_result.score)
    assert diff_result.retail_volume_submission_y == pytest.approx(exact_result.retail_volume_submission_y)
    assert diff_result.retail_volume_normalizer_y == pytest.approx(exact_result.retail_volume_normalizer_y)
    assert diff_result.arb_volume_submission_y == pytest.approx(exact_result.arb_volume_submission_y)
    assert diff_result.arb_volume_normalizer_y == pytest.approx(exact_result.arb_volume_normalizer_y)
    assert diff_result.average_bid_fee_submission == pytest.approx(exact_result.average_bid_fee_submission)
    assert diff_result.average_ask_fee_submission == pytest.approx(exact_result.average_ask_fee_submission)
    assert diff_result.average_bid_fee_normalizer == pytest.approx(exact_result.average_bid_fee_normalizer)
    assert diff_result.average_ask_fee_normalizer == pytest.approx(exact_result.average_ask_fee_normalizer)
