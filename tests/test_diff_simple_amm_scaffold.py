from arena_eval.diff_simple_amm import (
    AMMState,
    ChallengeTape,
    DiffMode,
    DiffSimpleAMMSimulatorConfig,
    FixedFeeDiffPolicy,
    PolicyOutput,
    PolicyState,
    RealisticTape,
    SimulatorState,
    SubmissionCompactDiffPolicy,
    build_challenge_tape,
    build_realistic_tape,
)
from arena_eval.exact_simple_amm import ExactSimpleAMMConfig


def test_diff_simple_amm_scaffold_imports() -> None:
    exact_config = ExactSimpleAMMConfig(n_steps=2, retail_arrival_rate=0.5, retail_mean_size=10.0)
    config = DiffSimpleAMMSimulatorConfig(mode=DiffMode.EXACT_PATH, seed=3, exact_config=exact_config)
    tape = ChallengeTape(
        gbm_normals=(0.0, 1.0),
        order_counts=(3, 2),
        order_sizes=((10.0, 11.0, 12.0), (9.0, 8.0)),
        order_side_uniforms=((0.2, 0.8, 0.4), (0.7, 0.1)),
        max_orders_per_step=3,
    )
    policy_output = PolicyOutput(bid_fee=0.003, ask_fee=0.004, state=PolicyState((1.0, 2.0)))
    state = SimulatorState(
        step=0,
        fair_price=1.0,
        submission=AMMState(1_000.0, 1_000.0, 0.003, 0.003),
        normalizer=AMMState(1_000.0, 1_000.0, 0.003, 0.003),
        submission_policy_state=policy_output.state,
        normalizer_policy_state=PolicyState(),
    )

    assert config.n_steps == 2
    assert tape.max_orders_per_step == 3
    assert state.submission.bid_fee == 0.003
    assert policy_output.state.values == (1.0, 2.0)
    assert FixedFeeDiffPolicy().bid_fee == 0.003
    assert SubmissionCompactDiffPolicy().params.base_fee > 0.0
    generated = build_challenge_tape(config=exact_config, seed=3)
    assert len(generated.gbm_normals) == exact_config.n_steps
    realistic_config = ExactSimpleAMMConfig.real_data_from_seed(3)
    realistic_tape = build_realistic_tape(config=realistic_config, seed=3)
    assert isinstance(realistic_tape, RealisticTape)
