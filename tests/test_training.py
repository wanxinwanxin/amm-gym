import numpy as np
import pytest

from amm_gym import AMMChallengeEnv, AMMFeeEnv
from amm_gym.sim.engine import SimConfig
from training.cem import CEMConfig, CEMTrainer
from training.algorithms.ppo import PPOConfig, PPOTrainer
from training.policies import FeatureLinearPolicySpace, SmoothedFeatureLinearPolicySpace, engineer_features
from training.run_first_pass import build_parser as build_first_pass_parser


def make_env() -> AMMFeeEnv:
    return AMMFeeEnv(config=SimConfig(n_steps=30), window_size=6)


def make_challenge_env() -> AMMChallengeEnv:
    return AMMChallengeEnv(config=SimConfig(n_steps=30), window_size=6)


def test_cem_training_smoke():
    trainer = CEMTrainer(
        env_factory=make_env,
        config=CEMConfig(
            population_size=8,
            elite_frac=0.25,
            iterations=2,
            eval_episodes=1,
            seed=7,
        ),
    )

    result = trainer.train()

    assert result.best_params.shape == (trainer.spec.param_dim,)
    assert np.isfinite(result.best_score)
    assert len(result.history) == 2


def test_cem_can_optimize_relative_objective():
    trainer = CEMTrainer(
        env_factory=make_env,
        config=CEMConfig(
            population_size=6,
            elite_frac=0.34,
            iterations=1,
            eval_episodes=1,
            seed=3,
            objective="edge_advantage",
        ),
    )

    result = trainer.train()

    assert np.isfinite(result.best_score)
    assert "selected_score" in result.history[0]


def test_ppo_training_smoke():
    pytest.importorskip("torch")

    trainer = PPOTrainer(
        env_factory=make_env,
        config=PPOConfig(
            total_timesteps=128,
            rollout_steps=64,
            epochs=1,
            minibatch_size=32,
            seed=5,
            validation_interval=1,
            train_seeds=(0, 1),
            validation_seeds=(100,),
        ),
    )

    result = trainer.train()
    action = result.policy.act(np.zeros(make_env().observation_space.shape, dtype=np.float32))

    assert len(result.history) == 2
    assert np.isfinite(result.best_validation_score)
    assert action.shape == make_env().action_space.shape


def test_feature_engineering_and_training_smoke():
    env = make_env()
    obs, _ = env.reset(seed=0)
    features = engineer_features(obs)
    space = FeatureLinearPolicySpace.from_env(env)

    assert features.ndim == 1
    assert features.shape[0] == space.feature_dim

    trainer = CEMTrainer(
        env_factory=make_env,
        config=CEMConfig(
            population_size=6,
            elite_frac=0.34,
            iterations=1,
            eval_episodes=1,
            seed=9,
            objective="edge_advantage",
        ),
        policy_space_factory=FeatureLinearPolicySpace.from_env,
    )
    result = trainer.train()

    assert np.isfinite(result.best_score)


def test_smoothed_feature_policy_resets_state():
    env = make_env()
    space = SmoothedFeatureLinearPolicySpace.from_env(env, smoothing_alpha=0.5)
    params = np.zeros(space.param_dim, dtype=np.float32)
    policy = space.build_policy(params)
    obs, _ = env.reset(seed=0)

    first = policy.act(obs)
    second = policy.act(obs)
    policy.reset()
    reset_first = policy.act(obs)

    np.testing.assert_allclose(first, reset_first)
    assert np.all(second <= 1.0)


def test_challenge_env_cem_smoke():
    trainer = CEMTrainer(
        env_factory=make_challenge_env,
        config=CEMConfig(
            population_size=4,
            elite_frac=0.5,
            iterations=1,
            eval_episodes=1,
            seed=13,
            objective="reward",
        ),
    )

    result = trainer.train()

    assert result.best_params.shape == (trainer.spec.param_dim,)
    assert np.isfinite(result.best_score)


def test_run_first_pass_parser_supports_challenge_env():
    parser = build_first_pass_parser()
    args = parser.parse_args(["--env-kind", "challenge"])

    assert args.env_kind == "challenge"
    assert args.scenario is None
