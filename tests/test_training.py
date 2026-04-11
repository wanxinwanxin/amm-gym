import numpy as np

from amm_gym import AMMFeeEnv
from amm_gym.sim.engine import SimConfig
from training.cem import CEMConfig, CEMTrainer


def make_env() -> AMMFeeEnv:
    return AMMFeeEnv(config=SimConfig(n_steps=30), window_size=6)


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
