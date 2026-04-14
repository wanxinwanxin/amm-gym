import json

import numpy as np

from amm_gym import AMMFeeEnv
from amm_gym.sim.engine import SimConfig
from training.cem import CEMConfig, CEMTrainer
from training.fixed_fee_benchmarks import (
    run_fixed_fee_sweep,
    save_benchmark_result,
)


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


def test_fixed_fee_benchmark_smoke(tmp_path):
    result = run_fixed_fee_sweep(
        num_seeds=2,
        episode_length=8,
        fee_bps_values=(0, 10),
    )

    assert [row.fee_bps for row in result.rows] == [0, 10]
    assert len(result.rows) == 2

    for row in result.rows:
        assert np.isfinite(row.mean_reward)
        assert np.isfinite(row.std_reward)
        assert np.isfinite(row.mean_edge)
        assert np.isfinite(row.std_edge)
        assert np.isfinite(row.mean_pnl)
        assert np.isfinite(row.std_pnl)

    output_path = tmp_path / "fixed_fee_benchmark.json"
    save_benchmark_result(result, output_path)
    payload = json.loads(output_path.read_text())

    assert payload["num_seeds"] == 2
    assert payload["episode_length"] == 8
    assert [row["fee_bps"] for row in payload["rows"]] == [0, 10]
    for row in payload["rows"]:
        assert np.isfinite(row["mean_reward"])
        assert np.isfinite(row["std_reward"])
        assert np.isfinite(row["mean_edge"])
        assert np.isfinite(row["std_edge"])
        assert np.isfinite(row["mean_pnl"])
        assert np.isfinite(row["std_pnl"])
