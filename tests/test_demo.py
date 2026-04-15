"""Smoke tests for builder-facing demo entrypoints."""

from __future__ import annotations

from demo.run_policy_benchmark import evaluate_policy, named_schedules
from amm_gym.baselines import benchmark_depth_policies


def test_policy_benchmark_smoke():
    row = evaluate_policy(
        policy_name="balanced",
        policy=benchmark_depth_policies()["balanced"],
        schedule_name="constant_low_vol",
        schedule=named_schedules()["constant_low_vol"],
        steps=20,
        window_size=6,
        seeds=2,
    )

    assert row["policy"] == "balanced"
    assert row["schedule"] == "constant_low_vol"
    assert isinstance(row["mean_reward"], float)
    assert isinstance(row["mean_pnl"], float)
