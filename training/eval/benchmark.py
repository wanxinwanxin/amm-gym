"""Scenario and split definitions for research evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Literal

from amm_gym import AMMChallengeEnv, AMMFeeEnv
from amm_gym.sim.engine import SimConfig

from demo.presets import DEFAULT_DEMO_STEPS, DEFAULT_WINDOW_SIZE, build_hackathon_demo_config
from training.eval.metrics import aggregate_episode_metrics, evaluate_episode


EnvKind = Literal["classic", "challenge"]


@dataclass(frozen=True)
class SeedSplit:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]


@dataclass(frozen=True)
class BenchmarkScenario:
    name: str
    env_kind: EnvKind
    schedule: tuple[tuple[int, float], ...] | None
    retail_arrival_rate: float
    steps: int = DEFAULT_DEMO_STEPS

    def env_factory(self, *, window_size: int = DEFAULT_WINDOW_SIZE):
        if self.env_kind == "challenge":
            def factory() -> AMMChallengeEnv:
                config = SimConfig(
                    n_steps=self.steps,
                    gbm_sigma=0.001,
                    volatility_schedule=self.schedule,
                    retail_arrival_rate=self.retail_arrival_rate,
                    retail_mean_size=18.0,
                    retail_size_sigma=0.7,
                    retail_buy_prob=0.5,
                    seed=0,
                )
                return AMMChallengeEnv(config=config, window_size=window_size)

            return factory

        def factory() -> AMMFeeEnv:
            config = build_hackathon_demo_config(seed=0, steps=self.steps, schedule=self.schedule)
            config = replace(config, retail_arrival_rate=self.retail_arrival_rate)
            return AMMFeeEnv(config=config, window_size=window_size)

        return factory


def default_seed_split() -> SeedSplit:
    return SeedSplit(
        train=tuple(range(32)),
        validation=tuple(range(100, 116)),
        test=tuple(range(1000, 1032)),
    )


def benchmark_scenarios(env_kind: EnvKind = "classic") -> dict[str, BenchmarkScenario]:
    if env_kind == "challenge":
        return {
            "challenge_single_regime": BenchmarkScenario(
                name="challenge_single_regime",
                env_kind="challenge",
                schedule=None,
                retail_arrival_rate=0.8,
                steps=256,
            ),
            "challenge_regime_shift": BenchmarkScenario(
                name="challenge_regime_shift",
                env_kind="challenge",
                schedule=((0, 0.0008), (96, 0.0035), (192, 0.0012)),
                retail_arrival_rate=0.8,
                steps=256,
            ),
        }
    return {
        "constant_low_vol": BenchmarkScenario(
            name="constant_low_vol",
            env_kind="classic",
            schedule=None,
            retail_arrival_rate=6.0,
        ),
        "regime_shift": BenchmarkScenario(
            name="regime_shift",
            env_kind="classic",
            schedule=((0, 0.0010), (40, 0.0035), (80, 0.0015)),
            retail_arrival_rate=6.0,
        ),
        "high_arrival_regime_shift": BenchmarkScenario(
            name="high_arrival_regime_shift",
            env_kind="classic",
            schedule=((0, 0.0010), (40, 0.0035), (80, 0.0015)),
            retail_arrival_rate=9.0,
        ),
    }


def evaluate_policy_across_scenarios(
    *,
    policy_name: str,
    policy,
    scenario_names: tuple[str, ...] | None = None,
    split_name: str = "test",
    window_size: int = DEFAULT_WINDOW_SIZE,
    seed_split: SeedSplit | None = None,
    env_kind: EnvKind = "classic",
) -> dict[str, object]:
    scenarios = benchmark_scenarios(env_kind=env_kind)
    if scenario_names is None:
        scenario_names = tuple(scenarios)
    split = seed_split or default_seed_split()
    seeds = getattr(split, split_name)
    results: dict[str, object] = {
        "policy": policy_name,
        "split": split_name,
        "env_kind": env_kind,
        "seed_split": asdict(split),
        "scenarios": {},
    }
    for scenario_name in scenario_names:
        scenario = scenarios[scenario_name]
        env_factory = scenario.env_factory(window_size=window_size)
        metrics = [evaluate_episode(env_factory, policy, seed=seed) for seed in seeds]
        results["scenarios"][scenario_name] = {
            "scenario": scenario_name,
            "config": {
                "steps": scenario.steps,
                "retail_arrival_rate": scenario.retail_arrival_rate,
                "schedule": scenario.schedule,
                "env_kind": scenario.env_kind,
            },
            "summary": aggregate_episode_metrics(metrics),
        }
    return results
