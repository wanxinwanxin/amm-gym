"""Cross-entropy-method baseline trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from training.policy import LinearPolicySpec, LinearTanhPolicy


EnvFactory = Callable[[], object]


@dataclass
class CEMConfig:
    population_size: int = 24
    elite_frac: float = 0.25
    iterations: int = 5
    eval_episodes: int = 2
    init_std: float = 0.5
    seed: int = 0


@dataclass
class TrainingResult:
    best_params: np.ndarray
    best_score: float
    history: list[dict[str, float]]


class CEMTrainer:
    """Train a linear policy against the public env interface."""

    def __init__(self, env_factory: EnvFactory, config: CEMConfig) -> None:
        self.env_factory = env_factory
        self.config = config

        env = env_factory()
        obs, _ = env.reset(seed=config.seed)
        self.spec = LinearPolicySpec(
            obs_dim=obs.shape[0],
            action_dim=int(np.asarray(env.action_space.low).shape[0]),
            action_low=np.asarray(env.action_space.low, dtype=np.float32),
            action_high=np.asarray(env.action_space.high, dtype=np.float32),
        )

    def train(self) -> TrainingResult:
        rng = np.random.default_rng(self.config.seed)
        mean = np.zeros(self.spec.param_dim, dtype=np.float64)
        std = np.full(self.spec.param_dim, self.config.init_std, dtype=np.float64)

        best_params = mean.copy()
        best_score = -np.inf
        history: list[dict[str, float]] = []

        elite_count = max(1, int(self.config.population_size * self.config.elite_frac))

        for iteration in range(self.config.iterations):
            population = rng.normal(
                loc=mean,
                scale=std,
                size=(self.config.population_size, self.spec.param_dim),
            )
            scores = np.array(
                [
                    self.evaluate_params(candidate, seed_offset=iteration * 1000 + idx)
                    for idx, candidate in enumerate(population)
                ],
                dtype=np.float64,
            )

            elite_idx = np.argsort(scores)[-elite_count:]
            elites = population[elite_idx]
            mean = elites.mean(axis=0)
            std = np.maximum(elites.std(axis=0), 1e-3)

            elite_best_idx = elite_idx[np.argmax(scores[elite_idx])]
            elite_best_score = float(scores[elite_best_idx])
            if elite_best_score > best_score:
                best_score = elite_best_score
                best_params = population[elite_best_idx].copy()

            history.append(
                {
                    "iteration": float(iteration),
                    "mean_score": float(scores.mean()),
                    "best_score": float(scores.max()),
                }
            )

        return TrainingResult(
            best_params=best_params,
            best_score=float(best_score),
            history=history,
        )

    def evaluate_params(self, params: np.ndarray, seed_offset: int = 0) -> float:
        policy = LinearTanhPolicy(self.spec, params.astype(np.float32))
        episode_rewards = []

        for episode in range(self.config.eval_episodes):
            env = self.env_factory()
            obs, _ = env.reset(seed=self.config.seed + seed_offset + episode)
            total_reward = 0.0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = policy.act(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward

            episode_rewards.append(total_reward)

        return float(np.mean(episode_rewards))
