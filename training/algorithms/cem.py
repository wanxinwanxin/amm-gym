"""Cross-entropy-method baseline trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from training.eval.metrics import evaluate_episode, score_episode_metrics
from training.policies import LinearPolicySpace, SearchSpace


EnvFactory = Callable[[], object]
PolicySpaceFactory = Callable[[object], SearchSpace]


@dataclass
class CEMConfig:
    population_size: int = 24
    elite_frac: float = 0.25
    iterations: int = 5
    eval_episodes: int = 2
    init_std: float = 0.5
    seed: int = 0
    objective: str = "reward"
    evaluation_seeds: tuple[int, ...] | None = None
    elite_reevaluation_seeds: tuple[int, ...] | None = None


@dataclass
class TrainingResult:
    best_params: np.ndarray
    best_score: float
    history: list[dict[str, float]]


class CEMTrainer:
    """Train a parameterized policy against the public env interface."""

    def __init__(
        self,
        env_factory: EnvFactory,
        config: CEMConfig,
        policy_space_factory: PolicySpaceFactory | None = None,
    ) -> None:
        self.env_factory = env_factory
        self.config = config

        env = env_factory()
        if policy_space_factory is None:
            self.policy_space = LinearPolicySpace.from_env(env)
        else:
            self.policy_space = policy_space_factory(env)
        self.spec = self.policy_space

    def train(self) -> TrainingResult:
        rng = np.random.default_rng(self.config.seed)
        mean = np.zeros(self.policy_space.param_dim, dtype=np.float64)
        std = np.full(self.policy_space.param_dim, self.config.init_std, dtype=np.float64)

        best_params = mean.copy()
        best_score = -np.inf
        history: list[dict[str, float]] = []

        elite_count = max(1, int(self.config.population_size * self.config.elite_frac))

        for iteration in range(self.config.iterations):
            population = rng.normal(
                loc=mean,
                scale=std,
                size=(self.config.population_size, self.policy_space.param_dim),
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

            train_best_idx = elite_idx[np.argmax(scores[elite_idx])]
            train_best_score = float(scores[train_best_idx])
            selected_idx = train_best_idx
            validation_score = np.nan
            if self.config.elite_reevaluation_seeds:
                validation_scores = np.array(
                    [
                        self.evaluate_params(
                            population[idx],
                            seeds=self.config.elite_reevaluation_seeds,
                        )
                        for idx in elite_idx
                    ],
                    dtype=np.float64,
                )
                best_validation_pos = int(np.argmax(validation_scores))
                selected_idx = elite_idx[best_validation_pos]
                validation_score = float(validation_scores[best_validation_pos])

            selected_score = (
                validation_score if np.isfinite(validation_score) else train_best_score
            )
            if selected_score > best_score:
                best_score = selected_score
                best_params = population[selected_idx].copy()

            entry = {
                "iteration": float(iteration),
                "mean_score": float(scores.mean()),
                "best_score": float(scores.max()),
                "selected_score": float(selected_score),
            }
            if np.isfinite(validation_score):
                entry["validation_best_score"] = float(validation_score)
            history.append(entry)

        return TrainingResult(
            best_params=best_params,
            best_score=float(best_score),
            history=history,
        )

    def evaluate_params(
        self,
        params: np.ndarray,
        *,
        seeds: Sequence[int] | None = None,
        seed_offset: int = 0,
    ) -> float:
        policy = self.policy_space.build_policy(params.astype(np.float32))
        episode_scores = []
        rollout_seeds = list(seeds) if seeds is not None else list(self._default_eval_seeds(seed_offset))

        for seed in rollout_seeds:
            metrics = evaluate_episode(self.env_factory, policy.act, seed=int(seed))
            episode_scores.append(score_episode_metrics(metrics, self.config.objective))

        return float(np.mean(episode_scores))

    def _default_eval_seeds(self, seed_offset: int) -> tuple[int, ...]:
        if self.config.evaluation_seeds is not None:
            return self.config.evaluation_seeds
        return tuple(self.config.seed + seed_offset + episode for episode in range(self.config.eval_episodes))
