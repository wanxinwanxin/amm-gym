"""Cheap random-search baseline for parameterized policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from training.algorithms.cem import CEMTrainer


@dataclass
class RandomSearchConfig:
    samples: int = 64
    scale: float = 0.75
    seed: int = 0


class RandomSearchTrainer:
    """Evaluate random policy parameters under the same policy-space interface."""

    def __init__(self, cem_trainer: CEMTrainer, config: RandomSearchConfig) -> None:
        self.cem_trainer = cem_trainer
        self.config = config

    def train(self) -> tuple[np.ndarray, float]:
        rng = np.random.default_rng(self.config.seed)
        params = rng.normal(
            loc=0.0,
            scale=self.config.scale,
            size=(self.config.samples, self.cem_trainer.policy_space.param_dim),
        )
        scores = np.asarray(
            [self.cem_trainer.evaluate_params(candidate) for candidate in params],
            dtype=np.float64,
        )
        best_idx = int(np.argmax(scores))
        return params[best_idx], float(scores[best_idx])
