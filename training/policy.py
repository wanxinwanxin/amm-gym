"""Simple policy objects for baseline training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LinearPolicySpec:
    obs_dim: int
    action_dim: int
    action_low: np.ndarray
    action_high: np.ndarray

    @property
    def param_dim(self) -> int:
        return self.action_dim * self.obs_dim + self.action_dim


class LinearTanhPolicy:
    """Small linear policy used by the CEM baseline.

    Actions are produced by a linear map followed by tanh and affine scaling
    into the environment action range.
    """

    def __init__(self, spec: LinearPolicySpec, params: np.ndarray) -> None:
        if params.shape != (spec.param_dim,):
            raise ValueError(
                f"expected params shape {(spec.param_dim,)}, got {params.shape}"
            )
        self.spec = spec
        split = spec.action_dim * spec.obs_dim
        self.weights = params[:split].reshape(spec.action_dim, spec.obs_dim)
        self.bias = params[split:]

    def act(self, obs: np.ndarray) -> np.ndarray:
        logits = self.weights @ obs + self.bias
        squashed = np.tanh(logits)
        low = self.spec.action_low
        high = self.spec.action_high
        action = low + 0.5 * (squashed + 1.0) * (high - low)
        return np.clip(action, low, high).astype(np.float32)
