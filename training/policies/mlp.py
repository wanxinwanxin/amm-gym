"""Small MLP policy family searched with black-box optimizers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MLPPolicySpace:
    obs_dim: int
    action_dim: int
    action_low: np.ndarray
    action_high: np.ndarray
    hidden_sizes: tuple[int, ...] = (32, 32)

    @classmethod
    def from_env(cls, env, hidden_sizes: tuple[int, ...] = (32, 32)) -> "MLPPolicySpace":
        obs, _ = env.reset(seed=0)
        return cls(
            obs_dim=int(obs.shape[0]),
            action_dim=int(np.asarray(env.action_space.low).shape[0]),
            action_low=np.asarray(env.action_space.low, dtype=np.float32),
            action_high=np.asarray(env.action_space.high, dtype=np.float32),
            hidden_sizes=hidden_sizes,
        )

    @property
    def layer_sizes(self) -> tuple[int, ...]:
        return (self.obs_dim, *self.hidden_sizes, self.action_dim)

    @property
    def param_dim(self) -> int:
        total = 0
        for in_dim, out_dim in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            total += in_dim * out_dim + out_dim
        return total

    def build_policy(self, params: np.ndarray) -> "MLPTanhPolicy":
        return MLPTanhPolicy(self, params)


class MLPTanhPolicy:
    """Tiny fully connected tanh policy for richer action mappings."""

    def __init__(self, spec: MLPPolicySpace, params: np.ndarray) -> None:
        if params.shape != (spec.param_dim,):
            raise ValueError(f"expected params shape {(spec.param_dim,)}, got {params.shape}")
        self.spec = spec
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        cursor = 0
        for in_dim, out_dim in zip(spec.layer_sizes[:-1], spec.layer_sizes[1:]):
            weight_count = in_dim * out_dim
            self.weights.append(params[cursor : cursor + weight_count].reshape(out_dim, in_dim))
            cursor += weight_count
            self.biases.append(params[cursor : cursor + out_dim])
            cursor += out_dim

    def act(self, obs: np.ndarray) -> np.ndarray:
        hidden = obs.astype(np.float32)
        for idx, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            hidden = weight @ hidden + bias
            hidden = np.tanh(hidden)
            if idx == len(self.weights) - 1:
                break
        low = self.spec.action_low
        high = self.spec.action_high
        action = low + 0.5 * (hidden + 1.0) * (high - low)
        return np.clip(action, low, high).astype(np.float32)
