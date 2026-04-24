"""Linear policy families for baseline search."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LinearPolicySpace:
    obs_dim: int
    action_dim: int
    action_low: np.ndarray
    action_high: np.ndarray

    @classmethod
    def from_env(cls, env) -> "LinearPolicySpace":
        obs, _ = env.reset(seed=0)
        return cls(
            obs_dim=int(obs.shape[0]),
            action_dim=int(np.asarray(env.action_space.low).shape[0]),
            action_low=np.asarray(env.action_space.low, dtype=np.float32),
            action_high=np.asarray(env.action_space.high, dtype=np.float32),
        )

    @property
    def param_dim(self) -> int:
        return self.action_dim * self.obs_dim + self.action_dim

    def build_policy(self, params: np.ndarray) -> "LinearTanhPolicy":
        return LinearTanhPolicy(self, params)


class LinearTanhPolicy:
    """Small linear policy used by the CEM baseline."""

    def __init__(self, spec: LinearPolicySpace, params: np.ndarray) -> None:
        if params.shape != (spec.param_dim,):
            raise ValueError(f"expected params shape {(spec.param_dim,)}, got {params.shape}")
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

    def reset(self) -> None:
        return None


def engineer_features(obs: np.ndarray) -> np.ndarray:
    """Build a compact nonlinear feature vector from the public observation."""
    obs = np.asarray(obs, dtype=np.float32)
    ws = obs.shape[0] - 15
    returns = obs[:ws]
    reserves_and_flow = obs[ws : ws + 7]
    last_action = obs[ws + 7 : ws + 13]
    rolling_vol = float(obs[ws + 13])
    progress = float(obs[ws + 14])

    imbalance = float(obs[ws + 2])
    lagged_edge = float(obs[ws + 3])
    exec_volume = float(obs[ws + 4])
    exec_count = float(obs[ws + 5])
    net_flow = float(obs[ws + 6])

    bid_action = last_action[:3]
    ask_action = last_action[3:]
    if returns.size:
        return_mean = float(np.mean(returns))
        return_std = float(np.std(returns))
        return_abs_mean = float(np.mean(np.abs(returns)))
        return_last = float(returns[-1])
        return_pos_sum = float(np.sum(np.clip(returns, 0.0, None)))
        return_neg_sum = float(np.sum(np.clip(returns, None, 0.0)))
    else:
        return_mean = return_std = return_abs_mean = return_last = 0.0
        return_pos_sum = return_neg_sum = 0.0

    engineered = np.array(
        [
            return_mean,
            return_std,
            return_abs_mean,
            return_last,
            return_pos_sum,
            return_neg_sum,
            imbalance,
            abs(imbalance),
            imbalance * imbalance,
            lagged_edge,
            exec_volume,
            exec_count,
            net_flow,
            abs(net_flow),
            imbalance * net_flow,
            rolling_vol,
            rolling_vol * rolling_vol,
            rolling_vol * progress,
            progress,
            progress * progress,
            float(np.mean(bid_action)),
            float(np.mean(ask_action)),
            float(np.mean(np.abs(last_action))),
            float(np.mean(np.abs(bid_action - ask_action))),
            float(np.linalg.norm(last_action)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([obs, reserves_and_flow, last_action, engineered]).astype(np.float32)


@dataclass
class FeatureLinearPolicySpace:
    feature_dim: int
    action_dim: int
    action_low: np.ndarray
    action_high: np.ndarray

    @classmethod
    def from_env(cls, env) -> "FeatureLinearPolicySpace":
        obs, _ = env.reset(seed=0)
        features = engineer_features(obs)
        return cls(
            feature_dim=int(features.shape[0]),
            action_dim=int(np.asarray(env.action_space.low).shape[0]),
            action_low=np.asarray(env.action_space.low, dtype=np.float32),
            action_high=np.asarray(env.action_space.high, dtype=np.float32),
        )

    @property
    def param_dim(self) -> int:
        return self.action_dim * self.feature_dim + self.action_dim

    def build_policy(self, params: np.ndarray) -> "FeatureLinearTanhPolicy":
        return FeatureLinearTanhPolicy(self, params)


class FeatureLinearTanhPolicy:
    """Linear policy on top of engineered trainer-side features."""

    def __init__(self, spec: FeatureLinearPolicySpace, params: np.ndarray) -> None:
        if params.shape != (spec.param_dim,):
            raise ValueError(f"expected params shape {(spec.param_dim,)}, got {params.shape}")
        self.spec = spec
        split = spec.action_dim * spec.feature_dim
        self.weights = params[:split].reshape(spec.action_dim, spec.feature_dim)
        self.bias = params[split:]

    def act(self, obs: np.ndarray) -> np.ndarray:
        features = engineer_features(obs)
        logits = self.weights @ features + self.bias
        squashed = np.tanh(logits)
        low = self.spec.action_low
        high = self.spec.action_high
        action = low + 0.5 * (squashed + 1.0) * (high - low)
        return np.clip(action, low, high).astype(np.float32)

    def reset(self) -> None:
        return None


@dataclass
class SmoothedFeatureLinearPolicySpace:
    base_space: FeatureLinearPolicySpace
    smoothing_alpha: float = 0.7

    @classmethod
    def from_env(cls, env, smoothing_alpha: float = 0.7) -> "SmoothedFeatureLinearPolicySpace":
        return cls(
            base_space=FeatureLinearPolicySpace.from_env(env),
            smoothing_alpha=float(smoothing_alpha),
        )

    @property
    def feature_dim(self) -> int:
        return self.base_space.feature_dim

    @property
    def action_dim(self) -> int:
        return self.base_space.action_dim

    @property
    def action_low(self) -> np.ndarray:
        return self.base_space.action_low

    @property
    def action_high(self) -> np.ndarray:
        return self.base_space.action_high

    @property
    def param_dim(self) -> int:
        return self.base_space.param_dim

    def build_policy(self, params: np.ndarray) -> "SmoothedFeatureLinearTanhPolicy":
        return SmoothedFeatureLinearTanhPolicy(
            base_policy=self.base_space.build_policy(params),
            action_low=self.action_low,
            action_high=self.action_high,
            smoothing_alpha=self.smoothing_alpha,
        )


@dataclass
class SmoothedFeatureLinearTanhPolicy:
    base_policy: FeatureLinearTanhPolicy
    action_low: np.ndarray
    action_high: np.ndarray
    smoothing_alpha: float = 0.7
    _prev_action: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._prev_action = np.zeros_like(self.action_low, dtype=np.float32)

    def act(self, obs: np.ndarray) -> np.ndarray:
        target_action = self.base_policy.act(obs)
        action = self.smoothing_alpha * self._prev_action + (1.0 - self.smoothing_alpha) * target_action
        clipped = np.clip(action, self.action_low, self.action_high).astype(np.float32)
        self._prev_action = clipped
        return clipped
