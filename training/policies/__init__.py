"""Policy families and heuristic baselines for AMM training."""

from typing import Protocol

import numpy as np

from training.policies.heuristic import research_benchmark_policies
from training.policies.linear import (
    FeatureLinearPolicySpace,
    FeatureLinearTanhPolicy,
    LinearPolicySpace,
    LinearTanhPolicy,
    SmoothedFeatureLinearPolicySpace,
    SmoothedFeatureLinearTanhPolicy,
    engineer_features,
)
from training.policies.mlp import MLPPolicySpace, MLPTanhPolicy


class SearchSpace(Protocol):
    param_dim: int

    def build_policy(self, params: np.ndarray): ...


__all__ = [
    "LinearPolicySpace",
    "LinearTanhPolicy",
    "FeatureLinearPolicySpace",
    "FeatureLinearTanhPolicy",
    "SmoothedFeatureLinearPolicySpace",
    "SmoothedFeatureLinearTanhPolicy",
    "MLPPolicySpace",
    "MLPTanhPolicy",
    "engineer_features",
    "SearchSpace",
    "research_benchmark_policies",
]
