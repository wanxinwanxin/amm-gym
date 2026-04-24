"""Backward-compatible imports for the baseline linear policy."""

from training.policies.linear import LinearPolicySpace as LinearPolicySpec
from training.policies.linear import LinearTanhPolicy

__all__ = ["LinearPolicySpec", "LinearTanhPolicy"]
