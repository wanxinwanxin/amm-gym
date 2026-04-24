"""Differentiable simple-AMM rewrite scaffold."""

from arena_eval.diff_simple_amm.challenge_dynamics import build_challenge_tape
from arena_eval.diff_simple_amm.realistic_dynamics import build_realistic_tape
from arena_eval.diff_simple_amm.objectives import (
    challenge_env_vector,
    expected_submission_edge,
    expected_submission_edge_batch,
    realistic_env_vector,
    smooth_submission_compact_batch_result,
    smooth_submission_compact_metrics,
    smooth_submission_compact_result,
    submission_compact_param_vector,
)
from arena_eval.diff_simple_amm.policies import DiffSimpleAMMPolicy
from arena_eval.diff_simple_amm.policies import FixedFeeDiffPolicy
from arena_eval.diff_simple_amm.policies import SubmissionCompactDiffPolicy
from arena_eval.diff_simple_amm.simulator import (
    DiffSimpleAMMSimulatorConfig,
    run_challenge_rollout,
    run_realistic_rollout,
    run_rollout,
)
from arena_eval.diff_simple_amm.types import (
    AMMState,
    ChallengeTape,
    DiffBatchResult,
    DiffMode,
    DiffSimulationResult,
    EventMask,
    PolicyOutput,
    PolicyState,
    RealisticTape,
    RetailOrder,
    SimulatorState,
    SmoothRelaxationConfig,
    TradeEvent,
)

__all__ = [
    "AMMState",
    "ChallengeTape",
    "DiffBatchResult",
    "DiffMode",
    "DiffSimpleAMMPolicy",
    "DiffSimpleAMMSimulatorConfig",
    "DiffSimulationResult",
    "EventMask",
    "FixedFeeDiffPolicy",
    "PolicyOutput",
    "PolicyState",
    "RealisticTape",
    "RetailOrder",
    "SimulatorState",
    "SmoothRelaxationConfig",
    "SubmissionCompactDiffPolicy",
    "TradeEvent",
    "build_challenge_tape",
    "build_realistic_tape",
    "challenge_env_vector",
    "expected_submission_edge",
    "expected_submission_edge_batch",
    "run_challenge_rollout",
    "realistic_env_vector",
    "run_realistic_rollout",
    "run_rollout",
    "smooth_submission_compact_batch_result",
    "smooth_submission_compact_metrics",
    "smooth_submission_compact_result",
    "submission_compact_param_vector",
]
