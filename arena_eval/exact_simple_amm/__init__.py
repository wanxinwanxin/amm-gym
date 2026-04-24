"""Exact replica of the Optimization Arena simple AMM challenge."""

from arena_eval.core.types import BatchResult, SimulationResult, TradeInfo
from arena_eval.exact_simple_amm.config import ExactSimpleAMMConfig
from arena_eval.exact_simple_amm.dynamics import EmpiricalImpactRetailTrader, RegimeSwitchingReturnProcess
from arena_eval.exact_simple_amm.strategies import FixedFeeStrategy
from arena_eval.exact_simple_amm.simulator import run_batch, run_seed, score_challenge

__all__ = [
    "BatchResult",
    "EmpiricalImpactRetailTrader",
    "ExactSimpleAMMConfig",
    "FixedFeeStrategy",
    "RegimeSwitchingReturnProcess",
    "SimulationResult",
    "TradeInfo",
    "run_batch",
    "run_seed",
    "score_challenge",
]
