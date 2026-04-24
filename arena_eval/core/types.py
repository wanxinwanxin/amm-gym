"""Shared evaluation dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TradeInfo:
    """Information exposed to fee strategies after a trade."""

    is_buy: bool
    amount_x: float
    amount_y: float
    timestamp: int
    reserve_x: float
    reserve_y: float


@dataclass(frozen=True)
class SimulationResult:
    """Aggregate result for one challenge-faithful simulation."""

    seed: int
    edge_submission: float
    edge_normalizer: float
    pnl_submission: float
    pnl_normalizer: float
    score: float
    retail_volume_submission_y: float
    retail_volume_normalizer_y: float
    arb_volume_submission_y: float
    arb_volume_normalizer_y: float
    average_bid_fee_submission: float
    average_ask_fee_submission: float
    average_bid_fee_normalizer: float
    average_ask_fee_normalizer: float

    @property
    def edge_advantage(self) -> float:
        return self.edge_submission - self.edge_normalizer

    @property
    def pnl_advantage(self) -> float:
        return self.pnl_submission - self.pnl_normalizer


@dataclass(frozen=True)
class BatchResult:
    """Aggregate result for multiple simulations."""

    seeds: tuple[int, ...]
    simulations: tuple[SimulationResult, ...]
    score: float
    edge_mean_submission: float
    edge_mean_normalizer: float
    edge_advantage_mean: float
    pnl_mean_submission: float
    pnl_mean_normalizer: float
    pnl_advantage_mean: float
    metadata: dict[str, object] = field(default_factory=dict)
