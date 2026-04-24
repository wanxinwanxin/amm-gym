"""Shared state types for the differentiable simple-AMM stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DiffMode(str, Enum):
    """Execution mode for the differentiable simulator."""

    EXACT_PATH = "exact_path"
    SMOOTH_TRAIN = "smooth_train"


@dataclass(frozen=True)
class EventMask:
    """Fixed-shape event activation mask for one simulation step."""

    active: tuple[float, ...]


@dataclass(frozen=True)
class ChallengeTape:
    """Exogenous challenge-mode randomness represented as explicit tapes."""

    gbm_normals: tuple[float, ...]
    order_counts: tuple[int, ...]
    order_sizes: tuple[tuple[float, ...], ...]
    order_side_uniforms: tuple[tuple[float, ...], ...]
    max_orders_per_step: int
    smooth_arrival_uniforms: tuple[tuple[float, ...], ...] = ()
    smooth_size_normals: tuple[tuple[float, ...], ...] = ()
    smooth_side_uniforms: tuple[tuple[float, ...], ...] = ()


@dataclass(frozen=True)
class RealisticTape:
    """Exogenous realistic-mode randomness represented as explicit tapes."""

    log_returns: tuple[float, ...]
    regimes: tuple[int, ...]
    return_percentiles: tuple[float, ...]
    order_counts: tuple[int, ...]
    impact_logs: tuple[tuple[float, ...], ...]
    max_orders_per_step: int
    smooth_arrival_uniforms: tuple[tuple[float, ...], ...] = ()
    smooth_impact_percentiles: tuple[tuple[float, ...], ...] = ()


@dataclass(frozen=True)
class RetailOrder:
    """Retail order decoded from an explicit challenge tape."""

    side: str
    size: float


@dataclass(frozen=True)
class AMMState:
    """Functional AMM state used by the diff simulator."""

    reserve_x: float
    reserve_y: float
    bid_fee: float
    ask_fee: float
    accumulated_fees_x: float = 0.0
    accumulated_fees_y: float = 0.0


@dataclass(frozen=True)
class PolicyState:
    """Opaque policy state placeholder for functional controller rewrites."""

    values: tuple[float, ...] = ()


@dataclass(frozen=True)
class PolicyOutput:
    """Next fee decision emitted by a functional policy."""

    bid_fee: float
    ask_fee: float
    state: PolicyState = field(default_factory=PolicyState)


@dataclass(frozen=True)
class TradeEvent:
    """Functional trade event representation for diff rollouts."""

    venue: str
    source: str
    is_buy: bool
    amount_x: float
    amount_y: float
    timestamp: int
    reserve_x: float
    reserve_y: float


@dataclass(frozen=True)
class SimulatorState:
    """Full diff-simulator state for a single rollout."""

    step: int
    fair_price: float
    submission: AMMState
    normalizer: AMMState
    submission_policy_state: PolicyState
    normalizer_policy_state: PolicyState
    edge_submission: float = 0.0
    edge_normalizer: float = 0.0
    retail_volume_submission_y: float = 0.0
    retail_volume_normalizer_y: float = 0.0
    arb_volume_submission_y: float = 0.0
    arb_volume_normalizer_y: float = 0.0
    bid_fee_submission_sum: float = 0.0
    ask_fee_submission_sum: float = 0.0
    bid_fee_normalizer_sum: float = 0.0
    ask_fee_normalizer_sum: float = 0.0


@dataclass(frozen=True)
class DiffSimulationResult:
    """Rollout aggregate for the differentiable simulator."""

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
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def edge_advantage(self) -> float:
        return self.edge_submission - self.edge_normalizer

    @property
    def pnl_advantage(self) -> float:
        return self.pnl_submission - self.pnl_normalizer


@dataclass(frozen=True)
class DiffBatchResult:
    """Batched aggregate for differentiable training and validation."""

    score: float
    edge_mean_submission: float
    edge_mean_normalizer: float
    edge_advantage_mean: float
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SmoothRelaxationConfig:
    """Configuration for smooth differentiable training-time relaxations."""

    gate_sharpness: float = 24.0
    clip_sharpness: float = 24.0
    arb_sharpness: float = 32.0
    arrival_sharpness: float = 20.0
    side_sharpness: float = 20.0
    min_trade_amount: float = 0.0001
