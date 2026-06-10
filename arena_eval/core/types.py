"""Shared evaluation dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field


SECONDS_PER_YEAR = 365.0 * 24.0 * 60.0 * 60.0


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
class IncomingSwap:
    """Information exposed to fee strategies *before* a swap executes — the
    simulator analog of a Uniswap v4 ``beforeSwap`` call. Lets a strategy price
    the swap from the pool's current state and the incoming swap's direction/size
    (which ``after_swap`` cannot see, since it fires post-execution on the
    completed trade).

    is_buy:   the incoming swap buys token x (retail buy / arb buying x).
    size:     swap notional in y (USD). ``None`` when endogenous (e.g. arb size,
              which the arbitrageur derives from the fee this call returns).
    reserve_x/reserve_y: the pool's reserves right now (spot = reserve_y/reserve_x).
    block:    the step index, treated as the block number for per-block logic.
    """

    is_buy: bool
    size: float | None
    reserve_x: float
    reserve_y: float
    block: int


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
    retail_edge_submission: float = 0.0
    retail_edge_normalizer: float = 0.0
    arb_loss_submission: float = 0.0
    arb_loss_normalizer: float = 0.0
    initial_value: float = 0.0
    initial_value_normalizer: float = 0.0
    episode_seconds: float = 0.0

    @property
    def edge_advantage(self) -> float:
        return self.edge_submission - self.edge_normalizer

    @property
    def pnl_advantage(self) -> float:
        return self.pnl_submission - self.pnl_normalizer

    @property
    def retail_edge_advantage(self) -> float:
        return self.retail_edge_submission - self.retail_edge_normalizer

    @property
    def net_edge_identity_error(self) -> float:
        return self.edge_submission - (self.retail_edge_submission - self.arb_loss_submission)

    def _return_on_initial_value(self, value: float) -> float:
        return float(value) / self.initial_value if self.initial_value > 0.0 else 0.0

    def _annualized_return(self, value: float) -> float:
        if self.initial_value <= 0.0 or self.episode_seconds <= 0.0:
            return 0.0
        return float(value) / self.initial_value * (SECONDS_PER_YEAR / self.episode_seconds)

    def _markout_bps(self, edge_value: float, volume_y: float) -> float:
        return 1.0e4 * float(edge_value) / float(volume_y) if volume_y > 0.0 else 0.0

    @property
    def edge_return_submission(self) -> float:
        return self._return_on_initial_value(self.edge_submission)

    @property
    def pnl_return_submission(self) -> float:
        return self._return_on_initial_value(self.pnl_submission)

    @property
    def retail_edge_return_submission(self) -> float:
        return self._return_on_initial_value(self.retail_edge_submission)

    @property
    def arb_loss_return_submission(self) -> float:
        return self._return_on_initial_value(self.arb_loss_submission)

    @property
    def annualized_edge_return_submission(self) -> float:
        return self._annualized_return(self.edge_submission)

    @property
    def annualized_pnl_return_submission(self) -> float:
        return self._annualized_return(self.pnl_submission)

    @property
    def annualized_retail_edge_return_submission(self) -> float:
        return self._annualized_return(self.retail_edge_submission)

    @property
    def annualized_arb_loss_return_submission(self) -> float:
        return self._annualized_return(self.arb_loss_submission)

    @property
    def retail_markout_bps_submission(self) -> float:
        return self._markout_bps(self.retail_edge_submission, self.retail_volume_submission_y)

    @property
    def arb_markout_bps_submission(self) -> float:
        return self._markout_bps(self.arb_loss_submission, self.arb_volume_submission_y)


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
    retail_edge_mean_submission: float = 0.0
    retail_edge_mean_normalizer: float = 0.0
    arb_loss_mean_submission: float = 0.0
    arb_loss_mean_normalizer: float = 0.0
    retail_volume_mean_submission_y: float = 0.0
    retail_volume_mean_normalizer_y: float = 0.0
    arb_volume_mean_submission_y: float = 0.0
    arb_volume_mean_normalizer_y: float = 0.0
    initial_value_mean: float = 0.0
    initial_value_mean_normalizer: float = 0.0
    episode_seconds_mean: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def retail_edge_advantage_mean(self) -> float:
        return self.retail_edge_mean_submission - self.retail_edge_mean_normalizer

    def _return_on_initial_value(self, value: float) -> float:
        return float(value) / self.initial_value_mean if self.initial_value_mean > 0.0 else 0.0

    def _annualized_return(self, value: float) -> float:
        if self.initial_value_mean <= 0.0 or self.episode_seconds_mean <= 0.0:
            return 0.0
        return float(value) / self.initial_value_mean * (SECONDS_PER_YEAR / self.episode_seconds_mean)

    def _markout_bps(self, edge_value: float, volume_y: float) -> float:
        return 1.0e4 * float(edge_value) / float(volume_y) if volume_y > 0.0 else 0.0

    @property
    def annualized_edge_return_mean_submission(self) -> float:
        return self._annualized_return(self.edge_mean_submission)

    @property
    def annualized_pnl_return_mean_submission(self) -> float:
        return self._annualized_return(self.pnl_mean_submission)

    @property
    def annualized_retail_edge_return_mean_submission(self) -> float:
        return self._annualized_return(self.retail_edge_mean_submission)

    @property
    def annualized_arb_loss_return_mean_submission(self) -> float:
        return self._annualized_return(self.arb_loss_mean_submission)

    @property
    def retail_markout_bps_mean_submission(self) -> float:
        return self._markout_bps(self.retail_edge_mean_submission, self.retail_volume_mean_submission_y)

    @property
    def arb_markout_bps_mean_submission(self) -> float:
        return self._markout_bps(self.arb_loss_mean_submission, self.arb_volume_mean_submission_y)
