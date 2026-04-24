"""Step-by-step simulation engine."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from amm_gym.sim.actors import Arbitrageur, OrderRouter, RetailTrader
from amm_gym.sim.ladder import DepthLadderAMM
from amm_gym.sim.price import GBMPriceProcess
from amm_gym.sim.quote_surface import ParametricQuoteSurfaceAMM
from amm_gym.sim.venues import VenueSpec, build_venue


@dataclass
class SimConfig:
    n_steps: int = 10_000
    initial_price: float = 100.0
    initial_x: float = 100.0
    initial_y: float = 10_000.0
    gbm_mu: float = 0.0
    gbm_sigma: float = 0.001
    gbm_dt: float = 1.0
    retail_arrival_rate: float = 5.0
    retail_mean_size: float = 2.0
    retail_size_sigma: float = 0.7
    retail_buy_prob: float = 0.5
    submission_band_bps: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
    submission_base_notional_y: float = 1_000.0
    volatility_schedule: tuple[tuple[int, float], ...] | None = None
    submission_venue: VenueSpec | None = None
    benchmark_venue: VenueSpec | None = None
    seed: int | None = None


@dataclass
class StepResult:
    timestamp: int
    fair_price: float
    spot_prices: dict[str, float]
    pnls: dict[str, float]
    edges: dict[str, float]
    n_retail_orders: int = 0
    retail_volume_y: dict[str, float] = field(default_factory=dict)
    arb_volume_y: dict[str, float] = field(default_factory=dict)
    execution_count: dict[str, int] = field(default_factory=dict)
    execution_volume_y: dict[str, float] = field(default_factory=dict)
    net_flow_y: dict[str, float] = field(default_factory=dict)
    ladder_depth_y: dict[str, float] = field(default_factory=dict)
    active_sigma: float = 0.0


class SimulationEngine:
    """Step-by-step AMM simulation with configurable submission and benchmark venues."""

    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self._reset_state(seed=config.seed)

    def _reset_state(self, seed: int | None) -> None:
        cfg = self.config
        self._volatility_schedule = self._normalize_volatility_schedule(
            cfg.volatility_schedule,
            cfg.gbm_sigma,
        )

        self.price_process = GBMPriceProcess(
            cfg.initial_price,
            cfg.gbm_mu,
            self._volatility_schedule[0][1],
            cfg.gbm_dt,
            seed=seed,
        )
        self.retail_trader = RetailTrader(
            cfg.retail_arrival_rate,
            cfg.retail_mean_size,
            cfg.retail_size_sigma,
            cfg.retail_buy_prob,
            seed=None if seed is None else seed + 1,
        )
        self.arbitrageur = Arbitrageur()
        self.router = OrderRouter()

        self.submission_spec = self._resolve_submission_spec(cfg)
        self.benchmark_spec = self._resolve_benchmark_spec(cfg)
        controllable_count = int(self.submission_spec.controllable) + int(
            self.benchmark_spec.controllable
        )
        if controllable_count > 1:
            raise ValueError("at most one venue may be controllable")

        self.submission_amm = build_venue(self.submission_spec)
        self.benchmark_amm = build_venue(self.benchmark_spec)
        # Compatibility aliases retained for existing demo/training code.
        self.amm_agent = self.submission_amm
        self.amm_norm = self.benchmark_amm

        self.initial_fair_price = cfg.initial_price
        self.initial_reserves = {
            "submission": (self.submission_spec.reserve_x, self.submission_spec.reserve_y),
            "benchmark": (self.benchmark_spec.reserve_x, self.benchmark_spec.reserve_y),
            "normalizer": (self.benchmark_spec.reserve_x, self.benchmark_spec.reserve_y),
            self.submission_amm.name: (self.submission_spec.reserve_x, self.submission_spec.reserve_y),
            self.benchmark_amm.name: (self.benchmark_spec.reserve_x, self.benchmark_spec.reserve_y),
        }
        self.edges: dict[str, float] = {"submission": 0.0, "benchmark": 0.0, "normalizer": 0.0}
        self.current_step = 0
        self.current_fair_price = cfg.initial_price
        self._controllable_action_dim = max(
            self.submission_spec.action_dim if self.submission_spec.controllable else 0,
            self.benchmark_spec.action_dim if self.benchmark_spec.controllable else 0,
        )
        self._pending_agent_action = np.zeros(self._controllable_action_dim, dtype=np.float32)

    def reset(self, seed: int | None = None) -> None:
        self._reset_state(seed=seed)

    def set_agent_action(self, action: np.ndarray) -> None:
        self._pending_agent_action = np.asarray(action, dtype=np.float32)

    def step(self) -> StepResult:
        t = self.current_step

        self._configure_controllable_venues()

        active_sigma = self._sigma_for_step(t)
        fair_price = self.price_process.step(sigma=active_sigma)

        step_arb_volume = {"submission": 0.0, "benchmark": 0.0, "normalizer": 0.0}
        step_retail_volume = {"submission": 0.0, "benchmark": 0.0, "normalizer": 0.0}
        step_execution_count = {"submission": 0, "benchmark": 0, "normalizer": 0}
        step_execution_volume = {"submission": 0.0, "benchmark": 0.0, "normalizer": 0.0}
        step_net_flow_y = {"submission": 0.0, "benchmark": 0.0, "normalizer": 0.0}

        for venue_key, amm in [("submission", self.submission_amm), ("benchmark", self.benchmark_amm)]:
            arb_result = self.arbitrageur.execute_arb(amm, fair_price, t)
            if arb_result is not None:
                step_arb_volume[venue_key] += arb_result.amount_y
                self.edges[venue_key] -= arb_result.profit
                self._record_venue_trade(
                    venue_key=venue_key,
                    amount_y=arb_result.amount_y,
                    signed_flow_y=-arb_result.amount_y if arb_result.side == "buy" else arb_result.amount_y,
                    is_arbitrage=True,
                )

        orders = self.retail_trader.generate_orders()
        routed_trades = self.router.route_orders(
            orders, self.submission_amm, self.benchmark_amm, fair_price, t
        )
        for trade in routed_trades:
            venue_key = "submission" if trade.amm_name == self.submission_amm.name else "benchmark"
            step_retail_volume[venue_key] += trade.amount_y
            step_execution_count[venue_key] += 1
            step_execution_volume[venue_key] += trade.amount_y
            if trade.amm_buys_x:
                step_net_flow_y[venue_key] -= trade.amount_y
                trade_edge = trade.amount_x * fair_price - trade.amount_y
            else:
                step_net_flow_y[venue_key] += trade.amount_y
                trade_edge = trade.amount_y - trade.amount_x * fair_price
            self.edges[venue_key] += trade_edge
            self._record_venue_trade(
                venue_key=venue_key,
                amount_y=trade.amount_y,
                signed_flow_y=-trade.amount_y if trade.amm_buys_x else trade.amount_y,
                is_arbitrage=False,
            )

        for mapping in [
            step_arb_volume,
            step_retail_volume,
            step_execution_count,
            step_execution_volume,
            step_net_flow_y,
            self.edges,
        ]:
            mapping["normalizer"] = mapping["benchmark"]

        self.current_fair_price = fair_price
        result = StepResult(
            timestamp=t,
            fair_price=fair_price,
            spot_prices={
                "submission": self.submission_amm.spot_price,
                "benchmark": self.benchmark_amm.spot_price,
                "normalizer": self.benchmark_amm.spot_price,
            },
            pnls=self._compute_pnls(fair_price),
            edges=dict(self.edges),
            n_retail_orders=len(orders),
            retail_volume_y=step_retail_volume,
            arb_volume_y=step_arb_volume,
            execution_count=step_execution_count,
            execution_volume_y=step_execution_volume,
            net_flow_y=step_net_flow_y,
            ladder_depth_y=self._venue_summary(self.submission_amm),
            active_sigma=active_sigma,
        )

        self.current_step += 1
        return result

    def _compute_pnls(self, fair_price: float) -> dict[str, float]:
        pnls = {}
        for venue_key, amm in [("submission", self.submission_amm), ("benchmark", self.benchmark_amm)]:
            init_x, init_y = self.initial_reserves[amm.name]
            init_value = init_x * self.initial_fair_price + init_y

            rx, ry = amm.reserves()
            fx = getattr(amm, "accumulated_fees_x", 0.0)
            fy = getattr(amm, "accumulated_fees_y", 0.0)
            curr_value = (rx * fair_price + ry) + (fx * fair_price + fy)
            pnls[venue_key] = curr_value - init_value
        pnls["normalizer"] = pnls["benchmark"]
        return pnls

    @property
    def done(self) -> bool:
        return self.current_step >= self.config.n_steps

    def _normalize_volatility_schedule(
        self,
        schedule: tuple[tuple[int, float], ...] | None,
        default_sigma: float,
    ) -> tuple[tuple[int, float], ...]:
        if not schedule:
            return ((0, float(default_sigma)),)

        normalized: list[tuple[int, float]] = []
        for start_step, sigma in schedule:
            step = int(start_step)
            vol = float(sigma)
            if step < 0:
                raise ValueError("volatility schedule step must be non-negative")
            if vol < 0.0:
                raise ValueError("volatility schedule sigma must be non-negative")
            normalized.append((step, vol))

        normalized.sort(key=lambda item: item[0])
        if normalized[0][0] != 0:
            normalized.insert(0, (0, float(default_sigma)))

        deduped: list[tuple[int, float]] = []
        for step, sigma in normalized:
            if deduped and deduped[-1][0] == step:
                deduped[-1] = (step, sigma)
            else:
                deduped.append((step, sigma))
        return tuple(deduped)

    def _sigma_for_step(self, step: int) -> float:
        sigma = self._volatility_schedule[0][1]
        for start_step, scheduled_sigma in self._volatility_schedule:
            if step < start_step:
                break
            sigma = scheduled_sigma
        return float(sigma)

    def _resolve_submission_spec(self, cfg: SimConfig) -> VenueSpec:
        if cfg.submission_venue is not None:
            return cfg.submission_venue
        return VenueSpec(
            kind="depth_ladder",
            name="submission",
            reserve_x=cfg.initial_x,
            reserve_y=cfg.initial_y,
            band_bps=cfg.submission_band_bps,
            base_notional_y=cfg.submission_base_notional_y,
            controllable=True,
        )

    def _resolve_benchmark_spec(self, cfg: SimConfig) -> VenueSpec:
        if cfg.benchmark_venue is not None:
            return cfg.benchmark_venue
        return VenueSpec(
            kind="cpmm",
            name="normalizer",
            reserve_x=cfg.initial_x,
            reserve_y=cfg.initial_y,
            bid_fee=0.003,
            ask_fee=0.003,
            controllable=False,
        )

    def _configure_controllable_venues(self) -> None:
        for spec, amm in [
            (self.submission_spec, self.submission_amm),
            (self.benchmark_spec, self.benchmark_amm),
        ]:
            if not hasattr(amm, "configure"):
                continue
            action = self._pending_agent_action if spec.controllable else spec.action_vector()
            if isinstance(amm, DepthLadderAMM):
                amm.configure(
                    reference_price=self.current_fair_price,
                    bid_raw=np.asarray(action[:3], dtype=np.float32),
                    ask_raw=np.asarray(action[3:], dtype=np.float32),
                )
            elif isinstance(amm, ParametricQuoteSurfaceAMM):
                amm.configure(
                    reference_price=self.current_fair_price,
                    action=np.asarray(action, dtype=np.float32),
                )

    def _record_venue_trade(
        self,
        *,
        venue_key: str,
        amount_y: float,
        signed_flow_y: float,
        is_arbitrage: bool,
    ) -> None:
        amm = self.submission_amm if venue_key == "submission" else self.benchmark_amm
        record_trade = getattr(amm, "record_trade", None)
        if callable(record_trade):
            record_trade(
                amount_y=float(amount_y),
                signed_flow_y=float(signed_flow_y),
                is_arbitrage=bool(is_arbitrage),
            )

    def _venue_summary(self, amm) -> dict[str, float]:
        if hasattr(amm, "current_ladder_summary"):
            return amm.current_ladder_summary()
        if hasattr(amm, "current_quote_summary"):
            return amm.current_quote_summary()
        return {
            "ask_near_depth_y": 0.0,
            "ask_far_depth_y": 0.0,
            "bid_near_depth_y": 0.0,
            "bid_far_depth_y": 0.0,
        }
