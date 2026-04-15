"""Step-by-step simulation engine."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from amm_gym.sim.amm import ConstantProductAMM
from amm_gym.sim.actors import Arbitrageur, OrderRouter, RetailTrader
from amm_gym.sim.ladder import DepthLadderAMM
from amm_gym.sim.price import GBMPriceProcess


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
    """Step-by-step AMM simulation with a ladder submission and CPMM normalizer."""

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

        self.amm_agent = DepthLadderAMM(
            name="submission",
            reserve_x=cfg.initial_x,
            reserve_y=cfg.initial_y,
            band_bps=cfg.submission_band_bps,
            base_notional_y=cfg.submission_base_notional_y,
        )
        self.amm_norm = ConstantProductAMM(
            name="normalizer",
            reserve_x=cfg.initial_x,
            reserve_y=cfg.initial_y,
            bid_fee=0.003,
            ask_fee=0.003,
        )

        self.initial_fair_price = cfg.initial_price
        self.initial_reserves = {
            "submission": (cfg.initial_x, cfg.initial_y),
            "normalizer": (cfg.initial_x, cfg.initial_y),
        }
        self.edges: dict[str, float] = {"submission": 0.0, "normalizer": 0.0}
        self.current_step = 0
        self.current_fair_price = cfg.initial_price
        self._pending_agent_action = np.zeros(6, dtype=np.float32)

    def reset(self, seed: int | None = None) -> None:
        self._reset_state(seed=seed)

    def set_agent_action(self, action: np.ndarray) -> None:
        self._pending_agent_action = np.asarray(action, dtype=np.float32)

    def step(self) -> StepResult:
        t = self.current_step

        self.amm_agent.configure(
            reference_price=self.current_fair_price,
            bid_raw=self._pending_agent_action[:3],
            ask_raw=self._pending_agent_action[3:],
        )

        active_sigma = self._sigma_for_step(t)
        fair_price = self.price_process.step(sigma=active_sigma)

        step_arb_volume = {"submission": 0.0, "normalizer": 0.0}
        step_retail_volume = {"submission": 0.0, "normalizer": 0.0}
        step_execution_count = {"submission": 0, "normalizer": 0}
        step_execution_volume = {"submission": 0.0, "normalizer": 0.0}
        step_net_flow_y = {"submission": 0.0, "normalizer": 0.0}

        for amm in [self.amm_agent, self.amm_norm]:
            arb_result = self.arbitrageur.execute_arb(amm, fair_price, t)
            if arb_result is not None:
                step_arb_volume[arb_result.amm_name] += arb_result.amount_y
                self.edges[arb_result.amm_name] -= arb_result.profit

        orders = self.retail_trader.generate_orders()
        routed_trades = self.router.route_orders(
            orders, self.amm_agent, self.amm_norm, fair_price, t
        )
        for trade in routed_trades:
            step_retail_volume[trade.amm_name] += trade.amount_y
            step_execution_count[trade.amm_name] += 1
            step_execution_volume[trade.amm_name] += trade.amount_y
            if trade.amm_buys_x:
                step_net_flow_y[trade.amm_name] -= trade.amount_y
                trade_edge = trade.amount_x * fair_price - trade.amount_y
            else:
                step_net_flow_y[trade.amm_name] += trade.amount_y
                trade_edge = trade.amount_y - trade.amount_x * fair_price
            self.edges[trade.amm_name] += trade_edge

        self.current_fair_price = fair_price
        result = StepResult(
            timestamp=t,
            fair_price=fair_price,
            spot_prices={
                "submission": self.amm_agent.spot_price,
                "normalizer": self.amm_norm.spot_price,
            },
            pnls=self._compute_pnls(fair_price),
            edges=dict(self.edges),
            n_retail_orders=len(orders),
            retail_volume_y=step_retail_volume,
            arb_volume_y=step_arb_volume,
            execution_count=step_execution_count,
            execution_volume_y=step_execution_volume,
            net_flow_y=step_net_flow_y,
            ladder_depth_y=self.amm_agent.current_ladder_summary(),
            active_sigma=active_sigma,
        )

        self.current_step += 1
        return result

    def _compute_pnls(self, fair_price: float) -> dict[str, float]:
        pnls = {}
        for amm in [self.amm_agent, self.amm_norm]:
            init_x, init_y = self.initial_reserves[amm.name]
            init_value = init_x * self.initial_fair_price + init_y

            rx, ry = amm.reserves()
            fx = getattr(amm, "accumulated_fees_x", 0.0)
            fy = getattr(amm, "accumulated_fees_y", 0.0)
            curr_value = (rx * fair_price + ry) + (fx * fair_price + fy)
            pnls[amm.name] = curr_value - init_value
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
