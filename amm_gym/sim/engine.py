"""Step-by-step simulation engine.

Matches the loop in amm_sim_rs/src/simulation/engine.rs:
  1. Generate new fair price via GBM
  2. Arbitrageur extracts profit from each AMM
  3. Retail orders arrive and are routed across AMMs
"""

from __future__ import annotations

from dataclasses import dataclass, field

from amm_gym.sim.amm import ConstantProductAMM
from amm_gym.sim.price import GBMPriceProcess
from amm_gym.sim.actors import (
    Arbitrageur,
    OrderRouter,
    RetailTrader,
)


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
    seed: int | None = None


@dataclass
class StepResult:
    timestamp: int
    fair_price: float
    spot_prices: dict[str, float]
    pnls: dict[str, float]
    fees: dict[str, tuple[float, float]]
    edges: dict[str, float]
    # Per-step trade stats
    n_retail_orders: int = 0
    retail_volume_y: dict[str, float] = field(default_factory=dict)
    arb_volume_y: dict[str, float] = field(default_factory=dict)


class SimulationEngine:
    """Step-by-step AMM simulation with two competing pools."""

    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self._reset_state(seed=config.seed)

    def _reset_state(self, seed: int | None) -> None:
        cfg = self.config

        self.price_process = GBMPriceProcess(
            cfg.initial_price, cfg.gbm_mu, cfg.gbm_sigma, cfg.gbm_dt,
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

        # Agent AMM (fees will be set externally by RL agent)
        self.amm_agent = ConstantProductAMM(
            name="submission",
            reserve_x=cfg.initial_x,
            reserve_y=cfg.initial_y,
            bid_fee=0.003,
            ask_fee=0.003,
        )

        # Normalizer AMM (fixed 30 bps)
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

    def reset(self, seed: int | None = None) -> None:
        """Reset for a new episode."""
        self._reset_state(seed=seed)

    def set_agent_fees(self, bid_fee: float, ask_fee: float) -> None:
        """Set the agent AMM's fees (called by RL agent before each step)."""
        self.amm_agent.fees.bid_fee = bid_fee
        self.amm_agent.fees.ask_fee = ask_fee

    def step(self) -> StepResult:
        """Execute one simulation step. Returns step result."""
        t = self.current_step

        # 1. Generate new fair price
        fair_price = self.price_process.step()
        self.current_fair_price = fair_price

        step_arb_volume: dict[str, float] = {"submission": 0.0, "normalizer": 0.0}
        step_retail_volume: dict[str, float] = {"submission": 0.0, "normalizer": 0.0}

        # 2. Arbitrageur trades on each AMM
        for amm in [self.amm_agent, self.amm_norm]:
            arb_result = self.arbitrageur.execute_arb(amm, fair_price, t)
            if arb_result is not None:
                step_arb_volume[arb_result.amm_name] += arb_result.amount_y
                self.edges[arb_result.amm_name] -= arb_result.profit

        # 3. Retail orders arrive and get routed
        orders = self.retail_trader.generate_orders()
        routed_trades = self.router.route_orders(
            orders, self.amm_agent, self.amm_norm, fair_price, t
        )
        for trade in routed_trades:
            step_retail_volume[trade.amm_name] += trade.amount_y
            if trade.amm_buys_x:
                trade_edge = trade.amount_x * fair_price - trade.amount_y
            else:
                trade_edge = trade.amount_y - trade.amount_x * fair_price
            self.edges[trade.amm_name] += trade_edge

        # 4. Capture step result
        result = StepResult(
            timestamp=t,
            fair_price=fair_price,
            spot_prices={
                "submission": self.amm_agent.spot_price,
                "normalizer": self.amm_norm.spot_price,
            },
            pnls=self._compute_pnls(fair_price),
            fees={
                "submission": (
                    self.amm_agent.fees.bid_fee,
                    self.amm_agent.fees.ask_fee,
                ),
                "normalizer": (
                    self.amm_norm.fees.bid_fee,
                    self.amm_norm.fees.ask_fee,
                ),
            },
            edges=dict(self.edges),
            n_retail_orders=len(orders),
            retail_volume_y=step_retail_volume,
            arb_volume_y=step_arb_volume,
        )

        self.current_step += 1
        return result

    def _compute_pnls(self, fair_price: float) -> dict[str, float]:
        """Compute running PnL for each AMM (reserves + fees at fair price)."""
        pnls = {}
        for amm in [self.amm_agent, self.amm_norm]:
            init_x, init_y = self.initial_reserves[amm.name]
            init_value = init_x * self.initial_fair_price + init_y

            rx, ry = amm.reserves()
            fx, fy = amm.accumulated_fees_x, amm.accumulated_fees_y
            curr_value = (rx * fair_price + ry) + (fx * fair_price + fy)

            pnls[amm.name] = curr_value - init_value
        return pnls

    @property
    def done(self) -> bool:
        return self.current_step >= self.config.n_steps
