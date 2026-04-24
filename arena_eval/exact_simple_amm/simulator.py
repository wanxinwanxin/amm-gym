"""Exact simple AMM simulator and scoring API."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Iterable

import numpy as np

from arena_eval.core.types import BatchResult, SimulationResult, TradeInfo
from arena_eval.exact_simple_amm.config import ExactSimpleAMMConfig
from arena_eval.exact_simple_amm.dynamics import EmpiricalImpactRetailTrader, RegimeSwitchingReturnProcess
from arena_eval.exact_simple_amm.strategies import ExactSimpleAMMStrategy, FixedFeeStrategy


DEFAULT_BASELINE_FEE = 0.003
MAX_FEE = 0.1
MIN_AMOUNT = 0.0001


def _clamp_fee(value: float) -> float:
    return float(min(MAX_FEE, max(0.0, value)))


def _safe_strategy_fees(
    current_bid: float,
    current_ask: float,
    callback,
) -> tuple[float, float]:
    try:
        bid_fee, ask_fee = callback()
    except Exception:
        return (current_bid, current_ask)
    return (_clamp_fee(float(bid_fee)), _clamp_fee(float(ask_fee)))


class GBMPriceProcess:
    """Geometric Brownian motion using the challenge update formula."""

    def __init__(self, initial_price: float, mu: float, sigma: float, dt: float, seed: int) -> None:
        self.current_price = float(initial_price)
        self.mu = float(mu)
        self.dt = float(dt)
        self.rng = np.random.default_rng(seed)
        self._set_sigma(float(sigma))

    def _set_sigma(self, sigma: float) -> None:
        self.sigma = sigma
        self.drift_term = (self.mu - 0.5 * sigma * sigma) * self.dt
        self.vol_term = sigma * math.sqrt(self.dt)

    def step(self) -> float:
        z = float(self.rng.standard_normal())
        exponent = self.drift_term + self.vol_term * z
        self.current_price *= math.exp(exponent)
        return self.current_price


@dataclass(frozen=True)
class RetailOrder:
    side: str
    size: float


class RetailTrader:
    """Retail flow generator with Poisson arrivals and lognormal sizes."""

    def __init__(
        self,
        arrival_rate: float,
        mean_size: float,
        size_sigma: float,
        buy_prob: float,
        seed: int,
    ) -> None:
        self.arrival_rate = float(arrival_rate)
        self.buy_prob = float(buy_prob)
        self.size_sigma = max(float(size_sigma), 0.01)
        mean = max(float(mean_size), 0.01)
        self._ln_mu = math.log(mean) - 0.5 * self.size_sigma**2
        self._ln_sigma = self.size_sigma
        self.rng = np.random.default_rng(seed)

    def generate_orders(
        self,
        *,
        fair_price: float,
        reference_amm=None,
    ) -> list[RetailOrder]:
        del fair_price, reference_amm
        if self.arrival_rate <= 0.0:
            return []
        n_orders = int(self.rng.poisson(self.arrival_rate))
        if n_orders == 0:
            return []
        sizes = self.rng.lognormal(self._ln_mu, self._ln_sigma, size=n_orders)
        sides = self.rng.random(size=n_orders)
        return [
            RetailOrder(side="buy" if side < self.buy_prob else "sell", size=float(size))
            for side, size in zip(sides, sizes)
        ]


class StrategyAMM:
    """Constant-product AMM with strategy-controlled fees."""

    def __init__(
        self,
        name: str,
        strategy: ExactSimpleAMMStrategy,
        reserve_x: float,
        reserve_y: float,
    ) -> None:
        self.name = name
        self.strategy = strategy
        self.reserve_x = float(reserve_x)
        self.reserve_y = float(reserve_y)
        self.bid_fee = DEFAULT_BASELINE_FEE
        self.ask_fee = DEFAULT_BASELINE_FEE
        self.accumulated_fees_x = 0.0
        self.accumulated_fees_y = 0.0

    @property
    def spot_price(self) -> float:
        if self.reserve_x == 0.0:
            return 0.0
        return self.reserve_y / self.reserve_x

    @property
    def k(self) -> float:
        return self.reserve_x * self.reserve_y

    def reserves(self) -> tuple[float, float]:
        return (self.reserve_x, self.reserve_y)

    def initialize(self) -> None:
        self.bid_fee, self.ask_fee = _safe_strategy_fees(
            self.bid_fee,
            self.ask_fee,
            lambda: self.strategy.after_initialize(self.reserve_x, self.reserve_y),
        )

    def quote_buy_x(self, amount_x: float) -> tuple[float, float]:
        if amount_x <= 0.0:
            return (0.0, 0.0)
        gamma = max(0.0, min(1.0, 1.0 - self.bid_fee))
        if gamma <= 0.0:
            return (0.0, 0.0)
        net_x = amount_x * gamma
        new_rx = self.reserve_x + net_x
        new_ry = self.k / new_rx
        y_out = self.reserve_y - new_ry
        if y_out > 0.0:
            return (y_out, amount_x * self.bid_fee)
        return (0.0, 0.0)

    def quote_sell_x(self, amount_x: float) -> tuple[float, float]:
        if amount_x <= 0.0 or amount_x >= self.reserve_x:
            return (0.0, 0.0)
        gamma = max(0.0, min(1.0, 1.0 - self.ask_fee))
        if gamma <= 0.0:
            return (0.0, 0.0)
        new_rx = self.reserve_x - amount_x
        new_ry = self.k / new_rx
        net_y = new_ry - self.reserve_y
        if net_y <= 0.0:
            return (0.0, 0.0)
        total_y = net_y / gamma
        return (total_y, total_y - net_y)

    def quote_x_for_y(self, amount_y: float) -> tuple[float, float]:
        if amount_y <= 0.0:
            return (0.0, 0.0)
        gamma = max(0.0, min(1.0, 1.0 - self.ask_fee))
        if gamma <= 0.0:
            return (0.0, 0.0)
        net_y = amount_y * gamma
        new_ry = self.reserve_y + net_y
        new_rx = self.k / new_ry
        x_out = self.reserve_x - new_rx
        if x_out > 0.0:
            return (x_out, amount_y * self.ask_fee)
        return (0.0, 0.0)

    def execute_buy_x(self, amount_x: float, timestamp: int) -> TradeInfo | None:
        y_out, fee_amount = self.quote_buy_x(amount_x)
        if y_out <= 0.0:
            return None
        net_x = amount_x - fee_amount
        self.reserve_x += net_x
        self.accumulated_fees_x += fee_amount
        self.reserve_y -= y_out
        trade = TradeInfo(
            is_buy=True,
            amount_x=float(amount_x),
            amount_y=float(y_out),
            timestamp=int(timestamp),
            reserve_x=self.reserve_x,
            reserve_y=self.reserve_y,
        )
        self.bid_fee, self.ask_fee = _safe_strategy_fees(
            self.bid_fee,
            self.ask_fee,
            lambda: self.strategy.after_swap(trade),
        )
        return trade

    def execute_sell_x(self, amount_x: float, timestamp: int) -> TradeInfo | None:
        total_y, fee_amount = self.quote_sell_x(amount_x)
        if total_y <= 0.0:
            return None
        net_y = total_y - fee_amount
        self.reserve_x -= amount_x
        self.reserve_y += net_y
        self.accumulated_fees_y += fee_amount
        trade = TradeInfo(
            is_buy=False,
            amount_x=float(amount_x),
            amount_y=float(total_y),
            timestamp=int(timestamp),
            reserve_x=self.reserve_x,
            reserve_y=self.reserve_y,
        )
        self.bid_fee, self.ask_fee = _safe_strategy_fees(
            self.bid_fee,
            self.ask_fee,
            lambda: self.strategy.after_swap(trade),
        )
        return trade

    def execute_buy_x_with_y(self, amount_y: float, timestamp: int) -> TradeInfo | None:
        x_out, fee_amount = self.quote_x_for_y(amount_y)
        if x_out <= 0.0:
            return None
        net_y = amount_y - fee_amount
        self.reserve_x -= x_out
        self.reserve_y += net_y
        self.accumulated_fees_y += fee_amount
        trade = TradeInfo(
            is_buy=False,
            amount_x=float(x_out),
            amount_y=float(amount_y),
            timestamp=int(timestamp),
            reserve_x=self.reserve_x,
            reserve_y=self.reserve_y,
        )
        self.bid_fee, self.ask_fee = _safe_strategy_fees(
            self.bid_fee,
            self.ask_fee,
            lambda: self.strategy.after_swap(trade),
        )
        return trade


@dataclass(frozen=True)
class RoutedTrade:
    amm_name: str
    source: str
    amount_y: float
    amount_x: float
    amm_buys_x: bool
    pre_spot_price: float
    post_spot_price: float
    pre_state: dict[str, float]
    post_state: dict[str, float]
    trade_info: TradeInfo


class OrderRouter:
    """Two-AMM optimal router matching the challenge formulas."""

    @staticmethod
    def _quote_state(submission: StrategyAMM, normalizer: StrategyAMM) -> dict[str, float]:
        return {
            "submission_mid": submission.spot_price,
            "submission_bid_fee": submission.bid_fee,
            "submission_ask_fee": submission.ask_fee,
            "normalizer_mid": normalizer.spot_price,
            "normalizer_bid_fee": normalizer.bid_fee,
            "normalizer_ask_fee": normalizer.ask_fee,
        }

    def split_buy_two_amms(self, amm1: StrategyAMM, amm2: StrategyAMM, total_y: float) -> tuple[float, float]:
        x1, y1 = amm1.reserves()
        x2, y2 = amm2.reserves()
        gamma1 = 1.0 - amm1.ask_fee
        gamma2 = 1.0 - amm2.ask_fee
        a1 = math.sqrt(x1 * gamma1 * y1)
        a2 = math.sqrt(x2 * gamma2 * y2)
        if a2 == 0.0:
            return (total_y, 0.0)
        r = a1 / a2
        numerator = r * (y2 + gamma2 * total_y) - y1
        denominator = gamma1 + r * gamma2
        y1_amount = total_y / 2.0 if denominator == 0.0 else numerator / denominator
        y1_amount = max(0.0, min(total_y, y1_amount))
        return (y1_amount, total_y - y1_amount)

    def split_sell_two_amms(self, amm1: StrategyAMM, amm2: StrategyAMM, total_x: float) -> tuple[float, float]:
        x1, y1 = amm1.reserves()
        x2, y2 = amm2.reserves()
        gamma1 = 1.0 - amm1.bid_fee
        gamma2 = 1.0 - amm2.bid_fee
        b1 = math.sqrt(y1 * gamma1 * x1)
        b2 = math.sqrt(y2 * gamma2 * x2)
        if b2 == 0.0:
            return (total_x, 0.0)
        r = b1 / b2
        numerator = r * (x2 + gamma2 * total_x) - x1
        denominator = gamma1 + r * gamma2
        x1_amount = total_x / 2.0 if denominator == 0.0 else numerator / denominator
        x1_amount = max(0.0, min(total_x, x1_amount))
        return (x1_amount, total_x - x1_amount)

    def route_orders(
        self,
        orders: list[RetailOrder],
        submission: StrategyAMM,
        normalizer: StrategyAMM,
        fair_price: float,
        timestamp: int,
    ) -> list[RoutedTrade]:
        trades: list[RoutedTrade] = []
        for order in orders:
            if order.side == "buy":
                y_submission, y_normalizer = self.split_buy_two_amms(submission, normalizer, order.size)
                if y_submission > MIN_AMOUNT:
                    pre_spot_price = submission.spot_price
                    pre_state = self._quote_state(submission, normalizer)
                    trade = submission.execute_buy_x_with_y(y_submission, timestamp)
                    if trade is not None:
                        trades.append(
                            RoutedTrade(
                                submission.name,
                                "retail",
                                y_submission,
                                trade.amount_x,
                                False,
                                pre_spot_price,
                                submission.spot_price,
                                pre_state,
                                self._quote_state(submission, normalizer),
                                trade,
                            )
                        )
                if y_normalizer > MIN_AMOUNT:
                    pre_spot_price = normalizer.spot_price
                    pre_state = self._quote_state(submission, normalizer)
                    trade = normalizer.execute_buy_x_with_y(y_normalizer, timestamp)
                    if trade is not None:
                        trades.append(
                            RoutedTrade(
                                normalizer.name,
                                "retail",
                                y_normalizer,
                                trade.amount_x,
                                False,
                                pre_spot_price,
                                normalizer.spot_price,
                                pre_state,
                                self._quote_state(submission, normalizer),
                                trade,
                            )
                        )
            else:
                total_x = order.size / fair_price
                x_submission, x_normalizer = self.split_sell_two_amms(submission, normalizer, total_x)
                if x_submission > MIN_AMOUNT:
                    pre_spot_price = submission.spot_price
                    pre_state = self._quote_state(submission, normalizer)
                    trade = submission.execute_buy_x(x_submission, timestamp)
                    if trade is not None:
                        trades.append(
                            RoutedTrade(
                                submission.name,
                                "retail",
                                trade.amount_y,
                                x_submission,
                                True,
                                pre_spot_price,
                                submission.spot_price,
                                pre_state,
                                self._quote_state(submission, normalizer),
                                trade,
                            )
                        )
                if x_normalizer > MIN_AMOUNT:
                    pre_spot_price = normalizer.spot_price
                    pre_state = self._quote_state(submission, normalizer)
                    trade = normalizer.execute_buy_x(x_normalizer, timestamp)
                    if trade is not None:
                        trades.append(
                            RoutedTrade(
                                normalizer.name,
                                "retail",
                                trade.amount_y,
                                x_normalizer,
                                True,
                                pre_spot_price,
                                normalizer.spot_price,
                                pre_state,
                                self._quote_state(submission, normalizer),
                                trade,
                            )
                        )
        return trades


@dataclass(frozen=True)
class ArbResult:
    amm_name: str
    source: str
    profit: float
    side: str
    amount_x: float
    amount_y: float
    pre_spot_price: float
    post_spot_price: float
    pre_state: dict[str, float]
    post_state: dict[str, float]
    trade_info: TradeInfo


class Arbitrageur:
    """Closed-form arbitrage against a constant-product AMM."""

    @staticmethod
    def _quote_state(amm: StrategyAMM) -> dict[str, float]:
        return {
            "mid": amm.spot_price,
            "bid_fee": amm.bid_fee,
            "ask_fee": amm.ask_fee,
        }

    def execute_arb(self, amm: StrategyAMM, fair_price: float, timestamp: int) -> ArbResult | None:
        spot_price = amm.spot_price
        if spot_price < fair_price:
            return self._buy_arb(amm, fair_price, timestamp)
        if spot_price > fair_price:
            return self._sell_arb(amm, fair_price, timestamp)
        return None

    def _buy_arb(self, amm: StrategyAMM, fair_price: float, timestamp: int) -> ArbResult | None:
        gamma = 1.0 - amm.ask_fee
        if gamma <= 0.0 or fair_price <= 0.0:
            return None
        rx, ry = amm.reserves()
        pre_spot_price = amm.spot_price
        pre_state = self._quote_state(amm)
        amount_x = rx - math.sqrt((rx * ry) / (gamma * fair_price))
        if amount_x <= 0.0:
            return None
        amount_x = min(amount_x, rx * 0.99)
        total_y, _ = amm.quote_sell_x(amount_x)
        if total_y <= 0.0:
            return None
        profit = amount_x * fair_price - total_y
        if profit <= 0.0:
            return None
        trade = amm.execute_sell_x(amount_x, timestamp)
        if trade is None:
            return None
        return ArbResult(
            amm.name,
            "arb",
            profit,
            "sell",
            amount_x,
            total_y,
            pre_spot_price,
            amm.spot_price,
            pre_state,
            self._quote_state(amm),
            trade,
        )

    def _sell_arb(self, amm: StrategyAMM, fair_price: float, timestamp: int) -> ArbResult | None:
        gamma = 1.0 - amm.bid_fee
        if gamma <= 0.0 or fair_price <= 0.0:
            return None
        rx, ry = amm.reserves()
        pre_spot_price = amm.spot_price
        pre_state = self._quote_state(amm)
        amount_x = (math.sqrt((rx * ry) * gamma / fair_price) - rx) / gamma
        if amount_x <= 0.0:
            return None
        amount_y, _ = amm.quote_buy_x(amount_x)
        if amount_y <= 0.0:
            return None
        profit = amount_y - amount_x * fair_price
        if profit <= 0.0:
            return None
        trade = amm.execute_buy_x(amount_x, timestamp)
        if trade is None:
            return None
        return ArbResult(
            amm.name,
            "arb",
            profit,
            "buy",
            amount_x,
            amount_y,
            pre_spot_price,
            amm.spot_price,
            pre_state,
            self._quote_state(amm),
            trade,
        )


@dataclass
class ExactSimpleAMMSimulator:
    """Stateful exact simple-AMM simulator."""

    config: ExactSimpleAMMConfig
    submission_strategy: ExactSimpleAMMStrategy
    normalizer_strategy: ExactSimpleAMMStrategy
    seed: int

    def __post_init__(self) -> None:
        self.submission = StrategyAMM(
            "submission",
            self.submission_strategy,
            self.config.initial_x,
            self.config.initial_y,
        )
        self.normalizer = StrategyAMM(
            "normalizer",
            self.normalizer_strategy,
            self.config.initial_x,
            self.config.initial_y,
        )
        self.price_process = self._build_price_process()
        self.retail_trader = self._build_retail_trader()
        self.arbitrageur = Arbitrageur()
        self.router = OrderRouter()
        self.submission.initialize()
        self.normalizer.initialize()
        self.current_step = 0
        self.current_fair_price = self.config.initial_price
        self.edge_submission = 0.0
        self.edge_normalizer = 0.0
        self.retail_volume_submission_y = 0.0
        self.retail_volume_normalizer_y = 0.0
        self.arb_volume_submission_y = 0.0
        self.arb_volume_normalizer_y = 0.0
        self.bid_fee_submission_sum = 0.0
        self.ask_fee_submission_sum = 0.0
        self.bid_fee_normalizer_sum = 0.0
        self.ask_fee_normalizer_sum = 0.0
        self.last_submission_trade: TradeInfo | None = None
        self.submission_trade_count = 0

    def _build_price_process(self):
        if self.config.price_process_kind == "gbm":
            return GBMPriceProcess(
                self.config.initial_price,
                self.config.gbm_mu,
                self.config.gbm_sigma,
                self.config.gbm_dt,
                seed=self.seed,
            )
        if self.config.price_process_kind == "regime_switching":
            if not self.config.regime_invcdf_path or not self.config.regime_transition_path:
                raise ValueError("Regime-switching evaluator requires regime CSV paths")
            return RegimeSwitchingReturnProcess(
                self.config.initial_price,
                self.config.regime_invcdf_path,
                self.config.regime_transition_path,
                start_regime=self.config.regime_start,
                seed=self.seed,
            )
        raise ValueError(f"Unsupported price_process_kind: {self.config.price_process_kind}")

    def _build_retail_trader(self):
        if self.config.retail_flow_kind == "lognormal_size":
            return RetailTrader(
                self.config.retail_arrival_rate,
                self.config.retail_mean_size,
                self.config.retail_size_sigma,
                self.config.retail_buy_prob,
                seed=self.seed + 1,
            )
        if self.config.retail_flow_kind == "empirical_impact":
            if not self.config.retail_impact_percentiles_path:
                raise ValueError("Empirical retail evaluator requires percentile CSV path")
            return EmpiricalImpactRetailTrader(
                self.config.retail_arrival_rate,
                self.config.retail_impact_percentiles_path,
                impact_column=self.config.retail_impact_column,
                reference_venue=self.config.retail_impact_reference_venue,
                scale_mode=self.config.retail_impact_scale_mode,
                initial_x=self.config.initial_x,
                initial_y=self.config.initial_y,
                seed=self.seed + 1,
            )
        raise ValueError(f"Unsupported retail_flow_kind: {self.config.retail_flow_kind}")

    def _retail_reference_amm(self) -> StrategyAMM:
        if self.config.retail_impact_reference_venue == "submission":
            return self.submission
        return self.normalizer

    @property
    def done(self) -> bool:
        return self.current_step >= self.config.n_steps

    def _record_fee_snapshot(self) -> None:
        self.bid_fee_submission_sum += self.submission.bid_fee
        self.ask_fee_submission_sum += self.submission.ask_fee
        self.bid_fee_normalizer_sum += self.normalizer.bid_fee
        self.ask_fee_normalizer_sum += self.normalizer.ask_fee

    def step_once(self) -> dict[str, object]:
        if self.done:
            raise RuntimeError("simulation is already complete")
        timestamp = self.current_step
        fair_price = self.price_process.step()
        self.current_fair_price = fair_price
        initial_value = self.config.initial_x * self.config.initial_price + self.config.initial_y
        prev_submission_trade_count = self.submission_trade_count
        self.last_submission_trade = None
        trade_events: list[dict[str, object]] = []

        def metric_snapshot() -> dict[str, float]:
            return {
                "edge_submission": self.edge_submission,
                "edge_normalizer": self.edge_normalizer,
                "pnl_submission": self._mark_to_market(self.submission) - initial_value,
                "pnl_normalizer": self._mark_to_market(self.normalizer) - initial_value,
            }

        def global_quote_state() -> dict[str, float]:
            return {
                "submission_mid": self.submission.spot_price,
                "submission_bid_fee": self.submission.bid_fee,
                "submission_ask_fee": self.submission.ask_fee,
                "normalizer_mid": self.normalizer.spot_price,
                "normalizer_bid_fee": self.normalizer.bid_fee,
                "normalizer_ask_fee": self.normalizer.ask_fee,
            }

        submission_arb = self.arbitrageur.execute_arb(self.submission, fair_price, timestamp)
        if submission_arb is not None:
            pre_metrics = metric_snapshot()
            self.arb_volume_submission_y += submission_arb.amount_y
            self.edge_submission -= submission_arb.profit
            self.last_submission_trade = submission_arb.trade_info
            self.submission_trade_count += 1
            post_metrics = metric_snapshot()
            trade_events.append(
                {
                    "venue": submission_arb.amm_name,
                    "source": submission_arb.source,
                    "trader_side": "buy_x" if submission_arb.side == "sell" else "sell_x",
                    "amount_x": submission_arb.amount_x,
                    "amount_y": submission_arb.amount_y,
                    "pre_spot_price": submission_arb.pre_spot_price,
                    "post_spot_price": submission_arb.post_spot_price,
                    "pre_state": {
                        **global_quote_state(),
                        f"{submission_arb.amm_name}_mid": submission_arb.pre_state["mid"],
                        f"{submission_arb.amm_name}_bid_fee": submission_arb.pre_state["bid_fee"],
                        f"{submission_arb.amm_name}_ask_fee": submission_arb.pre_state["ask_fee"],
                    },
                    "post_state": {
                        **global_quote_state(),
                        f"{submission_arb.amm_name}_mid": submission_arb.post_state["mid"],
                        f"{submission_arb.amm_name}_bid_fee": submission_arb.post_state["bid_fee"],
                        f"{submission_arb.amm_name}_ask_fee": submission_arb.post_state["ask_fee"],
                    },
                    "pre_metrics": pre_metrics,
                    "post_metrics": post_metrics,
                    "trade_info": submission_arb.trade_info,
                }
            )

        normalizer_arb = self.arbitrageur.execute_arb(self.normalizer, fair_price, timestamp)
        if normalizer_arb is not None:
            pre_metrics = metric_snapshot()
            self.arb_volume_normalizer_y += normalizer_arb.amount_y
            self.edge_normalizer -= normalizer_arb.profit
            post_metrics = metric_snapshot()
            trade_events.append(
                {
                    "venue": normalizer_arb.amm_name,
                    "source": normalizer_arb.source,
                    "trader_side": "buy_x" if normalizer_arb.side == "sell" else "sell_x",
                    "amount_x": normalizer_arb.amount_x,
                    "amount_y": normalizer_arb.amount_y,
                    "pre_spot_price": normalizer_arb.pre_spot_price,
                    "post_spot_price": normalizer_arb.post_spot_price,
                    "pre_state": {
                        **global_quote_state(),
                        f"{normalizer_arb.amm_name}_mid": normalizer_arb.pre_state["mid"],
                        f"{normalizer_arb.amm_name}_bid_fee": normalizer_arb.pre_state["bid_fee"],
                        f"{normalizer_arb.amm_name}_ask_fee": normalizer_arb.pre_state["ask_fee"],
                    },
                    "post_state": {
                        **global_quote_state(),
                        f"{normalizer_arb.amm_name}_mid": normalizer_arb.post_state["mid"],
                        f"{normalizer_arb.amm_name}_bid_fee": normalizer_arb.post_state["bid_fee"],
                        f"{normalizer_arb.amm_name}_ask_fee": normalizer_arb.post_state["ask_fee"],
                    },
                    "pre_metrics": pre_metrics,
                    "post_metrics": post_metrics,
                    "trade_info": normalizer_arb.trade_info,
                }
            )

        orders = self.retail_trader.generate_orders(
            fair_price=fair_price,
            reference_amm=self._retail_reference_amm(),
        )
        trades = self.router.route_orders(orders, self.submission, self.normalizer, fair_price, timestamp)
        for trade in trades:
            pre_metrics = metric_snapshot()
            trade_edge = (
                trade.amount_x * fair_price - trade.amount_y
                if trade.amm_buys_x
                else trade.amount_y - trade.amount_x * fair_price
            )
            if trade.amm_name == "submission":
                self.retail_volume_submission_y += trade.amount_y
                self.edge_submission += trade_edge
                self.last_submission_trade = trade.trade_info
                self.submission_trade_count += 1
            else:
                self.retail_volume_normalizer_y += trade.amount_y
                self.edge_normalizer += trade_edge
            post_metrics = metric_snapshot()
            trade_events.append(
                {
                    "venue": trade.amm_name,
                    "source": trade.source,
                    "trader_side": "sell_x" if trade.amm_buys_x else "buy_x",
                    "amount_x": trade.amount_x,
                    "amount_y": trade.amount_y,
                    "pre_spot_price": trade.pre_spot_price,
                    "post_spot_price": trade.post_spot_price,
                    "pre_state": trade.pre_state,
                    "post_state": trade.post_state,
                    "pre_metrics": pre_metrics,
                    "post_metrics": post_metrics,
                    "trade_info": trade.trade_info,
                }
            )

        self._record_fee_snapshot()
        self.current_step += 1
        return {
            "timestamp": timestamp,
            "fair_price": fair_price,
            "submission_trade_occurred": self.submission_trade_count > prev_submission_trade_count,
            "last_submission_trade": self.last_submission_trade,
            "n_orders": len(orders),
            "trade_events": tuple(trade_events),
        }

    def run(self) -> SimulationResult:
        while not self.done:
            self.step_once()
        return self.result()

    def result(self) -> SimulationResult:
        initial_value = self.config.initial_x * self.config.initial_price + self.config.initial_y
        pnl_submission = self._mark_to_market(self.submission) - initial_value
        pnl_normalizer = self._mark_to_market(self.normalizer) - initial_value
        steps = max(self.config.n_steps, 1)
        return SimulationResult(
            seed=self.seed,
            edge_submission=self.edge_submission,
            edge_normalizer=self.edge_normalizer,
            pnl_submission=pnl_submission,
            pnl_normalizer=pnl_normalizer,
            score=self.edge_submission,
            retail_volume_submission_y=self.retail_volume_submission_y,
            retail_volume_normalizer_y=self.retail_volume_normalizer_y,
            arb_volume_submission_y=self.arb_volume_submission_y,
            arb_volume_normalizer_y=self.arb_volume_normalizer_y,
            average_bid_fee_submission=self.bid_fee_submission_sum / steps,
            average_ask_fee_submission=self.ask_fee_submission_sum / steps,
            average_bid_fee_normalizer=self.bid_fee_normalizer_sum / steps,
            average_ask_fee_normalizer=self.ask_fee_normalizer_sum / steps,
        )

    def _mark_to_market(self, amm: StrategyAMM) -> float:
        rx, ry = amm.reserves()
        reserve_value = rx * self.current_fair_price + ry
        fee_value = amm.accumulated_fees_x * self.current_fair_price + amm.accumulated_fees_y
        return reserve_value + fee_value


def run_seed(
    submission_strategy: ExactSimpleAMMStrategy,
    seed: int,
    *,
    config: ExactSimpleAMMConfig | None = None,
    normalizer_strategy: ExactSimpleAMMStrategy | None = None,
    evaluator_kind: str = "challenge",
) -> SimulationResult:
    exact_config = config or ExactSimpleAMMConfig.for_evaluator(seed, evaluator_kind)
    simulator = ExactSimpleAMMSimulator(
        config=exact_config,
        submission_strategy=submission_strategy,
        normalizer_strategy=normalizer_strategy or FixedFeeStrategy(),
        seed=seed,
    )
    return simulator.run()


def run_batch(
    submission_strategy_factory,
    seeds: Iterable[int],
    *,
    normalizer_strategy_factory=None,
    evaluator_kind: str = "challenge",
) -> BatchResult:
    seed_values = tuple(int(seed) for seed in seeds)
    simulations = tuple(
        run_seed(
            submission_strategy_factory(),
            seed,
            normalizer_strategy=(normalizer_strategy_factory() if normalizer_strategy_factory else FixedFeeStrategy()),
            evaluator_kind=evaluator_kind,
        )
        for seed in seed_values
    )
    return BatchResult(
        seeds=seed_values,
        simulations=simulations,
        score=float(mean(sim.score for sim in simulations)),
        edge_mean_submission=float(mean(sim.edge_submission for sim in simulations)),
        edge_mean_normalizer=float(mean(sim.edge_normalizer for sim in simulations)),
        edge_advantage_mean=float(mean(sim.edge_advantage for sim in simulations)),
        pnl_mean_submission=float(mean(sim.pnl_submission for sim in simulations)),
        pnl_mean_normalizer=float(mean(sim.pnl_normalizer for sim in simulations)),
        pnl_advantage_mean=float(mean(sim.pnl_advantage for sim in simulations)),
        metadata={"n_simulations": len(simulations)},
    )


def score_challenge(submission_strategy_factory, *, n_simulations: int = 1000, evaluator_kind: str = "challenge") -> float:
    batch = run_batch(submission_strategy_factory, range(n_simulations), evaluator_kind=evaluator_kind)
    return batch.score
