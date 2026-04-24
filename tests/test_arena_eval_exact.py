from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from arena_eval.core.types import TradeInfo
from arena_eval.exact_simple_amm import (
    EmpiricalImpactRetailTrader,
    ExactSimpleAMMConfig,
    FixedFeeStrategy,
    RegimeSwitchingReturnProcess,
    run_batch,
    run_seed,
    score_challenge,
)
from arena_eval.exact_simple_amm.simulator import ExactSimpleAMMSimulator


class ErrorAfterSwapStrategy:
    def __init__(self, initial_bid: float = 0.003, initial_ask: float = 0.004) -> None:
        self.initial_bid = initial_bid
        self.initial_ask = initial_ask
        self.after_swap_calls = 0

    def after_initialize(self, initial_x: float, initial_y: float) -> tuple[float, float]:
        return (self.initial_bid, self.initial_ask)

    def after_swap(self, trade: TradeInfo) -> tuple[float, float]:
        self.after_swap_calls += 1
        raise RuntimeError("boom")


class CaptureTradesStrategy:
    def __init__(self, bid_fee: float = 0.003, ask_fee: float = 0.003) -> None:
        self.bid_fee = bid_fee
        self.ask_fee = ask_fee
        self.trades: list[TradeInfo] = []

    def after_initialize(self, initial_x: float, initial_y: float) -> tuple[float, float]:
        return (self.bid_fee, self.ask_fee)

    def after_swap(self, trade: TradeInfo) -> tuple[float, float]:
        self.trades.append(trade)
        return (self.bid_fee, self.ask_fee)


def test_exact_config_sampling_is_deterministic():
    first = ExactSimpleAMMConfig.from_seed(7)
    second = ExactSimpleAMMConfig.from_seed(7)

    assert first == second
    assert 0.000882 <= first.gbm_sigma <= 0.001008
    assert 0.6 <= first.retail_arrival_rate <= 1.0
    assert 19.0 <= first.retail_mean_size <= 21.0


def test_after_swap_errors_keep_previous_fees():
    strategy = ErrorAfterSwapStrategy(initial_bid=0.011, initial_ask=0.013)
    simulator = ExactSimpleAMMSimulator(
        config=ExactSimpleAMMConfig(n_steps=32, retail_arrival_rate=2.0, retail_mean_size=30.0),
        submission_strategy=strategy,
        normalizer_strategy=FixedFeeStrategy(),
        seed=3,
    )

    for _ in range(32):
        simulator.step_once()

    assert strategy.after_swap_calls > 0
    assert simulator.submission.bid_fee == pytest.approx(0.011)
    assert simulator.submission.ask_fee == pytest.approx(0.013)


def test_real_data_config_has_expected_artifacts():
    config = ExactSimpleAMMConfig.real_data_from_seed(7)

    assert config.evaluator_kind == "real_data"
    assert config.price_process_kind == "regime_switching"
    assert config.retail_flow_kind == "empirical_impact"
    assert Path(config.regime_invcdf_path).exists()
    assert Path(config.regime_transition_path).exists()
    assert Path(config.retail_impact_percentiles_path).exists()


def test_regime_switching_process_is_deterministic():
    config = ExactSimpleAMMConfig.real_data_from_seed(5)
    first = RegimeSwitchingReturnProcess(
        100.0,
        config.regime_invcdf_path,
        config.regime_transition_path,
        seed=11,
    )
    second = RegimeSwitchingReturnProcess(
        100.0,
        config.regime_invcdf_path,
        config.regime_transition_path,
        seed=11,
    )
    first_path = [first.step() for _ in range(5)]
    second_path = [second.step() for _ in range(5)]

    assert first_path == pytest.approx(second_path)


def test_empirical_impact_retail_trader_generates_y_notional_orders():
    config = ExactSimpleAMMConfig.real_data_from_seed(3)
    trader = EmpiricalImpactRetailTrader(
        arrival_rate=3.0,
        impact_percentiles_path=config.retail_impact_percentiles_path,
        initial_x=config.initial_x,
        initial_y=config.initial_y,
        seed=19,
    )
    reference_amm = type(
        "ReferenceAMM",
        (),
        {
            "reserve_x": config.initial_x,
            "reserve_y": config.initial_y,
            "bid_fee": 0.003,
            "ask_fee": 0.003,
            "spot_price": config.initial_price,
        },
    )()

    orders = trader.generate_orders(fair_price=config.initial_price, reference_amm=reference_amm)

    assert orders
    assert all(order.side in {"buy", "sell"} for order in orders)
    assert all(order.size > 0.0 for order in orders)


def test_after_swap_is_called_on_arbitrage_trades():
    strategy = CaptureTradesStrategy(bid_fee=0.0, ask_fee=0.0)
    simulator = ExactSimpleAMMSimulator(
        config=ExactSimpleAMMConfig(
            n_steps=4,
            retail_arrival_rate=0.0,
            gbm_sigma=0.02,
        ),
        submission_strategy=strategy,
        normalizer_strategy=FixedFeeStrategy(),
        seed=11,
    )

    for _ in range(4):
        simulator.step_once()

    assert len(strategy.trades) > 0


def test_step_once_exposes_trade_event_metadata():
    simulator = ExactSimpleAMMSimulator(
        config=ExactSimpleAMMConfig(
            n_steps=8,
            retail_arrival_rate=2.0,
            gbm_sigma=0.02,
        ),
        submission_strategy=FixedFeeStrategy(),
        normalizer_strategy=FixedFeeStrategy(),
        seed=5,
    )

    seen_event = None
    for _ in range(8):
        step_info = simulator.step_once()
        trade_events = step_info["trade_events"]
        if trade_events:
            seen_event = trade_events[0]
            break

    assert seen_event is not None
    assert seen_event["source"] in {"retail", "arb"}
    assert seen_event["venue"] in {"submission", "normalizer"}
    assert seen_event["pre_spot_price"] > 0.0
    assert seen_event["post_spot_price"] > 0.0


def test_real_data_simulator_runs():
    config = replace(ExactSimpleAMMConfig.real_data_from_seed(2), n_steps=8)
    simulator = ExactSimpleAMMSimulator(
        config=config,
        submission_strategy=FixedFeeStrategy(),
        normalizer_strategy=FixedFeeStrategy(),
        seed=2,
    )

    result = simulator.run()

    assert result.seed == 2
    assert isinstance(result.score, float)


def test_batch_score_matches_mean_submission_edge():
    batch = run_batch(lambda: FixedFeeStrategy(0.003, 0.003), range(5))

    expected = sum(sim.edge_submission for sim in batch.simulations) / len(batch.simulations)
    assert batch.score == pytest.approx(expected)
    assert batch.edge_mean_submission == pytest.approx(expected)


def test_run_seed_is_deterministic():
    first = run_seed(FixedFeeStrategy(), 17)
    second = run_seed(FixedFeeStrategy(), 17)

    assert first == second


def test_score_challenge_matches_run_batch_for_same_seed_count():
    expected = run_batch(lambda: FixedFeeStrategy(), range(8)).score
    actual = score_challenge(lambda: FixedFeeStrategy(), n_simulations=8)

    assert actual == pytest.approx(expected)


def test_score_challenge_supports_real_data_evaluator():
    score = score_challenge(lambda: FixedFeeStrategy(), n_simulations=2, evaluator_kind="real_data")

    assert isinstance(score, float)
