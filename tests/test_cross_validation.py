"""Formula checks and high-level simulation invariants."""

import math

import numpy as np
import pytest

from amm_gym.sim.amm import ConstantProductAMM
from amm_gym.sim.actors import Arbitrageur, OrderRouter
from amm_gym.sim.engine import SimConfig, SimulationEngine


class TestAMMFormulas:
    @pytest.mark.parametrize(
        "rx,ry,fee,amount_x",
        [
            (1000.0, 1000.0, 0.003, 10.0),
            (100.0, 10000.0, 0.003, 5.0),
            (1000.0, 1000.0, 0.05, 10.0),
        ],
    )
    def test_buy_x_formula(self, rx, ry, fee, amount_x):
        gamma = 1.0 - fee
        net_x = amount_x * gamma
        k = rx * ry
        new_rx = rx + net_x
        expected = ry - k / new_rx

        amm = ConstantProductAMM("test", rx, ry, bid_fee=fee, ask_fee=fee)
        y_out, fee_amt = amm.quote_buy_x(amount_x)
        assert y_out == pytest.approx(expected, rel=1e-12)
        assert fee_amt == pytest.approx(amount_x * fee, rel=1e-12)

    @pytest.mark.parametrize(
        "rx,ry,fee,amount_x",
        [
            (1000.0, 1000.0, 0.003, 10.0),
            (100.0, 10000.0, 0.003, 5.0),
            (1000.0, 1000.0, 0.05, 10.0),
        ],
    )
    def test_sell_x_formula(self, rx, ry, fee, amount_x):
        gamma = 1.0 - fee
        k = rx * ry
        new_rx = rx - amount_x
        new_ry = k / new_rx
        net_y = new_ry - ry
        expected_total = net_y / gamma

        amm = ConstantProductAMM("test", rx, ry, bid_fee=fee, ask_fee=fee)
        total_y, fee_amt = amm.quote_sell_x(amount_x)
        assert total_y == pytest.approx(expected_total, rel=1e-12)
        assert fee_amt == pytest.approx(expected_total - net_y, rel=1e-12)


class TestArbAndRouterFormulas:
    def test_cpmm_arbitrage_closed_form(self):
        rx, ry = 1000.0, 1000.0
        k = rx * ry
        fair_price = 1.2
        fee = 0.003
        gamma = 1.0 - fee

        expected_amount = rx - math.sqrt(k / (gamma * fair_price))
        amm = ConstantProductAMM("test", rx, ry, bid_fee=fee, ask_fee=fee)
        result = Arbitrageur().execute_arb(amm, fair_price, 0)

        assert result is not None
        assert result.amount_x == pytest.approx(expected_amount, rel=1e-9)

    def test_cpmm_router_formula(self):
        amm1 = ConstantProductAMM("a", 1000.0, 1000.0, bid_fee=0.003, ask_fee=0.003)
        amm2 = ConstantProductAMM("b", 1000.0, 1000.0, bid_fee=0.01, ask_fee=0.01)
        y1, y2 = OrderRouter().split_buy_two_amms(amm1, amm2, 100.0)
        assert y1 > y2


class TestSimulationProperties:
    def test_reset_with_same_seed_replays_episode(self):
        engine = SimulationEngine(SimConfig(n_steps=20))
        engine.reset(seed=42)
        first_run = [engine.step().fair_price for _ in range(10)]
        engine.reset(seed=42)
        second_run = [engine.step().fair_price for _ in range(10)]
        assert second_run == pytest.approx(first_run)

    def test_reset_without_seed_after_seeded_episode_is_not_deterministic(self):
        engine = SimulationEngine(SimConfig(n_steps=20))
        engine.reset(seed=42)
        seeded_run = [engine.step().fair_price for _ in range(10)]
        engine.reset()
        unseeded_run = [engine.step().fair_price for _ in range(10)]
        assert unseeded_run != seeded_run

    def test_zero_retail_rate_produces_no_orders_or_volume(self):
        engine = SimulationEngine(SimConfig(n_steps=50, retail_arrival_rate=0.0, seed=42))
        while not engine.done:
            result = engine.step()
            assert result.n_retail_orders == 0
            assert result.retail_volume_y["submission"] == 0.0
            assert result.retail_volume_y["normalizer"] == 0.0

    def test_aggressive_submission_changes_routed_volume(self):
        conservative = SimulationEngine(SimConfig(n_steps=300, seed=42))
        conservative.set_agent_action(np.array([-0.7, 0.7, 0.0, -0.7, 0.7, 0.0], dtype=np.float32))
        total_cons = 0.0
        while not conservative.done:
            result = conservative.step()
            total_cons += result.retail_volume_y["submission"]

        aggressive = SimulationEngine(SimConfig(n_steps=300, seed=42))
        aggressive.set_agent_action(np.array([0.7, -0.5, 0.0, 0.7, -0.5, 0.0], dtype=np.float32))
        total_aggr = 0.0
        while not aggressive.done:
            result = aggressive.step()
            total_aggr += result.retail_volume_y["submission"]

        assert total_aggr != total_cons

    def test_final_metrics_are_finite(self):
        engine = SimulationEngine(SimConfig(n_steps=500, seed=42))
        while not engine.done:
            result = engine.step()
        assert math.isfinite(result.edges["submission"])
        assert math.isfinite(result.pnls["submission"])
        assert math.isfinite(result.edges["normalizer"])
        assert math.isfinite(result.pnls["normalizer"])

    def test_volatility_schedule_switches_active_sigma_by_step(self):
        engine = SimulationEngine(
            SimConfig(
                n_steps=6,
                seed=42,
                volatility_schedule=((0, 0.001), (3, 0.005)),
            )
        )
        seen = []
        while not engine.done:
            result = engine.step()
            seen.append(result.active_sigma)
        assert seen[:3] == pytest.approx([0.001, 0.001, 0.001])
        assert seen[3:] == pytest.approx([0.005, 0.005, 0.005])

    def test_regime_shift_changes_realized_return_scale(self):
        engine = SimulationEngine(
            SimConfig(
                n_steps=120,
                seed=7,
                volatility_schedule=((0, 0.0005), (60, 0.004)),
            )
        )
        prices = [engine.current_fair_price]
        while not engine.done:
            prices.append(engine.step().fair_price)

        returns = np.diff(np.log(np.asarray(prices)))
        first_half = returns[:60]
        second_half = returns[60:]
        assert np.std(second_half) > np.std(first_half) * 2.0
