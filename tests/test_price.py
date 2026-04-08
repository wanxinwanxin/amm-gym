"""Tests for GBM price process."""

import math
import numpy as np
import pytest
from amm_gym.sim.price import GBMPriceProcess


class TestGBMFormula:
    def test_deterministic_with_seed(self):
        p1 = GBMPriceProcess(100.0, 0.0, 0.1, 1.0, seed=42)
        p2 = GBMPriceProcess(100.0, 0.0, 0.1, 1.0, seed=42)
        for _ in range(100):
            assert p1.step() == p2.step()

    def test_always_non_negative(self):
        """GBM produces positive prices; may underflow to 0.0 with extreme drift."""
        p = GBMPriceProcess(100.0, 0.0, 0.3, 1.0, seed=42)
        for _ in range(10_000):
            assert p.step() >= 0.0
            assert p.current_price >= 0.0

    def test_no_drift_mean(self):
        """With mu=0, E[S(t)] = S(0). Check over many samples."""
        n_trials = 50_000
        n_steps = 100
        final_prices = []
        for i in range(n_trials):
            p = GBMPriceProcess(100.0, 0.0, 0.01, 1.0, seed=i)
            for _ in range(n_steps):
                p.step()
            final_prices.append(p.current_price)

        mean_price = np.mean(final_prices)
        # E[S(T)] = S(0) * exp(mu*T) = 100 for mu=0
        assert mean_price == pytest.approx(100.0, rel=0.02)

    def test_volatility_scaling(self):
        """Higher sigma should produce wider distribution."""
        n = 10_000
        low_vol = []
        high_vol = []
        for i in range(n):
            p = GBMPriceProcess(100.0, 0.0, 0.01, 1.0, seed=i)
            for _ in range(100):
                p.step()
            low_vol.append(p.current_price)

            p2 = GBMPriceProcess(100.0, 0.0, 0.05, 1.0, seed=i)
            for _ in range(100):
                p2.step()
            high_vol.append(p2.current_price)

        assert np.std(high_vol) > np.std(low_vol)

    def test_reset(self):
        p = GBMPriceProcess(100.0, 0.0, 0.1, 1.0, seed=42)
        prices_a = [p.step() for _ in range(10)]

        p.reset(100.0, seed=42)
        prices_b = [p.step() for _ in range(10)]

        np.testing.assert_array_equal(prices_a, prices_b)
