"""Geometric Brownian Motion price process.

Matches amm_sim_rs/src/market/price_process.rs exactly.
Uses numpy RNG (not identical to Rust's Pcg64, so sequences differ
even with the same seed — but the formula is the same).
"""

from __future__ import annotations

import math

import numpy as np


class GBMPriceProcess:
    """GBM: dS = mu*S*dt + sigma*S*dW

    Discrete: S(t+1) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    """

    def __init__(
        self,
        initial_price: float,
        mu: float,
        sigma: float,
        dt: float,
        seed: int | None = None,
    ) -> None:
        self.current_price = initial_price
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self._set_sigma(sigma)
        self.rng = np.random.default_rng(seed)

    def _set_sigma(self, sigma: float) -> None:
        self.sigma = float(sigma)
        self.drift_term = (self.mu - 0.5 * self.sigma * self.sigma) * self.dt
        self.vol_term = self.sigma * math.sqrt(self.dt)

    def step(self, sigma: float | None = None) -> float:
        """Generate next price and return it."""
        if sigma is not None and sigma != self.sigma:
            self._set_sigma(sigma)
        z = self.rng.standard_normal()
        exponent = self.drift_term + self.vol_term * z
        self.current_price *= math.exp(exponent)
        return self.current_price

    def reset(self, initial_price: float, seed: int | None = None) -> None:
        self.current_price = initial_price
        if seed is not None:
            self.rng = np.random.default_rng(seed)
