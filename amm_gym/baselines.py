"""Baseline fee policies for benchmarking."""

from __future__ import annotations

import numpy as np


class StaticFeePolicy:
    """Always returns the same bid/ask fee."""

    def __init__(self, fee_bps: float) -> None:
        self.fee = fee_bps / 10_000.0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return np.array([self.fee, self.fee], dtype=np.float32)


class DecayHeuristic:
    """Widen fees after large trades, decay toward base fee.

    After each step, if recent volume is above average, increase fees.
    Otherwise decay toward base_fee_bps.
    """

    def __init__(
        self,
        base_fee_bps: float = 30.0,
        max_fee_bps: float = 100.0,
        widen_factor: float = 1.5,
        decay_rate: float = 0.95,
    ) -> None:
        self.base_fee = base_fee_bps / 10_000.0
        self.max_fee = max_fee_bps / 10_000.0
        self.widen_factor = widen_factor
        self.decay_rate = decay_rate
        self.current_fee = self.base_fee

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        # obs[ws+4] is EMA of recent retail volume (normalized)
        # obs[ws+5] is EMA of recent trade count (normalized)
        # Use trade count as proxy for "high activity"
        # Default window_size=10, so index 14+5=15 is count
        # We just use a simple heuristic based on the normalized count
        ws = len(obs) - 11  # infer window_size
        ema_count_norm = obs[ws + 5] if len(obs) > ws + 5 else 0.5

        if ema_count_norm > 1.0:
            # Above average activity -> widen
            self.current_fee = min(
                self.current_fee * self.widen_factor, self.max_fee
            )
        else:
            # Decay toward base
            self.current_fee = (
                self.decay_rate * self.current_fee
                + (1 - self.decay_rate) * self.base_fee
            )

        fee = float(np.clip(self.current_fee, 0.0001, 0.10))
        return np.array([fee, fee], dtype=np.float32)
