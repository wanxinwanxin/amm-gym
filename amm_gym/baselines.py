"""Baseline depth-ladder policies for benchmarking."""

from __future__ import annotations

import numpy as np


class StaticDepthPolicy:
    """Always returns the same 6D ladder control."""

    def __init__(
        self,
        bid_scale: float = 0.0,
        ask_scale: float = 0.0,
        bid_decay: float = 0.0,
        ask_decay: float = 0.0,
        bid_tilt: float = 0.0,
        ask_tilt: float = 0.0,
    ) -> None:
        self.action = np.array(
            [bid_scale, bid_decay, bid_tilt, ask_scale, ask_decay, ask_tilt],
            dtype=np.float32,
        )

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.action.copy()


class InventoryAwareDepthPolicy:
    """Simple heuristic that skews depth away from the heavier inventory side."""

    def __init__(self, base_scale: float = 0.0, base_decay: float = 0.0) -> None:
        self.base_scale = float(base_scale)
        self.base_decay = float(base_decay)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        ws = len(obs) - 15
        imbalance = float(obs[ws + 2]) if len(obs) > ws + 2 else 0.0
        bid_scale = np.clip(self.base_scale - imbalance, -1.0, 1.0)
        ask_scale = np.clip(self.base_scale + imbalance, -1.0, 1.0)
        bid_tilt = np.clip(-0.5 * imbalance, -1.0, 1.0)
        ask_tilt = np.clip(0.5 * imbalance, -1.0, 1.0)
        return np.array(
            [
                bid_scale,
                self.base_decay,
                bid_tilt,
                ask_scale,
                self.base_decay,
                ask_tilt,
            ],
            dtype=np.float32,
        )


def benchmark_depth_policies() -> dict[str, StaticDepthPolicy]:
    """Small hand-authored policy set for builder-facing benchmarks."""
    return {
        "balanced": StaticDepthPolicy(),
        "aggressive_near_mid": StaticDepthPolicy(
            bid_scale=0.6,
            ask_scale=0.6,
            bid_decay=0.7,
            ask_decay=0.7,
            bid_tilt=0.2,
            ask_tilt=0.2,
        ),
        "defensive_tail_thin": StaticDepthPolicy(
            bid_scale=-0.2,
            ask_scale=-0.2,
            bid_decay=1.0,
            ask_decay=1.0,
            bid_tilt=-0.2,
            ask_tilt=-0.2,
        ),
        "inventory_long_bias": StaticDepthPolicy(
            bid_scale=-0.5,
            ask_scale=0.5,
            bid_decay=0.2,
            ask_decay=0.4,
            bid_tilt=-0.3,
            ask_tilt=0.3,
        ),
    }
