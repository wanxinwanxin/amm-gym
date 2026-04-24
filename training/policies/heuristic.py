"""Research-facing heuristic policies."""

from __future__ import annotations

import numpy as np

from amm_gym.baselines import benchmark_depth_policies


def _clip_action(values: list[float]) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float32), -1.0, 1.0)


class InventorySkewPolicy:
    """Skew the ladder away from the heavier inventory side."""

    def __init__(self, scale_gain: float = 1.25, tilt_gain: float = 0.8, decay: float = 0.25) -> None:
        self.scale_gain = float(scale_gain)
        self.tilt_gain = float(tilt_gain)
        self.decay = float(decay)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        ws = len(obs) - 15
        imbalance = float(obs[ws + 2])
        bid_scale = -self.scale_gain * imbalance
        ask_scale = self.scale_gain * imbalance
        bid_tilt = -self.tilt_gain * imbalance
        ask_tilt = self.tilt_gain * imbalance
        return _clip_action([bid_scale, self.decay, bid_tilt, ask_scale, self.decay, ask_tilt])


class VolatilityAdaptivePolicy:
    """Shift from near-touch quoting to safer tails in high realized volatility."""

    def __init__(self, vol_threshold: float = 0.002, low_vol_scale: float = 0.45, high_vol_scale: float = -0.15) -> None:
        self.vol_threshold = float(vol_threshold)
        self.low_vol_scale = float(low_vol_scale)
        self.high_vol_scale = float(high_vol_scale)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        ws = len(obs) - 15
        rolling_vol = float(obs[ws + 13])
        progress = float(obs[ws + 14])
        scale = self.high_vol_scale if rolling_vol >= self.vol_threshold else self.low_vol_scale
        decay = np.interp(progress, [0.0, 1.0], [0.55, 0.15])
        tilt = np.clip((rolling_vol - self.vol_threshold) * 220.0, -0.35, 0.35)
        return _clip_action([scale, decay, -tilt, scale, decay, tilt])


class FlowAwarePolicy:
    """React to lagged flow and execution pressure, especially late in the episode."""

    def __init__(self, flow_gain: float = 12.0, inventory_gain: float = 0.7) -> None:
        self.flow_gain = float(flow_gain)
        self.inventory_gain = float(inventory_gain)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        ws = len(obs) - 15
        imbalance = float(obs[ws + 2])
        exec_rate = float(obs[ws + 5])
        net_flow = float(obs[ws + 6])
        progress = float(obs[ws + 14])

        base_scale = np.clip(0.15 + 0.8 * exec_rate - 0.25 * progress, -0.5, 0.7)
        skew = np.clip(-self.flow_gain * net_flow - self.inventory_gain * imbalance, -0.85, 0.85)
        base_decay = np.clip(0.55 - 0.35 * exec_rate + 0.25 * progress, -0.3, 0.9)
        return _clip_action(
            [
                base_scale + skew,
                base_decay,
                0.5 * skew,
                base_scale - skew,
                base_decay,
                -0.5 * skew,
            ]
        )


def research_benchmark_policies() -> dict[str, object]:
    """Policy registry used by the research benchmark harness."""
    policies = benchmark_depth_policies()
    policies.update(
        {
            "inventory_skew": InventorySkewPolicy(),
            "volatility_adaptive": VolatilityAdaptivePolicy(),
            "flow_aware": FlowAwarePolicy(),
        }
    )
    return policies
