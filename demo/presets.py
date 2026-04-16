"""Shared presets for hackathon-facing demo scripts."""

from __future__ import annotations

from amm_gym.sim.engine import SimConfig


DEFAULT_DEMO_SEED = 7
DEFAULT_DEMO_STEPS = 120
DEFAULT_WINDOW_SIZE = 10
DEFAULT_VOLATILITY_SCHEDULE: tuple[tuple[int, float], ...] = (
    (0, 0.0010),
    (40, 0.0035),
    (80, 0.0015),
)


def named_schedules() -> dict[str, tuple[tuple[int, float], ...] | None]:
    return {
        "constant_low_vol": None,
        "regime_shift": DEFAULT_VOLATILITY_SCHEDULE,
    }


def build_hackathon_demo_config(
    *,
    seed: int = DEFAULT_DEMO_SEED,
    steps: int = DEFAULT_DEMO_STEPS,
    schedule: tuple[tuple[int, float], ...] | None = DEFAULT_VOLATILITY_SCHEDULE,
) -> SimConfig:
    """Canonical config used across demo and training visuals."""
    return SimConfig(
        n_steps=steps,
        gbm_sigma=0.0010,
        volatility_schedule=schedule,
        retail_arrival_rate=6.0,
        retail_mean_size=2.4,
        retail_size_sigma=0.75,
        retail_buy_prob=0.5,
        submission_base_notional_y=1_200.0,
        seed=seed,
    )
