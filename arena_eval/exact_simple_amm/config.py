"""Configuration for the exact simple AMM evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT / "analysis" / "weth_usdc_90d"
DEFAULT_REGIME_INVCDF_PATH = ANALYSIS_DIR / "regimes_invcdf.csv"
DEFAULT_REGIME_TRANSITION_PATH = ANALYSIS_DIR / "regimes_transition_matrix.csv"
DEFAULT_RETAIL_IMPACT_PERCENTILES_PATH = ANALYSIS_DIR / "percentiles.csv"
EMPIRICAL_ROUTER_ARRIVAL_RATE = 186_085 / 645_123

@dataclass(frozen=True)
class ExactSimpleAMMConfig:
    n_steps: int = 10_000
    initial_price: float = 100.0
    initial_x: float = 100.0
    initial_y: float = 10_000.0
    gbm_mu: float = 0.0
    gbm_sigma: float = 0.000945
    gbm_dt: float = 1.0
    retail_arrival_rate: float = 0.8
    retail_mean_size: float = 20.0
    retail_size_sigma: float = 1.2
    retail_buy_prob: float = 0.5
    evaluator_kind: str = "challenge"
    price_process_kind: str = "gbm"
    retail_flow_kind: str = "lognormal_size"
    regime_invcdf_path: str | None = None
    regime_transition_path: str | None = None
    regime_start: int = 3
    retail_impact_percentiles_path: str | None = None
    retail_impact_column: str = "router_impact_log"
    retail_impact_reference_venue: str = "normalizer"
    retail_impact_scale_mode: str = "current_state"

    @classmethod
    def from_seed(cls, seed: int) -> "ExactSimpleAMMConfig":
        rng = np.random.default_rng(seed)
        return cls(
            gbm_sigma=float(rng.uniform(0.000882, 0.001008)),
            retail_arrival_rate=float(rng.uniform(0.6, 1.0)),
            retail_mean_size=float(rng.uniform(19.0, 21.0)),
        )

    @classmethod
    def real_data_from_seed(cls, seed: int) -> "ExactSimpleAMMConfig":
        return cls(
            evaluator_kind="real_data",
            price_process_kind="regime_switching",
            retail_flow_kind="empirical_impact",
            retail_arrival_rate=float(EMPIRICAL_ROUTER_ARRIVAL_RATE),
            regime_invcdf_path=str(DEFAULT_REGIME_INVCDF_PATH),
            regime_transition_path=str(DEFAULT_REGIME_TRANSITION_PATH),
            retail_impact_percentiles_path=str(DEFAULT_RETAIL_IMPACT_PERCENTILES_PATH),
        )

    @classmethod
    def for_evaluator(cls, seed: int, evaluator_kind: str) -> "ExactSimpleAMMConfig":
        if evaluator_kind == "challenge":
            return cls.from_seed(seed)
        if evaluator_kind == "real_data":
            return cls.real_data_from_seed(seed)
        raise ValueError(f"Unsupported evaluator_kind: {evaluator_kind}")
