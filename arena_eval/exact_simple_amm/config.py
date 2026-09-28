"""Configuration for the exact simple AMM evaluator."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from arena_eval.exact_simple_amm.dynamics import RetailVolScaling

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT / "analysis" / "weth_usdc_90d"
DEFAULT_REGIME_INVCDF_PATH = ANALYSIS_DIR / "regimes_invcdf.csv"
DEFAULT_REGIME_TRANSITION_PATH = ANALYSIS_DIR / "regimes_transition_matrix.csv"
DEFAULT_RETAIL_IMPACT_PERCENTILES_PATH = ANALYSIS_DIR / "percentiles.csv"
DEFAULT_RETAIL_USD_QUANTILES_PATH = ANALYSIS_DIR / "parent_order_usd_quantiles.csv"
EMPIRICAL_ROUTER_ARRIVAL_RATE = 186_085 / 645_123
# Strict-retail cohort: Uniswap first-party FE (web/mobile/extension, via the
# x-request-source surface tag in uniswap-labs.core.swaps) UNION MetaMask Swaps
# (87.5bps fee) — the SAME cohort used for §4 calibration and §5 validation —
# over the 30-day window 2026-04-21..2026-05-20. Replaces the broad 19-router
# cohort (6m values were 857_035/1_303_200 ≈ 0.6576/block, buy_prob 0.4413; the
# 19-router method is kept for robustness checks). From
# analysis/weth_usdc_90d/sql/parent_order_size_strict_30d.sql:
#   parent_count = 98_676, horizon_days = 30 -> 98_676/216_000 = 0.4568 / block
#   buy_eth / (buy_eth + sell_eth) = 45_659 / 98_676 = 0.4627
EMPIRICAL_PARENT_ORDER_ARRIVAL_RATE = 98_676 / 216_000
EMPIRICAL_PARENT_ORDER_BUY_PROB = 0.4627
# Volatility elasticity of retail flow. Uniswap front-end ETH/USDC swaps on mainnet
# (uniswap-labs.core.swaps, completed, 2024-11-07..2026-09-21) regressed in logs on the
# day's realized 12 s Binance ETHUSDT volatility, detrended with a 61-day rolling
# median: daily swap count elasticity 0.45, daily USD volume 1.2, so mean order size
# scales with exponent 1.2 - 0.45 = 0.75. Volatile-tercile days carry 1.6x the swaps
# and 2.8x the USD of calm-tercile days. Reference 3.7 bps/step is the mean day vol of
# the 2023-11..2025-11 replay sample, so the arrival multiplier averages ~1 over days.
# See analysis/weth_usdc_90d/retail_vol_elasticity.md and
# scripts/calibration/fit_retail_vol_elasticity.py.
EMPIRICAL_RETAIL_ARRIVAL_VOL_ELASTICITY = 0.45
EMPIRICAL_RETAIL_SIZE_VOL_ELASTICITY = 0.75
EMPIRICAL_RETAIL_VOL_REFERENCE_BPS = 3.7

@dataclass(frozen=True)
class ExactSimpleAMMConfig:
    n_steps: int = 10_000
    step_seconds: float = 12.0
    initial_price: float = 100.0
    initial_x: float = 100.0
    initial_y: float = 10_000.0
    submission_liquidity_fraction: float = 1.0
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
    retail_usd_quantiles_path: str | None = None
    # When True, the normalizer pool is re-synced to the prevailing fair price at
    # the top of every step (constant USDC depth D = normalizer_initial_y,
    # reserve_x = D / fair) instead of being arbed toward fair. It then represents
    # the efficient "rest of the market" — always correctly priced when retail
    # arrives, and no arbitrage ever hits it. Challenge mode leaves this False
    # (there the normalizer is a real competing baseline pool that must be arbed).
    normalizer_tracks_fair: bool = False
    # Arbitrageur passes against the submission pool per arb event. >1 lets the arb
    # re-trade as a per-swap dynamic fee decays (e.g. the Nezlobin surcharge). Default
    # 1 reproduces the single-pass behavior of every existing config exactly.
    arb_max_passes: int = 1
    # When True, run the submission arbitrageur a SECOND time at the bottom of the
    # block (after retail), realigning to fair. Needed by intra-block dynamic-fee
    # strategies whose surcharge is paid by the same-block backrun. Default False
    # leaves existing configs unchanged (arb only at the top of the block).
    bottom_of_block_arb: bool = False
    # Volatility elasticity of retail flow. Default 0/0 => constant-rate retail, bit-for-bit
    # unchanged. When either elasticity is non-zero the simulator tracks an EWMA of squared
    # per-step log fair-price returns (half-life retail_vol_half_life_steps; 1800 steps =
    # 6 h at 12 s) and hands the retail generator vol_bps = sqrt(EWMA) * 1e4 every step.
    # The generator scales its Poisson rate by ratio ** retail_arrival_vol_elasticity and
    # its USD order sizes by ratio ** retail_size_vol_elasticity, with
    # ratio = clip(vol_bps / retail_vol_reference_bps, retail_vol_ratio_min, retail_vol_ratio_max).
    # The tracker starts at retail_vol_initial_bps (None => the reference, multiplier 1);
    # pass the previous day's realized vol to avoid a warm-up bias in short runs.
    # Calibrated values: EMPIRICAL_RETAIL_*_VOL_ELASTICITY above.
    retail_arrival_vol_elasticity: float = 0.0
    retail_size_vol_elasticity: float = 0.0
    retail_vol_reference_bps: float = EMPIRICAL_RETAIL_VOL_REFERENCE_BPS
    retail_vol_ratio_min: float = 0.25
    retail_vol_ratio_max: float = 4.0
    retail_vol_half_life_steps: float = 1800.0
    retail_vol_initial_bps: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.submission_liquidity_fraction) or self.submission_liquidity_fraction <= 0.0:
            raise ValueError("submission_liquidity_fraction must be positive and finite")
        self.retail_vol_scaling  # validates the elasticity / reference / clip fields
        if not (math.isfinite(self.retail_vol_half_life_steps) and self.retail_vol_half_life_steps > 0.0):
            raise ValueError("retail_vol_half_life_steps must be positive and finite")
        if self.retail_vol_initial_bps is not None and not (
            math.isfinite(self.retail_vol_initial_bps) and self.retail_vol_initial_bps >= 0.0
        ):
            raise ValueError("retail_vol_initial_bps must be non-negative and finite")

    @property
    def retail_vol_scaling(self) -> RetailVolScaling:
        return RetailVolScaling(
            arrival_elasticity=float(self.retail_arrival_vol_elasticity),
            size_elasticity=float(self.retail_size_vol_elasticity),
            reference_bps=float(self.retail_vol_reference_bps),
            ratio_min=float(self.retail_vol_ratio_min),
            ratio_max=float(self.retail_vol_ratio_max),
        )

    @property
    def submission_initial_x(self) -> float:
        return float(self.initial_x) * float(self.submission_liquidity_fraction)

    @property
    def submission_initial_y(self) -> float:
        return float(self.initial_y) * float(self.submission_liquidity_fraction)

    @property
    def normalizer_initial_x(self) -> float:
        return float(self.initial_x)

    @property
    def normalizer_initial_y(self) -> float:
        return float(self.initial_y)

    @property
    def submission_initial_value(self) -> float:
        return self.submission_initial_x * float(self.initial_price) + self.submission_initial_y

    @property
    def normalizer_initial_value(self) -> float:
        return self.normalizer_initial_x * float(self.initial_price) + self.normalizer_initial_y

    @classmethod
    def from_seed(cls, seed: int) -> "ExactSimpleAMMConfig":
        rng = np.random.default_rng(seed)
        return cls(
            gbm_sigma=float(rng.uniform(0.000882, 0.001008)),
            retail_arrival_rate=float(rng.uniform(0.6, 1.0)),
            retail_mean_size=float(rng.uniform(19.0, 21.0)),
        )

    @classmethod
    def real_data_from_seed(
        cls,
        seed: int,
        retail_mode: str = "empirical_impact",
        *,
        vol_elastic_retail: bool = False,
    ) -> "ExactSimpleAMMConfig":
        """``vol_elastic_retail=True`` (empirical_usd_size only) switches on the calibrated
        volatility elasticity of retail arrivals and sizes; the default keeps the
        constant-rate flow every existing config uses."""
        if retail_mode == "empirical_usd_size":
            vol_kwargs = (
                dict(
                    retail_arrival_vol_elasticity=EMPIRICAL_RETAIL_ARRIVAL_VOL_ELASTICITY,
                    retail_size_vol_elasticity=EMPIRICAL_RETAIL_SIZE_VOL_ELASTICITY,
                    retail_vol_reference_bps=EMPIRICAL_RETAIL_VOL_REFERENCE_BPS,
                )
                if vol_elastic_retail
                else {}
            )
            return cls(
                evaluator_kind="real_data",
                price_process_kind="regime_switching",
                retail_flow_kind="empirical_usd_size",
                retail_arrival_rate=float(EMPIRICAL_PARENT_ORDER_ARRIVAL_RATE),
                retail_buy_prob=EMPIRICAL_PARENT_ORDER_BUY_PROB,
                regime_invcdf_path=str(DEFAULT_REGIME_INVCDF_PATH),
                regime_transition_path=str(DEFAULT_REGIME_TRANSITION_PATH),
                retail_usd_quantiles_path=str(DEFAULT_RETAIL_USD_QUANTILES_PATH),
                normalizer_tracks_fair=True,
                **vol_kwargs,
            )
        if vol_elastic_retail:
            raise ValueError("vol_elastic_retail is only supported with retail_mode='empirical_usd_size'")
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
