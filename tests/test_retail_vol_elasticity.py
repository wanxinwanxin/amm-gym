"""Volatility elasticity of retail flow: ``RetailVolScaling`` in the retail generators and
the EWMA volatility tracker in ``ExactSimpleAMMSimulator``.

Defaults (both elasticities 0) must reproduce the constant-rate flow exactly; the
opt-in path must scale arrivals and sizes as a power of (vol / reference)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from arena_eval.exact_simple_amm.config import (
    DEFAULT_RETAIL_USD_QUANTILES_PATH,
    EMPIRICAL_RETAIL_ARRIVAL_VOL_ELASTICITY,
    EMPIRICAL_RETAIL_SIZE_VOL_ELASTICITY,
    ExactSimpleAMMConfig,
)
from arena_eval.exact_simple_amm.dynamics import (
    EmpiricalImpactRetailTrader,
    EmpiricalUSDSizeRetailTrader,
    RetailVolScaling,
)
from arena_eval.exact_simple_amm.simulator import ExactSimpleAMMSimulator, RetailTrader
from arena_eval.exact_simple_amm.strategies import FixedFeeStrategy

QUANTILES = str(DEFAULT_RETAIL_USD_QUANTILES_PATH)
REF = 3.7


# ---------------------------------------------------------------------------------
# RetailVolScaling
# ---------------------------------------------------------------------------------
def test_default_scaling_is_inactive_identity():
    scaling = RetailVolScaling()
    assert not scaling.active
    assert scaling.multipliers(None) == (1.0, 1.0)
    assert scaling.multipliers(0.0) == (1.0, 1.0)
    assert scaling.multipliers(1e9) == (1.0, 1.0)


def test_multipliers_follow_power_law_and_clip():
    scaling = RetailVolScaling(arrival_elasticity=0.45, size_elasticity=0.75, reference_bps=REF)
    assert scaling.active
    assert scaling.multipliers(REF) == (1.0, 1.0)
    arrival, size = scaling.multipliers(2.0 * REF)
    assert arrival == pytest.approx(2.0**0.45)
    assert size == pytest.approx(2.0**0.75)
    # clipped at ratio_max=4 and ratio_min=0.25
    assert scaling.multipliers(1e6) == pytest.approx((4.0**0.45, 4.0**0.75))
    assert scaling.multipliers(0.0) == pytest.approx((0.25**0.45, 0.25**0.75))
    # a non-finite estimate never changes the flow
    assert scaling.multipliers(float("nan")) == (1.0, 1.0)


def test_scaling_validation():
    with pytest.raises(ValueError):
        RetailVolScaling(reference_bps=0.0)
    with pytest.raises(ValueError):
        RetailVolScaling(ratio_min=2.0, ratio_max=1.0)
    with pytest.raises(ValueError):
        RetailVolScaling(arrival_elasticity=float("inf"))


# ---------------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------------
def test_zero_elasticity_reproduces_constant_rate_orders_exactly():
    vols = np.random.default_rng(0).uniform(0.2, 25.0, size=300).tolist()

    base = EmpiricalUSDSizeRetailTrader(0.8, QUANTILES, buy_prob=0.46, seed=3)
    same = EmpiricalUSDSizeRetailTrader(0.8, QUANTILES, buy_prob=0.46, seed=3, vol_scaling=RetailVolScaling())
    assert [base.generate_orders(fair_price=100.0) for _ in vols] == [
        same.generate_orders(fair_price=100.0, vol_bps=v) for v in vols
    ]

    base_ln = RetailTrader(1.1, 20.0, 1.2, 0.5, seed=4)
    same_ln = RetailTrader(1.1, 20.0, 1.2, 0.5, seed=4, vol_scaling=RetailVolScaling())
    assert [base_ln.generate_orders(fair_price=100.0) for _ in vols] == [
        same_ln.generate_orders(fair_price=100.0, vol_bps=v) for v in vols
    ]


def test_size_elasticity_scales_sizes_deterministically():
    ref = EmpiricalUSDSizeRetailTrader(2.0, QUANTILES, seed=5)
    hot = EmpiricalUSDSizeRetailTrader(
        2.0, QUANTILES, seed=5, vol_scaling=RetailVolScaling(size_elasticity=1.0, reference_bps=REF)
    )
    n_seen = 0
    for _ in range(100):
        a = ref.generate_orders(fair_price=100.0)
        b = hot.generate_orders(fair_price=100.0, vol_bps=2.0 * REF)  # same Poisson draw, sizes x2
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert y.side == x.side
            assert y.size == pytest.approx(2.0 * x.size, rel=1e-12)
        n_seen += len(a)
    assert n_seen > 100


def test_arrival_elasticity_scales_poisson_rate():
    trader = EmpiricalUSDSizeRetailTrader(
        1.0, QUANTILES, seed=11, vol_scaling=RetailVolScaling(arrival_elasticity=1.0, reference_bps=REF)
    )
    n = 20_000
    at_ref = sum(len(trader.generate_orders(fair_price=100.0, vol_bps=REF)) for _ in range(n)) / n
    at_2x = sum(len(trader.generate_orders(fair_price=100.0, vol_bps=2.0 * REF)) for _ in range(n)) / n
    at_half = sum(len(trader.generate_orders(fair_price=100.0, vol_bps=0.5 * REF)) for _ in range(n)) / n
    assert at_ref == pytest.approx(1.0, abs=0.04)
    assert at_2x == pytest.approx(2.0, abs=0.06)
    assert at_half == pytest.approx(0.5, abs=0.03)


def test_impact_trader_takes_arrival_elasticity_only():
    with pytest.raises(ValueError):
        EmpiricalImpactRetailTrader(
            0.5,
            "unused.csv",
            initial_x=100.0,
            initial_y=10_000.0,
            vol_scaling=RetailVolScaling(size_elasticity=0.5),
        )


# ---------------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------------
def test_config_defaults_off_and_validated():
    cfg = ExactSimpleAMMConfig()
    assert not cfg.retail_vol_scaling.active
    with pytest.raises(ValueError):
        ExactSimpleAMMConfig(retail_vol_reference_bps=0.0)
    with pytest.raises(ValueError):
        ExactSimpleAMMConfig(retail_vol_half_life_steps=0.0)
    with pytest.raises(ValueError):
        ExactSimpleAMMConfig(retail_vol_ratio_min=3.0, retail_vol_ratio_max=2.0)
    with pytest.raises(ValueError):
        ExactSimpleAMMConfig(retail_vol_initial_bps=-1.0)


def test_real_data_config_opt_in():
    plain = ExactSimpleAMMConfig.real_data_from_seed(1, "empirical_usd_size")
    assert not plain.retail_vol_scaling.active
    elastic = ExactSimpleAMMConfig.real_data_from_seed(1, "empirical_usd_size", vol_elastic_retail=True)
    assert elastic.retail_vol_scaling == RetailVolScaling(
        arrival_elasticity=EMPIRICAL_RETAIL_ARRIVAL_VOL_ELASTICITY,
        size_elasticity=EMPIRICAL_RETAIL_SIZE_VOL_ELASTICITY,
        reference_bps=elastic.retail_vol_reference_bps,
    )
    # every other field is untouched
    assert replace(elastic, retail_arrival_vol_elasticity=0.0, retail_size_vol_elasticity=0.0) == plain
    with pytest.raises(ValueError):
        ExactSimpleAMMConfig.real_data_from_seed(1, "empirical_impact", vol_elastic_retail=True)


# ---------------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------------
def _run(cfg: ExactSimpleAMMConfig, seed: int):
    sim = ExactSimpleAMMSimulator(
        config=cfg,
        submission_strategy=FixedFeeStrategy(0.0005),
        normalizer_strategy=FixedFeeStrategy(0.0030),
        seed=seed,
    )
    records = []
    while not sim.done:
        records.append(sim.step_once())
    return sim.result(), records


def test_simulator_with_zero_elasticity_is_unchanged_by_tracker_settings():
    base = ExactSimpleAMMConfig(n_steps=400, retail_arrival_rate=1.5)
    tweaked = replace(base, retail_vol_reference_bps=0.9, retail_vol_half_life_steps=7.0, retail_vol_initial_bps=40.0)
    res_a, rec_a = _run(base, 21)
    res_b, rec_b = _run(tweaked, 21)
    assert res_a == res_b
    assert [r["n_orders"] for r in rec_a] == [r["n_orders"] for r in rec_b]
    assert [r["trade_events"] for r in rec_a] == [r["trade_events"] for r in rec_b]
    # the tracker still ran and reports a volatility estimate every step
    assert rec_a[0]["retail_vol_bps"] == pytest.approx(REF)  # no return yet: seeded at the reference
    assert rec_b[0]["retail_vol_bps"] == pytest.approx(40.0)  # ... or at retail_vol_initial_bps
    assert all(np.isfinite(r["retail_vol_bps"]) for r in rec_a)


def test_simulator_ewma_tracks_gbm_volatility():
    sigma = 0.0005  # 5 bps per step
    cfg = ExactSimpleAMMConfig(
        n_steps=6000,
        gbm_sigma=sigma,
        retail_arrival_rate=0.3,
        retail_vol_half_life_steps=300.0,
        retail_arrival_vol_elasticity=0.45,
    )
    _, records = _run(cfg, 5)
    vol = np.asarray([r["retail_vol_bps"] for r in records])
    assert vol[0] == pytest.approx(REF)
    tail = vol[-2000:]
    assert tail.mean() == pytest.approx(sigma * 1e4, rel=0.12)
    assert tail.min() > 2.5 and tail.max() < 9.0


def test_simulator_sends_more_retail_when_volatile():
    common = dict(
        n_steps=5000,
        retail_arrival_rate=0.5,
        retail_vol_half_life_steps=100.0,
        retail_arrival_vol_elasticity=1.0,
        retail_size_vol_elasticity=1.0,
        retail_vol_reference_bps=REF,
    )
    calm = ExactSimpleAMMConfig(gbm_sigma=0.0002, **common)  # 2 bps/step
    wild = ExactSimpleAMMConfig(gbm_sigma=0.0008, **common)  # 8 bps/step
    res_calm, rec_calm = _run(calm, 1)
    res_wild, rec_wild = _run(wild, 1)
    n_calm = sum(r["n_orders"] for r in rec_calm)
    n_wild = sum(r["n_orders"] for r in rec_wild)
    # expected rate ratio ~ (8/2) = 4 after warm-up; be loose on the Poisson noise
    assert 2.5 < n_wild / n_calm < 5.5
    # a 0.5/step rate at the reference => calm days run below it, wild days above it
    assert n_calm / calm.n_steps < 0.5 < n_wild / wild.n_steps
    assert res_wild.retail_volume_submission_y + res_wild.retail_volume_normalizer_y > (
        res_calm.retail_volume_submission_y + res_calm.retail_volume_normalizer_y
    )
