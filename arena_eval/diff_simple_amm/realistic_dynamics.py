"""Realistic-eval exogenous dynamics helpers for the diff simple-AMM stack."""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np

from arena_eval.exact_simple_amm.config import ExactSimpleAMMConfig
from arena_eval.diff_simple_amm.types import RealisticTape


@lru_cache(maxsize=8)
def _load_regime_invcdf(path: str) -> tuple[np.ndarray, np.ndarray]:
    rows = np.genfromtxt(Path(path), delimiter=",", names=True, dtype=float)
    pct_grid = np.asarray(rows["pct"], dtype=float)
    regime_columns = [name for name in rows.dtype.names if name.startswith("reg") and name.endswith("_bps")]
    regime_columns.sort()
    invcdf = np.column_stack([np.asarray(rows[name], dtype=float) * 1e-4 for name in regime_columns])
    return (pct_grid, invcdf)


@lru_cache(maxsize=8)
def _load_transition_matrix(path: str) -> np.ndarray:
    with Path(path).open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        matrix = [[float(value) for value in row[1:]] for row in reader]
    transition = np.asarray(matrix, dtype=float)
    row_sums = transition.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return transition / row_sums


@lru_cache(maxsize=8)
def _load_impact_percentiles(path: str, impact_column: str) -> tuple[np.ndarray, np.ndarray]:
    rows = np.genfromtxt(Path(path), delimiter=",", names=True, dtype=float)
    pct_grid = np.asarray(rows["pct"], dtype=float)
    values = np.asarray(rows[impact_column], dtype=float)
    return (pct_grid, values)


def load_realistic_artifacts(config: ExactSimpleAMMConfig) -> dict[str, np.ndarray]:
    """Load the empirical artifacts needed by the realistic evaluator."""

    if not config.regime_invcdf_path or not config.regime_transition_path or not config.retail_impact_percentiles_path:
        raise ValueError("realistic dynamics require regime and impact CSV paths")
    regime_pct_grid, regime_invcdf = _load_regime_invcdf(config.regime_invcdf_path)
    transition_matrix = _load_transition_matrix(config.regime_transition_path)
    impact_pct_grid, impact_values = _load_impact_percentiles(
        config.retail_impact_percentiles_path,
        config.retail_impact_column,
    )
    return {
        "regime_pct_grid": regime_pct_grid,
        "regime_invcdf": regime_invcdf,
        "transition_matrix": transition_matrix,
        "impact_pct_grid": impact_pct_grid,
        "impact_values": impact_values,
    }


def build_realistic_tape(*, config: ExactSimpleAMMConfig, seed: int) -> RealisticTape:
    """Materialize explicit realistic-mode randomness from the exact evaluator law."""

    artifacts = load_realistic_artifacts(config)
    transition = artifacts["transition_matrix"]
    regime_pct_grid = artifacts["regime_pct_grid"]
    regime_invcdf = artifacts["regime_invcdf"]
    impact_pct_grid = artifacts["impact_pct_grid"]
    impact_values = artifacts["impact_values"]

    price_rng = np.random.default_rng(seed)
    retail_rng = np.random.default_rng(seed + 1)
    smooth_rng = np.random.default_rng(seed + 101)

    n_regimes = int(transition.shape[0])
    regime = int(min(max(int(config.regime_start), 1), n_regimes))
    log_returns: list[float] = []
    regimes: list[int] = []
    return_percentiles: list[float] = []

    order_counts: list[int] = []
    impact_logs: list[tuple[float, ...]] = []
    max_orders_per_step = 0

    for _ in range(config.n_steps):
        transition_row = transition[regime - 1]
        regime = int(price_rng.choice(len(transition_row), p=transition_row)) + 1
        draw_pct = float(price_rng.random() * 100.0)
        log_return = float(np.interp(draw_pct, regime_pct_grid, regime_invcdf[:, regime - 1]))
        regimes.append(regime)
        return_percentiles.append(draw_pct)
        log_returns.append(log_return)

        count = int(retail_rng.poisson(config.retail_arrival_rate))
        order_counts.append(count)
        if count <= 0:
            impact_logs.append(())
            continue
        max_orders_per_step = max(max_orders_per_step, count)
        draw_pcts = retail_rng.random(size=count) * 100.0
        impacts = np.interp(draw_pcts, impact_pct_grid, impact_values)
        impact_logs.append(tuple(float(value) for value in impacts))

    width = max(max_orders_per_step, 1)
    smooth_arrival_uniforms = tuple(
        tuple(float(value) for value in smooth_rng.random(size=width))
        for _ in range(config.n_steps)
    )
    smooth_impact_percentiles = tuple(
        tuple(float(value) for value in (smooth_rng.random(size=width) * 100.0))
        for _ in range(config.n_steps)
    )

    return RealisticTape(
        log_returns=tuple(log_returns),
        regimes=tuple(regimes),
        return_percentiles=tuple(return_percentiles),
        order_counts=tuple(order_counts),
        impact_logs=tuple(impact_logs),
        max_orders_per_step=max_orders_per_step,
        smooth_arrival_uniforms=smooth_arrival_uniforms,
        smooth_impact_percentiles=smooth_impact_percentiles,
    )
