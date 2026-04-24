"""Alternative exogenous dynamics for the exact simple AMM evaluator."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Protocol

import numpy as np


class _ReferenceAMM(Protocol):
    reserve_x: float
    reserve_y: float
    bid_fee: float
    ask_fee: float

    @property
    def spot_price(self) -> float: ...


class RegimeSwitchingReturnProcess:
    """Empirical regime-switching log-return process sampled at 12s cadence."""

    def __init__(
        self,
        initial_price: float,
        invcdf_path: str | Path,
        trans_matrix_path: str | Path,
        *,
        start_regime: int = 3,
        seed: int | None = None,
    ) -> None:
        self.current_price = float(initial_price)
        self.invcdf_path = Path(invcdf_path)
        self.trans_matrix_path = Path(trans_matrix_path)
        self.pct_grid, self.invcdf = self._load_invcdf(self.invcdf_path)
        self.transition_matrix = self._load_transition_matrix(self.trans_matrix_path)
        self.regime = int(min(max(start_regime, 1), self.transition_matrix.shape[0]))
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _load_invcdf(path: Path) -> tuple[np.ndarray, np.ndarray]:
        rows = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
        pct_grid = np.asarray(rows["pct"], dtype=float)
        regime_columns = [name for name in rows.dtype.names if name.startswith("reg") and name.endswith("_bps")]
        regime_columns.sort()
        invcdf = np.column_stack([np.asarray(rows[name], dtype=float) * 1e-4 for name in regime_columns])
        return pct_grid, invcdf

    @staticmethod
    def _load_transition_matrix(path: Path) -> np.ndarray:
        with path.open(newline="") as handle:
            reader = csv.reader(handle)
            next(reader)
            matrix = [[float(value) for value in row[1:]] for row in reader]
        transition = np.asarray(matrix, dtype=float)
        row_sums = transition.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        return transition / row_sums

    def step(self, sigma: float | None = None) -> float:
        del sigma
        row = self.transition_matrix[self.regime - 1]
        next_regime = int(self.rng.choice(len(row), p=row)) + 1
        draw_pct = float(self.rng.random() * 100.0)
        log_return = float(np.interp(draw_pct, self.pct_grid, self.invcdf[:, next_regime - 1]))
        self.current_price *= math.exp(log_return)
        self.regime = next_regime
        return self.current_price

    def reset(self, initial_price: float, seed: int | None = None) -> None:
        self.current_price = float(initial_price)
        if seed is not None:
            self.rng = np.random.default_rng(seed)


class EmpiricalImpactRetailTrader:
    """Retail generator that samples target price impacts from empirical data."""

    def __init__(
        self,
        arrival_rate: float,
        impact_percentiles_path: str | Path,
        *,
        impact_column: str = "router_impact_log",
        reference_venue: str = "normalizer",
        scale_mode: str = "current_state",
        initial_x: float,
        initial_y: float,
        seed: int | None = None,
    ) -> None:
        self.arrival_rate = float(arrival_rate)
        self.impact_percentiles_path = Path(impact_percentiles_path)
        self.impact_column = impact_column
        self.reference_venue = reference_venue
        self.scale_mode = scale_mode
        self.initial_x = float(initial_x)
        self.initial_y = float(initial_y)
        self.initial_price = self.initial_y / max(self.initial_x, 1e-12)
        self.pct_grid, self.impact_values = self._load_percentiles(self.impact_percentiles_path, impact_column)
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _load_percentiles(path: Path, impact_column: str) -> tuple[np.ndarray, np.ndarray]:
        rows = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
        pct_grid = np.asarray(rows["pct"], dtype=float)
        values = np.asarray(rows[impact_column], dtype=float)
        return pct_grid, values

    def generate_orders(
        self,
        *,
        fair_price: float,
        reference_amm: _ReferenceAMM | None = None,
    ) -> list["RetailOrder"]:
        from arena_eval.exact_simple_amm.simulator import RetailOrder

        if self.arrival_rate <= 0.0:
            return []
        n_orders = int(self.rng.poisson(self.arrival_rate))
        if n_orders == 0:
            return []
        impacts = np.interp(self.rng.random(size=n_orders) * 100.0, self.pct_grid, self.impact_values)
        orders: list[RetailOrder] = []
        for impact in impacts.tolist():
            order = self._impact_to_order(
                float(impact),
                fair_price=float(fair_price),
                reference_amm=reference_amm,
            )
            if order is not None:
                orders.append(order)
        return orders

    def _impact_to_order(
        self,
        impact_log: float,
        *,
        fair_price: float,
        reference_amm: _ReferenceAMM | None,
    ) -> "RetailOrder | None":
        from arena_eval.exact_simple_amm.simulator import RetailOrder

        if abs(impact_log) <= 1e-12:
            return None
        reserve_x, reserve_y, bid_fee, ask_fee = self._reference_state(reference_amm)
        magnitude = abs(impact_log)
        exp_half = math.exp(0.5 * magnitude) - 1.0
        if impact_log > 0.0:
            gamma = max(1e-12, 1.0 - ask_fee)
            gross_y = reserve_y * exp_half / gamma
            return RetailOrder(side="buy", size=float(gross_y))
        gamma = max(1e-12, 1.0 - bid_fee)
        gross_x = reserve_x * exp_half / gamma
        gross_y_notional = gross_x * max(fair_price, 1e-12)
        return RetailOrder(side="sell", size=float(gross_y_notional))

    def _reference_state(self, reference_amm: _ReferenceAMM | None) -> tuple[float, float, float, float]:
        if reference_amm is None or self.scale_mode == "initial_state":
            return (self.initial_x, self.initial_y, 0.003, 0.003)
        return (
            float(reference_amm.reserve_x),
            float(reference_amm.reserve_y),
            float(reference_amm.bid_fee),
            float(reference_amm.ask_fee),
        )
