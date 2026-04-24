"""Configurable venue specs and construction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from amm_gym.sim.amm import ConstantProductAMM
from amm_gym.sim.ladder import DepthLadderAMM
from amm_gym.sim.quote_surface import (
    QUOTE_SURFACE_ACTION_DIM,
    ParametricQuoteSurfaceAMM,
)


VenueKind = Literal["cpmm", "depth_ladder", "quote_surface"]
Venue = ConstantProductAMM | DepthLadderAMM | ParametricQuoteSurfaceAMM


@dataclass(frozen=True)
class VenueSpec:
    kind: VenueKind
    name: str
    reserve_x: float
    reserve_y: float
    bid_fee: float | None = None
    ask_fee: float | None = None
    band_bps: tuple[float, ...] | None = None
    base_notional_y: float | None = None
    controllable: bool = False
    fixed_action: tuple[float, ...] | None = None

    @property
    def action_dim(self) -> int:
        if self.kind == "depth_ladder":
            return 6
        if self.kind == "quote_surface":
            return QUOTE_SURFACE_ACTION_DIM
        return 0

    def action_vector(self) -> np.ndarray:
        if self.action_dim == 0:
            return np.zeros(0, dtype=np.float32)
        if self.fixed_action is None:
            return np.zeros(self.action_dim, dtype=np.float32)
        action = np.asarray(self.fixed_action, dtype=np.float32)
        if action.shape != (self.action_dim,):
            raise ValueError(f"fixed_action must have shape ({self.action_dim},)")
        return np.clip(action, -1.0, 1.0)


def build_venue(spec: VenueSpec) -> Venue:
    if spec.kind == "cpmm":
        if spec.bid_fee is None or spec.ask_fee is None:
            raise ValueError("cpmm venue requires bid_fee and ask_fee")
        return ConstantProductAMM(
            name=spec.name,
            reserve_x=spec.reserve_x,
            reserve_y=spec.reserve_y,
            bid_fee=spec.bid_fee,
            ask_fee=spec.ask_fee,
        )

    if spec.kind == "depth_ladder":
        if spec.band_bps is None or spec.base_notional_y is None:
            raise ValueError("depth_ladder venue requires band_bps and base_notional_y")
        return DepthLadderAMM(
            name=spec.name,
            reserve_x=spec.reserve_x,
            reserve_y=spec.reserve_y,
            band_bps=spec.band_bps,
            base_notional_y=spec.base_notional_y,
        )

    if spec.kind == "quote_surface":
        return ParametricQuoteSurfaceAMM(
            name=spec.name,
            reserve_x=spec.reserve_x,
            reserve_y=spec.reserve_y,
        )

    raise ValueError(f"unsupported venue kind `{spec.kind}`")
