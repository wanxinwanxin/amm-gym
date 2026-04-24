"""Challenge-mode exogenous dynamics helpers for the diff simple-AMM stack."""

from __future__ import annotations

import math

import numpy as np

from arena_eval.exact_simple_amm.config import ExactSimpleAMMConfig
from arena_eval.diff_simple_amm.types import ChallengeTape


def build_challenge_tape(*, config: ExactSimpleAMMConfig, seed: int) -> ChallengeTape:
    """Materialize explicit challenge-mode randomness from the exact evaluator law."""

    price_rng = np.random.default_rng(seed)
    retail_rng = np.random.default_rng(seed + 1)
    smooth_rng = np.random.default_rng(seed + 101)
    size_sigma = max(float(config.retail_size_sigma), 0.01)
    mean_size = max(float(config.retail_mean_size), 0.01)
    ln_mu = math.log(mean_size) - 0.5 * size_sigma**2
    ln_sigma = size_sigma

    gbm_normals: list[float] = []
    order_counts: list[int] = []
    order_sizes: list[tuple[float, ...]] = []
    order_side_uniforms: list[tuple[float, ...]] = []
    max_orders_per_step = 0

    for _ in range(config.n_steps):
        gbm_normals.append(float(price_rng.standard_normal()))
        count = int(retail_rng.poisson(config.retail_arrival_rate))
        order_counts.append(count)
        if count <= 0:
            order_sizes.append(())
            order_side_uniforms.append(())
            continue
        max_orders_per_step = max(max_orders_per_step, count)
        sizes = tuple(float(value) for value in retail_rng.lognormal(ln_mu, ln_sigma, size=count))
        sides = tuple(float(value) for value in retail_rng.random(size=count))
        order_sizes.append(sizes)
        order_side_uniforms.append(sides)

    smooth_arrival_uniforms = tuple(
        tuple(float(value) for value in smooth_rng.random(size=max(max_orders_per_step, 1)))
        for _ in range(config.n_steps)
    )
    smooth_size_normals = tuple(
        tuple(float(value) for value in smooth_rng.standard_normal(size=max(max_orders_per_step, 1)))
        for _ in range(config.n_steps)
    )
    smooth_side_uniforms = tuple(
        tuple(float(value) for value in smooth_rng.random(size=max(max_orders_per_step, 1)))
        for _ in range(config.n_steps)
    )

    return ChallengeTape(
        gbm_normals=tuple(gbm_normals),
        order_counts=tuple(order_counts),
        order_sizes=tuple(order_sizes),
        order_side_uniforms=tuple(order_side_uniforms),
        max_orders_per_step=max_orders_per_step,
        smooth_arrival_uniforms=smooth_arrival_uniforms,
        smooth_size_normals=smooth_size_normals,
        smooth_side_uniforms=smooth_side_uniforms,
    )
