"""Gradient-based search over the differentiable simple-AMM surrogate."""

from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import mean
from typing import Callable

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - exercised only when jax is absent
    jax = None
    jnp = None

from arena_eval.diff_simple_amm import (
    SmoothRelaxationConfig,
    build_challenge_tape,
    build_realistic_tape,
    expected_submission_edge,
    submission_compact_param_vector,
)
from arena_eval.exact_simple_amm import FixedFeeStrategy, ExactSimpleAMMConfig, run_seed
from arena_policies.submission_safe import SubmissionCompactParams, SubmissionCompactStrategy

from .simple_amm_search import CandidateEvaluation


def _require_jax() -> None:
    if jax is None or jnp is None:
        raise RuntimeError("jax is required for differentiable simple-AMM search")


@dataclass(frozen=True)
class DiffSearchCase:
    seed: int
    config: ExactSimpleAMMConfig
    tape: object


@dataclass(frozen=True)
class GradientSearchConfig:
    train_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]
    test_seeds: tuple[int, ...] = ()
    evaluator_kind: str = "challenge"
    normalizer_fee: float = 0.003
    n_steps: int = 256
    policy_family: str = "submission_compact"


@dataclass(frozen=True)
class GradientSearchIteration:
    iteration: int
    train_objective: float
    gradient_norm: float
    fixed_validation: CandidateEvaluation
    fresh_validation: CandidateEvaluation | None
    fresh_validation_seeds: tuple[int, ...]
    params: SubmissionCompactParams


@dataclass(frozen=True)
class GradientSearchStudyResult:
    method: str
    history: tuple[GradientSearchIteration, ...]
    best_search: CandidateEvaluation
    best_validation: CandidateEvaluation
    validation_rerank: tuple[CandidateEvaluation, ...]


def build_diff_cases(
    seeds: tuple[int, ...],
    *,
    evaluator_kind: str,
    n_steps: int,
) -> tuple[DiffSearchCase, ...]:
    """Build explicit differentiable training cases for fixed seeds."""

    cases: list[DiffSearchCase] = []
    for seed in seeds:
        config = replace(ExactSimpleAMMConfig.for_evaluator(seed, evaluator_kind), n_steps=n_steps)
        tape = (
            build_challenge_tape(config=config, seed=seed)
            if evaluator_kind == "challenge"
            else build_realistic_tape(config=config, seed=seed)
        )
        cases.append(DiffSearchCase(seed=seed, config=config, tape=tape))
    return tuple(cases)


def evaluate_submission_compact_exact(
    params: SubmissionCompactParams,
    seeds: tuple[int, ...],
    *,
    evaluator_kind: str,
    normalizer_fee: float = 0.003,
    n_steps: int = 10_000,
) -> CandidateEvaluation:
    """Evaluate compact params on the exact evaluator with an optional step override."""

    normalized = params.normalized()
    simulations = [
        run_seed(
            SubmissionCompactStrategy(normalized),
            seed,
            config=replace(ExactSimpleAMMConfig.for_evaluator(seed, evaluator_kind), n_steps=n_steps),
            normalizer_strategy=FixedFeeStrategy(normalizer_fee, normalizer_fee),
        )
        for seed in seeds
    ]
    return CandidateEvaluation(
        params=normalized,
        score=float(mean(sim.score for sim in simulations)),
        edge_mean_submission=float(mean(sim.edge_submission for sim in simulations)),
        edge_mean_normalizer=float(mean(sim.edge_normalizer for sim in simulations)),
        edge_advantage_mean=float(mean(sim.edge_advantage for sim in simulations)),
        metadata={"n_simulations": len(simulations), "n_steps": n_steps},
    )


def gradient_ascent_search_with_validation(
    config: GradientSearchConfig,
    *,
    iterations: int,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    gradient_clip: float | None = None,
    fresh_validation_interval: int = 1,
    fresh_validation_seed_count: int = 0,
    rerank_top_k: int = 8,
    init_params: SubmissionCompactParams | None = None,
    init_strategy: str = "default",
    seed: int = 0,
    relaxation: SmoothRelaxationConfig = SmoothRelaxationConfig(),
    progress_callback: Callable[[GradientSearchIteration], None] | None = None,
) -> GradientSearchStudyResult:
    """Optimize compact-policy params by Adam over the smooth surrogate."""

    _require_jax()
    if config.policy_family != "submission_compact":
        raise ValueError("gradient search currently supports submission_compact only")

    train_cases = build_diff_cases(config.train_seeds, evaluator_kind=config.evaluator_kind, n_steps=config.n_steps)
    if not train_cases:
        raise ValueError("train_seeds must be non-empty")

    rng = np.random.default_rng(seed)
    fresh_rng = np.random.default_rng(seed + 1_000_003)
    params_vector = _initial_param_vector(rng, init_strategy=init_strategy, init_params=init_params)
    m = jnp.zeros_like(params_vector)
    v = jnp.zeros_like(params_vector)

    def objective(vector):
        values = [
            expected_submission_edge(
                vector,
                config=case.config,
                tape=case.tape,
                relaxation=relaxation,
            )
            for case in train_cases
        ]
        return jnp.mean(jnp.stack(values))

    history: list[GradientSearchIteration] = []
    snapshots: list[tuple[float, SubmissionCompactParams]] = []
    best_train_objective = float("-inf")
    best_train_params = _vector_to_submission_compact_params(params_vector)

    for iteration in range(iterations):
        train_value = float(objective(params_vector))
        grad = jax.grad(objective)(params_vector)
        grad_norm = float(jnp.linalg.norm(grad))
        if gradient_clip is not None and grad_norm > gradient_clip > 0.0:
            grad = grad * (gradient_clip / max(grad_norm, 1e-12))
            grad_norm = float(jnp.linalg.norm(grad))

        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * jnp.square(grad)
        t = iteration + 1
        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)
        params_vector = _project_vector(params_vector + learning_rate * m_hat / (jnp.sqrt(v_hat) + epsilon))

        params = _vector_to_submission_compact_params(params_vector)
        snapshots.append((train_value, params))
        if train_value > best_train_objective:
            best_train_objective = train_value
            best_train_params = params

        fixed_validation = evaluate_submission_compact_exact(
            params,
            config.validation_seeds,
            evaluator_kind=config.evaluator_kind,
            normalizer_fee=config.normalizer_fee,
            n_steps=config.n_steps,
        )
        fresh_validation_seeds = _sample_fresh_validation_seeds(
            fresh_rng,
            fresh_validation_interval=fresh_validation_interval,
            fresh_validation_seed_count=fresh_validation_seed_count,
            iteration=iteration,
        )
        fresh_validation = (
            evaluate_submission_compact_exact(
                params,
                fresh_validation_seeds,
                evaluator_kind=config.evaluator_kind,
                normalizer_fee=config.normalizer_fee,
                n_steps=config.n_steps,
            )
            if fresh_validation_seeds
            else None
        )
        record = GradientSearchIteration(
            iteration=iteration,
            train_objective=train_value,
            gradient_norm=grad_norm,
            fixed_validation=fixed_validation,
            fresh_validation=fresh_validation,
            fresh_validation_seeds=fresh_validation_seeds,
            params=params,
        )
        history.append(record)
        if progress_callback is not None:
            progress_callback(record)

    best_search = evaluate_submission_compact_exact(
        best_train_params,
        config.train_seeds,
        evaluator_kind=config.evaluator_kind,
        normalizer_fee=config.normalizer_fee,
        n_steps=config.n_steps,
    )
    best_search.metadata["smooth_objective"] = best_train_objective

    reranked = _rerank_snapshots(
        snapshots,
        validation_seeds=config.validation_seeds,
        evaluator_kind=config.evaluator_kind,
        normalizer_fee=config.normalizer_fee,
        n_steps=config.n_steps,
        top_k=rerank_top_k,
    )
    return GradientSearchStudyResult(
        method="adam",
        history=tuple(history),
        best_search=best_search,
        best_validation=reranked[0],
        validation_rerank=tuple(reranked),
    )


def _initial_param_vector(
    rng: np.random.Generator,
    *,
    init_strategy: str,
    init_params: SubmissionCompactParams | None,
):
    if init_params is not None:
        return submission_compact_param_vector(init_params)
    if init_strategy == "default":
        return submission_compact_param_vector(SubmissionCompactParams())
    if init_strategy == "random":
        lower, upper = _submission_compact_bounds_np()
        sampled = lower + rng.random(size=lower.shape[0]) * (upper - lower)
        return submission_compact_param_vector(_vector_to_submission_compact_params(sampled))
    raise ValueError(f"Unsupported init_strategy: {init_strategy}")


def _project_vector(vector):
    lower, upper = _submission_compact_bounds_np()
    bounded = np.clip(np.asarray(jax.device_get(vector), dtype=float), lower, upper)
    return submission_compact_param_vector(_vector_to_submission_compact_params(bounded))


def _vector_to_submission_compact_params(vector) -> SubmissionCompactParams:
    values = np.asarray(jax.device_get(vector), dtype=float).tolist()
    fields = dict(zip(SubmissionCompactParams.__dataclass_fields__.keys(), values, strict=True))
    return SubmissionCompactParams(**fields).normalized()


def _submission_compact_bounds_np() -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(
        jax.device_get(
            submission_compact_param_vector(
                SubmissionCompactParams(
                    base_fee=0.0,
                    min_fee=0.0,
                    max_fee=0.0,
                    flow_fast_decay=0.0,
                    flow_slow_decay=0.0,
                    size_fast_decay=0.0,
                    size_slow_decay=0.0,
                    gap_fast_decay=0.0,
                    gap_slow_decay=0.0,
                    toxicity_decay=0.0,
                    toxicity_weight=0.0,
                    base_spread=0.0,
                    flow_mid_weight=-0.1,
                    size_mid_weight=-0.1,
                    gap_mid_weight=-0.02,
                    skew_weight=-0.2,
                    toxicity_side_weight=0.0,
                    hot_gap_threshold=0.0,
                    big_trade_threshold=0.0001,
                    hot_fee_bump=0.0,
                )
            )
        ),
        dtype=float,
    )
    upper = np.asarray(
        jax.device_get(
            submission_compact_param_vector(
                SubmissionCompactParams(
                    base_fee=0.1,
                    min_fee=0.1,
                    max_fee=0.1,
                    flow_fast_decay=0.999,
                    flow_slow_decay=0.999,
                    size_fast_decay=0.999,
                    size_slow_decay=0.999,
                    gap_fast_decay=0.999,
                    gap_slow_decay=0.999,
                    toxicity_decay=0.999,
                    toxicity_weight=5.0,
                    base_spread=0.03,
                    flow_mid_weight=0.1,
                    size_mid_weight=0.2,
                    gap_mid_weight=0.02,
                    skew_weight=0.2,
                    toxicity_side_weight=0.2,
                    hot_gap_threshold=8.0,
                    big_trade_threshold=0.05,
                    hot_fee_bump=0.03,
                )
            )
        ),
        dtype=float,
    )
    return (lower, upper)


def _sample_fresh_validation_seeds(
    rng: np.random.Generator,
    *,
    fresh_validation_interval: int,
    fresh_validation_seed_count: int,
    iteration: int,
) -> tuple[int, ...]:
    if fresh_validation_seed_count <= 0 or fresh_validation_interval <= 0:
        return ()
    if (iteration + 1) % fresh_validation_interval != 0:
        return ()
    values = rng.integers(0, 2**31 - 1, size=fresh_validation_seed_count, endpoint=False)
    return tuple(int(value) for value in values.tolist())


def _rerank_snapshots(
    snapshots: list[tuple[float, SubmissionCompactParams]],
    *,
    validation_seeds: tuple[int, ...],
    evaluator_kind: str,
    normalizer_fee: float,
    n_steps: int,
    top_k: int,
) -> list[CandidateEvaluation]:
    deduped: list[tuple[float, SubmissionCompactParams]] = []
    seen: set[tuple[float, ...]] = set()
    for train_score, params in sorted(snapshots, key=lambda item: item[0], reverse=True):
        key = tuple(params.to_dict().values())
        if key in seen:
            continue
        seen.add(key)
        deduped.append((train_score, params))
        if len(deduped) >= top_k:
            break

    reranked = [
        evaluate_submission_compact_exact(
            params,
            validation_seeds,
            evaluator_kind=evaluator_kind,
            normalizer_fee=normalizer_fee,
            n_steps=n_steps,
        )
        for _, params in deduped
    ]
    reranked.sort(key=lambda item: item.score, reverse=True)
    return reranked
