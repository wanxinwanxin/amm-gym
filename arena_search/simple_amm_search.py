"""Parameter search over the exact simple-AMM replica."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np

from arena_eval.exact_simple_amm import FixedFeeStrategy, run_batch
from arena_policies import (
    BeliefStateControllerParams,
    BeliefStateControllerStrategy,
    InventoryToxicityParams,
    InventoryToxicityStrategy,
    LatentCompetitionParams,
    LatentCompetitionStrategy,
    LatentFairParams,
    LatentFairStrategy,
    LatentFlowParams,
    LatentFlowStrategy,
    LatentFullParams,
    LatentFullStrategy,
    LatentToxicityParams,
    LatentToxicityStrategy,
    PiecewiseControllerParams,
    PiecewiseControllerStrategy,
    ReactiveControllerParams,
    ReactiveControllerStrategy,
    SubmissionBasisParams,
    SubmissionBasisStrategy,
    SubmissionCompactParams,
    SubmissionCompactStrategy,
    SubmissionRegimeParams,
    SubmissionRegimeStrategy,
)


REACTIVE_PARAM_RANGES: dict[str, tuple[float, float]] = {
    "base_fee": (0.0001, 0.01),
    "base_spread": (0.0, 0.02),
    "flow_decay": (0.2, 0.99),
    "size_decay": (0.2, 0.99),
    "gap_decay": (0.2, 0.99),
    "toxicity_decay": (0.2, 0.995),
    "size_weight": (-0.02, 0.12),
    "gap_weight": (-0.002, 0.01),
    "gap_target": (0.0, 6.0),
    "flow_to_mid": (-0.05, 0.08),
    "flow_to_spread": (0.0, 0.08),
    "flow_to_skew": (-0.08, 0.08),
    "toxicity_to_mid": (0.0, 0.08),
    "toxicity_to_side": (0.0, 0.08),
    "buy_toxicity_weight": (0.0, 2.0),
    "sell_toxicity_weight": (0.0, 2.0),
}

INVENTORY_TOXICITY_PARAM_RANGES: dict[str, tuple[float, float]] = {
    "base_fee": (0.0001, 0.01),
    "base_spread": (0.0, 0.01),
    "inventory_weight": (0.0, 0.08),
    "inventory_target": (-0.5, 0.5),
    "flow_decay": (0.2, 0.99),
    "flow_skew_weight": (-0.05, 0.05),
    "toxicity_decay": (0.2, 0.995),
    "same_side_toxicity_weight": (0.0, 0.08),
    "reverse_reaction_weight": (0.0, 0.15),
    "reversal_time_scale": (1.0, 8.0),
    "toxicity_to_mid": (0.0, 0.08),
    "toxicity_to_spread": (0.0, 0.08),
    "toxicity_to_side": (0.0, 0.12),
    "min_fee": (0.0, 0.005),
    "max_fee": (0.005, 0.03),
}

PIECEWISE_CONTROLLER_PARAM_RANGES: dict[str, tuple[float, float]] = {
    "base_fee": (0.0001, 0.01),
    "base_spread": (0.0, 0.01),
    "signal_decay": (0.2, 0.99),
    "toxicity_decay": (0.2, 0.995),
    "small_trade_threshold": (0.0005, 0.01),
    "large_trade_threshold": (0.003, 0.03),
    "continuation_small": (-0.01, 0.02),
    "continuation_medium": (-0.01, 0.03),
    "continuation_large": (-0.01, 0.02),
    "reversal_small": (0.0, 0.04),
    "reversal_medium": (0.0, 0.06),
    "reversal_large": (0.0, 0.08),
    "continuation_to_same_side": (-2.0, 3.0),
    "continuation_to_cross_side": (-1.5, 2.0),
    "toxicity_to_mid": (0.0, 0.08),
    "toxicity_to_side": (0.0, 0.12),
}

BELIEF_STATE_PARAM_RANGES: dict[str, tuple[float, float]] = {
    "base_fee": (0.0005, 0.01),
    "min_fee": (0.0, 0.004),
    "max_fee": (0.004, 0.03),
    "fair_decay": (0.6, 0.995),
    "fair_size_weight": (0.1, 6.0),
    "flow_decay": (0.2, 0.995),
    "opportunity_decay": (0.2, 0.995),
    "opportunity_scale": (0.2, 10.0),
    "opportunity_weight": (-0.005, 0.01),
    "competition_decay": (0.2, 0.995),
    "competition_weight": (0.0, 0.02),
    "toxicity_decay": (0.2, 0.995),
    "reverse_toxicity_weight": (0.0, 3.0),
    "toxicity_weight": (0.0, 0.02),
    "flow_skew_weight": (-0.03, 0.03),
    "fair_gap_weight": (0.0, 0.05),
}

LATENT_FLOW_PARAM_RANGES: dict[str, tuple[float, float]] = {
    "base_fee": (0.0005, 0.01),
    "min_fee": (0.0, 0.004),
    "max_fee": (0.004, 0.03),
    "flow_decay": (0.2, 0.995),
    "opportunity_decay": (0.2, 0.995),
    "opportunity_scale": (0.2, 10.0),
    "opportunity_weight": (-0.005, 0.01),
    "flow_skew_weight": (-0.03, 0.03),
}

LATENT_FAIR_PARAM_RANGES: dict[str, tuple[float, float]] = {
    **LATENT_FLOW_PARAM_RANGES,
    "fair_decay": (0.6, 0.995),
    "fair_size_weight": (0.1, 6.0),
    "fair_gap_weight": (0.0, 0.05),
}

LATENT_TOXICITY_PARAM_RANGES: dict[str, tuple[float, float]] = {
    **LATENT_FAIR_PARAM_RANGES,
    "toxicity_decay": (0.2, 0.995),
    "reverse_toxicity_weight": (0.0, 3.0),
    "toxicity_weight": (0.0, 0.02),
}

LATENT_COMPETITION_PARAM_RANGES: dict[str, tuple[float, float]] = {
    **LATENT_TOXICITY_PARAM_RANGES,
    "competition_decay": (0.2, 0.995),
    "competition_weight": (0.0, 0.02),
}

LATENT_FULL_PARAM_RANGES: dict[str, tuple[float, float]] = {
    **LATENT_COMPETITION_PARAM_RANGES,
    "inventory_weight": (-0.03, 0.03),
    "inventory_target": (-0.5, 0.5),
}

SUBMISSION_COMPACT_PARAM_RANGES: dict[str, tuple[float, float]] = {
    "base_fee": (0.0005, 0.01),
    "min_fee": (0.0, 0.004),
    "max_fee": (0.004, 0.03),
    "flow_fast_decay": (0.2, 0.9),
    "flow_slow_decay": (0.8, 0.995),
    "size_fast_decay": (0.2, 0.9),
    "size_slow_decay": (0.8, 0.995),
    "gap_fast_decay": (0.2, 0.9),
    "gap_slow_decay": (0.85, 0.995),
    "toxicity_decay": (0.2, 0.995),
    "toxicity_weight": (0.0, 3.0),
    "base_spread": (0.0, 0.01),
    "flow_mid_weight": (-0.02, 0.08),
    "size_mid_weight": (0.0, 0.12),
    "gap_mid_weight": (0.0, 0.01),
    "skew_weight": (-0.1, 0.1),
    "toxicity_side_weight": (0.0, 0.12),
    "hot_gap_threshold": (0.6, 3.0),
    "big_trade_threshold": (0.001, 0.03),
    "hot_fee_bump": (0.0, 0.02),
}

SUBMISSION_REGIME_PARAM_RANGES: dict[str, tuple[float, float]] = {
    "base_fee": (0.0005, 0.01),
    "min_fee": (0.0, 0.004),
    "max_fee": (0.004, 0.03),
    "flow_fast_decay": (0.2, 0.9),
    "flow_slow_decay": (0.85, 0.995),
    "size_fast_decay": (0.2, 0.9),
    "size_slow_decay": (0.85, 0.995),
    "gap_fast_decay": (0.2, 0.9),
    "gap_slow_decay": (0.85, 0.995),
    "toxicity_decay": (0.2, 0.995),
    "toxicity_weight": (0.0, 3.0),
    "fair_fast_decay": (0.3, 0.95),
    "fair_slow_decay": (0.85, 0.995),
    "fair_size_weight": (0.2, 5.0),
    "base_spread": (0.0, 0.01),
    "calm_mid_shift": (-0.01, 0.01),
    "active_mid_shift": (-0.01, 0.02),
    "panic_mid_shift": (0.0, 0.03),
    "continuation_spread": (0.0, 0.01),
    "reversal_spread": (0.0, 0.02),
    "fair_weight": (0.0, 0.05),
    "inventory_weight": (-0.05, 0.05),
    "skew_weight": (-0.1, 0.1),
    "toxicity_side_weight": (0.0, 0.12),
    "active_gap_threshold": (0.8, 3.0),
    "panic_gap_threshold": (0.5, 2.0),
}

SUBMISSION_BASIS_PARAM_RANGES: dict[str, tuple[float, float]] = {
    "base_fee": (0.0005, 0.01),
    "min_fee": (0.0, 0.004),
    "max_fee": (0.004, 0.04),
    "flow_fast_decay": (0.2, 0.9),
    "flow_slow_decay": (0.85, 0.995),
    "size_fast_decay": (0.2, 0.9),
    "size_slow_decay": (0.85, 0.995),
    "gap_fast_decay": (0.2, 0.9),
    "gap_slow_decay": (0.85, 0.995),
    "toxicity_decay": (0.2, 0.995),
    "toxicity_weight": (0.0, 3.0),
    "fair_fast_decay": (0.3, 0.95),
    "fair_slow_decay": (0.85, 0.995),
    "fair_size_weight": (0.2, 6.0),
    "spread_base": (0.0, 0.01),
    "size_hinge_low": (0.001, 0.02),
    "size_hinge_high": (0.004, 0.04),
    "flow_hinge": (0.001, 0.02),
    "tox_hinge": (0.001, 0.03),
    "fair_hinge": (0.0001, 0.01),
    "gap_hinge": (0.8, 3.0),
    "basis_size_low_w": (0.0, 0.1),
    "basis_size_high_w": (0.0, 0.15),
    "basis_flow_w": (0.0, 0.12),
    "basis_gap_w": (0.0, 0.015),
    "basis_tox_w": (0.0, 0.12),
    "basis_fair_w": (0.0, 0.08),
    "basis_streak_w": (0.0, 0.02),
    "skew_flow_w": (-0.12, 0.12),
    "skew_inventory_w": (-0.06, 0.06),
    "skew_fair_w": (-0.08, 0.08),
    "side_tox_w": (0.0, 0.12),
}


@dataclass(frozen=True)
class PolicySpec:
    params_cls: type
    strategy_cls: type
    param_ranges: dict[str, tuple[float, float]]


POLICY_SPECS: dict[str, PolicySpec] = {
    "reactive": PolicySpec(
        params_cls=ReactiveControllerParams,
        strategy_cls=ReactiveControllerStrategy,
        param_ranges=REACTIVE_PARAM_RANGES,
    ),
    "inventory_toxicity": PolicySpec(
        params_cls=InventoryToxicityParams,
        strategy_cls=InventoryToxicityStrategy,
        param_ranges=INVENTORY_TOXICITY_PARAM_RANGES,
    ),
    "piecewise": PolicySpec(
        params_cls=PiecewiseControllerParams,
        strategy_cls=PiecewiseControllerStrategy,
        param_ranges=PIECEWISE_CONTROLLER_PARAM_RANGES,
    ),
    "belief_state": PolicySpec(
        params_cls=BeliefStateControllerParams,
        strategy_cls=BeliefStateControllerStrategy,
        param_ranges=BELIEF_STATE_PARAM_RANGES,
    ),
    "latent_flow": PolicySpec(
        params_cls=LatentFlowParams,
        strategy_cls=LatentFlowStrategy,
        param_ranges=LATENT_FLOW_PARAM_RANGES,
    ),
    "latent_fair": PolicySpec(
        params_cls=LatentFairParams,
        strategy_cls=LatentFairStrategy,
        param_ranges=LATENT_FAIR_PARAM_RANGES,
    ),
    "latent_toxicity": PolicySpec(
        params_cls=LatentToxicityParams,
        strategy_cls=LatentToxicityStrategy,
        param_ranges=LATENT_TOXICITY_PARAM_RANGES,
    ),
    "latent_competition": PolicySpec(
        params_cls=LatentCompetitionParams,
        strategy_cls=LatentCompetitionStrategy,
        param_ranges=LATENT_COMPETITION_PARAM_RANGES,
    ),
    "latent_full": PolicySpec(
        params_cls=LatentFullParams,
        strategy_cls=LatentFullStrategy,
        param_ranges=LATENT_FULL_PARAM_RANGES,
    ),
    "submission_compact": PolicySpec(
        params_cls=SubmissionCompactParams,
        strategy_cls=SubmissionCompactStrategy,
        param_ranges=SUBMISSION_COMPACT_PARAM_RANGES,
    ),
    "submission_regime": PolicySpec(
        params_cls=SubmissionRegimeParams,
        strategy_cls=SubmissionRegimeStrategy,
        param_ranges=SUBMISSION_REGIME_PARAM_RANGES,
    ),
    "submission_basis": PolicySpec(
        params_cls=SubmissionBasisParams,
        strategy_cls=SubmissionBasisStrategy,
        param_ranges=SUBMISSION_BASIS_PARAM_RANGES,
    ),
}


@dataclass(frozen=True)
class SearchConfig:
    seeds: tuple[int, ...]
    normalizer_fee: float = 0.003
    policy_family: str = "reactive"
    evaluator_kind: str = "challenge"


@dataclass(frozen=True)
class CandidateEvaluation:
    params: Any
    score: float
    edge_mean_submission: float
    edge_mean_normalizer: float
    edge_advantage_mean: float
    metadata: dict[str, object]


@dataclass(frozen=True)
class SearchIteration:
    iteration: int
    best_search: CandidateEvaluation
    fixed_validation: CandidateEvaluation
    fresh_validation: CandidateEvaluation | None
    fresh_validation_seeds: tuple[int, ...]


@dataclass(frozen=True)
class SearchStudyResult:
    method: str
    history: tuple[SearchIteration, ...]
    best_search: CandidateEvaluation
    best_validation: CandidateEvaluation
    validation_rerank: tuple[CandidateEvaluation, ...]


def evaluate_controller_params(params: Any, config: SearchConfig) -> CandidateEvaluation:
    return evaluate_params_on_seeds(
        params,
        config.seeds,
        normalizer_fee=config.normalizer_fee,
        evaluator_kind=config.evaluator_kind,
    )


def evaluate_params_on_seeds(
    params: Any,
    seeds: Iterable[int],
    *,
    normalizer_fee: float = 0.003,
    evaluator_kind: str = "challenge",
) -> CandidateEvaluation:
    normalized = params.normalized()
    strategy_cls = _strategy_cls_for_params(normalized)
    result = run_batch(
        lambda: strategy_cls(normalized),
        seeds,
        normalizer_strategy_factory=lambda: FixedFeeStrategy(normalizer_fee, normalizer_fee),
        evaluator_kind=evaluator_kind,
    )
    return CandidateEvaluation(
        params=normalized,
        score=result.score,
        edge_mean_submission=result.edge_mean_submission,
        edge_mean_normalizer=result.edge_mean_normalizer,
        edge_advantage_mean=result.edge_advantage_mean,
        metadata=result.metadata,
    )


def random_search(
    config: SearchConfig,
    *,
    n_candidates: int,
    seed: int = 0,
) -> list[CandidateEvaluation]:
    rng = np.random.default_rng(seed)
    evaluations = [evaluate_controller_params(_sample_params(rng, config.policy_family), config) for _ in range(n_candidates)]
    evaluations.sort(key=lambda item: item.score, reverse=True)
    return evaluations


def cross_entropy_search(
    config: SearchConfig,
    *,
    generations: int,
    population_size: int,
    elite_fraction: float = 0.2,
    seed: int = 0,
) -> list[CandidateEvaluation]:
    rng = np.random.default_rng(seed)
    spec = _policy_spec(config.policy_family)
    names = list(spec.param_ranges.keys())
    lows = np.asarray([spec.param_ranges[name][0] for name in names], dtype=float)
    highs = np.asarray([spec.param_ranges[name][1] for name in names], dtype=float)
    mean = 0.5 * (lows + highs)
    std = 0.25 * (highs - lows)
    best: list[CandidateEvaluation] = []

    for generation in range(generations):
        candidates: list[CandidateEvaluation] = []
        for _ in range(population_size):
            vector = rng.normal(mean, std)
            vector = np.clip(vector, lows, highs)
            params = spec.params_cls(**dict(zip(names, vector.tolist(), strict=True)))
            candidates.append(evaluate_controller_params(params, config))
        candidates.sort(key=lambda item: item.score, reverse=True)
        best.extend(candidates[: max(1, int(population_size * elite_fraction))])
        elites = candidates[: max(1, int(population_size * elite_fraction))]
        elite_matrix = np.asarray([[getattr(item.params, name) for name in names] for item in elites], dtype=float)
        mean = elite_matrix.mean(axis=0)
        std = np.maximum(elite_matrix.std(axis=0, ddof=0), 1e-4)
        best.sort(key=lambda item: item.score, reverse=True)
        best = best[: population_size]
        if generation == generations - 1:
            break
    best.sort(key=lambda item: item.score, reverse=True)
    return best


def random_search_with_validation(
    config: SearchConfig,
    *,
    fixed_validation_seeds: Iterable[int],
    rounds: int,
    candidates_per_round: int,
    rerank_top_k: int = 8,
    fresh_validation_interval: int = 1,
    fresh_validation_seed_count: int = 0,
    seed: int = 0,
    progress_callback: Callable[[SearchIteration], None] | None = None,
) -> SearchStudyResult:
    rng = np.random.default_rng(seed)
    fresh_rng = np.random.default_rng(seed + 1_000_003)
    fixed_validation_seeds = tuple(int(value) for value in fixed_validation_seeds)
    best: list[CandidateEvaluation] = []
    history: list[SearchIteration] = []

    for iteration in range(rounds):
        candidates = [evaluate_controller_params(_sample_params(rng, config.policy_family), config) for _ in range(candidates_per_round)]
        candidates.sort(key=lambda item: item.score, reverse=True)
        best.extend(candidates)
        best.sort(key=lambda item: item.score, reverse=True)
        best = _dedupe_candidates(best)[: max(rerank_top_k, candidates_per_round)]
        best_search = best[0]
        fixed_validation = evaluate_params_on_seeds(
            best_search.params,
            fixed_validation_seeds,
            normalizer_fee=config.normalizer_fee,
            evaluator_kind=config.evaluator_kind,
        )
        fresh_validation_seeds_current = _sample_fresh_validation_seeds(
            fresh_rng,
            fresh_validation_interval=fresh_validation_interval,
            fresh_validation_seed_count=fresh_validation_seed_count,
            iteration=iteration,
        )
        fresh_validation = (
            evaluate_params_on_seeds(
                best_search.params,
                fresh_validation_seeds_current,
                normalizer_fee=config.normalizer_fee,
                evaluator_kind=config.evaluator_kind,
            )
            if fresh_validation_seeds_current
            else None
        )
        history.append(
            SearchIteration(
                iteration=iteration,
                best_search=best_search,
                fixed_validation=fixed_validation,
                fresh_validation=fresh_validation,
                fresh_validation_seeds=fresh_validation_seeds_current,
            )
        )
        if progress_callback is not None:
            progress_callback(history[-1])

    reranked = _rerank_candidates(
        best,
        fixed_validation_seeds,
        normalizer_fee=config.normalizer_fee,
        evaluator_kind=config.evaluator_kind,
        top_k=rerank_top_k,
    )
    return SearchStudyResult(
        method="random",
        history=tuple(history),
        best_search=best[0],
        best_validation=reranked[0],
        validation_rerank=tuple(reranked),
    )


def cross_entropy_search_with_validation(
    config: SearchConfig,
    *,
    fixed_validation_seeds: Iterable[int],
    generations: int,
    population_size: int,
    elite_fraction: float = 0.2,
    rerank_top_k: int = 8,
    fresh_validation_interval: int = 1,
    fresh_validation_seed_count: int = 0,
    seed: int = 0,
    progress_callback: Callable[[SearchIteration], None] | None = None,
) -> SearchStudyResult:
    rng = np.random.default_rng(seed)
    fresh_rng = np.random.default_rng(seed + 1_000_003)
    fixed_validation_seeds = tuple(int(value) for value in fixed_validation_seeds)
    spec = _policy_spec(config.policy_family)
    names = list(spec.param_ranges.keys())
    lows = np.asarray([spec.param_ranges[name][0] for name in names], dtype=float)
    highs = np.asarray([spec.param_ranges[name][1] for name in names], dtype=float)
    mean = 0.5 * (lows + highs)
    std = 0.25 * (highs - lows)
    best: list[CandidateEvaluation] = []
    history: list[SearchIteration] = []

    for iteration in range(generations):
        candidates: list[CandidateEvaluation] = []
        for _ in range(population_size):
            vector = rng.normal(mean, std)
            vector = np.clip(vector, lows, highs)
            params = spec.params_cls(**dict(zip(names, vector.tolist(), strict=True)))
            candidates.append(evaluate_controller_params(params, config))
        candidates.sort(key=lambda item: item.score, reverse=True)
        elites = candidates[: max(1, int(population_size * elite_fraction))]
        best.extend(elites)
        best.sort(key=lambda item: item.score, reverse=True)
        best = _dedupe_candidates(best)[: max(population_size, rerank_top_k)]
        elite_matrix = np.asarray([[getattr(item.params, name) for name in names] for item in elites], dtype=float)
        mean = elite_matrix.mean(axis=0)
        std = np.maximum(elite_matrix.std(axis=0, ddof=0), 1e-4)

        best_search = best[0]
        fixed_validation = evaluate_params_on_seeds(
            best_search.params,
            fixed_validation_seeds,
            normalizer_fee=config.normalizer_fee,
            evaluator_kind=config.evaluator_kind,
        )
        fresh_validation_seeds_current = _sample_fresh_validation_seeds(
            fresh_rng,
            fresh_validation_interval=fresh_validation_interval,
            fresh_validation_seed_count=fresh_validation_seed_count,
            iteration=iteration,
        )
        fresh_validation = (
            evaluate_params_on_seeds(
                best_search.params,
                fresh_validation_seeds_current,
                normalizer_fee=config.normalizer_fee,
                evaluator_kind=config.evaluator_kind,
            )
            if fresh_validation_seeds_current
            else None
        )
        history.append(
            SearchIteration(
                iteration=iteration,
                best_search=best_search,
                fixed_validation=fixed_validation,
                fresh_validation=fresh_validation,
                fresh_validation_seeds=fresh_validation_seeds_current,
            )
        )
        if progress_callback is not None:
            progress_callback(history[-1])

    reranked = _rerank_candidates(
        best,
        fixed_validation_seeds,
        normalizer_fee=config.normalizer_fee,
        evaluator_kind=config.evaluator_kind,
        top_k=rerank_top_k,
    )
    return SearchStudyResult(
        method="cem",
        history=tuple(history),
        best_search=best[0],
        best_validation=reranked[0],
        validation_rerank=tuple(reranked),
    )


def _sample_params(rng: np.random.Generator, policy_family: str) -> Any:
    spec = _policy_spec(policy_family)
    values = {name: float(rng.uniform(low, high)) for name, (low, high) in spec.param_ranges.items()}
    return spec.params_cls(**values)


def _policy_spec(policy_family: str) -> PolicySpec:
    try:
        return POLICY_SPECS[policy_family]
    except KeyError as exc:
        raise ValueError(f"Unsupported policy_family: {policy_family}") from exc


def _strategy_cls_for_params(params: Any) -> type:
    if isinstance(params, ReactiveControllerParams):
        return ReactiveControllerStrategy
    if isinstance(params, InventoryToxicityParams):
        return InventoryToxicityStrategy
    if isinstance(params, PiecewiseControllerParams):
        return PiecewiseControllerStrategy
    if isinstance(params, BeliefStateControllerParams):
        return BeliefStateControllerStrategy
    if isinstance(params, LatentFlowParams):
        return LatentFlowStrategy
    if isinstance(params, LatentFairParams):
        return LatentFairStrategy
    if isinstance(params, LatentToxicityParams):
        return LatentToxicityStrategy
    if isinstance(params, LatentCompetitionParams):
        return LatentCompetitionStrategy
    if isinstance(params, LatentFullParams):
        return LatentFullStrategy
    if isinstance(params, SubmissionCompactParams):
        return SubmissionCompactStrategy
    if isinstance(params, SubmissionRegimeParams):
        return SubmissionRegimeStrategy
    if isinstance(params, SubmissionBasisParams):
        return SubmissionBasisStrategy
    raise TypeError(f"Unsupported params type: {type(params)!r}")


def _candidate_key(evaluation: CandidateEvaluation) -> tuple[float, ...]:
    return tuple(evaluation.params.to_dict().values())


def _dedupe_candidates(candidates: list[CandidateEvaluation]) -> list[CandidateEvaluation]:
    seen: set[tuple[float, ...]] = set()
    unique: list[CandidateEvaluation] = []
    for candidate in candidates:
        key = _candidate_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    unique.sort(key=lambda item: item.score, reverse=True)
    return unique


def _rerank_candidates(
    candidates: list[CandidateEvaluation],
    validation_seeds: tuple[int, ...],
    *,
    normalizer_fee: float,
    evaluator_kind: str,
    top_k: int,
) -> list[CandidateEvaluation]:
    unique = _dedupe_candidates(candidates)[:top_k]
    reranked = [
        evaluate_params_on_seeds(
            candidate.params,
            validation_seeds,
            normalizer_fee=normalizer_fee,
            evaluator_kind=evaluator_kind,
        )
        for candidate in unique
    ]
    reranked.sort(key=lambda item: item.score, reverse=True)
    return reranked


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
