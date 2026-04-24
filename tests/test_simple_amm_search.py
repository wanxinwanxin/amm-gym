from __future__ import annotations

from arena_search import (
    SearchConfig,
    cross_entropy_search,
    cross_entropy_search_with_validation,
    evaluate_controller_params,
    random_search,
    random_search_with_validation,
)
from arena_policies import (
    BeliefStateControllerParams,
    InventoryToxicityParams,
    LatentCompetitionParams,
    LatentFairParams,
    LatentFlowParams,
    LatentFullParams,
    LatentToxicityParams,
    PiecewiseControllerParams,
    ReactiveControllerParams,
    SubmissionBasisParams,
    SubmissionCompactParams,
    SubmissionRegimeParams,
)


def test_evaluate_controller_params_runs():
    evaluation = evaluate_controller_params(
        ReactiveControllerParams(),
        SearchConfig(seeds=(0, 1, 2)),
    )
    assert isinstance(evaluation.score, float)
    assert evaluation.edge_mean_submission == evaluation.score


def test_evaluate_inventory_toxicity_params_runs():
    evaluation = evaluate_controller_params(
        InventoryToxicityParams(),
        SearchConfig(seeds=(0, 1), policy_family="inventory_toxicity"),
    )
    assert isinstance(evaluation.score, float)
    assert evaluation.edge_mean_submission == evaluation.score


def test_evaluate_piecewise_params_runs():
    evaluation = evaluate_controller_params(
        PiecewiseControllerParams(),
        SearchConfig(seeds=(0, 1), policy_family="piecewise"),
    )
    assert isinstance(evaluation.score, float)
    assert evaluation.edge_mean_submission == evaluation.score


def test_evaluate_belief_state_params_runs():
    evaluation = evaluate_controller_params(
        BeliefStateControllerParams(),
        SearchConfig(seeds=(0, 1), policy_family="belief_state"),
    )
    assert isinstance(evaluation.score, float)
    assert evaluation.edge_mean_submission == evaluation.score


def test_evaluate_latent_ladder_params_run():
    cases = (
        (LatentFlowParams(), "latent_flow"),
        (LatentFairParams(), "latent_fair"),
        (LatentToxicityParams(), "latent_toxicity"),
        (LatentCompetitionParams(), "latent_competition"),
        (LatentFullParams(), "latent_full"),
    )

    for params, family in cases:
        evaluation = evaluate_controller_params(
            params,
            SearchConfig(seeds=(0,), policy_family=family),
        )
        assert isinstance(evaluation.score, float)
        assert evaluation.edge_mean_submission == evaluation.score


def test_evaluate_submission_safe_params_run():
    cases = (
        (SubmissionCompactParams(), "submission_compact"),
        (SubmissionRegimeParams(), "submission_regime"),
        (SubmissionBasisParams(), "submission_basis"),
    )

    for params, family in cases:
        evaluation = evaluate_controller_params(
            params,
            SearchConfig(seeds=(0,), policy_family=family),
        )
        assert isinstance(evaluation.score, float)
        assert evaluation.edge_mean_submission == evaluation.score


def test_evaluate_piecewise_params_runs_on_real_data_evaluator():
    evaluation = evaluate_controller_params(
        PiecewiseControllerParams(),
        SearchConfig(seeds=(0,), policy_family="piecewise", evaluator_kind="real_data"),
    )
    assert isinstance(evaluation.score, float)
    assert evaluation.edge_mean_submission == evaluation.score


def test_random_search_sorts_candidates():
    evaluations = random_search(
        SearchConfig(seeds=(0, 1)),
        n_candidates=3,
        seed=7,
    )
    assert len(evaluations) == 3
    assert evaluations[0].score >= evaluations[-1].score


def test_cross_entropy_search_returns_ranked_candidates():
    evaluations = cross_entropy_search(
        SearchConfig(seeds=(0,)),
        generations=1,
        population_size=3,
        seed=11,
    )
    assert evaluations
    assert evaluations[0].score >= evaluations[-1].score


def test_random_search_with_validation_tracks_history():
    study = random_search_with_validation(
        SearchConfig(seeds=(0, 1)),
        fixed_validation_seeds=(10, 11),
        rounds=2,
        candidates_per_round=2,
        rerank_top_k=2,
        fresh_validation_interval=1,
        fresh_validation_seed_count=2,
        seed=5,
    )
    assert len(study.history) == 2
    assert study.history[0].fixed_validation.score == study.history[0].fixed_validation.edge_mean_submission
    assert study.history[0].fresh_validation is not None
    assert len(study.validation_rerank) == 2
    assert study.best_validation.score == study.validation_rerank[0].score


def test_cem_search_with_validation_tracks_fixed_and_fresh_curves():
    study = cross_entropy_search_with_validation(
        SearchConfig(seeds=(0,)),
        fixed_validation_seeds=(20,),
        generations=2,
        population_size=3,
        rerank_top_k=2,
        fresh_validation_interval=2,
        fresh_validation_seed_count=1,
        seed=13,
    )
    assert len(study.history) == 2
    assert study.history[0].fresh_validation is None
    assert study.history[1].fresh_validation is not None
    assert len(study.validation_rerank) <= 2
    assert study.best_validation.score == study.validation_rerank[0].score
