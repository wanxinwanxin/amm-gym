from __future__ import annotations

import math

import pytest

pytest.importorskip("jax")

from arena_search.diff_simple_amm_search import GradientSearchConfig, gradient_ascent_search_with_validation


@pytest.mark.parametrize("evaluator_kind", ["challenge", "real_data"])
def test_gradient_search_smoke_runs(evaluator_kind: str) -> None:
    study = gradient_ascent_search_with_validation(
        GradientSearchConfig(
            train_seeds=(0, 1),
            validation_seeds=(2,),
            evaluator_kind=evaluator_kind,
            n_steps=8,
        ),
        iterations=2,
        learning_rate=0.01,
        gradient_clip=5.0,
        fresh_validation_seed_count=0,
        rerank_top_k=2,
        seed=3,
    )

    assert study.method == "adam"
    assert len(study.history) == 2
    assert len(study.validation_rerank) >= 1
    assert math.isfinite(study.best_search.score)
    assert math.isfinite(study.best_validation.score)
