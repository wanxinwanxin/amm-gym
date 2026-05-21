"""Tests for the back-prop training stack.

Two responsibilities:
  1. Batched-rollout parity: the new vmap'd `compact_metrics_realistic_batched`
     entrypoint matches per-seed `compact_metrics` bit-exactly.
  2. Trainer sanity: a short Adam run reduces the loss with finite, nonzero
     gradients throughout (a lightweight version of the spec's DONE criteria).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from arena_eval.diff_simple_amm import build_realistic_tape
from arena_eval.diff_simple_amm.objectives import submission_compact_param_vector
from arena_eval.diff_simple_amm.tape_smooth import (
    compact_metrics,
    compact_metrics_realistic_batched,
    realistic_tapes_to_batched_arrays,
)
from arena_eval.exact_simple_amm import ExactSimpleAMMConfig
from arena_policies.submission_safe import SubmissionCompactParams

from scripts.train_backprop_policy import (
    DEFAULT_SEEDS,
    build_batched_arrays,
    check_done_criteria,
    initial_params,
    train,
)


# ----- batched-rollout parity -----
BATCH_SEEDS = (3, 8, 11, 13, 19, 23, 29, 31)
N_STEPS = 64


def _params_vec():
    return submission_compact_param_vector(SubmissionCompactParams()).astype(jnp.float64)


def test_batched_rollout_matches_per_seed_compact_realistic() -> None:
    """Per-seed `compact_metrics` and the vmap'd batched version must agree.

    Required for the trainer: the loss is `mean(batched.edge_submission)`, and
    the spec demands "at least one new parity test that exercises the batched
    rollout".
    """
    cfg = replace(ExactSimpleAMMConfig.real_data_from_seed(BATCH_SEEDS[0]), n_steps=N_STEPS)
    tapes = []
    per_seed = []
    params = _params_vec()
    for s in BATCH_SEEDS:
        cfg_s = replace(ExactSimpleAMMConfig.real_data_from_seed(s), n_steps=N_STEPS)
        tape = build_realistic_tape(config=cfg_s, seed=s)
        tapes.append(tape)
        per_seed.append(compact_metrics(cfg_s, tape, params))

    batched = realistic_tapes_to_batched_arrays(tapes)
    batched_metrics = compact_metrics_realistic_batched(cfg, batched, params)

    edges_batched = np.asarray(batched_metrics["edge_submission"])
    pnls_batched = np.asarray(batched_metrics["pnl_submission"])
    rvols_batched = np.asarray(batched_metrics["retail_volume_submission_y"])
    for i, m in enumerate(per_seed):
        assert float(edges_batched[i]) == pytest.approx(
            float(m["edge_submission"]), rel=1e-9, abs=1e-12
        ), f"edge mismatch seed={BATCH_SEEDS[i]}"
        assert float(pnls_batched[i]) == pytest.approx(
            float(m["pnl_submission"]), rel=1e-9, abs=1e-12
        ), f"pnl mismatch seed={BATCH_SEEDS[i]}"
        assert float(rvols_batched[i]) == pytest.approx(
            float(m["retail_volume_submission_y"]), rel=1e-9, abs=1e-12
        ), f"retail_volume mismatch seed={BATCH_SEEDS[i]}"


def test_batched_rollout_jit_grad_is_finite_and_nonzero() -> None:
    cfg, batched = build_batched_arrays(BATCH_SEEDS, N_STEPS)
    params = _params_vec()

    @jax.jit
    def loss_fn(p):
        m = compact_metrics_realistic_batched(cfg, batched, p)
        return -jnp.mean(m["edge_submission"])

    @jax.jit
    def grad_fn(p):
        return jax.grad(loss_fn)(p)

    val = loss_fn(params)
    grad = grad_fn(params)
    assert jnp.isfinite(val)
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.linalg.norm(grad)) > 0.0


# ----- trainer end-to-end -----
def test_train_short_run_reduces_loss(tmp_path: Path) -> None:
    """A 30-step Adam run should already produce a clear loss decrease with
    finite, nonzero gradients throughout. Done criteria are evaluated against
    a shorter horizon than the full-spec 200-step run but should comfortably
    hold for >=20% reduction since the policy is far from optimal."""
    plot = tmp_path / "curve.png"
    json_path = tmp_path / "curve.json"
    summary = train(
        seeds=DEFAULT_SEEDS,
        steps=30,
        lr=5e-3,
        n_steps=64,
        plot_path=plot,
        json_path=json_path,
        verbose=False,
    )
    assert summary["gradients_all_finite"], "gradients went non-finite"
    assert summary["gradients_all_positive"], "some |grad| step was zero"
    assert summary["final_loss"] < summary["initial_loss"], "loss did not decrease"
    # At least a 20% reduction on this short run too — checks the optimization
    # is making real progress and not just oscillating.
    assert summary["twenty_percent_reduction"], (
        f"insufficient improvement: {summary['initial_loss']:+.4e} -> "
        f"{summary['final_loss']:+.4e}"
    )
    assert plot.exists() and plot.stat().st_size > 0
    assert json_path.exists()


def test_check_done_criteria_helper() -> None:
    """Smoke-check the criteria evaluator on synthetic loss curves."""
    # Strictly decreasing curve (initial >> final, MA monotone)
    losses = np.linspace(1.0, 0.1, 100)
    s = check_done_criteria(losses)
    assert s["twenty_percent_reduction"]
    assert s["monotone_ma_second_half"]

    # Increasing curve fails both
    losses_bad = np.linspace(0.1, 1.0, 100)
    s2 = check_done_criteria(losses_bad)
    assert not s2["twenty_percent_reduction"]
    assert not s2["monotone_ma_second_half"]


def test_train_full_run_meets_all_done_criteria(tmp_path: Path) -> None:
    """End-to-end 200-step training run satisfies all four DONE criteria.

    This is the canonical validation that back-prop works on the tape-smooth
    simulator: final_loss <= 0.8*initial_loss, 20-step MA monotone over the
    second half, and gradients finite-and-nonzero throughout.
    """
    plot = tmp_path / "curve.png"
    json_path = tmp_path / "curve.json"
    summary = train(
        seeds=DEFAULT_SEEDS,
        steps=200,
        lr=5e-3,
        lr_min_frac=0.02,
        n_steps=64,
        plot_path=plot,
        json_path=json_path,
        verbose=False,
    )
    assert summary["twenty_percent_reduction"], (
        f"Loss reduction below 20%: {summary['initial_loss']:+.4e} -> "
        f"{summary['final_loss']:+.4e}"
    )
    assert summary["monotone_ma_second_half"], (
        f"MA not monotone in 2nd half (max upward step: {summary['ma_max_up_step']:.4e}, "
        f"tolerance: {summary['ma_tolerance']:.4e})"
    )
    assert summary["gradients_all_finite"], "non-finite gradient detected"
    assert summary["gradients_all_positive"], "zero gradient detected"


def test_initial_params_is_finite() -> None:
    p = initial_params()
    assert p.dtype == jnp.float64
    assert p.shape == (20,)
    assert bool(jnp.all(jnp.isfinite(p)))
