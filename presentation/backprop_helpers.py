"""Helpers for the back-prop env presentation notebook.

Provides:
- `parity_table`: run exact + tape_smooth on K seeds for SubmissionCompact and
  return a tidy DataFrame for parity-scatter plots.
- `plot_parity_scatter`: side-by-side scatter (exact vs diff) for edge / pnl /
  avg-bid-fee, with `y=x` reference and tolerance annotation.
- `run_quick_training`: thin wrapper around `scripts.train_backprop_policy.train`
  that returns the loss curve + grad norms + summary in a notebook-friendly dict.
- `plot_training_curves`: loss (left) + |grad| (right) using the existing palette.
- `evaluate_compact_params`: evaluate `SubmissionCompactParams` (or a param vec)
  against a fixed seed set via tape_smooth, returning per-seed edge.
- `plot_held_out_eval`: grouped bar chart (init vs trained) of per-seed edge.

These follow the same style/colour discipline as `helpers.py`.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from presentation.helpers import STYLE, _apply_style

# ---------------------------------------------------------------------------
# Local style: keep palette compatible with `helpers.py`
# ---------------------------------------------------------------------------
BACKPROP_STYLE = {
    "exact": {"color": STYLE["observed"]["color"], "label": "Exact simulator"},
    "diff": {"color": STYLE["realistic"]["color"], "label": "tape_smooth (diff)"},
    "init": {"color": STYLE["challenge"]["color"], "label": "Initial params"},
    "trained": {"color": STYLE["realistic"]["color"], "label": "Trained params"},
    "loss": {"color": STYLE["observed"]["color"], "label": "loss = -mean(edge_submission)"},
    "loss_ma": {"color": STYLE["challenge"]["color"], "label": "20-step MA"},
    "grad": {"color": STYLE["realistic"]["color"], "label": "|grad|"},
}


# ===================================================================
# Section 1: parity tape_smooth vs exact
# ===================================================================

def parity_table(
    seeds: tuple[int, ...],
    n_steps: int = 64,
    params=None,
) -> pd.DataFrame:
    """For each seed, evaluate `SubmissionCompactParams` in both simulators.

    Returns a DataFrame with columns:
      seed, metric, exact, diff, abs_diff, rel_diff
    where metric in {edge_submission, pnl_submission, average_bid_fee_submission}.

    Uses the batched tape_smooth entrypoint (one jit call) for the diff side
    and per-seed `run_seed` for the exact side.
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from arena_eval.diff_simple_amm import build_realistic_tape
    from arena_eval.diff_simple_amm.objectives import (
        submission_compact_param_vector,
    )
    from arena_eval.diff_simple_amm.tape_smooth import (
        compact_metrics_realistic_batched,
        realistic_tapes_to_batched_arrays,
    )
    from arena_eval.exact_simple_amm import (
        ExactSimpleAMMConfig,
        FixedFeeStrategy,
        run_seed,
    )
    from arena_policies.submission_safe import (
        SubmissionCompactParams,
        SubmissionCompactStrategy,
    )

    if params is None:
        params_dc = SubmissionCompactParams()
    else:
        params_dc = params
    param_vec = submission_compact_param_vector(params_dc).astype(jnp.float64)

    metric_keys = (
        "edge_submission",
        "pnl_submission",
        "average_bid_fee_submission",
    )

    tapes = []
    exact_vals: dict[str, list[float]] = {k: [] for k in metric_keys}
    for s in seeds:
        cfg = replace(ExactSimpleAMMConfig.real_data_from_seed(s), n_steps=n_steps)
        tape = build_realistic_tape(config=cfg, seed=s)
        tapes.append(tape)
        ex = run_seed(
            SubmissionCompactStrategy(params_dc),
            s,
            config=cfg,
            normalizer_strategy=FixedFeeStrategy(),
        )
        exact_vals["edge_submission"].append(float(ex.edge_submission))
        exact_vals["pnl_submission"].append(float(ex.pnl_submission))
        exact_vals["average_bid_fee_submission"].append(
            float(ex.average_bid_fee_submission)
        )

    cfg0 = replace(ExactSimpleAMMConfig.real_data_from_seed(seeds[0]), n_steps=n_steps)
    batched = realistic_tapes_to_batched_arrays(tapes)
    batched_metrics = compact_metrics_realistic_batched(cfg0, batched, param_vec)
    diff_vals = {k: np.asarray(batched_metrics[k]) for k in metric_keys}

    rows = []
    for k in metric_keys:
        for i, s in enumerate(seeds):
            ex_v = exact_vals[k][i]
            d_v = float(diff_vals[k][i])
            abs_d = abs(d_v - ex_v)
            scale = max(abs(ex_v), 1e-12)
            rows.append(
                {
                    "seed": s,
                    "metric": k,
                    "exact": ex_v,
                    "diff": d_v,
                    "abs_diff": abs_d,
                    "rel_diff": abs_d / scale,
                }
            )
    return pd.DataFrame(rows)


def plot_parity_scatter(
    df: pd.DataFrame,
    *,
    metric_titles: dict[str, str] | None = None,
    fig: plt.Figure | None = None,
) -> plt.Figure:
    """Side-by-side scatter (exact vs diff) per metric with a y=x reference.

    Each subplot shows the K seeds for one metric. Annotates the max relative
    error visible on the panel.
    """
    metrics = ["edge_submission", "pnl_submission", "average_bid_fee_submission"]
    titles = metric_titles or {
        "edge_submission": "edge_submission",
        "pnl_submission": "pnl_submission",
        "average_bid_fee_submission": "average_bid_fee_submission",
    }

    if fig is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    else:
        axes = fig.axes

    for ax, m in zip(axes, metrics):
        sub = df[df["metric"] == m]
        x = sub["exact"].values
        y = sub["diff"].values
        ax.scatter(
            x, y,
            s=72,
            color=BACKPROP_STYLE["diff"]["color"],
            edgecolors="white",
            zorder=5,
        )
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        pad = 0.05 * max(hi - lo, 1e-12)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                ls="--", color="#999", lw=1, label="y = x")
        max_rel = float(sub["rel_diff"].max())
        ax.text(
            0.04, 0.94,
            f"max rel error = {max_rel:.2e}",
            transform=ax.transAxes,
            fontsize=9.5,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      alpha=0.85, edgecolor="#ccc"),
        )
        ax.set_xlabel(f"Exact {titles[m]}", fontsize=11)
        ax.set_ylabel(f"tape_smooth {titles[m]}", fontsize=11)
        ax.set_title(titles[m], fontsize=12, fontweight="bold")
        _apply_style(ax)

    fig.tight_layout()
    return fig


# ===================================================================
# Section 2: backprop training
# ===================================================================

def run_quick_training(
    *,
    seeds: tuple[int, ...] = (3, 8, 11, 13, 19, 23, 29, 31),
    steps: int = 200,
    lr: float = 5e-3,
    lr_min_frac: float = 0.02,
    n_steps: int = 64,
    tmp_dir: Path | None = None,
) -> dict:
    """Wrap `scripts.train_backprop_policy.train` for notebook usage.

    Trains, then re-reads the JSON sidecar to surface the loss curve, grad
    norms, and the trained param vector. Plot/json get written to `tmp_dir`
    (defaults to a scratch path under `plots/`).
    """
    import json
    from scripts.train_backprop_policy import train

    if tmp_dir is None:
        import tempfile
        tmp_dir = Path(tempfile.mkdtemp(prefix="amm_backprop_"))
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    plot_path = tmp_dir / "backprop_training_curve.png"
    json_path = tmp_dir / "backprop_training_curve.json"

    # `train` calls `matplotlib.use("Agg")` internally to draw the PNG sidecar.
    # Preserve the caller's backend so inline plots in subsequent notebook
    # cells keep rendering.
    import matplotlib
    prev_backend = matplotlib.get_backend()
    try:
        summary = train(
            seeds=seeds,
            steps=steps,
            lr=lr,
            lr_min_frac=lr_min_frac,
            n_steps=n_steps,
            plot_path=plot_path,
            json_path=json_path,
            verbose=False,
        )
    finally:
        matplotlib.use(prev_backend, force=True)
    with json_path.open("r") as f:
        payload = json.load(f)
    return {
        "summary": summary,
        "loss_curve": np.asarray(payload["loss_curve"]),
        "grad_norms": np.asarray(payload["grad_norms"]),
        "final_params": np.asarray(payload["final_params"]),
        "json_path": json_path,
        "plot_path": plot_path,
    }


def plot_training_curves(
    loss_curve: np.ndarray,
    grad_norms: np.ndarray,
    summary: dict,
    *,
    fig: plt.Figure | None = None,
) -> plt.Figure:
    """Loss curve (left) + |grad| log-scale (right), styled to match helpers.py."""
    if fig is None:
        fig, (ax_loss, ax_grad) = plt.subplots(1, 2, figsize=(14, 5))
    else:
        ax_loss, ax_grad = fig.axes

    steps = np.arange(len(loss_curve))
    ax_loss.plot(
        steps, loss_curve,
        lw=1.6,
        color=BACKPROP_STYLE["loss"]["color"],
        label=BACKPROP_STYLE["loss"]["label"],
    )
    window = int(summary.get("ma_window", 20))
    if window > 1 and loss_curve.size > window:
        kernel = np.ones(window) / window
        ma = np.convolve(loss_curve, kernel, mode="valid")
        ax_loss.plot(
            np.arange(window - 1, window - 1 + ma.size),
            ma,
            lw=1.6,
            color=BACKPROP_STYLE["loss_ma"]["color"],
            label=f"{window}-step MA",
        )
    ax_loss.axvline(
        len(loss_curve) // 2,
        color="grey", ls=":", alpha=0.5, label="step M/2",
    )
    ax_loss.set_xlabel("Adam step", fontsize=11)
    ax_loss.set_ylabel("loss", fontsize=11)
    pct = 100.0 * summary["improvement"]
    ax_loss.set_title(
        f"Backprop loss curve\n"
        f"{summary['initial_loss']:+.3e} → {summary['final_loss']:+.3e}  "
        f"({pct:.1f}% reduction)",
        fontsize=12, fontweight="bold",
    )
    _apply_style(ax_loss)

    ax_grad.semilogy(
        steps, grad_norms,
        lw=1.4,
        color=BACKPROP_STYLE["grad"]["color"],
        label=BACKPROP_STYLE["grad"]["label"],
    )
    ax_grad.set_xlabel("Adam step", fontsize=11)
    ax_grad.set_ylabel("|grad|  (L2, log scale)", fontsize=11)
    finite_str = "finite ✓" if summary.get("gradients_all_finite") else "non-finite ✗"
    nonzero_str = "nonzero ✓" if summary.get("gradients_all_positive") else "zero ✗"
    ax_grad.set_title(
        f"Gradient norm vs step ({finite_str}, {nonzero_str})",
        fontsize=12, fontweight="bold",
    )
    _apply_style(ax_grad)

    fig.tight_layout()
    return fig


# ===================================================================
# Held-out evaluation
# ===================================================================

def evaluate_compact_params(
    param_vec: np.ndarray | "jnp.ndarray",
    seeds: tuple[int, ...],
    n_steps: int = 64,
) -> pd.DataFrame:
    """Evaluate a SubmissionCompact param vector on `seeds` (realistic mode).

    Returns columns: seed, edge_submission, pnl_submission, retail_volume_submission_y.
    Uses the batched tape_smooth entrypoint.
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from arena_eval.diff_simple_amm import build_realistic_tape
    from arena_eval.diff_simple_amm.tape_smooth import (
        compact_metrics_realistic_batched,
        realistic_tapes_to_batched_arrays,
    )
    from arena_eval.exact_simple_amm import ExactSimpleAMMConfig

    cfg0 = replace(ExactSimpleAMMConfig.real_data_from_seed(seeds[0]), n_steps=n_steps)
    tapes = [
        build_realistic_tape(
            config=replace(ExactSimpleAMMConfig.real_data_from_seed(s), n_steps=n_steps),
            seed=s,
        )
        for s in seeds
    ]
    batched = realistic_tapes_to_batched_arrays(tapes)
    pvec = jnp.asarray(np.asarray(param_vec), dtype=jnp.float64)
    m = compact_metrics_realistic_batched(cfg0, batched, pvec)
    return pd.DataFrame(
        {
            "seed": list(seeds),
            "edge_submission": np.asarray(m["edge_submission"]),
            "pnl_submission": np.asarray(m["pnl_submission"]),
            "retail_volume_submission_y": np.asarray(m["retail_volume_submission_y"]),
        }
    )


def plot_held_out_eval(
    init_df: pd.DataFrame,
    trained_df: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Grouped bar chart: per-seed edge_submission for initial vs trained params."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    seeds = init_df["seed"].tolist()
    x = np.arange(len(seeds))
    width = 0.38

    ax.bar(
        x - width / 2, init_df["edge_submission"].values, width,
        color=BACKPROP_STYLE["init"]["color"],
        label=BACKPROP_STYLE["init"]["label"], alpha=0.85,
    )
    ax.bar(
        x + width / 2, trained_df["edge_submission"].values, width,
        color=BACKPROP_STYLE["trained"]["color"],
        label=BACKPROP_STYLE["trained"]["label"], alpha=0.85,
    )
    ax.axhline(0, color="#444", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("Held-out seed", fontsize=11)
    ax.set_ylabel("edge_submission", fontsize=11)
    init_mean = float(init_df["edge_submission"].mean())
    trained_mean = float(trained_df["edge_submission"].mean())
    ax.set_title(
        f"Held-out evaluation: edge_submission per seed\n"
        f"mean init = {init_mean:+.4e}  →  mean trained = {trained_mean:+.4e}",
        fontsize=12, fontweight="bold",
    )
    _apply_style(ax)
    return ax
