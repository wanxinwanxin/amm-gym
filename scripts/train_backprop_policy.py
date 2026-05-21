"""Back-prop training of SubmissionCompactParams on the tape-smooth simulator.

Loss = -mean(edge_submission) across K realistic-mode seeds, batched in a
single jit/vmap call. Adam optimizer (pure-jnp). Outputs:

- `plots/backprop_training_curve.png`     — loss vs. step
- prints initial/final loss, % improvement, and gradient diagnostics
- saves the trained param vector + history to `plots/backprop_training_curve.json`

Example:
    .venv/bin/python scripts/train_backprop_policy.py --steps 200 --lr 5e-3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from arena_eval.diff_simple_amm import build_realistic_tape
from arena_eval.diff_simple_amm.objectives import (
    submission_compact_bounds,
    submission_compact_param_vector,
)
from arena_eval.diff_simple_amm.tape_smooth import (
    compact_metrics_realistic_batched,
    realistic_tapes_to_batched_arrays,
)
from arena_eval.exact_simple_amm import ExactSimpleAMMConfig
from arena_policies.submission_safe import SubmissionCompactParams


DEFAULT_SEEDS = (3, 8, 11, 13, 19, 23, 29, 31)
DEFAULT_STEPS = 200
DEFAULT_LR = 5e-3
DEFAULT_N_STEPS = 64
DEFAULT_PLOT_PATH = ROOT / "plots" / "backprop_training_curve.png"
DEFAULT_JSON_PATH = ROOT / "plots" / "backprop_training_curve.json"


def build_batched_arrays(seeds, n_steps):
    """Build (cfg, batched_arrays) for the trainer.

    All chosen seeds share the same realistic cfg geometry, so we use cfg from
    `seeds[0]` and stack per-seed tapes.
    """
    cfg = replace(ExactSimpleAMMConfig.real_data_from_seed(seeds[0]), n_steps=n_steps)
    tapes = [
        build_realistic_tape(
            config=replace(ExactSimpleAMMConfig.real_data_from_seed(s), n_steps=n_steps),
            seed=s,
        )
        for s in seeds
    ]
    return cfg, realistic_tapes_to_batched_arrays(tapes)


def initial_params() -> jnp.ndarray:
    return submission_compact_param_vector(SubmissionCompactParams()).astype(jnp.float64)


def project_to_bounds(p: jnp.ndarray) -> jnp.ndarray:
    """Clip params to the documented compact bounds, plus a few hardening clips
    (decays in (0, 1), fees >= 0). These are required for stability — a few
    decay coefficients drifting to exactly 1.0 destabilizes the recursion.
    """
    lower, upper = submission_compact_bounds()
    lower = lower.astype(jnp.float64)
    upper = upper.astype(jnp.float64)
    # Decays must stay strictly in (0, 1) (we lift the upper bound to 0.999 already)
    # but clip lower to a small positive to avoid zero-multiplier degeneracy.
    decay_idx = jnp.asarray([3, 4, 5, 6, 7, 8, 9], dtype=jnp.int32)
    lower_d = lower.at[decay_idx].max(0.01)
    return jnp.clip(p, lower_d, upper)


def adam_init(params: jnp.ndarray):
    return {
        "m": jnp.zeros_like(params),
        "v": jnp.zeros_like(params),
        "t": jnp.asarray(0, dtype=jnp.int32),
    }


def adam_step(params, state, grad, *, lr, b1=0.9, b2=0.999, eps=1e-8):
    t = state["t"] + 1
    m = b1 * state["m"] + (1 - b1) * grad
    v = b2 * state["v"] + (1 - b2) * (grad * grad)
    m_hat = m / (1 - b1 ** t)
    v_hat = v / (1 - b2 ** t)
    new_params = params - lr * m_hat / (jnp.sqrt(v_hat) + eps)
    return new_params, {"m": m, "v": v, "t": t}


def build_train_step(cfg, batched_arrays, lr: float):
    """Returns a jit-compiled (params, state) -> (params, state, loss, grad) step."""

    def loss_fn(p):
        m = compact_metrics_realistic_batched(cfg, batched_arrays, p)
        return -jnp.mean(m["edge_submission"])

    value_and_grad = jax.value_and_grad(loss_fn)

    @jax.jit
    def step_fn(params, state):
        loss, grad = value_and_grad(params)
        new_params, new_state = adam_step(params, state, grad, lr=lr)
        new_params = project_to_bounds(new_params)
        return new_params, new_state, loss, grad

    return step_fn, loss_fn


def moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or arr.size < window:
        return arr.copy()
    kernel = np.ones(window, dtype=arr.dtype) / window
    return np.convolve(arr, kernel, mode="valid")


def check_done_criteria(loss_curve: np.ndarray) -> dict:
    """Apply the spec's three quantitative criteria to a loss history."""
    initial_loss = float(loss_curve[0])
    final_loss = float(loss_curve[-1])
    improvement = (initial_loss - final_loss) / max(abs(initial_loss), 1e-12)
    twenty_percent_ok = final_loss <= 0.8 * initial_loss

    # 20-step moving average over second half should be monotone-decreasing
    half = len(loss_curve) // 2
    second_half = loss_curve[half:]
    window = min(20, max(2, second_half.size // 4))
    ma = moving_average(second_half, window)
    monotone_decreasing = bool(np.all(np.diff(ma) <= 1e-12))

    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "improvement": improvement,
        "twenty_percent_reduction": bool(twenty_percent_ok),
        "monotone_ma_second_half": monotone_decreasing,
        "ma_window": int(window),
    }


def plot_loss_curve(loss_curve, grad_norms, *, out_path: Path, summary: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_loss, ax_grad) = plt.subplots(2, 1, figsize=(7.5, 6.0), sharex=True)
    steps = np.arange(len(loss_curve))
    ax_loss.plot(steps, loss_curve, lw=1.5, color="tab:blue", label="loss = -mean(edge_submission)")
    window = summary.get("ma_window", 20)
    if window > 1 and loss_curve.size > window:
        ma = moving_average(loss_curve, window)
        ax_loss.plot(
            np.arange(window - 1, window - 1 + ma.size),
            ma,
            lw=1.5,
            color="tab:red",
            label=f"{window}-step MA",
        )
    ax_loss.axvline(len(loss_curve) // 2, color="grey", ls=":", alpha=0.5, label="step M/2")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title(
        f"Backprop training of SubmissionCompactParams\n"
        f"initial {summary['initial_loss']:+.4e} -> final {summary['final_loss']:+.4e} "
        f"({100*summary['improvement']:.1f}% reduction)"
    )
    ax_loss.legend(loc="best", fontsize=9)
    ax_loss.grid(True, alpha=0.3)

    ax_grad.semilogy(steps, grad_norms, lw=1.0, color="tab:green")
    ax_grad.set_xlabel("Adam step")
    ax_grad.set_ylabel("|grad|  (log)")
    ax_grad.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def train(
    *,
    seeds=DEFAULT_SEEDS,
    steps: int = DEFAULT_STEPS,
    lr: float = DEFAULT_LR,
    n_steps: int = DEFAULT_N_STEPS,
    plot_path: Path = DEFAULT_PLOT_PATH,
    json_path: Path = DEFAULT_JSON_PATH,
    verbose: bool = True,
) -> dict:
    cfg, batched = build_batched_arrays(seeds, n_steps)
    params = initial_params()
    state = adam_init(params)

    step_fn, loss_fn = build_train_step(cfg, batched, lr)
    if verbose:
        print(f"K={len(seeds)} seeds, n_steps={n_steps}, lr={lr}, M={steps}")
        print(f"params shape: {params.shape}")

    losses = np.zeros(steps + 1, dtype=np.float64)
    grad_norms = np.zeros(steps + 1, dtype=np.float64)

    t_compile_start = time.time()
    init_loss = float(loss_fn(params))
    losses[0] = init_loss
    grad_norms[0] = float(jnp.linalg.norm(jax.grad(loss_fn)(params)))
    if verbose:
        print(f"  step    0  loss={init_loss:+.6e}  |grad|={grad_norms[0]:.4e}  "
              f"(setup={time.time()-t_compile_start:.2f}s)")

    t_train_start = time.time()
    for i in range(1, steps + 1):
        params, state, loss, grad = step_fn(params, state)
        losses[i] = float(loss)
        grad_norms[i] = float(jnp.linalg.norm(grad))
        if verbose and (i <= 5 or i % max(1, steps // 10) == 0):
            print(f"  step {i:4d}  loss={float(loss):+.6e}  |grad|={grad_norms[i]:.4e}")

    if verbose:
        print(f"  total training time: {time.time()-t_train_start:.2f}s "
              f"({(time.time()-t_train_start)*1000/steps:.1f} ms/step)")

    summary = check_done_criteria(losses)
    summary.update(
        {
            "seeds": list(seeds),
            "steps": int(steps),
            "lr": float(lr),
            "n_steps_per_episode": int(n_steps),
            "gradients_all_finite": bool(np.all(np.isfinite(grad_norms))),
            "gradients_all_positive": bool(np.all(grad_norms > 0.0)),
        }
    )

    plot_path = Path(plot_path)
    json_path = Path(json_path)
    plot_loss_curve(losses, grad_norms, out_path=plot_path, summary=summary)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as f:
        json.dump(
            {
                "summary": summary,
                "loss_curve": losses.tolist(),
                "grad_norms": grad_norms.tolist(),
                "final_params": np.asarray(params).tolist(),
            },
            f,
            indent=2,
        )

    if verbose:
        print(f"\nDone criteria check:")
        print(f"  initial -> final: {summary['initial_loss']:+.6e} -> {summary['final_loss']:+.6e}")
        print(f"  improvement: {100*summary['improvement']:.2f}%  (>= 20% required? "
              f"{summary['twenty_percent_reduction']})")
        print(f"  MA monotone over 2nd half: {summary['monotone_ma_second_half']} "
              f"(window={summary['ma_window']})")
        print(f"  gradients finite throughout: {summary['gradients_all_finite']}")
        print(f"  gradients nonzero throughout: {summary['gradients_all_positive']}")
        print(f"  plot: {plot_path}")
        print(f"  json: {json_path}")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS,
                        help="n_steps per episode (tape length)")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT_PATH)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON_PATH)
    args = parser.parse_args()

    train(
        seeds=tuple(args.seeds),
        steps=args.steps,
        lr=args.lr,
        n_steps=args.n_steps,
        plot_path=args.plot,
        json_path=args.json,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
