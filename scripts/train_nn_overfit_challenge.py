"""Overfit a recurrent NN policy on challenge-mode training tapes.

Capability check for the diff sim + backprop pipeline:

  Hypothesis : our simulator + optimizer work.
  Expected   : with enough capacity and training, mean(edge_submission) on the
               training seed set should at least clear the top public submission
               score (~+522 over the upstream leaderboard distribution).
  Rejected if: we cannot get near +522 on training data.

Loss: -mean(edge_submission) over K challenge-mode training seeds, evaluated
in a single jit/vmap batched call. Adam with cosine LR + global-norm gradient
clipping (BPTT through up to 10k steps is the main pathology to control).

Example (smoke):
    .venv/bin/python scripts/train_nn_overfit_challenge.py \
        --seeds 0 --steps 20 --n-steps 128 --hidden-dim 16 --lr 3e-3

Example (real overfit attempt):
    .venv/bin/python scripts/train_nn_overfit_challenge.py \
        --seeds 0 1 2 ... 63 --steps 2000 --n-steps 10000 \
        --hidden-dim 64 --lr 1e-3 --grad-clip 1.0
"""

from __future__ import annotations

import argparse
import json
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

from arena_eval.diff_simple_amm import build_challenge_tape
from arena_eval.diff_simple_amm.tape_smooth import (
    challenge_tapes_to_batched_arrays,
    metrics_challenge_batched,
)
from arena_eval.exact_simple_amm import ExactSimpleAMMConfig
from arena_eval.exact_simple_amm.oracle import (
    FixedFeeClairvoyantController,
    GreedyStepOracleController,
    run_clairvoyant_seed,
)

from presentation.nn_overfit_challenge import (
    NNPolicyConfig,
    init_params,
    initial_fees,
    initial_hidden_state,
    make_nn_after_event,
    param_count,
)


def compute_references(
    seeds: tuple[int, ...],
    cfg: ExactSimpleAMMConfig,
    *,
    oracle_seed_cap: int = 8,
    fee_grid_size: int = 41,
    verbose: bool = True,
) -> dict:
    """Per-tape reference numbers for the bounds on training data:
      - fixed 30bps : per-seed floor (static baseline policy)
      - greedy oracle: per-seed upper bound (same-step lookahead clairvoyant)

    Oracle is expensive (~40s per length-10k seed), so we cap how many seeds
    we run it on (averaging over that subset).
    """
    seeds = tuple(seeds)
    fixed = FixedFeeClairvoyantController(0.003, 0.003)
    oracle = GreedyStepOracleController(fee_grid_size=fee_grid_size)

    fixed_edges = []
    for s in seeds:
        seed_cfg = replace(cfg, n_steps=cfg.n_steps)
        r = run_clairvoyant_seed(fixed, s, config=seed_cfg)
        fixed_edges.append(float(r.edge_submission))
    fixed_mean = float(np.mean(fixed_edges))
    if verbose:
        print(f"  fixed-30bps reference (K={len(seeds)}): mean edge = {fixed_mean:+.4f}")

    oracle_seeds = seeds[: min(oracle_seed_cap, len(seeds))]
    oracle_edges = []
    t0 = time.time()
    for s in oracle_seeds:
        seed_cfg = replace(cfg, n_steps=cfg.n_steps)
        r = run_clairvoyant_seed(oracle, s, config=seed_cfg)
        oracle_edges.append(float(r.edge_submission))
    oracle_mean = float(np.mean(oracle_edges)) if oracle_edges else None
    if verbose:
        print(f"  greedy-oracle reference (K={len(oracle_seeds)}): mean edge = "
              f"{oracle_mean:+.4f}  ({time.time()-t0:.1f}s)")

    return {
        "fixed_30bps_mean": fixed_mean,
        "fixed_30bps_per_seed": dict(zip(seeds, fixed_edges)),
        "oracle_mean": oracle_mean,
        "oracle_per_seed": dict(zip(oracle_seeds, oracle_edges)),
        "oracle_seed_cap": int(oracle_seed_cap),
    }


DEFAULT_PLOT_PATH = ROOT / "plots" / "nn_overfit_challenge_curve.png"
DEFAULT_JSON_PATH = ROOT / "plots" / "nn_overfit_challenge_curve.json"


def build_batched_arrays(seeds, n_steps):
    """Build (cfg, batched_arrays) using each seed's own challenge config.

    Each upstream seed picks its own gbm_sigma / lambda / mean_size via
    `ExactSimpleAMMConfig.from_seed`. We use seeds[0]'s cfg as the static cfg
    passed to the rollout — the rollout reads gbm_mu / gbm_sigma / gbm_dt /
    retail_buy_prob from cfg, but with `jax.vmap` over per-seed *tapes* the
    seed-specific randomness lives in the tape itself. For a faithful test
    against the leaderboard distribution we therefore force all seeds to use
    *seeds[0]*'s gbm_sigma (else metrics would mix mismatched cfgs).

    NOTE: this is fine for the overfit experiment because we'll just declare
    the training distribution to be `from_seed(seeds[0])` and draw K seeds
    from that. Each seed still draws its own GBM path + retail flow.
    """
    cfg = replace(ExactSimpleAMMConfig.from_seed(seeds[0]), n_steps=n_steps)
    tapes = [
        build_challenge_tape(config=cfg, seed=s)
        for s in seeds
    ]
    return cfg, challenge_tapes_to_batched_arrays(tapes)


def adam_init(params: dict):
    leaves = jax.tree_util.tree_leaves(params)
    treedef = jax.tree_util.tree_structure(params)
    m = jax.tree_util.tree_unflatten(treedef, [jnp.zeros_like(l) for l in leaves])
    v = jax.tree_util.tree_unflatten(treedef, [jnp.zeros_like(l) for l in leaves])
    return {"m": m, "v": v, "t": jnp.asarray(0, dtype=jnp.int32)}


def _global_norm(tree) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(jnp.asarray(l) ** 2) for l in leaves))


def _clip_by_global_norm(tree, clip: float):
    g = _global_norm(tree)
    factor = jnp.minimum(1.0, clip / jnp.maximum(g, 1e-12))
    return jax.tree_util.tree_map(lambda x: x * factor, tree), g


def adam_step(params, state, grad, *, lr, b1=0.9, b2=0.999, eps=1e-8):
    t = state["t"] + 1
    m = jax.tree_util.tree_map(lambda m_, g_: b1 * m_ + (1 - b1) * g_, state["m"], grad)
    v = jax.tree_util.tree_map(lambda v_, g_: b2 * v_ + (1 - b2) * (g_ * g_), state["v"], grad)
    bc1 = 1.0 - b1 ** t
    bc2 = 1.0 - b2 ** t
    new_params = jax.tree_util.tree_map(
        lambda p_, m_, v_: p_ - lr * (m_ / bc1) / (jnp.sqrt(v_ / bc2) + eps),
        params, m, v,
    )
    return new_params, {"m": m, "v": v, "t": t}


def sgdm_init(params: dict):
    """SGD-with-momentum optimizer state."""
    leaves = jax.tree_util.tree_leaves(params)
    treedef = jax.tree_util.tree_structure(params)
    v = jax.tree_util.tree_unflatten(treedef, [jnp.zeros_like(l) for l in leaves])
    return {"v": v, "t": jnp.asarray(0, dtype=jnp.int32)}


def sgdm_step(params, state, grad, *, lr, momentum=0.9):
    """Heavy-ball SGD-with-momentum. Plain — no per-parameter scaling."""
    t = state["t"] + 1
    v = jax.tree_util.tree_map(
        lambda v_, g_: momentum * v_ + g_,
        state["v"], grad,
    )
    new_params = jax.tree_util.tree_map(
        lambda p_, v_: p_ - lr * v_,
        params, v,
    )
    return new_params, {"v": v, "t": t}


def cosine_lr(step_idx, *, lr_peak, lr_min, total_steps):
    progress = jnp.clip(step_idx / jnp.maximum(total_steps, 1), 0.0, 1.0)
    cos = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
    return lr_min + (lr_peak - lr_min) * cos


def train(
    *,
    seeds: tuple[int, ...],
    steps: int,
    lr: float,
    lr_min_frac: float,
    n_steps: int,
    hidden_dim: int,
    feature_dim: int,
    base_fee: float,
    fee_amplitude: float,
    init_scale: float,
    grad_clip: float,
    plot_path: Path,
    json_path: Path,
    nn_seed: int,
    oracle_seed_cap: int = 8,
    skip_references: bool = False,
    optimizer: str = "adam",
    sgd_momentum: float = 0.9,
    verbose: bool = True,
) -> dict:
    nn_cfg = NNPolicyConfig(
        hidden_dim=hidden_dim,
        feature_dim=feature_dim,
        base_fee=base_fee,
        fee_amplitude=fee_amplitude,
        init_scale=init_scale,
    )
    after_event = make_nn_after_event(nn_cfg)
    init_state = initial_hidden_state(nn_cfg)
    params = init_params(nn_cfg, seed=nn_seed)
    if optimizer == "adam":
        opt_state = adam_init(params)
    elif optimizer == "sgdm":
        opt_state = sgdm_init(params)
    else:
        raise ValueError(f"unknown optimizer {optimizer!r}")

    cfg, batched = build_batched_arrays(seeds, n_steps)

    references = None
    if not skip_references:
        if verbose:
            print("Computing per-tape reference numbers ...")
        references = compute_references(
            tuple(seeds), cfg,
            oracle_seed_cap=oracle_seed_cap,
            verbose=verbose,
        )

    init_bid, init_ask = initial_fees(nn_cfg)

    def loss_fn(p):
        m = metrics_challenge_batched(
            cfg, batched,
            after_event=after_event,
            params=p,
            initial_policy_state=init_state,
            initial_bid=init_bid,
            initial_ask=init_ask,
        )
        return -jnp.mean(m["edge_submission"])

    value_and_grad = jax.value_and_grad(loss_fn)
    lr_min = lr * lr_min_frac

    if optimizer == "adam":
        opt_step = lambda p, s, g, *, lr_: adam_step(p, s, g, lr=lr_)
    else:  # sgdm
        opt_step = lambda p, s, g, *, lr_: sgdm_step(p, s, g, lr=lr_, momentum=sgd_momentum)

    @jax.jit
    def step_fn(params, opt_state):
        loss, grad = value_and_grad(params)
        clipped, raw_norm = _clip_by_global_norm(grad, grad_clip)
        cur_lr = cosine_lr(opt_state["t"], lr_peak=lr, lr_min=lr_min, total_steps=steps)
        new_params, new_opt = opt_step(params, opt_state, clipped, lr_=cur_lr)
        return new_params, new_opt, loss, raw_norm

    if verbose:
        print(f"K={len(seeds)} seeds, n_steps={n_steps}, hidden={hidden_dim}, "
              f"|params|={param_count(params)}, opt={optimizer}, "
              f"lr={lr}->{lr_min:.2e}, clip={grad_clip}")

    losses = np.zeros(steps + 1, dtype=np.float64)
    grad_norms = np.zeros(steps + 1, dtype=np.float64)

    t0 = time.time()
    init_loss = float(loss_fn(params))
    losses[0] = init_loss
    init_grad = jax.grad(loss_fn)(params)
    grad_norms[0] = float(_global_norm(init_grad))
    if verbose:
        print(f"  step    0  loss={init_loss:+.6e}  |grad|={grad_norms[0]:.4e}  "
              f"(compile+initial-eval={time.time()-t0:.2f}s)")

    t1 = time.time()
    for i in range(1, steps + 1):
        params, opt_state, loss, raw_norm = step_fn(params, opt_state)
        losses[i] = float(loss)
        grad_norms[i] = float(raw_norm)
        if verbose and (i <= 5 or i % max(1, steps // 10) == 0):
            print(f"  step {i:4d}  loss={float(loss):+.6e}  "
                  f"edge_mean={-float(loss):+.6e}  |grad|={grad_norms[i]:.4e}")

    total_train = time.time() - t1
    if verbose:
        print(f"  total training: {total_train:.2f}s  ({total_train*1000/max(steps,1):.1f} ms/step)")

    summary = {
        "seeds": list(seeds),
        "n_steps": int(n_steps),
        "hidden_dim": int(hidden_dim),
        "feature_dim": int(feature_dim),
        "param_count": int(param_count(params)),
        "lr_peak": float(lr),
        "lr_min": float(lr_min),
        "grad_clip": float(grad_clip),
        "optimizer": str(optimizer),
        "sgd_momentum": float(sgd_momentum) if optimizer == "sgdm" else None,
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "initial_edge_mean": -float(losses[0]),
        "final_edge_mean": -float(losses[-1]),
        "gradients_all_finite": bool(np.all(np.isfinite(grad_norms))),
        "gradients_all_positive": bool(np.all(grad_norms > 0.0)),
        "compile_plus_init_seconds": float(t1 - t0),
        "train_seconds": float(total_train),
        "references": references,
    }

    plot_path = Path(plot_path)
    json_path = Path(json_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    # Plot (loss + grad norm) -- matches the train_backprop_policy style.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax_loss, ax_grad) = plt.subplots(2, 1, figsize=(7.5, 6.0), sharex=True)
    x = np.arange(len(losses))
    ax_loss.plot(x, -losses, lw=1.5, color="tab:blue", label="mean(edge_submission)")
    ax_loss.axhline(522.0, color="tab:red", ls="--", lw=1.0, label="public top (≈+522, 1k-seed distrib.)")
    if references is not None:
        fixed_mean = references["fixed_30bps_mean"]
        ax_loss.axhline(fixed_mean, color="tab:gray", ls=":", lw=1.0,
                        label=f"fixed 30bps  ({fixed_mean:+.1f})")
        if references["oracle_mean"] is not None:
            ax_loss.axhline(references["oracle_mean"], color="tab:green", ls="--", lw=1.0,
                            label=f"greedy oracle  ({references['oracle_mean']:+.1f}, "
                                  f"K={len(references['oracle_per_seed'])})")
    ax_loss.set_ylabel("edge_submission (training mean)")
    ax_loss.set_title(
        f"NN overfit on challenge mode  |  init {summary['initial_edge_mean']:+.3e} → "
        f"final {summary['final_edge_mean']:+.3e}  (K={len(seeds)}, n_steps={n_steps})"
    )
    ax_loss.legend(loc="best", fontsize=9)
    ax_loss.grid(True, alpha=0.3)
    ax_grad.semilogy(x, np.maximum(grad_norms, 1e-30), lw=1.0, color="tab:green")
    ax_grad.set_xlabel("Adam step")
    ax_grad.set_ylabel("|grad|  (pre-clip, log)")
    ax_grad.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)

    with json_path.open("w") as f:
        flat_params = {k: np.asarray(v).tolist() for k, v in params.items()}
        json.dump(
            {
                "summary": summary,
                "loss_curve": losses.tolist(),
                "edge_curve": (-losses).tolist(),
                "grad_norms": grad_norms.tolist(),
                "final_params": flat_params,
                "nn_config": {
                    "hidden_dim": nn_cfg.hidden_dim,
                    "feature_dim": nn_cfg.feature_dim,
                    "base_fee": nn_cfg.base_fee,
                    "fee_amplitude": nn_cfg.fee_amplitude,
                    "min_fee": nn_cfg.min_fee,
                    "max_fee": nn_cfg.max_fee,
                    "init_scale": nn_cfg.init_scale,
                },
            },
            f,
            indent=2,
        )

    if verbose:
        print(f"\n  initial -> final edge:  "
              f"{summary['initial_edge_mean']:+.4f} -> {summary['final_edge_mean']:+.4f}")
        if references is not None:
            fmean = references["fixed_30bps_mean"]
            omean = references["oracle_mean"]
            final = summary["final_edge_mean"]
            print(f"  fixed-30bps mean = {fmean:+.4f}")
            if omean is not None:
                gap_closed = (final - fmean) / max(omean - fmean, 1e-12)
                print(f"  oracle mean      = {omean:+.4f}  (cap K={summary['references']['oracle_seed_cap']})")
                print(f"  gap closed: {100*gap_closed:.1f}% of (oracle - 30bps)")
        print(f"  plot: {plot_path}")
        print(f"  json: {json_path}")

    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-min-frac", type=float, default=0.05)
    p.add_argument("--n-steps", type=int, default=10000,
                   help="tape length per episode (upstream uses 10_000)")
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--feature-dim", type=int, default=32)
    p.add_argument("--base-fee", type=float, default=0.003)
    p.add_argument("--fee-amplitude", type=float, default=0.02)
    p.add_argument("--init-scale", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--nn-seed", type=int, default=0)
    p.add_argument("--plot", type=Path, default=DEFAULT_PLOT_PATH)
    p.add_argument("--json", type=Path, default=DEFAULT_JSON_PATH)
    p.add_argument("--oracle-seed-cap", type=int, default=8,
                   help="max number of seeds to run the greedy oracle on (~40s each at n=10k)")
    p.add_argument("--skip-references", action="store_true",
                   help="don't compute fixed-30bps / oracle references (use for quick iter)")
    p.add_argument("--optimizer", choices=["adam", "sgdm"], default="adam")
    p.add_argument("--sgd-momentum", type=float, default=0.9,
                   help="momentum coefficient (only used with --optimizer sgdm)")
    args = p.parse_args()

    train(
        seeds=tuple(args.seeds),
        steps=args.steps,
        lr=args.lr,
        lr_min_frac=args.lr_min_frac,
        n_steps=args.n_steps,
        hidden_dim=args.hidden_dim,
        feature_dim=args.feature_dim,
        base_fee=args.base_fee,
        fee_amplitude=args.fee_amplitude,
        init_scale=args.init_scale,
        grad_clip=args.grad_clip,
        plot_path=args.plot,
        json_path=args.json,
        nn_seed=args.nn_seed,
        oracle_seed_cap=args.oracle_seed_cap,
        skip_references=args.skip_references,
        optimizer=args.optimizer,
        sgd_momentum=args.sgd_momentum,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
