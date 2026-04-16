"""CLI entrypoint for the first baseline trainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from amm_gym import AMMFeeEnv
from amm_gym.baselines import benchmark_depth_policies
from demo.common import collect_rollout
from demo.presets import (
    DEFAULT_DEMO_SEED,
    DEFAULT_DEMO_STEPS,
    DEFAULT_WINDOW_SIZE,
    build_hackathon_demo_config,
)
from training.cem import CEMConfig, CEMTrainer
from training.policy import LinearTanhPolicy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a baseline AMM agent")
    parser.add_argument("--steps", type=int, default=DEFAULT_DEMO_STEPS)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--elite-frac", type=float, default=0.25)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=DEFAULT_DEMO_SEED)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--comparison-plot", type=Path, default=None)
    return parser


def make_env_factory(args: argparse.Namespace):
    def factory() -> AMMFeeEnv:
        config = build_hackathon_demo_config(seed=args.seed, steps=args.steps)
        return AMMFeeEnv(config=config, window_size=args.window_size)

    return factory


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    trainer = CEMTrainer(
        env_factory=make_env_factory(args),
        config=CEMConfig(
            population_size=args.population_size,
            elite_frac=args.elite_frac,
            iterations=args.iterations,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
        ),
    )
    result = trainer.train()

    policy = LinearTanhPolicy(trainer.spec, result.best_params.astype(np.float32))
    eval_env = make_env_factory(args)()
    trace = collect_rollout(eval_env, policy.act, seed=args.seed + 50_000)
    info = trace.final_info

    payload = {
        "best_score": result.best_score,
        "evaluation_reward": trace.total_reward,
        "final_edge": info["edge"],
        "final_pnl": info["pnl"],
        "history": result.history,
    }

    print(json.dumps(payload, indent=2, sort_keys=True))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.output,
            best_params=result.best_params,
            best_score=result.best_score,
        )

    if args.plot is not None:
        save_training_plot(args.plot, result.history)

    if args.comparison_plot is not None:
        baseline_env = make_env_factory(args)()
        baseline_policy = benchmark_depth_policies()["balanced"]
        baseline_trace = collect_rollout(baseline_env, baseline_policy, seed=args.seed + 50_000)
        save_comparison_plot(args.comparison_plot, baseline_trace, trace)


def save_training_plot(path: Path, history: list[dict[str, float]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "matplotlib is required for --plot. Install with `pip install -e .[demo]`."
        ) from exc

    iterations = [entry["iteration"] for entry in history]
    mean_scores = [entry["mean_score"] for entry in history]
    best_scores = [entry["best_score"] for entry in history]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(iterations, mean_scores, marker="o", color="#2563eb", label="population mean")
    ax.plot(iterations, best_scores, marker="o", color="#dc2626", label="population best")
    ax.set_title("CEM training progress on the canonical hackathon scenario")
    ax.set_xlabel("iteration")
    ax.set_ylabel("episode reward")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    print(f"plot={path}")


def save_comparison_plot(path: Path, baseline_trace, trained_trace) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "matplotlib is required for --comparison-plot. Install with `pip install -e .[demo]`."
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")

    baseline_cum_reward = np.cumsum(np.asarray(baseline_trace.rewards))
    trained_cum_reward = np.cumsum(np.asarray(trained_trace.rewards))
    axes[0, 0].plot(baseline_trace.steps, baseline_cum_reward, color="#64748b", label="balanced baseline")
    axes[0, 0].plot(trained_trace.steps, trained_cum_reward, color="#2563eb", label="trained policy")
    axes[0, 0].set_title("Cumulative reward on the same seeded scenario")
    axes[0, 0].set_ylabel("cumulative reward")
    axes[0, 0].legend(loc="best")

    axes[0, 1].plot(baseline_trace.steps, baseline_trace.edges, color="#64748b", label="baseline edge")
    axes[0, 1].plot(trained_trace.steps, trained_trace.edges, color="#2563eb", label="trained edge")
    axes[0, 1].plot(baseline_trace.steps, baseline_trace.pnls, color="#f59e0b", alpha=0.55, label="baseline pnl")
    axes[0, 1].plot(trained_trace.steps, trained_trace.pnls, color="#dc2626", alpha=0.75, label="trained pnl")
    axes[0, 1].set_title("Value outcomes")
    axes[0, 1].set_ylabel("value")
    axes[0, 1].legend(loc="best")

    axes[1, 0].plot(
        baseline_trace.steps,
        np.cumsum(np.asarray(baseline_trace.execution_volumes)),
        color="#64748b",
        label="baseline submission volume",
    )
    axes[1, 0].plot(
        trained_trace.steps,
        np.cumsum(np.asarray(trained_trace.execution_volumes)),
        color="#2563eb",
        label="trained submission volume",
    )
    axes[1, 0].set_title("How much retail flow the submission venue wins")
    axes[1, 0].set_xlabel("step")
    axes[1, 0].set_ylabel("cumulative submission volume (Y)")
    axes[1, 0].legend(loc="best")

    axes[1, 1].plot(baseline_trace.steps, baseline_trace.bid_near, color="#64748b", label="baseline bid near")
    axes[1, 1].plot(baseline_trace.steps, baseline_trace.ask_near, color="#94a3b8", label="baseline ask near")
    axes[1, 1].plot(trained_trace.steps, trained_trace.bid_near, color="#2563eb", label="trained bid near")
    axes[1, 1].plot(trained_trace.steps, trained_trace.ask_near, color="#dc2626", label="trained ask near")
    axes[1, 1].set_title("Control behavior near the touch")
    axes[1, 1].set_xlabel("step")
    axes[1, 1].set_ylabel("near depth (Y)")
    axes[1, 1].legend(loc="best")

    fig.suptitle("Baseline vs trained policy on one canonical hackathon episode")

    fig.tight_layout()
    fig.savefig(path)
    print(f"plot={path}")


if __name__ == "__main__":
    main()
