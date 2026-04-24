"""CLI entrypoint for the research baseline trainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from amm_gym import AMMChallengeEnv, AMMFeeEnv
from demo.common import collect_rollout
from demo.presets import DEFAULT_DEMO_SEED, DEFAULT_WINDOW_SIZE
from training.algorithms.cem import CEMConfig, CEMTrainer
from training.eval import benchmark_scenarios, default_seed_split, evaluate_policy_across_scenarios
from training.eval.metrics import aggregate_episode_metrics, evaluate_episode
from training.policies import (
    FeatureLinearPolicySpace,
    LinearPolicySpace,
    MLPPolicySpace,
    SmoothedFeatureLinearPolicySpace,
    research_benchmark_policies,
)


OBJECTIVES = ["reward", "edge", "edge_advantage", "pnl", "pnl_advantage", "balanced"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a baseline AMM agent")
    parser.add_argument("--env-kind", default="classic", choices=["classic", "challenge"])
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--elite-frac", type=float, default=0.25)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=DEFAULT_DEMO_SEED)
    parser.add_argument(
        "--policy-family",
        default="linear",
        choices=["linear", "feature_linear", "feature_linear_smooth", "mlp"],
    )
    parser.add_argument("--objective", default=None, choices=OBJECTIVES)
    parser.add_argument("--hidden-sizes", default="32,32")
    parser.add_argument("--smoothing-alpha", type=float, default=0.7)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--comparison-plot", type=Path, default=None)
    return parser


def parse_hidden_sizes(raw: str) -> tuple[int, ...]:
    if not raw.strip():
        return (32, 32)
    return tuple(int(part) for part in raw.split(",") if part.strip())


def make_env_factory(args: argparse.Namespace):
    scenario = benchmark_scenarios(env_kind=args.env_kind)[args.scenario]
    return scenario.env_factory(window_size=args.window_size)


def make_policy_space_factory(args: argparse.Namespace):
    if args.policy_family == "linear":
        return LinearPolicySpace.from_env
    if args.policy_family == "feature_linear":
        return FeatureLinearPolicySpace.from_env
    if args.policy_family == "feature_linear_smooth":
        def factory(env):
            return SmoothedFeatureLinearPolicySpace.from_env(env, smoothing_alpha=args.smoothing_alpha)

        return factory

    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)

    def factory(env):
        return MLPPolicySpace.from_env(env, hidden_sizes=hidden_sizes)

    return factory


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    scenarios = benchmark_scenarios(env_kind=args.env_kind)
    if args.scenario is None:
        args.scenario = "challenge_single_regime" if args.env_kind == "challenge" else "regime_shift"
    elif args.scenario not in scenarios:
        parser.error(
            f"--scenario must be one of {sorted(scenarios)} for env kind `{args.env_kind}`"
        )
    if args.objective is None:
        args.objective = "reward" if args.env_kind == "challenge" else "edge_advantage"
    seed_split = default_seed_split()

    trainer = CEMTrainer(
        env_factory=make_env_factory(args),
        config=CEMConfig(
            population_size=args.population_size,
            elite_frac=args.elite_frac,
            iterations=args.iterations,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
            objective=args.objective,
            evaluation_seeds=seed_split.train[: max(args.eval_episodes, 1)],
            elite_reevaluation_seeds=seed_split.validation,
        ),
        policy_space_factory=make_policy_space_factory(args),
    )
    result = trainer.train()

    policy = trainer.policy_space.build_policy(result.best_params.astype(np.float32))
    eval_factory = make_env_factory(args)
    validation_metrics = [
        evaluate_episode(eval_factory, policy.act, seed=seed) for seed in seed_split.validation
    ]
    test_metrics = [evaluate_episode(eval_factory, policy.act, seed=seed) for seed in seed_split.test]
    env_probe = eval_factory()
    if isinstance(env_probe, AMMChallengeEnv):
        baseline_action = np.zeros(env_probe.action_space.shape, dtype=np.float32)

        def baseline_policy(_obs: np.ndarray) -> np.ndarray:
            return baseline_action.copy()
    else:
        baseline_policy = research_benchmark_policies()["balanced"]
    baseline_validation_metrics = [
        evaluate_episode(eval_factory, baseline_policy, seed=seed) for seed in seed_split.validation
    ]

    payload = {
        "scenario": args.scenario,
        "policy_family": args.policy_family,
        "objective": args.objective,
        "best_score": result.best_score,
        "history": result.history,
        "validation_summary": aggregate_episode_metrics(validation_metrics),
        "test_summary": aggregate_episode_metrics(test_metrics),
        "baseline_validation_summary": aggregate_episode_metrics(baseline_validation_metrics),
        "scenario_benchmark": evaluate_policy_across_scenarios(
            policy_name=f"trained_{args.policy_family}",
            policy=policy.act,
            scenario_names=(args.scenario,),
            split_name="test",
            window_size=args.window_size,
            seed_split=seed_split,
            env_kind=args.env_kind,
        ),
    }

    condensed_payload = {
        "scenario": args.scenario,
        "policy_family": args.policy_family,
        "env_kind": args.env_kind,
        "objective": args.objective,
        "best_score": result.best_score,
        "validation_edge_advantage_mean": payload["validation_summary"]["edge_advantage_mean"],
        "validation_edge_advantage_win_rate": payload["validation_summary"]["edge_advantage_win_rate"],
        "test_edge_advantage_mean": payload["test_summary"]["edge_advantage_mean"],
        "test_edge_advantage_win_rate": payload["test_summary"]["edge_advantage_win_rate"],
    }
    print(json.dumps(condensed_payload, indent=2, sort_keys=True))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.output,
            best_params=result.best_params,
            best_score=result.best_score,
        )
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True))

    if args.plot is not None:
        save_training_plot(args.plot, result.history)

    if args.comparison_plot is not None:
        baseline_trace = collect_rollout(eval_factory(), baseline_policy, seed=seed_split.validation[0])
        trained_trace = collect_rollout(eval_factory(), policy.act, seed=seed_split.validation[0])
        save_comparison_plot(args.comparison_plot, baseline_trace, trained_trace)


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
    selected_scores = [entry["selected_score"] for entry in history]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(iterations, mean_scores, marker="o", color="#2563eb", label="population mean")
    ax.plot(iterations, best_scores, marker="o", color="#dc2626", label="population best")
    ax.plot(iterations, selected_scores, marker="o", color="#0f766e", label="selected score")
    ax.set_title("CEM training progress on the research benchmark")
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

    baseline_edge_adv = np.asarray(baseline_trace.edges) - np.asarray(baseline_trace.normalizer_edges)
    trained_edge_adv = np.asarray(trained_trace.edges) - np.asarray(trained_trace.normalizer_edges)
    axes[0, 1].plot(baseline_trace.steps, baseline_edge_adv, color="#64748b", label="baseline edge advantage")
    axes[0, 1].plot(trained_trace.steps, trained_edge_adv, color="#2563eb", label="trained edge advantage")
    axes[0, 1].set_title("Edge advantage vs normalizer")
    axes[0, 1].set_ylabel("submission edge - normalizer edge")
    axes[0, 1].legend(loc="best")

    axes[1, 0].plot(
        baseline_trace.steps,
        np.cumsum(np.asarray(baseline_trace.retail_volumes)),
        color="#64748b",
        label="baseline retail volume",
    )
    axes[1, 0].plot(
        trained_trace.steps,
        np.cumsum(np.asarray(trained_trace.retail_volumes)),
        color="#2563eb",
        label="trained retail volume",
    )
    axes[1, 0].set_title("Submission retail volume won")
    axes[1, 0].set_xlabel("step")
    axes[1, 0].set_ylabel("cumulative retail volume (Y)")
    axes[1, 0].legend(loc="best")

    axes[1, 1].plot(baseline_trace.steps, baseline_trace.bid_near, color="#64748b", label="baseline bid near")
    axes[1, 1].plot(baseline_trace.steps, baseline_trace.ask_near, color="#94a3b8", label="baseline ask near")
    axes[1, 1].plot(trained_trace.steps, trained_trace.bid_near, color="#2563eb", label="trained bid near")
    axes[1, 1].plot(trained_trace.steps, trained_trace.ask_near, color="#dc2626", label="trained ask near")
    axes[1, 1].set_title("Near-touch control behavior")
    axes[1, 1].set_xlabel("step")
    axes[1, 1].set_ylabel("near depth (Y)")
    axes[1, 1].legend(loc="best")

    fig.suptitle("Balanced baseline vs trained policy")
    fig.tight_layout()
    fig.savefig(path)
    print(f"plot={path}")


if __name__ == "__main__":
    main()
