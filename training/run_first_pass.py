"""Run the first strategy-iteration pass across a small model/objective grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.run_baseline import make_env_factory, make_policy_space_factory
from training.algorithms.cem import CEMConfig, CEMTrainer
from training.eval import benchmark_scenarios, default_seed_split
from training.eval.metrics import aggregate_episode_metrics, evaluate_episode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the first research pass over candidate trainers")
    parser.add_argument("--env-kind", default="classic", choices=["classic", "challenge"])
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--population-size", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=None)
    return parser


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
    seed_split = default_seed_split()
    candidates = [
        ("linear", "edge_advantage", "32,32"),
        ("feature_linear", "edge_advantage", "32,32"),
        ("linear", "balanced", "32,32"),
        ("feature_linear", "balanced", "32,32"),
        ("mlp", "balanced", "32,32"),
    ]
    rows: list[dict[str, object]] = []
    for family, objective, hidden_sizes in candidates:
        candidate_args = argparse.Namespace(
            env_kind=args.env_kind,
            scenario=args.scenario,
            window_size=args.window_size,
            population_size=args.population_size,
            elite_frac=0.25,
            iterations=args.iterations,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
            policy_family=family,
            objective=objective,
            hidden_sizes=hidden_sizes,
        )
        trainer = CEMTrainer(
            env_factory=make_env_factory(candidate_args),
            config=CEMConfig(
                population_size=args.population_size,
                elite_frac=0.25,
                iterations=args.iterations,
                eval_episodes=args.eval_episodes,
                seed=args.seed,
                objective=objective,
                evaluation_seeds=seed_split.train[: max(args.eval_episodes, 1)],
                elite_reevaluation_seeds=seed_split.validation,
            ),
            policy_space_factory=make_policy_space_factory(candidate_args),
        )
        result = trainer.train()
        policy = trainer.policy_space.build_policy(result.best_params.astype("float32"))
        env_factory = make_env_factory(candidate_args)
        validation_metrics = [
            evaluate_episode(env_factory, policy.act, seed=seed) for seed in seed_split.validation
        ]
        test_metrics = [
            evaluate_episode(env_factory, policy.act, seed=seed) for seed in seed_split.test
        ]
        validation_summary = aggregate_episode_metrics(validation_metrics)
        test_summary = aggregate_episode_metrics(test_metrics)
        rows.append(
            {
                "policy_family": family,
                "objective": objective,
                "best_score": result.best_score,
                "validation_edge_advantage_mean": validation_summary["edge_advantage_mean"],
                "validation_edge_advantage_win_rate": validation_summary["edge_advantage_win_rate"],
                "test_edge_advantage_mean": test_summary["edge_advantage_mean"],
                "test_edge_advantage_win_rate": test_summary["edge_advantage_win_rate"],
                "test_pnl_advantage_mean": test_summary["pnl_advantage"]["mean"],
            }
        )

    rows.sort(key=lambda row: float(row["validation_edge_advantage_mean"]), reverse=True)
    print(json.dumps(rows, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
