"""CLI entrypoint for the first baseline trainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from amm_gym import AMMFeeEnv
from amm_gym.sim.engine import SimConfig
from training.cem import CEMConfig, CEMTrainer
from training.policy import LinearTanhPolicy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a baseline AMM agent")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--elite-frac", type=float, default=0.25)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def make_env_factory(args: argparse.Namespace):
    def factory() -> AMMFeeEnv:
        config = SimConfig(n_steps=args.steps)
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
    obs, info = eval_env.reset(seed=args.seed + 50_000)
    total_reward = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = policy.act(obs)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward

    payload = {
        "best_score": result.best_score,
        "evaluation_reward": total_reward,
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


if __name__ == "__main__":
    main()

