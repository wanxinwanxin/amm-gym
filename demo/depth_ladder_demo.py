"""Demo runner for the submission depth ladder."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from amm_gym import AMMFeeEnv
from amm_gym.baselines import benchmark_depth_policies
from amm_gym.sim.engine import SimConfig


def named_schedules() -> dict[str, tuple[tuple[int, float], ...] | None]:
    return {
        "constant_low_vol": None,
        "regime_shift": ((0, 0.001), (40, 0.0035), (80, 0.0015)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a short ladder demo episode")
    parser.add_argument("--policy", default="aggressive_near_mid", choices=sorted(benchmark_depth_policies()))
    parser.add_argument("--schedule", default="regime_shift", choices=sorted(named_schedules()))
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--plot", type=Path, default=None, help="Optional PNG output path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    policy = benchmark_depth_policies()[args.policy]
    schedule = named_schedules()[args.schedule]

    env = AMMFeeEnv(
        config=SimConfig(n_steps=args.steps, volatility_schedule=schedule, seed=args.seed),
        window_size=args.window_size,
    )
    obs, _ = env.reset(seed=args.seed)

    steps: list[int] = []
    spots: list[float] = []
    edges: list[float] = []
    pnls: list[float] = []
    imbalances: list[float] = []
    ask_near: list[float] = []
    ask_far: list[float] = []
    bid_near: list[float] = []
    bid_far: list[float] = []

    terminated = False
    truncated = False
    info: dict[str, float] = {}
    total_reward = 0.0

    while not (terminated or truncated):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        ws = env.window_size
        steps.append(int(info["step"]))
        spots.append(float(info["spot_price"]))
        edges.append(float(info["edge"]))
        pnls.append(float(info["pnl"]))
        imbalances.append(float(obs[ws + 2]))
        ask_near.append(float(info["ask_near_depth_y"]))
        ask_far.append(float(info["ask_far_depth_y"]))
        bid_near.append(float(info["bid_near_depth_y"]))
        bid_far.append(float(info["bid_far_depth_y"]))
        total_reward += reward

    print(f"policy={args.policy}")
    print(f"schedule={args.schedule}")
    print(f"total_reward={total_reward:.6f}")
    print(f"final_edge={float(info['edge']):.6f}")
    print(f"final_pnl={float(info['pnl']):.6f}")
    print(
        "final_ladder="
        f"bid_near={bid_near[-1]:.2f}, bid_far={bid_far[-1]:.2f}, "
        f"ask_near={ask_near[-1]:.2f}, ask_far={ask_far[-1]:.2f}"
    )

    if args.plot is not None:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise SystemExit(
                "matplotlib is required for --plot. Install with `pip install -e .[demo]`."
            ) from exc

        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

        axes[0].plot(steps, spots, label="submission spot")
        axes[0].set_ylabel("price")
        axes[0].legend(loc="best")

        axes[1].plot(steps, bid_near, label="bid near depth")
        axes[1].plot(steps, bid_far, label="bid far depth")
        axes[1].plot(steps, ask_near, label="ask near depth")
        axes[1].plot(steps, ask_far, label="ask far depth")
        axes[1].set_ylabel("depth (Y)")
        axes[1].legend(loc="best")

        axes[2].plot(steps, imbalances, label="inventory imbalance")
        axes[2].set_ylabel("imbalance")
        axes[2].legend(loc="best")

        axes[3].plot(steps, edges, label="edge")
        axes[3].plot(steps, pnls, label="pnl")
        axes[3].set_ylabel("value")
        axes[3].set_xlabel("step")
        axes[3].legend(loc="best")

        fig.tight_layout()
        fig.savefig(args.plot)
        print(f"plot={args.plot}")


if __name__ == "__main__":
    main()
