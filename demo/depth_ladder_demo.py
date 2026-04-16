"""Demo runner for the submission depth ladder."""

from __future__ import annotations

import argparse
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
    named_schedules,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a short ladder demo episode")
    parser.add_argument("--policy", default="aggressive_near_mid", choices=sorted(benchmark_depth_policies()))
    parser.add_argument("--schedule", default="regime_shift", choices=sorted(named_schedules()))
    parser.add_argument("--steps", type=int, default=DEFAULT_DEMO_STEPS)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_DEMO_SEED)
    parser.add_argument("--plot", type=Path, default=None, help="Optional PNG output path.")
    return parser


def collect_ladder_snapshots(
    *,
    policy,
    seed: int,
    steps: int,
    window_size: int,
    schedule,
    snapshot_steps: list[int],
) -> dict[int, dict[str, np.ndarray]]:
    env = AMMFeeEnv(
        config=build_hackathon_demo_config(seed=seed, steps=steps, schedule=schedule),
        window_size=window_size,
    )
    obs, _ = env.reset(seed=seed)
    ladder = env.engine.amm_agent
    captured: dict[int, dict[str, np.ndarray]] = {}

    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = np.asarray(policy(obs), dtype=np.float32)
        obs, _, terminated, truncated, info = env.step(action)
        step = int(info["step"])
        if step in snapshot_steps:
            centers_bps = 0.5 * (ladder._band_lower + ladder.band_rel) * 10_000.0
            captured[step] = {
                "centers_bps": centers_bps.copy(),
                "bid_depth_y": (ladder.bid_depth_x * ladder.reference_price).copy(),
                "ask_depth_y": ladder.ask_depth_y.copy(),
                "reference_price": np.array([ladder.reference_price], dtype=np.float64),
            }
    return captured


def add_regime_shading(ax, schedule, max_step: int) -> None:
    if not schedule:
        return
    colors = ["#e0f2fe", "#fee2e2", "#ecfccb", "#ede9fe"]
    for idx, (start, sigma) in enumerate(schedule):
        end = schedule[idx + 1][0] if idx + 1 < len(schedule) else max_step + 1
        ax.axvspan(start, end, color=colors[idx % len(colors)], alpha=0.25, linewidth=0)
        ax.text(
            (start + end - 1) / 2.0,
            0.98,
            f"sigma={sigma:.4f}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
            color="#334155",
        )


def plot_ladder_snapshot(ax, title: str, snapshot: dict[str, np.ndarray]) -> None:
    centers = snapshot["centers_bps"]
    bid = snapshot["bid_depth_y"]
    ask = snapshot["ask_depth_y"]
    ax.barh(centers, -bid, height=10.0, color="#2563eb", alpha=0.85, label="bid depth")
    ax.barh(centers, ask, height=10.0, color="#dc2626", alpha=0.85, label="ask depth")
    ax.axvline(0.0, color="#0f172a", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("posted depth (Y)")
    ax.set_ylabel("distance from mid (bps)")


def main() -> None:
    args = build_parser().parse_args()
    policy = benchmark_depth_policies()[args.policy]
    schedule = named_schedules()[args.schedule]

    env = AMMFeeEnv(
        config=build_hackathon_demo_config(
            seed=args.seed,
            steps=args.steps,
            schedule=schedule,
        ),
        window_size=args.window_size,
    )
    trace = collect_rollout(env, policy, seed=args.seed)
    info = trace.final_info
    snapshot_steps = sorted({10, args.steps // 2, max(args.steps - 5, 0)})
    ladder_snapshots = collect_ladder_snapshots(
        policy=policy,
        seed=args.seed,
        steps=args.steps,
        window_size=args.window_size,
        schedule=schedule,
        snapshot_steps=snapshot_steps,
    )

    print(f"policy={args.policy}")
    print(f"schedule={args.schedule}")
    print(f"total_reward={trace.total_reward:.6f}")
    print(f"final_edge={float(info['edge']):.6f}")
    print(f"final_pnl={float(info['pnl']):.6f}")
    print(
        "final_ladder="
        f"bid_near={trace.bid_near[-1]:.2f}, bid_far={trace.bid_far[-1]:.2f}, "
        f"ask_near={trace.ask_near[-1]:.2f}, ask_far={trace.ask_far[-1]:.2f}"
    )

    if args.plot is not None:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise SystemExit(
                "matplotlib is required for --plot. Install with `pip install -e .[demo]`."
            ) from exc

        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(14, 11))
        grid = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.0, 1.0])

        ax_price = fig.add_subplot(grid[0, :])
        ax_price.plot(trace.steps, trace.fair_prices, label="fair price", color="#111827")
        ax_price.plot(trace.steps, trace.submission_spots, label="submission spot", color="#2563eb")
        ax_price.plot(
            trace.steps,
            trace.normalizer_spots,
            label="normalizer spot",
            color="#dc2626",
            alpha=0.85,
        )
        add_regime_shading(ax_price, schedule, max(trace.steps))
        ax_price.set_title("How the simulator works: market moves, venues respond, and flow routes")
        ax_price.set_ylabel("price")
        ax_price.legend(loc="upper right")

        snapshot_axes = [
            fig.add_subplot(grid[1, 0]),
            fig.add_subplot(grid[1, 1]),
            fig.add_subplot(grid[2, 0]),
        ]
        for ax, step in zip(snapshot_axes, snapshot_steps):
            plot_ladder_snapshot(ax, f"Ladder at step {step}", ladder_snapshots[step])
        snapshot_axes[0].legend(loc="lower right")

        ax_flow = fig.add_subplot(grid[2, 1])
        sub_cum = np.cumsum(np.asarray(trace.execution_volumes))
        norm_cum = np.cumsum(np.asarray(trace.normalizer_execution_volumes))
        ax_flow.plot(trace.steps, sub_cum, color="#2563eb", linewidth=2.0, label="submission cumulative volume")
        ax_flow.plot(
            trace.steps,
            norm_cum,
            color="#dc2626",
            linewidth=2.0,
            label="normalizer cumulative volume",
        )
        ax_flow.set_title("Which venue gets the retail flow?")
        ax_flow.set_xlabel("step")
        ax_flow.set_ylabel("cumulative execution volume (Y)")
        ax_flow.legend(loc="upper left")
        ax_flow.text(
            0.03,
            0.97,
            f"final edge: {info['edge']:.3f}\n"
            f"final pnl: {info['pnl']:.1f}\n"
            f"submission volume: {sub_cum[-1]:.1f}\n"
            f"normalizer volume: {norm_cum[-1]:.1f}",
            transform=ax_flow.transAxes,
            va="top",
            bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "boxstyle": "round,pad=0.4"},
        )

        fig.tight_layout()
        fig.savefig(args.plot)
        print(f"plot={args.plot}")


if __name__ == "__main__":
    main()
