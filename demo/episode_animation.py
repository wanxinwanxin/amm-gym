"""Animated rollout demo for a single market-making episode."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from amm_gym import AMMFeeEnv
from amm_gym.baselines import benchmark_depth_policies
from demo.common import RolloutTrace, collect_rollout
from demo.depth_ladder_demo import add_regime_shading
from demo.presets import (
    DEFAULT_DEMO_SEED,
    DEFAULT_DEMO_STEPS,
    DEFAULT_WINDOW_SIZE,
    build_hackathon_demo_config,
    named_schedules,
)
from training.cem import CEMConfig, CEMTrainer
from training.policy import LinearTanhPolicy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render an animated episode demo as a GIF.")
    parser.add_argument("--policy", default="trained", help="`trained` or a benchmark policy name.")
    parser.add_argument("--schedule", default="regime_shift", choices=sorted(named_schedules()))
    parser.add_argument("--steps", type=int, default=DEFAULT_DEMO_STEPS)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_DEMO_SEED)
    parser.add_argument("--frame-step", type=int, default=2)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--elite-frac", type=float, default=0.25)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("demo/artifacts/trained_episode.gif"))
    return parser


def make_env(*, seed: int, steps: int, window_size: int, schedule) -> AMMFeeEnv:
    return AMMFeeEnv(
        config=build_hackathon_demo_config(seed=seed, steps=steps, schedule=schedule),
        window_size=window_size,
    )


def build_policy(args: argparse.Namespace):
    if args.policy != "trained":
        return benchmark_depth_policies()[args.policy], None

    def env_factory() -> AMMFeeEnv:
        return make_env(
            seed=args.seed,
            steps=args.steps,
            window_size=args.window_size,
            schedule=named_schedules()[args.schedule],
        )

    trainer = CEMTrainer(
        env_factory=env_factory,
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
    return policy.act, result


def render_animation(
    path: Path,
    trace: RolloutTrace,
    *,
    schedule,
    policy_label: str,
    fps: int,
    frame_step: int,
    trainer_result=None,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "matplotlib with Pillow support is required. Install with `pip install -e .[demo]`."
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)

    frame_indices = list(range(0, len(trace.steps), max(frame_step, 1)))
    if frame_indices[-1] != len(trace.steps) - 1:
        frame_indices.append(len(trace.steps) - 1)

    cum_reward = np.cumsum(np.asarray(trace.rewards))
    cum_submission_retail = np.cumsum(np.asarray(trace.retail_volumes))
    cum_normalizer_retail = np.cumsum(np.asarray(trace.normalizer_retail_volumes))
    cum_submission_arb = np.cumsum(np.asarray(trace.arb_volumes))
    cum_normalizer_arb = np.cumsum(np.asarray(trace.normalizer_arb_volumes))
    submission_edges = np.asarray(trace.edges)
    normalizer_edges = np.asarray(trace.normalizer_edges)
    max_ladder_depth = max(
        np.max(np.asarray(trace.bid_band_depths)) if trace.bid_band_depths else 0.0,
        np.max(np.asarray(trace.ask_band_depths)) if trace.ask_band_depths else 0.0,
    )
    fig = plt.figure(figsize=(13, 8.5), constrained_layout=True)
    outer = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0])
    ax_price = fig.add_subplot(outer[0, 0])
    ax_ladder = fig.add_subplot(outer[0, 1])
    ax_edge = fig.add_subplot(outer[1, 0])
    outcome_grid = outer[1, 1].subgridspec(2, 1, hspace=0.08)
    ax_retail = fig.add_subplot(outcome_grid[0, 0])
    ax_arb = fig.add_subplot(outcome_grid[1, 0], sharex=ax_retail)

    add_regime_shading(ax_price, schedule, max(trace.steps))
    ax_price.plot(trace.steps, trace.fair_prices, color="#cbd5e1", linewidth=1.2, alpha=0.8)
    ax_price.plot(trace.steps, trace.submission_spots, color="#bfdbfe", linewidth=1.2, alpha=0.8)
    ax_price.plot(trace.steps, trace.normalizer_spots, color="#fecaca", linewidth=1.2, alpha=0.8)
    fair_line, = ax_price.plot([], [], color="#111827", linewidth=2.2, label="fair price")
    submission_line, = ax_price.plot([], [], color="#2563eb", linewidth=2.0, label="submission spot")
    normalizer_line, = ax_price.plot([], [], color="#dc2626", linewidth=2.0, label="normalizer spot")
    time_cursor = ax_price.axvline(trace.steps[0], color="#0f172a", linestyle="--", linewidth=1.2)
    ax_price.set_title("Market context")
    ax_price.set_ylabel("price")
    ax_price.legend(loc="upper left")

    band_centers_bps = np.asarray(trace.band_centers_bps)
    band_widths_bps = np.asarray(trace.band_widths_bps)
    bid_x_positions = -band_centers_bps
    ask_x_positions = band_centers_bps
    ladder_bid = ax_ladder.bar(
        bid_x_positions,
        np.zeros_like(band_centers_bps),
        width=band_widths_bps,
        color="#2563eb",
        alpha=0.85,
        label="bid depth",
        align="center",
    )
    ladder_ask = ax_ladder.bar(
        ask_x_positions,
        np.zeros_like(band_centers_bps),
        width=band_widths_bps,
        color="#dc2626",
        alpha=0.85,
        label="ask depth",
        align="center",
    )
    max_distance_bps = float(np.max(band_centers_bps + 0.5 * band_widths_bps))
    ax_ladder.axvline(0.0, color="#0f172a", linewidth=1.0)
    ax_ladder.set_xlim(-1.1 * max_distance_bps, 1.1 * max_distance_bps)
    ax_ladder.set_ylim(0.0, 1.15 * max_ladder_depth)
    ax_ladder.set_title("Action translated into a liquidity ladder")
    ax_ladder.set_xlabel("distance from mid (bps)")
    ax_ladder.set_ylabel("posted depth (Y)")
    ax_ladder.legend(loc="upper right")

    ax_edge.plot(trace.steps, submission_edges, color="#bfdbfe", linewidth=1.2, alpha=0.9)
    ax_edge.plot(trace.steps, normalizer_edges, color="#fecaca", linewidth=1.2, alpha=0.9)
    submission_edge_line, = ax_edge.plot([], [], color="#2563eb", linewidth=2.2, label="submission edge")
    normalizer_edge_line, = ax_edge.plot([], [], color="#dc2626", linewidth=2.2, label="normalizer edge")
    edge_cursor = ax_edge.axvline(trace.steps[0], color="#0f172a", linestyle="--", linewidth=1.2)
    ax_edge.set_title("Cumulative edge over the episode")
    ax_edge.set_xlabel("step")
    ax_edge.set_ylabel("edge")
    ax_edge.legend(loc="upper left")

    ax_retail.plot(trace.steps, cum_submission_retail, color="#bfdbfe", linewidth=1.2, alpha=0.9)
    ax_retail.plot(trace.steps, cum_normalizer_retail, color="#fecaca", linewidth=1.2, alpha=0.9)
    submission_retail_line, = ax_retail.plot(
        [], [], color="#2563eb", linewidth=2.2, label="submission retail volume"
    )
    normalizer_retail_line, = ax_retail.plot(
        [], [], color="#dc2626", linewidth=2.2, label="normalizer retail volume"
    )
    retail_cursor = ax_retail.axvline(trace.steps[0], color="#0f172a", linestyle="--", linewidth=1.2)
    ax_retail.set_title("Cumulative retail order volume by venue")
    ax_retail.set_ylabel("retail volume (Y)")
    ax_retail.legend(loc="upper left", fontsize=9)

    ax_arb.plot(trace.steps, cum_submission_arb, color="#bfdbfe", linewidth=1.2, alpha=0.9)
    ax_arb.plot(trace.steps, cum_normalizer_arb, color="#fecaca", linewidth=1.2, alpha=0.9)
    submission_arb_line, = ax_arb.plot(
        [], [], color="#2563eb", linewidth=2.2, label="submission arb volume"
    )
    normalizer_arb_line, = ax_arb.plot(
        [], [], color="#dc2626", linewidth=2.2, label="normalizer arb volume"
    )
    arb_cursor = ax_arb.axvline(trace.steps[0], color="#0f172a", linestyle="--", linewidth=1.2)
    ax_arb.set_title("Cumulative arbitrage volume by venue")
    ax_arb.set_xlabel("step")
    ax_arb.set_ylabel("arb volume (Y)")
    ax_arb.legend(loc="upper left", fontsize=9)

    subtitle = "one seeded episode"
    if trainer_result is not None:
        subtitle = (
            f"trained over {len(trainer_result.history)} CEM iterations, "
            f"best score {trainer_result.best_score:.2f}"
        )
    fig.suptitle(f"AMM demo animation: {policy_label} policy on {subtitle}")

    def update(frame_idx: int):
        step_slice = slice(0, frame_idx + 1)
        fair_line.set_data(trace.steps[step_slice], trace.fair_prices[step_slice])
        submission_line.set_data(trace.steps[step_slice], trace.submission_spots[step_slice])
        normalizer_line.set_data(trace.steps[step_slice], trace.normalizer_spots[step_slice])
        time_cursor.set_xdata([trace.steps[frame_idx], trace.steps[frame_idx]])
        submission_edge_line.set_data(trace.steps[step_slice], submission_edges[step_slice])
        normalizer_edge_line.set_data(trace.steps[step_slice], normalizer_edges[step_slice])
        edge_cursor.set_xdata([trace.steps[frame_idx], trace.steps[frame_idx]])
        submission_retail_line.set_data(trace.steps[step_slice], cum_submission_retail[step_slice])
        normalizer_retail_line.set_data(trace.steps[step_slice], cum_normalizer_retail[step_slice])
        retail_cursor.set_xdata([trace.steps[frame_idx], trace.steps[frame_idx]])
        submission_arb_line.set_data(trace.steps[step_slice], cum_submission_arb[step_slice])
        normalizer_arb_line.set_data(trace.steps[step_slice], cum_normalizer_arb[step_slice])
        arb_cursor.set_xdata([trace.steps[frame_idx], trace.steps[frame_idx]])

        for patch, height in zip(ladder_bid.patches, np.asarray(trace.bid_band_depths[frame_idx])):
            patch.set_height(float(height))
        for patch, height in zip(ladder_ask.patches, np.asarray(trace.ask_band_depths[frame_idx])):
            patch.set_height(float(height))
        ax_edge.set_title(
            "Cumulative edge over the episode\n"
            f"step {trace.steps[frame_idx]}  sigma {trace.active_sigmas[frame_idx]:.4f}  "
            f"reward {cum_reward[frame_idx]:.2f}  sub pnl {trace.pnls[frame_idx]:.1f}  "
            f"norm pnl {trace.normalizer_pnls[frame_idx]:.1f}"
        )
        ax_ladder.set_title(
            "Current submission ladder\n"
            f"bid near/far {trace.bid_near[frame_idx]:.0f}/{trace.bid_far[frame_idx]:.0f}  "
            f"ask near/far {trace.ask_near[frame_idx]:.0f}/{trace.ask_far[frame_idx]:.0f}"
        )
        return (
            fair_line,
            submission_line,
            normalizer_line,
            time_cursor,
            submission_edge_line,
            normalizer_edge_line,
            edge_cursor,
            submission_retail_line,
            normalizer_retail_line,
            retail_cursor,
            submission_arb_line,
            normalizer_arb_line,
            arb_cursor,
            *ladder_bid.patches,
            *ladder_ask.patches,
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        interval=1000 / max(fps, 1),
        blit=False,
        repeat=False,
    )
    animation.save(path, writer=PillowWriter(fps=max(fps, 1)))
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    schedule = named_schedules()[args.schedule]

    if args.policy != "trained" and args.policy not in benchmark_depth_policies():
        available = ", ".join(["trained", *sorted(benchmark_depth_policies())])
        raise SystemExit(f"unknown policy `{args.policy}`. Choose one of: {available}")

    policy, trainer_result = build_policy(args)
    eval_seed = args.seed + 50_000 if args.policy == "trained" else args.seed
    env = make_env(seed=args.seed, steps=args.steps, window_size=args.window_size, schedule=schedule)
    trace = collect_rollout(env, policy, seed=eval_seed)
    render_animation(
        args.output,
        trace,
        schedule=schedule,
        policy_label=args.policy,
        fps=args.fps,
        frame_step=args.frame_step,
        trainer_result=trainer_result,
    )
    print(f"animation={args.output}")


if __name__ == "__main__":
    main()
