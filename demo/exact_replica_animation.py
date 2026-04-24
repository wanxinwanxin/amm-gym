"""Animated exact-replica visualization for the simple AMM challenge."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

from arena_eval.exact_simple_amm import ExactSimpleAMMConfig, FixedFeeStrategy
from arena_eval.exact_simple_amm.simulator import ExactSimpleAMMSimulator
from arena_policies import ReactiveControllerParams, ReactiveControllerStrategy


EPS = 1e-9


@dataclass(frozen=True)
class VenueView:
    mid: float
    bid_fee: float
    ask_fee: float
    edge: float
    pnl: float
    retail_edge: float
    arb_edge: float
    retail_pnl: float
    arb_pnl: float


@dataclass(frozen=True)
class TradeOverlay:
    venue: str
    source: str
    trader_side: str
    pre_spot_price: float
    post_spot_price: float
    amount_x: float
    amount_y: float


@dataclass(frozen=True)
class AnimationFrame:
    step: int
    phase: str
    fair_price: float
    submission: VenueView
    normalizer: VenueView
    overlay: TradeOverlay | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a GIF for the exact simple-AMM replica.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--strategy", choices=("reactive", "fixed"), default="reactive")
    parser.add_argument("--bid-fee", type=float, default=0.003)
    parser.add_argument("--ask-fee", type=float, default=0.003)
    parser.add_argument("--params-json", type=Path, default=None)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("demo/artifacts/exact_replica.gif"))
    return parser


def load_reactive_params(path: Path | None) -> ReactiveControllerParams:
    if path is None:
        return ReactiveControllerParams()
    raw = json.loads(path.read_text())
    if "params" in raw:
        raw = raw["params"]
    return ReactiveControllerParams(**raw)


def build_submission_strategy(args: argparse.Namespace):
    if args.strategy == "fixed":
        return FixedFeeStrategy(bid_fee=args.bid_fee, ask_fee=args.ask_fee)
    return ReactiveControllerStrategy(load_reactive_params(args.params_json))


def _build_venue_view(state: dict[str, object], venue: str) -> VenueView:
    return VenueView(
        mid=float(state[f"{venue}_mid"]),
        bid_fee=float(state[f"{venue}_bid_fee"]),
        ask_fee=float(state[f"{venue}_ask_fee"]),
        edge=float(state["edge_submission"] if venue == "submission" else state["edge_normalizer"]),
        pnl=float(state["pnl_submission"] if venue == "submission" else state["pnl_normalizer"]),
        retail_edge=float(state[f"{venue}_retail_edge"]),
        arb_edge=float(state[f"{venue}_arb_edge"]),
        retail_pnl=float(state[f"{venue}_retail_pnl"]),
        arb_pnl=float(state[f"{venue}_arb_pnl"]),
    )


def build_animation_frames(simulator: ExactSimpleAMMSimulator) -> list[AnimationFrame]:
    initial_value = simulator.config.initial_x * simulator.config.initial_price + simulator.config.initial_y
    running_breakdown = {
        "submission_retail_edge": 0.0,
        "submission_arb_edge": 0.0,
        "submission_retail_pnl": 0.0,
        "submission_arb_pnl": 0.0,
        "normalizer_retail_edge": 0.0,
        "normalizer_arb_edge": 0.0,
        "normalizer_retail_pnl": 0.0,
        "normalizer_arb_pnl": 0.0,
    }
    frames: list[AnimationFrame] = [
        AnimationFrame(
            step=-1,
            phase="initial",
            fair_price=float(simulator.current_fair_price),
            submission=VenueView(
                mid=float(simulator.submission.spot_price),
                bid_fee=float(simulator.submission.bid_fee),
                ask_fee=float(simulator.submission.ask_fee),
                edge=0.0,
                pnl=float(simulator._mark_to_market(simulator.submission) - initial_value),
                retail_edge=0.0,
                arb_edge=0.0,
                retail_pnl=0.0,
                arb_pnl=0.0,
            ),
            normalizer=VenueView(
                mid=float(simulator.normalizer.spot_price),
                bid_fee=float(simulator.normalizer.bid_fee),
                ask_fee=float(simulator.normalizer.ask_fee),
                edge=0.0,
                pnl=float(simulator._mark_to_market(simulator.normalizer) - initial_value),
                retail_edge=0.0,
                arb_edge=0.0,
                retail_pnl=0.0,
                arb_pnl=0.0,
            ),
            overlay=None,
        )
    ]
    while not simulator.done:
        step_info = simulator.step_once()
        fair_price = float(step_info["fair_price"])
        trade_events = tuple(step_info["trade_events"])
        step = int(step_info["timestamp"])
        if not trade_events:
            idle_state = {
                "submission_mid": simulator.submission.spot_price,
                "submission_bid_fee": simulator.submission.bid_fee,
                "submission_ask_fee": simulator.submission.ask_fee,
                "normalizer_mid": simulator.normalizer.spot_price,
                "normalizer_bid_fee": simulator.normalizer.bid_fee,
                "normalizer_ask_fee": simulator.normalizer.ask_fee,
                "edge_submission": simulator.edge_submission,
                "edge_normalizer": simulator.edge_normalizer,
                "pnl_submission": simulator._mark_to_market(simulator.submission)
                - (simulator.config.initial_x * simulator.config.initial_price + simulator.config.initial_y),
                "pnl_normalizer": simulator._mark_to_market(simulator.normalizer)
                - (simulator.config.initial_x * simulator.config.initial_price + simulator.config.initial_y),
                **running_breakdown,
            }
            frames.append(
                AnimationFrame(
                    step=step,
                    phase="idle",
                    fair_price=fair_price,
                    submission=_build_venue_view(idle_state, "submission"),
                    normalizer=_build_venue_view(idle_state, "normalizer"),
                    overlay=None,
                )
            )
            continue

        for event in trade_events:
            overlay = TradeOverlay(
                venue=str(event["venue"]),
                source=str(event["source"]),
                trader_side=str(event["trader_side"]),
                pre_spot_price=float(event["pre_spot_price"]),
                post_spot_price=float(event["post_spot_price"]),
                amount_x=float(event["amount_x"]),
                amount_y=float(event["amount_y"]),
            )
            pre_state = dict(event["pre_state"])
            pre_metrics = dict(event["pre_metrics"])
            pre_state["edge_submission"] = pre_metrics["edge_submission"]
            pre_state["edge_normalizer"] = pre_metrics["edge_normalizer"]
            pre_state["pnl_submission"] = pre_metrics["pnl_submission"]
            pre_state["pnl_normalizer"] = pre_metrics["pnl_normalizer"]
            pre_state.update(running_breakdown)
            post_state = dict(event["post_state"])
            post_metrics = dict(event["post_metrics"])
            post_state["edge_submission"] = post_metrics["edge_submission"]
            post_state["edge_normalizer"] = post_metrics["edge_normalizer"]
            post_state["pnl_submission"] = post_metrics["pnl_submission"]
            post_state["pnl_normalizer"] = post_metrics["pnl_normalizer"]
            post_state.update(running_breakdown)

            venue_prefix = str(event["venue"])
            source = str(event["source"])
            edge_delta = (
                post_metrics["edge_submission"] - pre_metrics["edge_submission"]
                if venue_prefix == "submission"
                else post_metrics["edge_normalizer"] - pre_metrics["edge_normalizer"]
            )
            pnl_delta = (
                post_metrics["pnl_submission"] - pre_metrics["pnl_submission"]
                if venue_prefix == "submission"
                else post_metrics["pnl_normalizer"] - pre_metrics["pnl_normalizer"]
            )
            running_breakdown[f"{venue_prefix}_{source}_edge"] += edge_delta
            running_breakdown[f"{venue_prefix}_{source}_pnl"] += pnl_delta
            post_state.update(running_breakdown)

            frames.append(
                AnimationFrame(
                    step=step,
                    phase="trade",
                    fair_price=fair_price,
                    submission=_build_venue_view(pre_state, "submission"),
                    normalizer=_build_venue_view(pre_state, "normalizer"),
                    overlay=overlay,
                )
            )
            frames.append(
                AnimationFrame(
                    step=step,
                    phase="quote_update",
                    fair_price=fair_price,
                    submission=_build_venue_view(post_state, "submission"),
                    normalizer=_build_venue_view(post_state, "normalizer"),
                    overlay=overlay,
                )
            )
    return frames


def _quote_prices(mid_price: float, bid_fee: float, ask_fee: float) -> tuple[float, float]:
    bid_price = mid_price * max(1.0 - bid_fee, 0.0)
    ask_price = mid_price / max(1.0 - ask_fee, EPS)
    return (bid_price, ask_price)


def _price_limits(frames: list[AnimationFrame]) -> tuple[float, float]:
    values: list[float] = []
    for frame in frames:
        for venue in (frame.submission, frame.normalizer):
            bid, ask = _quote_prices(venue.mid, venue.bid_fee, venue.ask_fee)
            values.extend((venue.mid, bid, ask))
        values.append(frame.fair_price)
        if frame.overlay is not None:
            values.extend((frame.overlay.pre_spot_price, frame.overlay.post_spot_price))
    low = min(values)
    high = max(values)
    pad = max((high - low) * 0.08, frame.fair_price * 0.002)
    return (low - pad, high + pad)


def _metric_limits(frames: list[AnimationFrame], venue: str, field: str) -> tuple[float, float]:
    values = [getattr(getattr(frame, venue), field) for frame in frames]
    low = min(values)
    high = max(values)
    if abs(high - low) < EPS:
        pad = max(abs(high) * 0.1, 1.0)
        return (low - pad, high + pad)
    pad = 0.08 * (high - low)
    return (low - pad, high + pad)


def _draw_quote_panel(ax, frame: AnimationFrame, venue_name: str, price_limits: tuple[float, float], total_steps: int) -> None:
    ax.cla()
    venue = frame.submission if venue_name == "submission" else frame.normalizer
    bid, ask = _quote_prices(venue.mid, venue.bid_fee, venue.ask_fee)
    ax.set_xlim(*price_limits)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    ax.set_xlabel("price")
    ax.set_title(f"{venue_name.title()} quote reaction")
    ax.axvline(frame.fair_price, color="#64748b", linestyle=":", linewidth=1.2)
    ax.axvline(venue.mid, color="#111827", linewidth=2.0)
    ax.axvline(bid, color="#2563eb", linewidth=2.0)
    ax.axvline(ask, color="#dc2626", linewidth=2.0)
    ax.text(
        0.02,
        0.95,
        (
            f"step {max(frame.step + 1, 0)}/{total_steps}  phase: {frame.phase}"
            if frame.phase != "initial"
            else f"step 0/{total_steps}  phase: initial"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cbd5e1"},
    )

    if frame.overlay is not None and frame.overlay.venue == venue_name:
        color = "#16a34a" if frame.overlay.source == "retail" else "#dc2626"
        if frame.phase == "trade":
            ax.axvline(frame.overlay.post_spot_price, color="#111827", linestyle="--", linewidth=1.6, alpha=0.85)
            ax.annotate(
                "",
                xy=(frame.overlay.post_spot_price, 0.62),
                xytext=(frame.overlay.pre_spot_price, 0.62),
                arrowprops={"arrowstyle": "->", "lw": 2.4, "color": color},
            )
            ax.text(
                0.5 * (frame.overlay.pre_spot_price + frame.overlay.post_spot_price),
                0.69,
                f"{frame.overlay.source} {frame.overlay.trader_side.replace('_', ' ')}",
                color=color,
                ha="center",
                va="bottom",
                fontsize=9,
            )
        elif frame.phase == "quote_update":
            ax.text(
                0.5,
                0.63,
                "quote update",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="#0f766e",
            )
    elif frame.phase == "idle":
        ax.text(0.5, 0.62, "no executed trade", transform=ax.transAxes, ha="center", va="center", color="#64748b")
    elif frame.phase == "initial":
        ax.text(0.5, 0.62, "initial quotes", transform=ax.transAxes, ha="center", va="center", color="#64748b")

    legend_handles = [
        ax.plot([], [], color="#64748b", linestyle=":", linewidth=1.2, label="fair")[0],
        ax.plot([], [], color="#111827", linewidth=2.0, label="mid")[0],
        ax.plot([], [], color="#2563eb", linewidth=2.0, label="bid")[0],
        ax.plot([], [], color="#dc2626", linewidth=2.0, label="ask")[0],
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=9)


def _metric_limits_for_fields(frames: list[AnimationFrame], venue: str, fields: tuple[str, ...]) -> tuple[float, float]:
    values: list[float] = []
    for frame in frames:
        venue_view = getattr(frame, venue)
        values.extend(float(getattr(venue_view, field)) for field in fields)
    low = min(values)
    high = max(values)
    if abs(high - low) < EPS:
        pad = max(abs(high) * 0.1, 1.0)
        return (low - pad, high + pad)
    pad = 0.08 * (high - low)
    return (low - pad, high + pad)


def _draw_stats_panel(
    ax,
    frames: list[AnimationFrame],
    frame_idx: int,
    venue_name: str,
    edge_limits: tuple[float, float],
) -> None:
    ax.cla()
    steps = [frame.step for frame in frames[: frame_idx + 1]]
    retail_edges = [getattr(getattr(frame, venue_name), "retail_edge") for frame in frames[: frame_idx + 1]]
    arb_edges = [getattr(getattr(frame, venue_name), "arb_edge") for frame in frames[: frame_idx + 1]]
    total_edges = [getattr(getattr(frame, venue_name), "edge") for frame in frames[: frame_idx + 1]]

    ax.plot(steps, retail_edges, color="#16a34a", linewidth=2.0, label="retail edge")
    ax.plot(steps, arb_edges, color="#dc2626", linewidth=2.0, label="arb edge")
    ax.plot(steps, total_edges, color="#111827", linewidth=2.0, linestyle="--", label="total edge")
    ax.axvline(steps[-1], color="#0f172a", linestyle="--", linewidth=1.1)
    ax.set_ylim(*edge_limits)
    ax.set_title(f"{venue_name.title()} cumulative edge by source")
    ax.set_xlabel("step")
    ax.set_ylabel("edge")
    ax.tick_params(axis="y", colors="#334155")
    ax.grid(alpha=0.25)

    retail_handle = ax.plot([], [], color="#16a34a", linewidth=2.0, label="retail edge")[0]
    arb_handle = ax.plot([], [], color="#dc2626", linewidth=2.0, label="arb edge")[0]
    total_handle = ax.plot([], [], color="#111827", linewidth=2.0, linestyle="--", label="total edge")[0]
    ax.legend(handles=[retail_handle, arb_handle, total_handle], loc="upper left", fontsize=9)


def render_animation(
    path: Path,
    frames: list[AnimationFrame],
    *,
    fps: int,
    frame_step: int,
    strategy_label: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "matplotlib with Pillow support is required. Install with `pip install -e .[demo]`."
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    sampled_indices = list(range(0, len(frames), max(frame_step, 1)))
    if sampled_indices[-1] != len(frames) - 1:
        sampled_indices.append(len(frames) - 1)

    price_limits = _price_limits(frames)
    submission_edge_limits = _metric_limits_for_fields(frames, "submission", ("retail_edge", "arb_edge", "edge"))
    normalizer_edge_limits = _metric_limits_for_fields(frames, "normalizer", ("retail_edge", "arb_edge", "edge"))

    fig = plt.figure(figsize=(15, 8.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0])
    ax_submission_quote = fig.add_subplot(grid[0, 0])
    ax_normalizer_quote = fig.add_subplot(grid[1, 0])
    ax_submission_stats = fig.add_subplot(grid[0, 1])
    ax_normalizer_stats = fig.add_subplot(grid[1, 1])
    fig.suptitle(f"Exact simple-AMM replica: {strategy_label}", fontsize=15)

    total_steps = max(frame.step for frame in frames if frame.step >= 0) + 1

    def update(sampled_idx: int):
        frame = frames[sampled_idx]
        _draw_quote_panel(ax_submission_quote, frame, "submission", price_limits, total_steps)
        _draw_quote_panel(ax_normalizer_quote, frame, "normalizer", price_limits, total_steps)
        _draw_stats_panel(
            ax_submission_stats,
            frames,
            sampled_idx,
            "submission",
            submission_edge_limits,
        )
        _draw_stats_panel(
            ax_normalizer_stats,
            frames,
            sampled_idx,
            "normalizer",
            normalizer_edge_limits,
        )
        return tuple(
            artist
            for ax in (
                ax_submission_quote,
                ax_normalizer_quote,
                ax_submission_stats,
                ax_normalizer_stats,
            )
            for artist in ax.get_children()
        )

    animation = FuncAnimation(fig, update, frames=sampled_indices, interval=1000 / max(fps, 1), blit=False)
    animation.save(path, writer=PillowWriter(fps=max(fps, 1)))
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    config = replace(ExactSimpleAMMConfig.from_seed(args.seed), n_steps=args.steps)
    simulator = ExactSimpleAMMSimulator(
        config=config,
        submission_strategy=build_submission_strategy(args),
        normalizer_strategy=FixedFeeStrategy(),
        seed=args.seed,
    )
    frames = build_animation_frames(simulator)
    strategy_label = args.strategy
    if args.params_json is not None:
        strategy_label = f"{strategy_label} ({args.params_json.stem})"
    render_animation(
        args.output,
        frames,
        fps=args.fps,
        frame_step=args.frame_step,
        strategy_label=strategy_label,
    )
    print(f"animation={args.output}")


if __name__ == "__main__":
    main()
