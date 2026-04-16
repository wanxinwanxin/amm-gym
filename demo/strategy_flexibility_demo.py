"""Single visual contrasting 2D fee control with the 6D ladder strategy."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from amm_gym.sim.ladder import DepthLadderAMM
from demo.presets import build_hackathon_demo_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot one visual comparing simple fee control to 6D ladder control"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("demo/artifacts/strategy_flexibility.png"),
    )
    parser.add_argument("--reference-price", type=float, default=100.0)
    return parser


def fee_curve(reference_price: float, bid_fee_bps: float, ask_fee_bps: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    y = np.array(
        [
            reference_price * (1.0 - bid_fee_bps / 10_000.0),
            reference_price,
            reference_price * (1.0 + ask_fee_bps / 10_000.0),
        ],
        dtype=np.float64,
    )
    return x, y


def ladder_profile(reference_price: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = build_hackathon_demo_config(seed=7)
    ladder = DepthLadderAMM(
        name="submission",
        reserve_x=cfg.initial_x,
        reserve_y=cfg.initial_y,
        band_bps=cfg.submission_band_bps,
        base_notional_y=cfg.submission_base_notional_y,
    )
    action = np.array([0.8, 0.5, -0.6, 0.2, 0.9, 0.7], dtype=np.float32)
    ladder.configure(reference_price=reference_price, bid_raw=action[:3], ask_raw=action[3:])

    band_centers = np.concatenate(
        (
            -0.5 * (ladder._band_lower + ladder.band_rel)[::-1] * 10_000.0,
            0.5 * (ladder._band_lower + ladder.band_rel) * 10_000.0,
        )
    )
    bid_depth_y = (ladder.bid_depth_x * reference_price)[::-1]
    ask_depth_y = ladder.ask_depth_y.copy()
    depths = np.concatenate((bid_depth_y, ask_depth_y))
    prices = reference_price * (1.0 + band_centers / 10_000.0)
    return band_centers, prices, depths


def main() -> None:
    args = build_parser().parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "matplotlib is required for this demo. Install with `pip install -e .[demo]`."
        ) from exc

    fee_x, fee_y = fee_curve(args.reference_price, bid_fee_bps=30.0, ask_fee_bps=30.0)
    band_centers, ladder_prices, ladder_depths = ladder_profile(args.reference_price)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), width_ratios=(1, 1.35))

    axes[0].plot(fee_x, fee_y, color="#0f172a", linewidth=2.5, marker="o")
    axes[0].axhline(args.reference_price, color="#94a3b8", linestyle="--", linewidth=1.0)
    axes[0].set_xticks([-1.0, 0.0, 1.0], labels=["bid quote", "mid", "ask quote"])
    axes[0].set_ylabel("quoted price")
    axes[0].set_title("Default 2D fee control")
    axes[0].text(
        0.02,
        0.04,
        "Only widens or tightens one bid and one ask.\nNo control over where depth sits away from mid.",
        transform=axes[0].transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "boxstyle": "round,pad=0.4"},
    )

    colors = np.where(band_centers < 0.0, "#2563eb", "#dc2626")
    axes[1].bar(band_centers, ladder_depths, width=12.0, color=colors, alpha=0.85)
    axes[1].axvline(0.0, color="#0f172a", linewidth=1.2)
    axes[1].set_xlabel("distance from mid (bps)")
    axes[1].set_ylabel("posted depth (Y)")
    axes[1].set_title("Current 6D submission ladder control")
    axes[1].text(
        0.02,
        0.96,
        "One action shapes scale, decay, and tilt independently on bid and ask.\nThat changes both quote aggressiveness and where liquidity is concentrated.",
        transform=axes[1].transAxes,
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "boxstyle": "round,pad=0.4"},
    )

    for x_pos, price, depth in zip(band_centers, ladder_prices, ladder_depths):
        axes[1].text(
            x_pos,
            depth + max(ladder_depths) * 0.02,
            f"{price:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    fig.suptitle("Why the ladder strategy is more expressive than plain bid/ask fees")
    fig.tight_layout()
    fig.savefig(args.output)
    print(f"plot={args.output}")


if __name__ == "__main__":
    main()
