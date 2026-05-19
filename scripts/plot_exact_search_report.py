"""Visualize exact-search report JSON artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile


def _load_report(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise SystemExit(f"report not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in report {path}: {exc}") from exc


def _candidate_score(candidate: dict[str, object] | None) -> float | None:
    if not candidate:
        return None
    score = candidate.get("score")
    return float(score) if score is not None else None


def _candidate_metric(candidate: dict[str, object] | None, key: str) -> float | None:
    if not candidate:
        return None
    value = candidate.get(key)
    return float(value) if value is not None else None


def _default_output_path(report_path: Path) -> Path:
    return report_path.with_name(f"{report_path.stem}_viz.png")


def save_exact_search_report_plot(report_path: Path, output_path: Path) -> None:
    cache_root = Path(tempfile.gettempdir()) / "amm-gym-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "matplotlib is required for exact-search plots. Install with `pip install -e .[demo]`."
        ) from exc

    payload = _load_report(report_path)
    history = payload.get("history")
    if not isinstance(history, list) or not history:
        raise SystemExit(f"report has no history records: {report_path}")

    iterations = [int(record["iteration"]) for record in history]
    best_search_scores = [_candidate_score(record.get("best_search")) for record in history]
    fixed_validation_scores = [_candidate_score(record.get("fixed_validation")) for record in history]
    fresh_validation_scores = [_candidate_score(record.get("fresh_validation")) for record in history]

    search_minus_fixed = [
        (search - fixed) if search is not None and fixed is not None else None
        for search, fixed in zip(best_search_scores, fixed_validation_scores)
    ]
    fresh_minus_fixed = [
        (fresh - fixed) if fresh is not None and fixed is not None else None
        for fresh, fixed in zip(fresh_validation_scores, fixed_validation_scores)
    ]

    fixed_advantage = [
        _candidate_metric(record.get("fixed_validation"), "edge_advantage_mean") for record in history
    ]
    fresh_advantage = [
        _candidate_metric(record.get("fresh_validation"), "edge_advantage_mean") for record in history
    ]
    fixed_retail_return = [
        _candidate_metric(record.get("fixed_validation"), "annualized_retail_edge_return_mean_submission")
        for record in history
    ]
    fresh_retail_return = [
        _candidate_metric(record.get("fresh_validation"), "annualized_retail_edge_return_mean_submission")
        for record in history
    ]
    fixed_arb_loss_return = [
        _candidate_metric(record.get("fixed_validation"), "annualized_arb_loss_return_mean_submission")
        for record in history
    ]
    fresh_arb_loss_return = [
        _candidate_metric(record.get("fresh_validation"), "annualized_arb_loss_return_mean_submission")
        for record in history
    ]

    best_search = _candidate_score(payload.get("best_search"))
    best_validation = _candidate_score(payload.get("best_validation"))
    best_test = _candidate_score(payload.get("best_test"))
    final_candidate = payload.get("best_test") or payload.get("best_validation") or payload.get("best_search")
    final_initial_value = _candidate_metric(final_candidate, "initial_value_mean")
    final_episode_seconds = _candidate_metric(final_candidate, "episode_seconds_mean")
    final_retail_markout = _candidate_metric(final_candidate, "retail_markout_bps_mean_submission")
    final_arb_markout = _candidate_metric(final_candidate, "arb_markout_bps_mean_submission")

    method = str(payload.get("method", "unknown")).upper()
    objective = str(payload.get("objective", "score"))
    search_cfg = payload.get("search", {})
    if not isinstance(search_cfg, dict):
        search_cfg = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 11), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[2.0, 1.15, 0.95], hspace=0.28, wspace=0.2)
    ax_scores = fig.add_subplot(gs[0, :])
    ax_gap = fig.add_subplot(gs[1, 0])
    ax_components = fig.add_subplot(gs[1, 1])
    ax_summary = fig.add_subplot(gs[2, 0])
    ax_notes = fig.add_subplot(gs[2, 1])

    ax_scores.plot(
        iterations,
        best_search_scores,
        color="#2563eb",
        marker="o",
        linewidth=2.4,
        label="best in-sample search score",
    )
    ax_scores.plot(
        iterations,
        fixed_validation_scores,
        color="#dc2626",
        marker="o",
        linewidth=2.2,
        label="fixed validation",
    )
    ax_scores.plot(
        iterations,
        fresh_validation_scores,
        color="#0f766e",
        marker="o",
        linewidth=2.2,
        linestyle="--",
        label="fresh validation",
    )
    if best_test is not None:
        ax_scores.axhline(
            best_test,
            color="#7c3aed",
            linestyle=":",
            linewidth=2,
            label="final fresh validation / test",
        )
    ax_scores.set_title("Exact search score vs validation over iterations")
    ax_scores.set_xlabel("iteration")
    ax_scores.set_ylabel(objective)
    ax_scores.grid(alpha=0.25)
    ax_scores.legend(loc="best")

    if best_validation is not None:
        best_idx = max(range(len(fixed_validation_scores)), key=lambda idx: fixed_validation_scores[idx])
        ax_scores.scatter(
            iterations[best_idx],
            fixed_validation_scores[best_idx],
            s=90,
            color="#dc2626",
            zorder=4,
        )
        ax_scores.annotate(
            f"best fixed val {best_validation:.2f}",
            (iterations[best_idx], fixed_validation_scores[best_idx]),
            xytext=(8, 10),
            textcoords="offset points",
            color="#7f1d1d",
            fontsize=9,
        )

    ax_gap.axhline(0.0, color="#94a3b8", linewidth=1)
    ax_gap.plot(
        iterations,
        search_minus_fixed,
        color="#1d4ed8",
        marker="o",
        linewidth=2,
        label="search - fixed",
    )
    ax_gap.plot(
        iterations,
        fresh_minus_fixed,
        color="#059669",
        marker="o",
        linewidth=2,
        linestyle="--",
        label="fresh - fixed",
    )
    ax_gap.set_title("Generalization gap")
    ax_gap.set_xlabel("iteration")
    ax_gap.set_ylabel("score delta")
    ax_gap.grid(alpha=0.25)
    ax_gap.legend(loc="best")

    if any(value is not None for value in fixed_retail_return + fresh_retail_return + fixed_arb_loss_return + fresh_arb_loss_return):
        fixed_retail_pct = [100.0 * value if value is not None else None for value in fixed_retail_return]
        fresh_retail_pct = [100.0 * value if value is not None else None for value in fresh_retail_return]
        fixed_arb_pct = [100.0 * value if value is not None else None for value in fixed_arb_loss_return]
        fresh_arb_pct = [100.0 * value if value is not None else None for value in fresh_arb_loss_return]
        ax_components.plot(
            iterations,
            fixed_retail_pct,
            color="#16a34a",
            marker="o",
            linewidth=2.2,
            label="fixed val retail capture",
        )
        ax_components.plot(
            iterations,
            fresh_retail_pct,
            color="#15803d",
            marker="o",
            linewidth=2.0,
            linestyle="--",
            label="fresh val retail capture",
        )
        ax_components.plot(
            iterations,
            fixed_arb_pct,
            color="#dc2626",
            marker="o",
            linewidth=2.2,
            label="fixed val arb loss",
        )
        ax_components.plot(
            iterations,
            fresh_arb_pct,
            color="#b91c1c",
            marker="o",
            linewidth=2.0,
            linestyle="--",
            label="fresh val arb loss",
        )
        ax_components.set_title("Annualized Component Returns")
        ax_components.set_xlabel("iteration")
        ax_components.set_ylabel("% of initial pool value / year")
        ax_components.grid(alpha=0.25)
        ax_components.legend(loc="best")
    else:
        ax_components.set_title("Annualized Component Returns")
        ax_components.text(0.5, 0.5, "component metrics unavailable", ha="center", va="center")
        ax_components.axis("off")

    summary_labels: list[str] = []
    summary_values: list[float] = []
    summary_colors: list[str] = []
    for label, value, color in (
        ("best search", best_search, "#2563eb"),
        ("best fixed val", best_validation, "#dc2626"),
        ("best test", best_test, "#7c3aed"),
    ):
        if value is not None:
            summary_labels.append(label)
            summary_values.append(value)
            summary_colors.append(color)

    if summary_labels:
        bars = ax_summary.bar(summary_labels, summary_values, color=summary_colors)
        for bar, value in zip(bars, summary_values):
            ax_summary.text(
                bar.get_x() + bar.get_width() / 2.0,
                value,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax_summary.set_title("Final score snapshot")
    ax_summary.set_ylabel(objective)
    ax_summary.tick_params(axis="x", rotation=12)
    ax_summary.grid(axis="y", alpha=0.25)

    ax_notes.axis("off")
    note_lines = []
    if final_initial_value is not None:
        note_lines.append(f"initial pool value: {final_initial_value:,.0f} quote units")
    if final_episode_seconds is not None:
        note_lines.append(f"episode length: {final_episode_seconds / 3600.0:.1f} hours")
    if final_retail_markout is not None:
        note_lines.append(f"retail markout: {final_retail_markout:.2f} bps")
    if final_arb_markout is not None:
        note_lines.append(f"arb loss rate: {final_arb_markout:.2f} bps")
    if not note_lines:
        note_lines.append("enhanced reporting unavailable")
    ax_notes.set_title("Economic Interpretation")
    ax_notes.text(
        0.02,
        0.98,
        "\n".join(note_lines),
        ha="left",
        va="top",
        fontsize=10,
        color="#334155",
    )

    if any(value is not None for value in fixed_advantage + fresh_advantage):
        advantage_text = [
            f"fixed adv last: {fixed_advantage[-1]:.1f}" if fixed_advantage[-1] is not None else "fixed adv last: n/a",
            f"fresh adv last: {fresh_advantage[-1]:.1f}" if fresh_advantage[-1] is not None else "fresh adv last: n/a",
        ]
    else:
        advantage_text = ["edge advantage unavailable"]

    fig.suptitle(
        (
            f"Exact Search Report: {method} | "
            f"search={len(search_cfg.get('search_seeds', []))} seeds | "
            f"fixed val={len(search_cfg.get('validation_seeds', []))} seeds | "
            f"fresh batch={search_cfg.get('fresh_validation_seed_count', 'n/a')}"
        ),
        fontsize=14,
    )
    fig.text(
        0.99,
        0.015,
        " | ".join(advantage_text),
        ha="right",
        va="bottom",
        fontsize=9,
        color="#475569",
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot exact-search report JSON as a PNG summary.")
    parser.add_argument("report", type=Path, help="Path to exact-search report JSON.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to <report_stem>_viz.png next to the report.",
    )
    args = parser.parse_args()

    output_path = args.output or _default_output_path(args.report)
    save_exact_search_report_plot(args.report, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
