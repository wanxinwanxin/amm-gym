"""Optional plotting helpers for benchmark summaries."""

from __future__ import annotations

from pathlib import Path


def save_edge_advantage_plot(path: Path, rows: list[dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "matplotlib is required for benchmark plots. Install with `pip install -e .[demo]`."
        ) from exc

    labels = [str(row["policy"]) for row in rows]
    means = [float(row["edge_advantage_mean"]) for row in rows]
    win_rates = [100.0 * float(row["edge_advantage_win_rate"]) for row in rows]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(labels, means, color="#2563eb")
    axes[0].set_title("Mean edge advantage vs normalizer")
    axes[0].set_ylabel("edge advantage")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(labels, win_rates, color="#0f766e")
    axes[1].set_title("Win rate vs normalizer")
    axes[1].set_ylabel("percent of seeds with positive edge advantage")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path)
