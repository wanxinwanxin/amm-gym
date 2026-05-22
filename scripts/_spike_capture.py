"""One-shot capture of the §5 simulator-markout array for spike analysis."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKTREE / "presentation"))
sys.path.insert(0, str(WORKTREE))

import numpy as np  # noqa: E402

from calibration_helpers import load_final, run_calibrated_markout  # noqa: E402


def main() -> None:
    final = load_final()
    seeds = tuple(final["seeds_holdout"])
    n_steps = int(final["n_steps"])
    sim = run_calibrated_markout(final, seeds=seeds, n_steps=n_steps)
    markouts = np.asarray(sim["markouts_bps"], dtype=float)

    out_dir = WORKTREE / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "spike_markouts.npy", markouts)
    print(f"saved {markouts.size:,} markouts to {out_dir/'spike_markouts.npy'}")
    print(f"  n_seeds={sim['n_seeds']}  n_steps={sim['n_steps']}")
    print(f"  agg markout = {sim['aggregate_markout_bps']:+.3f} bps")
    print(f"  vol share submission = {sim['volume_share_submission']*100:.2f}%")
    print(f"  min/p1/p5/p50/p95/p99/max = "
          f"{markouts.min():.3f} / "
          f"{np.percentile(markouts, 1):.3f} / "
          f"{np.percentile(markouts, 5):.3f} / "
          f"{np.percentile(markouts, 50):.3f} / "
          f"{np.percentile(markouts, 95):.3f} / "
          f"{np.percentile(markouts, 99):.3f} / "
          f"{markouts.max():.3f} bps")


if __name__ == "__main__":
    main()
