"""Quantify the gap between exact and diff simulators across seeds.

Compares per-seed scores of the same policy on three stacks:

- exact_simple_amm (ground truth)
- diff_simple_amm EXACT_PATH (parity replay)
- diff_simple_amm SMOOTH_TRAIN (differentiable surrogate)

Reports mean/stdev gaps and Pearson correlation, for both challenge and
realistic evaluators on FixedFee and SubmissionCompact policies.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arena_eval.diff_simple_amm import (
    DiffMode,
    DiffSimpleAMMSimulatorConfig,
    FixedFeeDiffPolicy,
    SubmissionCompactDiffPolicy,
    build_challenge_tape,
    build_realistic_tape,
    realistic_env_vector,
    run_challenge_rollout,
    run_realistic_rollout,
    smooth_submission_compact_result,
    submission_compact_param_vector,
)
from arena_eval.diff_simple_amm.objectives import challenge_env_vector
from arena_eval.exact_simple_amm import ExactSimpleAMMConfig, FixedFeeStrategy, run_seed
from arena_policies.submission_safe import SubmissionCompactParams, SubmissionCompactStrategy


def _gap_stats(label: str, exact_vals: list[float], cmp_vals: list[float]) -> dict[str, object]:
    diffs = [c - e for e, c in zip(exact_vals, cmp_vals)]
    rel = [
        (c - e) / abs(e)
        for e, c in zip(exact_vals, cmp_vals)
        if abs(e) > 1e-9
    ]
    pearson = (
        float(np.corrcoef(exact_vals, cmp_vals)[0, 1])
        if len(exact_vals) > 1 and statistics.pstdev(exact_vals) > 0 and statistics.pstdev(cmp_vals) > 0
        else float("nan")
    )
    return {
        "label": label,
        "n": len(exact_vals),
        "exact_mean": statistics.mean(exact_vals),
        "exact_stdev": statistics.pstdev(exact_vals),
        "cmp_mean": statistics.mean(cmp_vals),
        "cmp_stdev": statistics.pstdev(cmp_vals),
        "diff_mean": statistics.mean(diffs),
        "diff_stdev": statistics.pstdev(diffs),
        "abs_rel_diff_mean": statistics.mean(abs(r) for r in rel) if rel else float("nan"),
        "abs_rel_diff_max": max((abs(r) for r in rel), default=float("nan")),
        "pearson": pearson,
    }


def _print_block(stats: dict[str, object]) -> None:
    print(f"== {stats['label']} (n={stats['n']}) ==")
    print(f"  exact  mean={stats['exact_mean']:+.4f}  stdev={stats['exact_stdev']:.4f}")
    print(f"  cmp    mean={stats['cmp_mean']:+.4f}  stdev={stats['cmp_stdev']:.4f}")
    print(f"  diff   mean={stats['diff_mean']:+.4f}  stdev={stats['diff_stdev']:.4f}")
    print(f"  |rel|  mean={stats['abs_rel_diff_mean']:.3f}  max={stats['abs_rel_diff_max']:.3f}")
    print(f"  pearson={stats['pearson']:.3f}")
    print()


def _run_mode(
    *,
    mode: str,
    seeds: tuple[int, ...],
    n_steps: int,
    policy_label: str,
    params: SubmissionCompactParams | None,
) -> dict[str, object]:
    """Run a single (mode, policy) sweep and return gap stats vs exact."""

    exact_scores: list[float] = []
    diff_exact_scores: list[float] = []
    smooth_scores: list[float] = []

    is_realistic = mode == "realistic"
    diff_param_vector = submission_compact_param_vector(params or SubmissionCompactParams())

    for seed in seeds:
        if is_realistic:
            cfg = replace(ExactSimpleAMMConfig.real_data_from_seed(seed), n_steps=n_steps)
            env = realistic_env_vector(cfg)
            tape = build_realistic_tape(config=cfg, seed=seed)
        else:
            cfg = replace(ExactSimpleAMMConfig.from_seed(seed), n_steps=n_steps)
            env = challenge_env_vector(cfg)
            tape = build_challenge_tape(config=cfg, seed=seed)

        if params is None:
            exact_strategy = FixedFeeStrategy()
            diff_submission = FixedFeeDiffPolicy()
        else:
            exact_strategy = SubmissionCompactStrategy(params)
            diff_submission = SubmissionCompactDiffPolicy(params=params)

        e = run_seed(
            exact_strategy,
            seed,
            config=cfg,
            normalizer_strategy=FixedFeeStrategy(),
        )
        exact_scores.append(float(e.score))

        run_diff = run_realistic_rollout if is_realistic else run_challenge_rollout
        d_exact = run_diff(
            config=DiffSimpleAMMSimulatorConfig(
                mode=DiffMode.EXACT_PATH, seed=seed, exact_config=cfg
            ),
            tape=tape,
            submission_policy=diff_submission,
            normalizer_policy=FixedFeeDiffPolicy(),
        )
        diff_exact_scores.append(float(d_exact.score))

        smooth = smooth_submission_compact_result(
            diff_param_vector,
            config=cfg,
            tape=tape,
            env_vector=env,
            seed=seed,
        )
        smooth_scores.append(float(smooth.score))

    label_prefix = f"{mode.upper()} / {policy_label}"
    return {
        "mode": mode,
        "policy": policy_label,
        "exact_scores": exact_scores,
        "diff_exact_scores": diff_exact_scores,
        "smooth_scores": smooth_scores,
        "diff_exact_vs_exact": _gap_stats(
            f"{label_prefix} / diff_exact_path vs exact",
            exact_scores,
            diff_exact_scores,
        ),
        "smooth_vs_exact": _gap_stats(
            f"{label_prefix} / smooth_train vs exact",
            exact_scores,
            smooth_scores,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments" / "diff_faithfulness_calibration.json",
    )
    args = parser.parse_args()

    seeds = tuple(range(args.seed_start, args.seed_start + args.n_seeds))
    print(f"seeds={seeds[0]}..{seeds[-1]} ({len(seeds)})  n_steps={args.n_steps}")
    started = time.time()

    fixed_params = SubmissionCompactParams(
        base_fee=0.003, min_fee=0.003, max_fee=0.003
    ).normalized()
    compact_params = SubmissionCompactParams(
        base_fee=0.004,
        flow_fast_decay=0.61,
        flow_slow_decay=0.9,
        skew_weight=0.07,
        hot_fee_bump=0.005,
    ).normalized()

    runs = []
    for mode in ("challenge", "realistic"):
        runs.append(
            _run_mode(
                mode=mode,
                seeds=seeds,
                n_steps=args.n_steps,
                policy_label="FixedFee(0.003)",
                params=None,
            )
        )
        runs.append(
            _run_mode(
                mode=mode,
                seeds=seeds,
                n_steps=args.n_steps,
                policy_label="FixedFee(0.003) via compact-vector",
                params=fixed_params,
            )
        )
        runs.append(
            _run_mode(
                mode=mode,
                seeds=seeds,
                n_steps=args.n_steps,
                policy_label="SubmissionCompact",
                params=compact_params,
            )
        )

    elapsed = time.time() - started
    print(f"\nFinished in {elapsed:.1f}s\n")
    for run in runs:
        _print_block(run["diff_exact_vs_exact"])
        _print_block(run["smooth_vs_exact"])

    payload = {
        "config": {
            "n_seeds": args.n_seeds,
            "n_steps": args.n_steps,
            "seed_start": args.seed_start,
            "elapsed_seconds": elapsed,
        },
        "runs": [
            {
                "mode": run["mode"],
                "policy": run["policy"],
                "exact_scores": run["exact_scores"],
                "diff_exact_scores": run["diff_exact_scores"],
                "smooth_scores": run["smooth_scores"],
                "diff_exact_vs_exact": run["diff_exact_vs_exact"],
                "smooth_vs_exact": run["smooth_vs_exact"],
            }
            for run in runs
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
