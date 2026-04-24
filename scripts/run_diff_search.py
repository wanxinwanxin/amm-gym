"""Run gradient-based search on the differentiable simple-AMM surrogate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arena_search import (
    GradientSearchConfig,
    evaluate_submission_compact_exact,
    gradient_ascent_search_with_validation,
)


def _seed_range(start: int, count: int) -> tuple[int, ...]:
    return tuple(range(start, start + count))


def _candidate_payload(candidate) -> dict[str, object]:
    return {
        "score": candidate.score,
        "edge_mean_submission": candidate.edge_mean_submission,
        "edge_mean_normalizer": candidate.edge_mean_normalizer,
        "edge_advantage_mean": candidate.edge_advantage_mean,
        "params": candidate.params.to_dict(),
        "metadata": candidate.metadata,
    }


def _format_score(value: float | None) -> str:
    return "na" if value is None else f"{value:.3f}"


class ProgressLogger:
    def __init__(self, path: Path, *, heartbeat_minutes: int = 15) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", buffering=1)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._start_time = time.time()
        self._thread: threading.Thread | None = None
        self.heartbeat_minutes = max(int(heartbeat_minutes), 1)

    def start(self) -> None:
        self.log("run started")
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.log("run finished")
        self._handle.close()

    def log(self, message: str) -> None:
        elapsed_min = (time.time() - self._start_time) / 60.0
        with self._lock:
            self._handle.write(f"[{elapsed_min:7.2f}m] {message}\n")
            self._handle.flush()

    def _heartbeat_loop(self) -> None:
        schedule = [1, 2, 3, 5]
        next_mark = self.heartbeat_minutes
        while not self._stop.is_set():
            elapsed_sec = time.time() - self._start_time
            elapsed_min = elapsed_sec / 60.0
            while schedule and elapsed_min >= schedule[0]:
                mark = schedule.pop(0)
                self.log(f"heartbeat reached {mark}m")
            while elapsed_min >= next_mark:
                self.log(f"heartbeat reached {next_mark}m")
                next_mark += self.heartbeat_minutes
            self._stop.wait(1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradient search on the differentiable simple-AMM surrogate")
    parser.add_argument("--evaluator-kind", choices=("challenge", "real_data"), default="challenge")
    parser.add_argument("--policy-family", choices=("submission_compact",), default="submission_compact")
    parser.add_argument("--train-seed-start", type=int, default=0)
    parser.add_argument("--train-seed-count", type=int, default=16)
    parser.add_argument("--validation-seed-start", type=int, default=1000)
    parser.add_argument("--validation-seed-count", type=int, default=32)
    parser.add_argument("--test-seed-start", type=int, default=2000)
    parser.add_argument("--test-seed-count", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--search-rng-seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--normalizer-fee", type=float, default=0.003)
    parser.add_argument("--fresh-validation-interval", type=int, default=1)
    parser.add_argument("--fresh-validation-seed-count", type=int, default=32)
    parser.add_argument("--rerank-top-k", type=int, default=8)
    parser.add_argument("--init-strategy", choices=("default", "random"), default="default")
    parser.add_argument("--log-progress", action="store_true")
    parser.add_argument("--progress-log", type=Path, default=None)
    parser.add_argument("--heartbeat-minutes", type=int, default=15)
    parser.add_argument("--output", type=Path, default=Path("experiments/diff_search_report.json"))
    args = parser.parse_args()

    train_seeds = _seed_range(args.train_seed_start, args.train_seed_count)
    validation_seeds = _seed_range(args.validation_seed_start, args.validation_seed_count)
    test_seeds = _seed_range(args.test_seed_start, args.test_seed_count)
    config = GradientSearchConfig(
        train_seeds=train_seeds,
        validation_seeds=validation_seeds,
        test_seeds=test_seeds,
        evaluator_kind=args.evaluator_kind,
        normalizer_fee=args.normalizer_fee,
        n_steps=args.n_steps,
        policy_family=args.policy_family,
    )

    progress_log_path = args.progress_log
    if args.log_progress and progress_log_path is None:
        progress_log_path = args.output.with_suffix(".log")
    progress_logger = (
        ProgressLogger(progress_log_path, heartbeat_minutes=args.heartbeat_minutes)
        if progress_log_path is not None
        else None
    )
    if progress_logger is not None:
        progress_logger.start()

    start_time = time.time()

    def progress_callback(record) -> None:
        if not args.log_progress:
            return
        elapsed = time.time() - start_time
        fresh_score = record.fresh_validation.score if record.fresh_validation else None
        line = (
            f"[iter {record.iteration:02d}] elapsed={elapsed / 60.0:.1f}m "
            f"train={record.train_objective:.3f} "
            f"grad_norm={record.gradient_norm:.3f} "
            f"fixed_val={record.fixed_validation.score:.3f} "
            f"fresh_val={_format_score(fresh_score)}"
        )
        print(line, flush=True)
        if progress_logger is not None:
            progress_logger.log(line)

    try:
        study = gradient_ascent_search_with_validation(
            config,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            gradient_clip=args.gradient_clip,
            fresh_validation_interval=args.fresh_validation_interval,
            fresh_validation_seed_count=args.fresh_validation_seed_count,
            rerank_top_k=args.rerank_top_k,
            init_strategy=args.init_strategy,
            seed=args.search_rng_seed,
            progress_callback=progress_callback,
        )
        final_test = study.best_validation
        if test_seeds:
            final_test = evaluate_submission_compact_exact(
                study.best_validation.params,
                test_seeds,
                evaluator_kind=args.evaluator_kind,
                normalizer_fee=args.normalizer_fee,
                n_steps=args.n_steps,
            )

        payload = {
            "method": study.method,
            "objective": "mean(smooth edge_submission surrogate)",
            "policy_family": args.policy_family,
            "evaluator_kind": args.evaluator_kind,
            "normalizer_fee": args.normalizer_fee,
            "search": {
                "rng_seed": args.search_rng_seed,
                "train_seeds": list(train_seeds),
                "validation_seeds": list(validation_seeds),
                "test_seeds": list(test_seeds),
                "n_steps": args.n_steps,
                "iterations": args.iterations,
                "learning_rate": args.learning_rate,
                "gradient_clip": args.gradient_clip,
                "fresh_validation_interval": args.fresh_validation_interval,
                "fresh_validation_seed_count": args.fresh_validation_seed_count,
                "rerank_top_k": args.rerank_top_k,
                "init_strategy": args.init_strategy,
            },
            "best_search": _candidate_payload(study.best_search),
            "best_validation": _candidate_payload(study.best_validation),
            "best_test": _candidate_payload(final_test),
            "validation_rerank": [_candidate_payload(candidate) for candidate in study.validation_rerank],
            "history": [
                {
                    "iteration": record.iteration,
                    "train_objective": record.train_objective,
                    "gradient_norm": record.gradient_norm,
                    "params": record.params.to_dict(),
                    "fixed_validation": _candidate_payload(record.fixed_validation),
                    "fresh_validation": _candidate_payload(record.fresh_validation) if record.fresh_validation else None,
                    "fresh_validation_seeds": list(record.fresh_validation_seeds),
                }
                for record in study.history
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(args.output)
        if progress_logger is not None:
            progress_logger.log(f"final report written to {args.output}")
    finally:
        if progress_logger is not None:
            progress_logger.stop()


if __name__ == "__main__":
    main()
