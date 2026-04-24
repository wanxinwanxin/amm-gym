"""Run exact-score search with fixed and fresh validation tracking."""

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

from arena_search import SearchConfig, cross_entropy_search_with_validation, evaluate_params_on_seeds, random_search_with_validation


def _seed_range(start: int, count: int) -> tuple[int, ...]:
    return tuple(range(start, start + count))


def _candidate_payload(candidate) -> dict[str, object]:
    return {
        "score": candidate.score,
        "edge_mean_submission": candidate.edge_mean_submission,
        "edge_mean_normalizer": candidate.edge_mean_normalizer,
        "edge_advantage_mean": candidate.edge_advantage_mean,
        "params": candidate.params.to_dict(),
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
    parser = argparse.ArgumentParser(description="Search exact simple-AMM policies on the challenge score")
    parser.add_argument("--method", choices=("random", "cem"), default="cem")
    parser.add_argument(
        "--policy-family",
        choices=(
            "reactive",
            "inventory_toxicity",
            "piecewise",
            "belief_state",
            "latent_flow",
            "latent_fair",
            "latent_toxicity",
            "latent_competition",
            "latent_full",
            "submission_compact",
            "submission_regime",
            "submission_basis",
        ),
        default="reactive",
    )
    parser.add_argument(
        "--evaluator-kind",
        choices=("challenge", "real_data"),
        default="challenge",
        help="Dynamics backend to train and validate on",
    )
    parser.add_argument(
        "--cross-eval-kind",
        action="append",
        choices=("challenge", "real_data"),
        default=[],
        help="Additional evaluator(s) to score the final selected model on",
    )
    parser.add_argument("--search-seed-start", type=int, default=0)
    parser.add_argument("--search-seed-count", type=int, default=64)
    parser.add_argument("--validation-seed-start", type=int, default=1000)
    parser.add_argument("--validation-seed-count", type=int, default=128)
    parser.add_argument("--test-seed-start", type=int, default=2000)
    parser.add_argument("--test-seed-count", type=int, default=256)
    parser.add_argument("--search-rng-seed", type=int, default=0)
    parser.add_argument("--fresh-validation-interval", type=int, default=1)
    parser.add_argument("--fresh-validation-seed-count", type=int, default=64)
    parser.add_argument("--rerank-top-k", type=int, default=8)
    parser.add_argument("--normalizer-fee", type=float, default=0.003)
    parser.add_argument("--rounds", type=int, default=8, help="Random-search rounds")
    parser.add_argument("--candidates-per-round", type=int, default=16, help="Random-search batch size")
    parser.add_argument("--generations", type=int, default=8, help="CEM generations")
    parser.add_argument("--population-size", type=int, default=24, help="CEM population size")
    parser.add_argument("--elite-fraction", type=float, default=0.2, help="CEM elite fraction")
    parser.add_argument(
        "--log-progress",
        action="store_true",
        help="Print a one-line progress summary after each search iteration",
    )
    parser.add_argument(
        "--progress-log",
        type=Path,
        default=None,
        help="Optional file path for heartbeat and iteration progress logs",
    )
    parser.add_argument(
        "--heartbeat-minutes",
        type=int,
        default=15,
        help="Heartbeat interval in minutes for the progress log after the initial 1/2/3/5m marks",
    )
    parser.add_argument("--output", type=Path, default=Path("experiments/exact_search_report.json"))
    args = parser.parse_args()

    search_seeds = _seed_range(args.search_seed_start, args.search_seed_count)
    validation_seeds = _seed_range(args.validation_seed_start, args.validation_seed_count)
    test_seeds = _seed_range(args.test_seed_start, args.test_seed_count)
    config = SearchConfig(
        seeds=search_seeds,
        normalizer_fee=args.normalizer_fee,
        policy_family=args.policy_family,
        evaluator_kind=args.evaluator_kind,
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
            f"search={record.best_search.score:.3f} "
            f"fixed_val={record.fixed_validation.score:.3f} "
            f"fresh_val={_format_score(fresh_score)}"
        )
        print(line, flush=True)
        if progress_logger is not None:
            progress_logger.log(line)

    try:
        if args.method == "random":
            study = random_search_with_validation(
                config,
                fixed_validation_seeds=validation_seeds,
                rounds=args.rounds,
                candidates_per_round=args.candidates_per_round,
                rerank_top_k=args.rerank_top_k,
                fresh_validation_interval=args.fresh_validation_interval,
                fresh_validation_seed_count=args.fresh_validation_seed_count,
                seed=args.search_rng_seed,
                progress_callback=progress_callback,
            )
        else:
            study = cross_entropy_search_with_validation(
                config,
                fixed_validation_seeds=validation_seeds,
                generations=args.generations,
                population_size=args.population_size,
                elite_fraction=args.elite_fraction,
                rerank_top_k=args.rerank_top_k,
                fresh_validation_interval=args.fresh_validation_interval,
                fresh_validation_seed_count=args.fresh_validation_seed_count,
                seed=args.search_rng_seed,
                progress_callback=progress_callback,
            )

        final_test = study.best_validation
        if test_seeds:
            final_test = evaluate_params_on_seeds(
                study.best_validation.params,
                test_seeds,
                normalizer_fee=args.normalizer_fee,
                evaluator_kind=args.evaluator_kind,
            )

        cross_eval = {}
        for evaluator_kind in sorted(set(args.cross_eval_kind)):
            cross_eval[evaluator_kind] = _candidate_payload(
                evaluate_params_on_seeds(
                    study.best_validation.params,
                    test_seeds,
                    normalizer_fee=args.normalizer_fee,
                    evaluator_kind=evaluator_kind,
                )
            )

        payload = {
            "method": study.method,
            "objective": "mean(edge_submission)",
            "policy_family": args.policy_family,
            "evaluator_kind": args.evaluator_kind,
            "normalizer_fee": args.normalizer_fee,
            "search": {
                "rng_seed": args.search_rng_seed,
                "search_seeds": list(search_seeds),
                "validation_seeds": list(validation_seeds),
                "test_seeds": list(test_seeds),
                "fresh_validation_interval": args.fresh_validation_interval,
                "fresh_validation_seed_count": args.fresh_validation_seed_count,
                "rerank_top_k": args.rerank_top_k,
                "rounds": args.rounds if args.method == "random" else None,
                "candidates_per_round": args.candidates_per_round if args.method == "random" else None,
                "generations": args.generations if args.method == "cem" else None,
                "population_size": args.population_size if args.method == "cem" else None,
                "elite_fraction": args.elite_fraction if args.method == "cem" else None,
            },
            "best_search": _candidate_payload(study.best_search),
            "best_validation": _candidate_payload(study.best_validation),
            "best_test": _candidate_payload(final_test),
            "cross_eval": cross_eval,
            "validation_rerank": [_candidate_payload(candidate) for candidate in study.validation_rerank],
            "history": [
                {
                    "iteration": record.iteration,
                    "best_search": _candidate_payload(record.best_search),
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
