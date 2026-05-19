"""Run random or CEM search over the exact simple-AMM replica."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arena_search import SearchConfig, cross_entropy_search, random_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize a reactive controller on the simple-AMM replica")
    parser.add_argument("--method", choices=("random", "cem"), default="random")
    parser.add_argument("--seed-count", type=int, default=64)
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--population-size", type=int, default=16)
    parser.add_argument(
        "--submission-liquidity-fraction",
        type=float,
        default=1.0,
        help="Initial submission-pool liquidity as a fraction of the normalizer pool",
    )
    parser.add_argument("--output", type=Path, default=Path("experiments/simple_amm_best_params.json"))
    args = parser.parse_args()

    config = SearchConfig(
        seeds=tuple(range(args.seed_count)),
        submission_liquidity_fraction=args.submission_liquidity_fraction,
    )
    if args.method == "random":
        evaluations = random_search(config, n_candidates=args.candidates, seed=args.search_seed)
    else:
        evaluations = cross_entropy_search(
            config,
            generations=args.generations,
            population_size=args.population_size,
            seed=args.search_seed,
        )
    best = evaluations[0]
    payload = {
        "score": best.score,
        "edge_mean_submission": best.edge_mean_submission,
        "edge_mean_normalizer": best.edge_mean_normalizer,
        "edge_advantage_mean": best.edge_advantage_mean,
        "retail_edge_mean_submission": best.retail_edge_mean_submission,
        "retail_edge_mean_normalizer": best.retail_edge_mean_normalizer,
        "arb_loss_mean_submission": best.arb_loss_mean_submission,
        "arb_loss_mean_normalizer": best.arb_loss_mean_normalizer,
        "annualized_edge_return_mean_submission": best.annualized_edge_return_mean_submission,
        "annualized_retail_edge_return_mean_submission": best.annualized_retail_edge_return_mean_submission,
        "annualized_arb_loss_return_mean_submission": best.annualized_arb_loss_return_mean_submission,
        "retail_markout_bps_mean_submission": best.retail_markout_bps_mean_submission,
        "arb_markout_bps_mean_submission": best.arb_markout_bps_mean_submission,
        "initial_value_mean": best.initial_value_mean,
        "episode_seconds_mean": best.episode_seconds_mean,
        "submission_liquidity_fraction": args.submission_liquidity_fraction,
        "params": best.params.to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(args.output)


if __name__ == "__main__":
    main()
