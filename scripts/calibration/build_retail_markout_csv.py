"""Pull retail-only (router-routed) per-swap markouts and USD volumes on the
5bp WETH/USDC pool, day-by-day under the BigQuery bytes-billed cap, then
aggregate to:

  analysis/weth_usdc_90d/markout_5bp_pool_retail.csv
    -> per-swap rows (block_date, usd_amount, markout_15s_bps, markout_next_bps)

  analysis/weth_usdc_90d/markout_5bp_pool_summary_retail.csv
    -> summary stats including USD-weighted quantiles of markout_15s

The 5bp Uniswap V3 pool's full-7d window 2026-05-14..2026-05-20 returns ~0
markouts on day 20, so we use 2026-05-14..2026-05-19 (6d).

Convention reminder: markout_15s is the LP-profitability reference (15s after
the trade) and is the default validation metric. markout_next (next-block mid)
is also pulled for trader-t-cost contexts but is not the primary metric here.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from google.cloud import bigquery


REPO = Path(__file__).resolve().parent.parent.parent
ANALYSIS = REPO / "analysis" / "weth_usdc_90d"
sys.path.insert(0, str(ANALYSIS))
from router_cohorts import retail_router_addresses  # noqa: E402

PER_SWAP_PATH = ANALYSIS / "markout_5bp_pool_retail.csv"
SUMMARY_PATH = ANALYSIS / "markout_5bp_pool_summary_retail.csv"

POOL_5BP = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
ROUTERS = retail_router_addresses()

DAYS = [
    "2026-05-14", "2026-05-15", "2026-05-16",
    "2026-05-17", "2026-05-18", "2026-05-19",
]


def day_query(day: str) -> str:
    routers = ",".join(f"'{r}'" for r in ROUTERS)
    return f"""
    SELECT
      block_date,
      usd_amount,
      markout_next * 1e4 AS next_bps,
      markout_15s  * 1e4 AS m15_bps
    FROM `uniswap-labs.research.markout_prod`
    WHERE block_date = DATE '{day}'
      AND liquidity_pool_address = '{POOL_5BP}'
      AND markout_next IS NOT NULL
      AND markout_15s  IS NOT NULL
      AND usd_amount   IS NOT NULL
      AND LOWER(transaction_to_address) IN UNNEST([{routers}])
    """


def weighted_percentile(values: np.ndarray, weights: np.ndarray, pct: float) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cum = np.cumsum(w)
    cutoff = pct / 100.0 * cum[-1]
    idx = np.searchsorted(cum, cutoff)
    idx = min(idx, len(v) - 1)
    return float(v[idx])


def main() -> None:
    client = bigquery.Client()
    rows_all: list[tuple[str, float, float, float]] = []
    for day in DAYS:
        print(f"querying {day} …")
        job_config = bigquery.QueryJobConfig(maximum_bytes_billed=2 * 1024**3)
        rows = list(client.query(day_query(day), job_config=job_config).result())
        for r in rows:
            if r.usd_amount is None or r.next_bps is None or r.m15_bps is None:
                continue
            rows_all.append((day, float(r.usd_amount), float(r.next_bps), float(r.m15_bps)))
        print(f"  {day}: n={len(rows)}")

    print(f"\ntotal: n={len(rows_all)}")

    PER_SWAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PER_SWAP_PATH.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["block_date", "usd_amount", "markout_next_bps", "markout_15s_bps"])
        for r in rows_all:
            w.writerow([r[0], f"{r[1]:.4f}", f"{r[2]:.6f}", f"{r[3]:.6f}"])
    print(f"wrote {PER_SWAP_PATH}")

    usd     = np.array([r[1] for r in rows_all], dtype=np.float64)
    nxt_bps = np.array([r[2] for r in rows_all], dtype=np.float64)
    m15_bps = np.array([r[3] for r in rows_all], dtype=np.float64)

    # Per-swap-mean (unweighted) and USD-weighted summary stats for markout_15s.
    usd_total = usd.sum()
    usd_w_mean_15 = float((m15_bps * usd).sum() / usd_total)
    usd_w_mean_nx = float((nxt_bps * usd).sum() / usd_total)

    print(f"\nmarkout_15s:  per-swap mean = {m15_bps.mean():+.3f}  usd-weighted = {usd_w_mean_15:+.3f}  bps")
    print(f"markout_next: per-swap mean = {nxt_bps.mean():+.3f}  usd-weighted = {usd_w_mean_nx:+.3f}  bps")

    pct_grid = [1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99]
    with SUMMARY_PATH.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "markout_15s_bps", "markout_next_bps"])
        w.writerow(["n_swaps",       len(rows_all),                    len(rows_all)])
        w.writerow(["usd_total",     f"{usd_total:.2f}",               f"{usd_total:.2f}"])
        w.writerow(["mean_per_swap", f"{m15_bps.mean():.3f}",          f"{nxt_bps.mean():.3f}"])
        w.writerow(["mean_usd_w",    f"{usd_w_mean_15:.3f}",           f"{usd_w_mean_nx:.3f}"])
        w.writerow(["std_per_swap",  f"{m15_bps.std():.3f}",           f"{nxt_bps.std():.3f}"])
        for p in pct_grid:
            w.writerow([f"p{p}_per_swap",
                        f"{np.percentile(m15_bps, p):.3f}",
                        f"{np.percentile(nxt_bps, p):.3f}"])
        for p in pct_grid:
            w.writerow([f"p{p}_usd_w",
                        f"{weighted_percentile(m15_bps, usd, p):.3f}",
                        f"{weighted_percentile(nxt_bps, usd, p):.3f}"])
    print(f"wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
