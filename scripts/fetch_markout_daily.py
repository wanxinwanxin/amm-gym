"""Fetch per-day markout statistics for the Uniswap V3 5bp WETH/USDC pool from
`uniswap-labs.research.markout_prod`, in 30-day chunks via the `bq` CLI, and
write a CSV of daily aggregates that downstream code can roll up to arbitrary
windows.

Sign convention (verified on sample rows from 2026-05-07 -- see report):
    markout_next ≈ (benchmark - executed_price_with_binance) / benchmark
                 = LP-perspective markout (positive = LP gain).
    benchmark is the Binance midprice at the next swap on the same pool;
    markout_15s uses Binance midprice 15s after the swap.
    The lp_pnl_*_dollar fields are USD-denominated LP profit per swap.

USD-weighted markout (bps) = SUM(markout * usd_amount) / SUM(usd_amount) * 1e4
                            = SUM(markout_dollar) / SUM(usd_amount) * 1e4

Output: a CSV with one row per day with columns
    block_date, n_swaps, simple_avg_next_bps, usd_weighted_next_bps,
    simple_avg_15s_bps, usd_weighted_15s_bps, total_usd,
    lp_pnl_next_dollar, lp_pnl_15s_dollar
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

POOL = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
PROJECT_TABLE = "`uniswap-labs.research.markout_prod`"

DAILY_SQL = """
SELECT
  block_date,
  COUNT(*) AS n_swaps,
  AVG(markout_next) * 1e4 AS simple_avg_next_bps,
  SAFE_DIVIDE(SUM(markout_next * usd_amount), SUM(usd_amount)) * 1e4 AS usd_weighted_next_bps,
  AVG(markout_15s) * 1e4 AS simple_avg_15s_bps,
  SAFE_DIVIDE(SUM(markout_15s * usd_amount), SUM(usd_amount)) * 1e4 AS usd_weighted_15s_bps,
  SUM(usd_amount) AS total_usd,
  SUM(markout_next_dollar) AS lp_pnl_next_dollar,
  SUM(markout_15s_dollar) AS lp_pnl_15s_dollar
FROM {table}
WHERE block_date BETWEEN DATE '{start}' AND DATE '{end}'
  AND LOWER(liquidity_pool_address) = '{pool}'
  AND markout_next IS NOT NULL
GROUP BY block_date
ORDER BY block_date
""".strip()


def daterange(start: date, end: date, step_days: int) -> Iterable[tuple[date, date]]:
    cur = start
    while cur <= end:
        chunk_end = min(end, cur + timedelta(days=step_days - 1))
        yield cur, chunk_end
        cur = chunk_end + timedelta(days=1)


def run_bq(sql: str, max_bytes: int) -> list[dict]:
    cmd = [
        "bq",
        "query",
        "--use_legacy_sql=false",
        "--format=json",
        f"--maximum_bytes_billed={max_bytes}",
        "--quiet",
        sql,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"bq error (rc={res.returncode}):\n{res.stderr}\nstdout:\n{res.stdout}"
        )
    if not res.stdout.strip():
        return []
    return json.loads(res.stdout)


FIELDS = [
    "block_date",
    "n_swaps",
    "simple_avg_next_bps",
    "usd_weighted_next_bps",
    "simple_avg_15s_bps",
    "usd_weighted_15s_bps",
    "total_usd",
    "lp_pnl_next_dollar",
    "lp_pnl_15s_dollar",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk-days", type=int, default=30)
    ap.add_argument(
        "--max-bytes",
        type=int,
        default=20_000_000_000,
        help="Per-query bytes-billed cap (default 20 GB)",
    )
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows_seen = 0
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for chunk_start, chunk_end in daterange(start, end, args.chunk_days):
            sql = DAILY_SQL.format(
                table=PROJECT_TABLE,
                start=chunk_start.isoformat(),
                end=chunk_end.isoformat(),
                pool=POOL,
            )
            rows = run_bq(sql, args.max_bytes)
            for row in rows:
                writer.writerow({k: row.get(k) for k in FIELDS})
                rows_seen += 1
            print(
                f"[{chunk_start}..{chunk_end}] fetched {len(rows)} day-rows "
                f"(total so far: {rows_seen})",
                flush=True,
            )
            fh.flush()


if __name__ == "__main__":
    main()
