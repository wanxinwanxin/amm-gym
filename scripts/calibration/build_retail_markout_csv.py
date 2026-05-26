"""Pull retail-only (router-routed) per-swap markout distribution on the 5bp pool
for the 6 days of the calibration window (2026-05-14..2026-05-19; 2026-05-20 has
no markout populated for the pool), aggregate to 1001 quantile points, and write:

  analysis/weth_usdc_90d/markout_5bp_pool_percentiles_retail.csv   (1001 quantiles)
  analysis/weth_usdc_90d/markout_5bp_pool_summary_retail.csv       (summary stats)

Each day is queried under the bytes-billed cap; results are streamed and
concatenated locally.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from google.cloud import bigquery


REPO = Path(__file__).resolve().parent.parent.parent
ANALYSIS = REPO / "analysis" / "weth_usdc_90d"
PCT_PATH = ANALYSIS / "markout_5bp_pool_percentiles_retail.csv"
SUMMARY_PATH = ANALYSIS / "markout_5bp_pool_summary_retail.csv"

POOL_5BP = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
ROUTERS = [
    "0xef1c6e67703c7bd7107eed8303fbe6ec2554bf6b",
    "0x3fc91a3afd70395cd496c647d5a6cc9d4b2b7fad",
    "0x66a9893cc07d91d95644aedd05d03f95e1dba8af",
    "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45",
    "0xe592427a0aece92de3edee1f18e0157c05861564",
    "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",
    "0x1111111254fb6c44bac0bed2854e76f90643097d",
    "0x1111111254eeb25477b68fb85ed929f73a960582",
    "0x111111125421ca6dc452d289314280a0f8842a65",
    "0xdef1c0ded9bec7f1a1670819833240f027b25eff",
    "0x0000000000001ff3684f28c67538d4d072c22734",
    "0x9008d19f58aabd9ed0d60971565aa8510560ab41",
    "0xdef171fe48cf0115b1d80b88dc8eab59176fee57",
    "0x6a000f20005980200259b80c5102003040001068",
    "0x6131b5fae19ea4f9d964eac0408e4408b66337b5",
    "0x617dee16b86534a5d792a4d7a62fb491b544111e",
    "0xbdb3ba9ffe392549e1f8658dd2630c141fdf47b6",
    "0x51c72848c68a965f66fa7a88855f9f7784502a7f",
    "0x881d40237659c251811cec9c364ef91dc08d300c",
]

DAYS = [
    "2026-05-14", "2026-05-15", "2026-05-16",
    "2026-05-17", "2026-05-18", "2026-05-19",
]


def day_query(day: str) -> str:
    routers = ",".join(f"'{r}'" for r in ROUTERS)
    return f"""
    SELECT markout_next * 1e4 AS bps
    FROM `uniswap-labs.research.markout_prod`
    WHERE block_date = DATE '{day}'
      AND liquidity_pool_address = '{POOL_5BP}'
      AND markout_next IS NOT NULL
      AND LOWER(transaction_to_address) IN UNNEST([{routers}])
    """


def main() -> None:
    client = bigquery.Client()
    all_bps: list[float] = []
    for day in DAYS:
        print(f"querying {day} …")
        job_config = bigquery.QueryJobConfig(maximum_bytes_billed=2 * 1024**3)
        rows = list(client.query(day_query(day), job_config=job_config).result())
        vals = [float(r.bps) for r in rows if r.bps is not None]
        all_bps.extend(vals)
        print(f"  {day}: n={len(vals)}  avg={np.mean(vals):+.3f} bps  median={np.median(vals):+.3f} bps")

    arr = np.array(all_bps, dtype=np.float64)
    print(f"\ntotal: n={len(arr)}  avg={arr.mean():+.3f} bps  median={np.median(arr):+.3f} bps")
    print(f"       std={arr.std():.3f} bps  p5={np.percentile(arr,5):+.3f}  p95={np.percentile(arr,95):+.3f}")

    # 1001 percentile points (matches existing markout_5bp_pool_percentiles.csv shape)
    pcts = np.arange(0, 100.0 + 1e-9, 0.1)
    qs = np.percentile(arr, pcts)
    PCT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PCT_PATH.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pct", "markout_next_bps"])
        for p, q in zip(pcts, qs):
            w.writerow([f"{p:.1f}", f"{q:.6f}"])
    print(f"\nwrote {PCT_PATH}")

    # Summary in the same format as markout_5bp_pool_summary.csv
    with SUMMARY_PATH.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "markout_next_bps"])
        w.writerow(["n_swaps", len(arr)])
        w.writerow(["avg", f"{arr.mean():.3f}"])
        w.writerow(["std", f"{arr.std():.3f}"])
        for p in [0, 1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99, 100]:
            w.writerow([f"p{p}", f"{np.percentile(arr, p):.3f}"])
    print(f"wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
