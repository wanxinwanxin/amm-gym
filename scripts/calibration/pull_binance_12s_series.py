"""Pull the raw time-ordered 12s Binance ETHUSDT mid series (2 years) and cache it.

This is the data source for the historical-replay arb-only lab: instead of a
synthetic martingale (GBM) or the semi-parametric regime process (same CDF, IID
within a regime), we replay the ACTUAL Binance series — slicing it into real ~17h
windows — so the fair path carries the real serial structure (autocorrelated
jumps, vol clustering) the regime model assumes away.

Scans ~45 GB in BigQuery (~$0.28 on-demand); downloads ~5.26M rows via the
Storage API and writes a compact parquet cache. Run once:

    .venv/bin/python scripts/calibration/pull_binance_12s_series.py            # bills uniswap-allium
    .venv/bin/python scripts/calibration/pull_binance_12s_series.py uniswap-labs-dev

Output: analysis/weth_usdc_90d/binance_eth_12s_mid_2y.parquet  (cols: ts int64, mid float32)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from google.cloud import bigquery

REPO = Path(__file__).resolve().parents[2]
SQL = REPO / "analysis" / "weth_usdc_90d" / "sql" / "binance_eth_12s_mid_2y.sql"
OUT = REPO / "analysis" / "weth_usdc_90d" / "binance_eth_12s_mid_2y.parquet"
BUCKET_S = 12


def main(bill_project: str = "uniswap-allium") -> None:
    sql = SQL.read_text()
    print(f"billing project: {bill_project}\nrunning query (scans ~45 GB)...", flush=True)
    client = bigquery.Client(project=bill_project)
    job = client.query(sql)
    df = job.result().to_dataframe(create_bqstorage_client=True)
    print(f"  scanned {job.total_bytes_processed / 1e9:.1f} GB, {len(df):,} rows", flush=True)

    df = df.sort_values("ts").reset_index(drop=True)
    ts = df["ts"].to_numpy(np.int64)
    mid = df["mid"].to_numpy(np.float64)

    # diagnostics ---------------------------------------------------------
    span_days = (ts[-1] - ts[0]) / 86400.0
    expected = int((ts[-1] - ts[0]) // BUCKET_S) + 1
    gaps = np.diff(ts)
    n_gap = int((gaps != BUCKET_S).sum())
    big_gaps = gaps[gaps != BUCKET_S]
    logret = np.diff(np.log(mid))
    print(f"  date span : {span_days:.1f} days  ({pd.to_datetime(ts[0], unit='s')} .. "
          f"{pd.to_datetime(ts[-1], unit='s')})")
    print(f"  buckets   : {len(ts):,} present / {expected:,} expected "
          f"({100*len(ts)/expected:.2f}% coverage)")
    print(f"  gaps      : {n_gap:,} non-12s steps; "
          f"max gap {big_gaps.max()/60 if n_gap else 0:.1f} min; "
          f"gaps>1min: {int((big_gaps > 60).sum()) if n_gap else 0}")
    print(f"  logret    : std {logret.std()*1e4:.2f} bp, "
          f"min {logret.min()*1e4:.1f} bp, max {logret.max()*1e4:.1f} bp, "
          f"mean {logret.mean()*1e4:.4f} bp")

    out = pd.DataFrame({"ts": ts.astype(np.int64), "mid": mid.astype(np.float32)})
    out.to_parquet(OUT, index=False, compression="zstd")
    print(f"\nwrote {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "uniswap-allium")
