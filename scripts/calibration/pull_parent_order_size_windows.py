"""Run analysis/weth_usdc_90d/sql/router_parent_order_size_windows.sql and
dump the result to router_parent_order_size_windows.csv, then derive the
1001-point parent_order_usd_quantiles.csv (the slice the simulator samples
retail order USDs from).

Re-run after touching the router cohort list — the SQL filters parent
orders by the retail_routers cohort, so the size distribution changes
when the cohort changes.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from google.cloud import bigquery


REPO = Path(__file__).resolve().parent.parent.parent
SQL_PATH = REPO / "analysis" / "weth_usdc_90d" / "sql" / "router_parent_order_size_windows.sql"
FULL_CSV = REPO / "analysis" / "weth_usdc_90d" / "router_parent_order_size_windows.csv"
QUANT_CSV = REPO / "analysis" / "weth_usdc_90d" / "parent_order_usd_quantiles.csv"

# Slice that the simulator samples from
QUANT_WINDOW = "6m"
QUANT_MODE = "strict"
QUANT_SIDE = "all"


def main() -> None:
    sql = SQL_PATH.read_text()
    client = bigquery.Client()
    job_config = bigquery.QueryJobConfig(maximum_bytes_billed=80 * 1024**3)
    print(f"running {SQL_PATH.name} …")
    job = client.query(sql, job_config=job_config)
    rows = job.result()
    schema = [f.name for f in rows.schema]

    FULL_CSV.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with FULL_CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(schema)
        for r in rows:
            w.writerow([r[c] for c in schema])
            n += 1
    print(f"wrote {FULL_CSV}: {n:,} rows")
    print(f"bytes_billed:    {job.total_bytes_billed / 1e9:.2f} GB")
    print(f"bytes_processed: {job.total_bytes_processed / 1e9:.2f} GB")

    # Derive the simulator's input quantile file:
    #   (window=6m, mode=strict, side=all) → parent_order_usd_quantiles.csv
    import pandas as pd
    df = pd.read_csv(FULL_CSV)
    sel = df[
        (df["window_name"] == QUANT_WINDOW)
        & (df["mode"]        == QUANT_MODE)
        & (df["side_group"]  == QUANT_SIDE)
    ][["pct", "size_usd"]].sort_values("pct").reset_index(drop=True)
    if len(sel) != 10001:
        print(f"WARN: expected 10001 quantile rows for ({QUANT_WINDOW},{QUANT_MODE},{QUANT_SIDE}), got {len(sel)}")
    sel.to_csv(QUANT_CSV, index=False)
    print(f"wrote {QUANT_CSV}: {len(sel):,} quantile points "
          f"(window={QUANT_WINDOW}, mode={QUANT_MODE}, side={QUANT_SIDE})")

    # Print a summary of the size shifts
    print()
    print("Quantile snapshot (6m, strict, all):")
    print(f"  {'pct':>5}  {'size_usd':>12}")
    for p in [50, 75, 90, 95, 99, 99.9, 100]:
        row = sel[sel["pct"].round(2) == round(p, 2)]
        if len(row):
            print(f"  {p:>5}  ${row['size_usd'].iloc[0]:>11,.0f}")


if __name__ == "__main__":
    main()
