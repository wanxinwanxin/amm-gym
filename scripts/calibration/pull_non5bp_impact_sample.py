"""Run analysis/weth_usdc_90d/sql/non5bp_impact_sample_v3_pool_mid_7d.sql and
dump the result to non5bp_impact_sample_v3_pool_mid_7d.csv. Streamed row by
row so the BQ Python client doesn't load everything into memory at once.

Re-run after touching the router cohort list — the SQL inlines the cohort
addresses, so the sample's filter changes with each cohort revision.
"""
from __future__ import annotations

import csv
from pathlib import Path

from google.cloud import bigquery


REPO = Path(__file__).resolve().parent.parent.parent
SQL_PATH = REPO / "analysis" / "weth_usdc_90d" / "sql" / "non5bp_impact_sample_v3_pool_mid_7d.sql"
OUT_PATH = REPO / "analysis" / "weth_usdc_90d" / "non5bp_impact_sample_v3_pool_mid_7d.csv"


def main() -> None:
    sql = SQL_PATH.read_text()
    client = bigquery.Client()
    job_config = bigquery.QueryJobConfig(maximum_bytes_billed=80 * 1024**3)
    print(f"running {SQL_PATH.name} …")
    job = client.query(sql, job_config=job_config)
    rows = job.result()  # iterates without loading whole result
    schema = [f.name for f in rows.schema]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with OUT_PATH.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(schema)
        for r in rows:
            w.writerow([r[c] for c in schema])
            n += 1
            if n % 5000 == 0:
                print(f"  {n:,} rows…")
    print(f"wrote {OUT_PATH}: {n:,} rows")
    print(f"bytes_billed:    {job.total_bytes_billed / 1e9:.2f} GB")
    print(f"bytes_processed: {job.total_bytes_processed / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
