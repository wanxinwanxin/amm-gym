"""Convert the BigQuery JSON dump from non5bp_impact_sample_v3_pool_mid_7d.sql
into a CSV at analysis/weth_usdc_90d/non5bp_impact_sample_v3_pool_mid_7d.csv.

Also joins with the existing non5bp_impact_sample_7d.csv (fair-price-referenced)
on tx_hash so we can compare pool_mid_pre_bps vs fair_bps side by side for
overlapping txs.

Standard library only.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: convert_pool_mid_sample.py <bq_json_file>", file=sys.stderr)
        sys.exit(2)
    bq_path = Path(sys.argv[1])
    repo = Path(__file__).resolve().parent.parent.parent
    out_csv = repo / "analysis/weth_usdc_90d/non5bp_impact_sample_v3_pool_mid_7d.csv"
    fair_csv = repo / "analysis/weth_usdc_90d/non5bp_impact_sample_7d.csv"

    raw = bq_path.read_text()
    rows = json.loads(raw)
    print(f"loaded {len(rows)} rows from BigQuery JSON dump")

    fieldnames = [
        "tx_hash",
        "block_timestamp",
        "size_usd",
        "n_legs",
        "n_distinct_pools",
        "side",
        "n_distinct_sides",
        "fee_tier_sample",
        "effective_exec_usdc_per_weth",
        "pool_mid_pre_blended",
        "observed_spread_pool_bps",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            ts = r.get("block_timestamp")
            if isinstance(ts, dict):
                ts = ts.get("value")
            w.writerow({
                "tx_hash": r["tx_hash"],
                "block_timestamp": ts,
                "size_usd": r["size_usd"],
                "n_legs": r["n_legs"],
                "n_distinct_pools": r["n_distinct_pools"],
                "side": r["side"],
                "n_distinct_sides": r["n_distinct_sides"],
                "fee_tier_sample": r["fee_tier_sample"],
                "effective_exec_usdc_per_weth": r["effective_exec_usdc_per_weth"],
                "pool_mid_pre_blended": r["pool_mid_pre_blended"],
                "observed_spread_pool_bps": r["observed_spread_pool_bps"],
            })
    print(f"wrote {out_csv}")

    # Build joined view on tx_hash
    fair: dict[str, dict] = {}
    if fair_csv.exists():
        with fair_csv.open() as fh:
            for row in csv.DictReader(fh):
                fair[row["tx_hash"]] = row
        print(f"loaded {len(fair)} rows from {fair_csv.name}")

    joined_csv = repo / "analysis/weth_usdc_90d/non5bp_impact_sample_pool_vs_fair_7d.csv"
    overlap = 0
    with joined_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "tx_hash", "block_timestamp", "size_usd_v3", "size_usd_fair",
            "side_v3", "side_fair",
            "spread_pool_bps", "spread_fair_bps",
            "pool_mid_pre_blended", "fair_minus_pool_bps_pre",  # informational
        ])
        for r in rows:
            tx = r["tx_hash"]
            f = fair.get(tx)
            if not f:
                continue
            overlap += 1
            ts = r.get("block_timestamp")
            if isinstance(ts, dict):
                ts = ts.get("value")
            try:
                size_fair = float(f["size_usd"])
            except (TypeError, ValueError):
                size_fair = ""
            try:
                spread_fair = float(f["observed_spread_bps"])
            except (TypeError, ValueError):
                spread_fair = ""
            w.writerow([
                tx, ts, r["size_usd"], size_fair,
                r["side"], f["side"],
                r["observed_spread_pool_bps"], spread_fair,
                r["pool_mid_pre_blended"], "",
            ])
    print(f"joined {overlap} rows -> {joined_csv}")


if __name__ == "__main__":
    main()
