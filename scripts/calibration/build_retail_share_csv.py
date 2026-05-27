"""Pull retail (router-routed) WETH/USDC volume and fees by 5bp-pool vs.
everywhere-else, day-by-day under the BigQuery bytes-billed cap, then
aggregate to a single retail-volume-share and retail-fee-share number.

The numerator is volume/fees on the 5bp Uniswap V3 pool
(0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640). The denominator is all
router-routed WETH/USDC volume/fees across every protocol and pool in
`uniswap-allium.ethereum.dex_trades`.

Output:
  analysis/weth_usdc_90d/retail_5bp_share_summary.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

from google.cloud import bigquery


REPO = Path(__file__).resolve().parent.parent.parent
SQL_PATH = REPO / "analysis" / "weth_usdc_90d" / "sql" / "retail_5bp_volume_fee_share.sql"
OUT_PATH = REPO / "analysis" / "weth_usdc_90d" / "retail_5bp_share_summary.csv"

DAYS = [
    "2026-05-14", "2026-05-15", "2026-05-16",
    "2026-05-17", "2026-05-18", "2026-05-19", "2026-05-20",
]


def main() -> None:
    client = bigquery.Client()
    sql = SQL_PATH.read_text()

    tot_vol_5bp = tot_vol_other = 0.0
    tot_fee_5bp = tot_fee_other = 0.0

    for day in DAYS:
        job_config = bigquery.QueryJobConfig(
            maximum_bytes_billed=2 * 1024**3,
            query_parameters=[bigquery.ScalarQueryParameter("target_date", "DATE", day)],
        )
        rows = list(client.query(sql, job_config=job_config).result())
        d_v5 = d_vo = d_f5 = d_fo = 0.0
        for r in rows:
            if r.pool_group == "5bp":
                d_v5 = float(r.volume_usd or 0.0)
                d_f5 = float(r.fees_usd   or 0.0)
            else:
                d_vo = float(r.volume_usd or 0.0)
                d_fo = float(r.fees_usd   or 0.0)
        tot_vol_5bp   += d_v5
        tot_vol_other += d_vo
        tot_fee_5bp   += d_f5
        tot_fee_other += d_fo
        print(f"{day}: vol_5bp=${d_v5/1e6:.2f}M  vol_other=${d_vo/1e6:.2f}M  "
              f"fee_5bp=${d_f5/1e3:.1f}K  fee_other=${d_fo/1e3:.1f}K")

    tot_vol = tot_vol_5bp + tot_vol_other
    tot_fee = tot_fee_5bp + tot_fee_other
    vol_share = tot_vol_5bp / tot_vol if tot_vol > 0 else float("nan")
    fee_share = tot_fee_5bp / tot_fee if tot_fee > 0 else float("nan")

    print()
    print(f"total 7d retail volume: ${tot_vol/1e6:.2f}M "
          f"(5bp ${tot_vol_5bp/1e6:.2f}M, other ${tot_vol_other/1e6:.2f}M)")
    print(f"total 7d retail fees:   ${tot_fee/1e3:.1f}K "
          f"(5bp ${tot_fee_5bp/1e3:.1f}K, other ${tot_fee_other/1e3:.1f}K)")
    print(f"retail volume share at 5bp: {vol_share*100:.2f}%")
    print(f"retail fee   share at 5bp: {fee_share*100:.2f}%")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value"])
        w.writerow(["window", f"{DAYS[0]}..{DAYS[-1]}"])
        w.writerow(["retail_volume_usd_5bp",   f"{tot_vol_5bp:.2f}"])
        w.writerow(["retail_volume_usd_other", f"{tot_vol_other:.2f}"])
        w.writerow(["retail_volume_usd_total", f"{tot_vol:.2f}"])
        w.writerow(["retail_fees_usd_5bp",     f"{tot_fee_5bp:.2f}"])
        w.writerow(["retail_fees_usd_other",   f"{tot_fee_other:.2f}"])
        w.writerow(["retail_fees_usd_total",   f"{tot_fee:.2f}"])
        w.writerow(["retail_volume_share_5bp", f"{vol_share:.6f}"])
        w.writerow(["retail_fee_share_5bp",    f"{fee_share:.6f}"])
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
