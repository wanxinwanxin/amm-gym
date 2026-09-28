"""Fit the volatility elasticity of retail flow behind
``ExactSimpleAMMConfig.retail_arrival_vol_elasticity`` / ``retail_size_vol_elasticity``.

Question: does more retail show up, and does it trade larger, on volatile days? The
exact simulator used a constant Poisson rate and a fixed size distribution on every
step, which biases any comparison between fee designs that react to volatility and
designs that do not (the time-average quote is not the retail-weighted quote).

Data
  analysis/weth_usdc_90d/fe_eth_usdc_daily_activity.csv
      completed Uniswap front-end ETH/USDC swaps on mainnet per UTC day: distinct
      txs, USD shown to the user, distinct wallets. From
      analysis/weth_usdc_90d/sql/fe_eth_usdc_daily_activity.sql (``--pull`` re-runs it).
  analysis/weth_usdc_90d/binance_eth_day_vol.csv
      realized volatility of Binance ETHUSDT 12 s mids per UTC day, bps per 12 s step
      (std of contiguous 12 s log returns). ``--rebuild-vol`` recomputes it from
      binance_eth_12s_mid_3y.parquet (falls back to the tracked _2y file, which only
      covers the first year of the activity window); ``--check-vol`` recomputes and
      compares without writing.

Fit
  log(activity_d) = a + e * log(vol_d) by OLS, raw and detrended (both series minus a
  centred 61-day rolling median in logs, to strip secular trends in front-end usage).
  Also a regime table on the study's volatility terciles (cut points from the
  2023-11-07..2025-11-06 replay sample) and a last-12-months check.

Output
  analysis/weth_usdc_90d/retail_vol_elasticity_fit.json and a markdown summary on stdout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
ANALYSIS = REPO / "analysis" / "weth_usdc_90d"
SQL_PATH = ANALYSIS / "sql" / "fe_eth_usdc_daily_activity.sql"
ACTIVITY_CSV = ANALYSIS / "fe_eth_usdc_daily_activity.csv"
DAY_VOL_CSV = ANALYSIS / "binance_eth_day_vol.csv"
FIT_JSON = ANALYSIS / "retail_vol_elasticity_fit.json"
PARQUETS = [ANALYSIS / "binance_eth_12s_mid_3y.parquet", ANALYSIS / "binance_eth_12s_mid_2y.parquet"]

STEP_S = 12
MIN_RETURNS_PER_DAY = 6_000          # of 7_200 possible 12 s returns; drops days with large gaps
TERCILE_SAMPLE = ("2023-11-07", "2025-11-06")   # replay sample whose vol terciles define the regimes
DETREND_WINDOW_DAYS = 61


# ---------------------------------------------------------------------------------
def pull_activity() -> pd.DataFrame:
    from google.cloud import bigquery

    client = bigquery.Client(project="uniswap-labs")
    job = client.query(SQL_PATH.read_text(), job_config=bigquery.QueryJobConfig(maximum_bytes_billed=40 * 1024**3))
    df = job.result().to_dataframe()
    df["usd"] = df["usd"].astype(float).round(2)
    df["day_utc"] = pd.to_datetime(df["day_utc"]).dt.strftime("%Y-%m-%d")
    print(f"pulled {len(df)} days; scanned {job.total_bytes_processed / 1e9:.1f} GB")
    return df


def day_vol_from_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).sort_values("ts")
    ts = df["ts"].to_numpy(np.int64)
    mid = df["mid"].to_numpy(np.float64)
    contiguous = np.diff(ts) == STEP_S
    r = np.diff(np.log(mid))[contiguous]
    day = pd.to_datetime(ts[1:][contiguous], unit="s").normalize()
    g = pd.Series(r).groupby(day.values)
    out = pd.DataFrame({"vol_bps_per_block": g.std(ddof=0) * 1e4, "n_returns": g.size()})
    out = out[out["n_returns"] >= MIN_RETURNS_PER_DAY]
    out.index = pd.to_datetime(out.index).strftime("%Y-%m-%d")
    out.index.name = "day"
    return out.reset_index()


def rebuild_day_vol() -> pd.DataFrame:
    for path in PARQUETS:
        if path.exists():
            print(f"day vol from {path.name}")
            return day_vol_from_parquet(path)
    raise FileNotFoundError("no binance_eth_12s_mid_*.parquet under analysis/weth_usdc_90d")


# ---------------------------------------------------------------------------------
def _ols_slope(x: pd.Series, y: pd.Series) -> float:
    return float(np.polyfit(x.to_numpy(), y.to_numpy(), 1)[0])


def fit(activity: pd.DataFrame, day_vol: pd.DataFrame) -> dict:
    m = activity.merge(day_vol[["day", "vol_bps_per_block"]], left_on="day_utc", right_on="day", how="inner")
    m["day_utc"] = pd.to_datetime(m["day_utc"])
    m = m.set_index("day_utc").sort_index()
    lv = np.log(m["vol_bps_per_block"])
    lv_d = lv - lv.rolling(DETREND_WINDOW_DAYS, center=True, min_periods=30).median()

    elasticities: dict[str, dict[str, float]] = {}
    for col in ("swaps", "usd", "wallets"):
        ly = np.log(m[col])
        ly_d = ly - ly.rolling(DETREND_WINDOW_DAYS, center=True, min_periods=30).median()
        ok = ly_d.notna() & lv_d.notna()
        elasticities[col] = {
            "raw": _ols_slope(lv, ly),
            "raw_corr": float(np.corrcoef(lv, ly)[0, 1]),
            "detrended": _ols_slope(lv_d[ok], ly_d[ok]),
            "detrended_corr": float(np.corrcoef(lv_d[ok], ly_d[ok])[0, 1]),
        }

    sample = day_vol[(day_vol["day"] >= TERCILE_SAMPLE[0]) & (day_vol["day"] <= TERCILE_SAMPLE[1])]["vol_bps_per_block"]
    cuts = tuple(float(x) for x in np.quantile(sample, [1 / 3, 2 / 3]))
    regime = np.where(m["vol_bps_per_block"] <= cuts[0], "calm", np.where(m["vol_bps_per_block"] <= cuts[1], "mid", "volatile"))

    def regime_table(frame: pd.DataFrame, reg: np.ndarray) -> dict:
        g = frame.groupby(reg).agg(
            days=("swaps", "size"),
            vol_bps=("vol_bps_per_block", "mean"),
            swaps=("swaps", "mean"),
            usd=("usd", "mean"),
            wallets=("wallets", "mean"),
        )
        table = {k: {c: float(v) for c, v in row.items()} for k, row in g.iterrows()}
        table["volatile_over_calm"] = {c: float(g.loc["volatile", c] / g.loc["calm", c]) for c in ("swaps", "usd", "wallets")}
        return table

    last_year_start = m.index.max() - pd.Timedelta(days=365)
    recent = m[m.index >= last_year_start]
    recent_regime = regime[m.index >= last_year_start]

    e_count = elasticities["swaps"]["detrended"]
    e_usd = elasticities["usd"]["detrended"]
    return {
        "sample": {
            "days": int(len(m)),
            "first_day": m.index.min().strftime("%Y-%m-%d"),
            "last_day": m.index.max().strftime("%Y-%m-%d"),
            "detrend_window_days": DETREND_WINDOW_DAYS,
        },
        "elasticities": elasticities,
        "regime_cut_points_bps": cuts,
        "regime_table": regime_table(m, regime),
        "regime_table_last_12_months": regime_table(recent, recent_regime),
        "recommended": {
            "retail_arrival_vol_elasticity": round(e_count, 2),
            "retail_size_vol_elasticity": round(e_usd - e_count, 2),
            "retail_vol_reference_bps": round(float(sample.mean()), 2),
            "note": "arrival = detrended swap-count elasticity; size = USD elasticity minus count elasticity "
            "(mean size elasticity); reference = mean day vol of the replay sample so the multiplier averages ~1",
        },
    }


def markdown(res: dict) -> str:
    e = res["elasticities"]
    t = res["regime_table"]
    t12 = res["regime_table_last_12_months"]
    c = res["regime_cut_points_bps"]
    lines = [
        f"sample: {res['sample']['days']} days, {res['sample']['first_day']} .. {res['sample']['last_day']}",
        "",
        "| series | elasticity (raw) | corr | elasticity (detrended) | corr |",
        "|---|---|---|---|---|",
    ]
    for col in ("swaps", "usd", "wallets"):
        lines.append(f"| {col} | {e[col]['raw']:.2f} | {e[col]['raw_corr']:.2f} | {e[col]['detrended']:.2f} | {e[col]['detrended_corr']:.2f} |")
    lines += [
        "",
        f"regimes: calm <= {c[0]:.2f}, mid <= {c[1]:.2f}, volatile > {c[1]:.2f} bps/step",
        "",
        "| regime | days | vol bps | swaps/day | USD/day | wallets/day |",
        "|---|---|---|---|---|---|",
    ]
    for k in ("calm", "mid", "volatile"):
        r = t[k]
        lines.append(f"| {k} | {r['days']:.0f} | {r['vol_bps']:.2f} | {r['swaps']:.0f} | ${r['usd'] / 1e6:.1f}m | {r['wallets']:.0f} |")
    v = t["volatile_over_calm"]
    v12 = t12["volatile_over_calm"]
    lines += [
        "",
        f"volatile / calm: swaps {v['swaps']:.2f}x, USD {v['usd']:.2f}x, wallets {v['wallets']:.2f}x "
        f"(last 12 months: {v12['swaps']:.2f}x, {v12['usd']:.2f}x, {v12['wallets']:.2f}x)",
        "",
        "recommended: " + ", ".join(f"{k} = {val}" for k, val in res["recommended"].items() if k != "note"),
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pull", action="store_true", help="re-run the BigQuery SQL and rewrite the activity CSV")
    ap.add_argument("--rebuild-vol", action="store_true", help="recompute the day-vol CSV from the 12 s parquet")
    ap.add_argument("--check-vol", action="store_true", help="recompute day vol and compare with the CSV, no write")
    args = ap.parse_args()

    if args.pull:
        activity = pull_activity()
        activity.to_csv(ACTIVITY_CSV, index=False)
        print(f"wrote {ACTIVITY_CSV}")
    else:
        activity = pd.read_csv(ACTIVITY_CSV)

    if args.rebuild_vol:
        day_vol = rebuild_day_vol()
        day_vol[["day", "vol_bps_per_block"]].round(3).to_csv(DAY_VOL_CSV, index=False)
        print(f"wrote {DAY_VOL_CSV}: {len(day_vol)} days")
    else:
        day_vol = pd.read_csv(DAY_VOL_CSV)
        if args.check_vol:
            fresh = rebuild_day_vol()
            both = day_vol.merge(fresh, on="day", suffixes=("_csv", "_new"))
            diff = (both["vol_bps_per_block_csv"] - both["vol_bps_per_block_new"]).abs()
            print(
                f"check-vol: {len(both)} common days of {len(day_vol)} csv / {len(fresh)} recomputed; "
                f"max |diff| {diff.max():.4f} bps, median {diff.median():.4f}"
            )

    res = fit(activity, day_vol)
    FIT_JSON.write_text(json.dumps(res, indent=1))
    print(markdown(res))
    print(f"\nwrote {FIT_JSON}")


if __name__ == "__main__":
    main()
