"""Sensitivity + diagnostics for the §9 Guidestar backtest.

Three independent blocks, all written to a single JSON cache
(analysis/weth_usdc_90d/guidestar_sensitivity_cache.json) that the notebook §9
loads and plots. Re-run offline to refresh.

  1. by_size_diag  — per-size-bin RETAIL markout with sample counts, MEDIAN, a
     leave-one-out (drop the single most extreme trade) mean, and the most
     extreme single-trade bps. Answers "is the mid-x-axis mean-markout spike
     outlier / small-sample driven?".
  2. arrival_decomp — final LP markout decomposed into retail(+) vs arb/LVR(−)
     components AND retail volume captured, across the retail-arrival multiplier.
     Grounds the arrival-rate intuition (the breakeven chart only stores totals).
  3. sizedist      — vary the retail size distribution and recompute final
     markout/retail/arb/volume per pool:
        mean_sweep : multiply every order size by m (shifts mean & sd together).
        std_sweep  : log-spread around the fixed median, size' = med*(size/med)^s
                     (s<1 narrows, s>1 widens; median pinned).

Same engine, normalizer, metric, and pools as guidestar_backtest.py.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arena_eval.exact_simple_amm.config import ExactSimpleAMMConfig
from arena_eval.exact_simple_amm.simulator import ExactSimpleAMMSimulator
from arena_eval.exact_simple_amm.strategies import FixedFeeStrategy
from scripts.calibration.guidestar_backtest import (
    A, NORM_PHI, NORM_D, INIT_PX, STEP_S, MARKOUT_S, FEE_INIT_BPS, BASE_ARRIVAL,
    N_STEPS, PRIMARY_DEPTH, POOL_ORDER, make_strat, markout,
)

CACHE = A / "guidestar_sensitivity_cache.json"
QUANT_DEFAULT = str(A / "parent_order_usd_quantiles.csv")
SEEDS = tuple(range(40, 64))               # 24 paired seeds
ARRIVAL_MULTS = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 11.0, 16.0]
MEAN_MULTS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
STD_SPREADS = [0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
SIZE_EDGES = np.logspace(0, 6, 19)         # same 18 log bins as the by_size chart


# ---- engine (quantiles path is overridable, unlike guidestar_backtest.make_cfg) ----
def make_cfg(depth_usd: float, arrival_mult: float, quantiles_path: str) -> ExactSimpleAMMConfig:
    return ExactSimpleAMMConfig(
        n_steps=N_STEPS, initial_price=INIT_PX, initial_x=NORM_D / INIT_PX, initial_y=NORM_D,
        submission_liquidity_fraction=depth_usd / NORM_D, evaluator_kind="real_data",
        price_process_kind="regime_switching", retail_flow_kind="empirical_usd_size",
        retail_arrival_rate=BASE_ARRIVAL * arrival_mult, retail_buy_prob=0.4627,
        regime_invcdf_path=str(A / "regimes_invcdf.csv"),
        regime_transition_path=str(A / "regimes_transition_matrix.csv"),
        retail_usd_quantiles_path=quantiles_path,
        normalizer_tracks_fair=True,
    )


def run_pool(name, depth_usd, seed, arrival_mult=1.0, quantiles_path=QUANT_DEFAULT):
    sim = ExactSimpleAMMSimulator(config=make_cfg(depth_usd, arrival_mult, quantiles_path),
                                  submission_strategy=make_strat(name),
                                  normalizer_strategy=FixedFeeStrategy(NORM_PHI, NORM_PHI), seed=seed)
    fair, trades = [], []
    while not sim.done:
        out = sim.step_once()
        fair.append(out["fair_price"])
        for ev in out["trade_events"]:
            if ev["venue"] == "submission":
                trades.append((out["timestamp"], ev["source"], ev["trader_side"], ev["amount_x"], ev["amount_y"]))
    return np.asarray(fair), trades


def run_agg(name, depth_usd, seeds, arrival_mult=1.0, quantiles_path=QUANT_DEFAULT,
            collect_retail=False, collect_all=False):
    """Mean-per-sim total/retail/arb markout + retail volume; optional per-trade arrays.
    collect_retail -> retail-only (notl, mk); collect_all -> every trade (notl, mk, is_retail)."""
    tot = ret = arb = rvol = 0.0
    rnotl, rmk = [], []
    anotl, amk, aret = [], [], []
    for s in seeds:
        fair, trades = run_pool(name, depth_usd, s, arrival_mult, quantiles_path)
        for (t, src, m, notl) in markout(fair, trades):
            tot += m
            is_ret = src == "retail"
            if is_ret:
                ret += m
                rvol += notl
                if collect_retail:
                    rnotl.append(notl)
                    rmk.append(m)
            else:
                arb += m
            if collect_all:
                anotl.append(notl)
                amk.append(m)
                aret.append(is_ret)
    n = len(seeds)
    out = dict(total=tot / n, retail=ret / n, arb=arb / n, volume=rvol / n)
    if collect_retail:
        out["rnotl"] = np.asarray(rnotl)
        out["rmk"] = np.asarray(rmk)
    if collect_all:
        out["anotl"] = np.asarray(anotl)
        out["amk"] = np.asarray(amk)
        out["aret"] = np.asarray(aret, dtype=bool)
    return out


# ---- size-distribution transforms (write a temp quantile CSV the sim reads) ----
def _load_quants():
    rows = np.genfromtxt(QUANT_DEFAULT, delimiter=",", names=True, dtype=float)
    return np.asarray(rows["pct"], float), np.asarray(rows["size_usd"], float)


def _write_quants(pct, sz, path):
    with open(path, "w") as f:
        f.write("pct,size_usd\n")
        for p, s in zip(pct, sz):
            f.write(f"{p},{s}\n")


def _realized_stats(pct, sz):
    draws = np.interp(np.linspace(0, 100, 200_000), pct, sz)
    pos = draws[draws > 0]
    return float(draws.mean()), float(np.std(np.log10(pos)))


# ---- block 1: by-size diagnostic --------------------------------------------
def block_by_size():
    print("block 1: by_size diagnostic ...")
    centers = np.sqrt(SIZE_EDGES[:-1] * SIZE_EDGES[1:])
    out = {"centers": centers.round(2).tolist(), "edges": SIZE_EDGES.round(3).tolist()}
    nseeds = len(SEEDS)
    for nm in POOL_ORDER:
        agg = run_agg(nm, PRIMARY_DEPTH, SEEDS, collect_retail=True, collect_all=True)
        notl, mk = agg["rnotl"], agg["rmk"]
        ok = notl > 1.0
        notl, mk = notl[ok], mk[ok]
        bps = 1e4 * mk / notl
        n_, mean_, med_, drop1_, maxabs_, p25_, p75_ = ([] for _ in range(7))
        # all-trades (retail + arb) stats for the total-markout($) centre panel
        an, am, ar = agg["anotl"], agg["amk"], agg["aret"]
        all_n, arb_n, tot_sum, arb_sum, top1, top5 = ([] for _ in range(6))
        for lo, hi in zip(SIZE_EDGES[:-1], SIZE_EDGES[1:]):
            sel = (notl >= lo) & (notl < hi)
            b = bps[sel]
            n_.append(int(b.size))
            if b.size >= 1:
                mean_.append(float(np.mean(b)))
                med_.append(float(np.median(b)))
                p25_.append(float(np.percentile(b, 25)))
                p75_.append(float(np.percentile(b, 75)))
                j = int(np.argmax(np.abs(b)))
                maxabs_.append(float(b[j]))
                rest = np.delete(b, j)
                drop1_.append(float(np.mean(rest)) if rest.size else None)
            else:
                mean_.append(None); med_.append(None); p25_.append(None)
                p75_.append(None); maxabs_.append(None); drop1_.append(None)
            asel = (an >= lo) & (an < hi)
            am_b = am[asel]
            all_n.append(int(am_b.size))
            arb_n.append(int((asel & ~ar).sum()))
            tot_sum.append(float(am_b.sum() / nseeds))                  # matches the chart's total_usd
            arb_sum.append(float(am[asel & ~ar].sum() / nseeds))
            if am_b.size >= 1:
                absm = np.abs(am_b)
                s_abs = float(absm.sum())
                order = np.sort(absm)[::-1]
                top1.append(float(order[0] / s_abs) if s_abs > 0 else None)
                top5.append(float(order[:5].sum() / s_abs) if s_abs > 0 else None)
            else:
                top1.append(None); top5.append(None)
        out[nm] = dict(count=n_, mean_bps=mean_, median_bps=med_, mean_drop1_bps=drop1_,
                       max_abs_bps=maxabs_, p25_bps=p25_, p75_bps=p75_,
                       all_count=all_n, arb_count=arb_n, total_usd=tot_sum, arb_usd=arb_sum,
                       top1_abs_share=top1, top5_abs_share=top5)
    return out


# ---- block 2: arrival-rate decomposition ------------------------------------
def block_arrival():
    print("block 2: arrival decomposition ...")
    out = {"mults": ARRIVAL_MULTS}
    for nm in POOL_ORDER:
        tot, ret, arb, vol = [], [], [], []
        for mlt in ARRIVAL_MULTS:
            a = run_agg(nm, PRIMARY_DEPTH, SEEDS, arrival_mult=mlt)
            tot.append(round(a["total"], 2)); ret.append(round(a["retail"], 2))
            arb.append(round(a["arb"], 2)); vol.append(round(a["volume"], 1))
        out[nm] = dict(total=tot, retail=ret, arb=arb, volume=vol)
    return out


# ---- block 3: size-distribution sensitivity ---------------------------------
def block_sizedist():
    print("block 3: size-distribution sensitivity ...")
    pct, sz = _load_quants()
    median = float(np.interp(50.0, pct, sz))
    size_cap = float(sz.max())          # empirical p100 ($1.36M): never simulate orders beyond observed support
    tmpdir = Path(tempfile.mkdtemp(prefix="gs_sizedist_"))
    base_mean, base_lstd = _realized_stats(pct, sz)

    mean_sweep = {"mults": MEAN_MULTS, "means": [], "log10std": []}
    for nm in POOL_ORDER:
        mean_sweep[nm] = dict(total=[], retail=[], arb=[], volume=[])
    for m in MEAN_MULTS:
        sz2 = sz * m
        p = tmpdir / f"mean_{m}.csv"
        _write_quants(pct, sz2, p)
        rmean, rlstd = _realized_stats(pct, sz2)
        mean_sweep["means"].append(round(rmean, 2)); mean_sweep["log10std"].append(round(rlstd, 3))
        for nm in POOL_ORDER:
            a = run_agg(nm, PRIMARY_DEPTH, SEEDS, quantiles_path=str(p))
            mean_sweep[nm]["total"].append(round(a["total"], 2)); mean_sweep[nm]["retail"].append(round(a["retail"], 2))
            mean_sweep[nm]["arb"].append(round(a["arb"], 2)); mean_sweep[nm]["volume"].append(round(a["volume"], 1))

    std_sweep = {"spreads": STD_SPREADS, "means": [], "log10std": [], "median": round(median, 2)}
    for nm in POOL_ORDER:
        std_sweep[nm] = dict(total=[], retail=[], arb=[], volume=[])
    for s in STD_SPREADS:
        with np.errstate(divide="ignore", invalid="ignore"):
            sz2 = median * np.power(np.where(sz > 0, sz, median) / median, s)
        sz2 = np.clip(sz2, 0.0, size_cap)    # cap at empirical max: widening spread must not invent $B whale orders
        p = tmpdir / f"std_{s}.csv"
        _write_quants(pct, sz2, p)
        rmean, rlstd = _realized_stats(pct, sz2)
        std_sweep["means"].append(round(rmean, 2)); std_sweep["log10std"].append(round(rlstd, 3))
        for nm in POOL_ORDER:
            a = run_agg(nm, PRIMARY_DEPTH, SEEDS, quantiles_path=str(p))
            std_sweep[nm]["total"].append(round(a["total"], 2)); std_sweep[nm]["retail"].append(round(a["retail"], 2))
            std_sweep[nm]["arb"].append(round(a["arb"], 2)); std_sweep[nm]["volume"].append(round(a["volume"], 1))

    return dict(base_mean=round(base_mean, 2), base_log10std=round(base_lstd, 3),
                size_cap=round(size_cap, 0), mean_sweep=mean_sweep, std_sweep=std_sweep)


def main():
    cache = dict(
        meta=dict(seeds=len(SEEDS), n_steps=N_STEPS, step_seconds=STEP_S, markout_seconds=MARKOUT_S,
                  fee_init_bps=FEE_INIT_BPS, primary_depth=PRIMARY_DEPTH, pool_order=POOL_ORDER,
                  normalizer="§8 full-market φ=4.65bps D=$275M", base_arrival=BASE_ARRIVAL),
        by_size_diag=block_by_size(),
        arrival_decomp=block_arrival(),
        sizedist=block_sizedist(),
    )
    CACHE.write_text(json.dumps(cache))
    print(f"\nwrote {CACHE} ({CACHE.stat().st_size/1e3:.0f} KB)")
    # quick console summary
    ad = cache["arrival_decomp"]
    print("\narrival decomposition (final markout $, $1M):")
    print(f"  {'mult':>5}", *[f"{m:>6.0f}" for m in ad['mults']])
    for nm in POOL_ORDER:
        print(f"  {nm[:18]:18s} tot", *[f"{v:6.0f}" for v in ad[nm]['total']])
        print(f"  {'':18s} ret", *[f"{v:6.0f}" for v in ad[nm]['retail']])
        print(f"  {'':18s} arb", *[f"{v:6.0f}" for v in ad[nm]['arb']])
    print("\nsize-dist MEAN sweep (final total markout $):  means", cache["sizedist"]["mean_sweep"]["means"])
    for nm in POOL_ORDER:
        print(f"  {nm[:18]:18s}", cache["sizedist"]["mean_sweep"][nm]["total"])
    print("\nsize-dist STD sweep (final total markout $):  log10std", cache["sizedist"]["std_sweep"]["log10std"])
    for nm in POOL_ORDER:
        print(f"  {nm[:18]:18s}", cache["sizedist"]["std_sweep"][nm]["total"])


def main_sizedist_only():
    """Recompute only block 3 (size-distribution) and splice into the existing cache."""
    cache = json.loads(CACHE.read_text())
    cache["sizedist"] = block_sizedist()
    CACHE.write_text(json.dumps(cache))
    print(f"\nrewrote sizedist in {CACHE}")
    sd = cache["sizedist"]
    print("MEAN sweep means:", sd["mean_sweep"]["means"])
    for nm in POOL_ORDER:
        print(f"  {nm[:18]:18s} total", sd["mean_sweep"][nm]["total"])
    print(f"STD sweep (cap ${sd['size_cap']:,.0f}) log10std:", sd["std_sweep"]["log10std"], " means:", sd["std_sweep"]["means"])
    for nm in POOL_ORDER:
        print(f"  {nm[:18]:18s} total", sd["std_sweep"][nm]["total"])


def main_bysize_only():
    cache = json.loads(CACHE.read_text())
    cache["by_size_diag"] = block_by_size()
    CACHE.write_text(json.dumps(cache))
    print(f"\nrewrote by_size_diag in {CACHE}")
    g = cache["by_size_diag"][POOL_ORDER[0]]
    ctr = cache["by_size_diag"]["centers"]
    print(f"{'center$':>9} {'all_n':>6} {'arb_n':>6} {'total$':>9} {'arb$':>9} {'top1%':>6} {'top5%':>6}")
    for i, x in enumerate(ctr):
        f = lambda v, p="6.2f": (("%" + p) % v) if v is not None else "   nan"
        print(f"{x:9.1f} {g['all_count'][i]:6d} {g['arb_count'][i]:6d} {g['total_usd'][i]:9.1f} {g['arb_usd'][i]:9.1f} "
              f"{f(g['top1_abs_share'][i]) if g['top1_abs_share'][i] is not None else '   nan'} "
              f"{f(g['top5_abs_share'][i]) if g['top5_abs_share'][i] is not None else '   nan'}")


if __name__ == "__main__":
    if "--sizedist-only" in sys.argv:
        main_sizedist_only()
    elif "--bysize-only" in sys.argv:
        main_bysize_only()
    else:
        main()
