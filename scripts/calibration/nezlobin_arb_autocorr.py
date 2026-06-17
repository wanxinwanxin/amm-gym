"""Microstructure test: is ARB DIRECTION autocorrelated under a martingale fair price?

The fee-skew LVR mechanism (Nezlobin doc) does NOT need fair-price momentum. It needs
the *arbitrage direction* to be predictable. After a buy arb the pool mid is left at
(1-f_a)*fair < fair (the arber stops when after-fee marginal price = fair). So fair sits
at the ASK edge of the no-arb band; any up move triggers another buy arb, while a sell
arb needs fair to traverse the whole spread. => conditional on a buy arb, the next arb is
more likely a buy than a sell, EVEN WHEN fair is a martingale.

This script measures, at base config (regime-switching = martingale by construction),
the bottom-of-block-arb env, $1M depth, 16 paired seeds:

  (1) FLAT 4.5/4.5 pool (no skew): the TOB arb direction sequence ->
        - unconditional P(buy) / P(sell)
        - P(next arb buy | this buy), P(next arb sell | this buy)  [the user's claim]
        - lag-1 autocorrelation of signed arb direction
        - block-open offset (mid-fair)/fair vs the last arb direction (does it persist?)
  (2) SKEW-ONLY Nezlobin (surcharge_on=False, exception_on=False): does the resting
      skew land on the side the TOB arb actually takes? (continuation-correct rate,
      skew magnitude), and arb markout vs flat (the realized LVR capture).

Prints a table + caches arrays for plotting.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arena_eval.exact_simple_amm.simulator import ExactSimpleAMMSimulator
from arena_eval.exact_simple_amm.strategies import FixedFeeStrategy
from arena_eval.exact_simple_amm.nezlobin_dynamic_fee import NezlobinDynamicFeeStrategy
from scripts.calibration.nezlobin_backtest import (
    A, NORM_PHI, MARKOUT_S, STEP_S, PRIMARY_DEPTH, make_cfg,
)

CACHE = A / "nezlobin_arb_autocorr_cache.json"
SEEDS = tuple(range(40, 56))


def run(strat_factory):
    """Per block (step): the top-of-block arb (first arb event), if any.

    Returns list of per-seed dicts of parallel arrays indexed by block:
      dir   : +1 buy arb (mid<fair), -1 sell arb (mid>fair), 0 no TOB arb
      mid0  : pool mid at block open (pre_spot_price of the TOB arb; NaN if no arb)
      fair  : fair price that block
      askf, bidf : resting fees the TOB arb paid (pre_state), in bps
      arb_mk: 15s-forward markout $ of the TOB arb (NaN if no arb)
    """
    seeds_out = []
    for s in SEEDS:
        sim = ExactSimpleAMMSimulator(config=make_cfg(PRIMARY_DEPTH), submission_strategy=strat_factory(),
                                      normalizer_strategy=FixedFeeStrategy(NORM_PHI, NORM_PHI), seed=s)
        fair_path = []
        rec = []  # (block_idx, dir, mid0, fair, askf_bps, bidf_bps, ax, ay, side, tstamp)
        bi = 0
        while not sim.done:
            out = sim.step_once()
            t = out["timestamp"]
            fair_path.append(out["fair_price"])
            tob = None
            for e in out["trade_events"]:
                if e["venue"] == "submission" and e["source"] == "arb":
                    tob = e
                    break  # first arb of the block == top-of-block arb
            if tob is None:
                rec.append((bi, 0, np.nan, out["fair_price"], np.nan, np.nan, 0.0, 0.0, "", t))
            else:
                d = 1 if tob["trader_side"] == "buy_x" else -1
                af = 1e4 * tob["pre_state"].get("submission_ask_fee", np.nan)
                bf = 1e4 * tob["pre_state"].get("submission_bid_fee", np.nan)
                rec.append((bi, d, tob["pre_spot_price"], out["fair_price"], af, bf,
                            tob["amount_x"], tob["amount_y"], tob["trader_side"], t))
            bi += 1
        fair_path = np.asarray(fair_path)
        # 15s-forward markout for the TOB arb of each block
        n = len(fair_path); off = MARKOUT_S / STEP_S
        arb_mk = np.full(len(rec), np.nan)
        for k, (_, d, _m, _f, _af, _bf, ax, ay, side, t) in enumerate(rec):
            if d == 0 or ax <= 0:
                continue
            tf = t + off; i0 = int(np.floor(tf))
            if i0 + 1 >= n:
                continue
            f15 = fair_path[i0] + (tf - i0) * (fair_path[i0 + 1] - fair_path[i0])
            arb_mk[k] = (ay - ax * f15) if side == "buy_x" else (ax * f15 - ay)
        seeds_out.append(dict(
            dir=np.array([r[1] for r in rec]),
            mid0=np.array([r[2] for r in rec], float),
            fair=np.array([r[3] for r in rec], float),
            askf=np.array([r[4] for r in rec], float),
            bidf=np.array([r[5] for r in rec], float),
            arb_mk=arb_mk,
        ))
    return seeds_out


def analyze_autocorr(seeds_out):
    """Pool transitions across seeds (no cross-seed adjacency)."""
    nbuy = nsell = nnone = 0
    # transition counts conditional on this-block arb direction
    cnt = {("buy", "buy"): 0, ("buy", "sell"): 0, ("buy", "none"): 0,
           ("sell", "buy"): 0, ("sell", "sell"): 0, ("sell", "none"): 0}
    signed = []  # signed dir of arb-blocks for lag-1 autocorr (consecutive arb blocks)
    off_by_lastdir = {"buy": [], "sell": []}  # next-block-open offset bps by last arb dir
    for so in seeds_out:
        d = so["dir"]; mid0 = so["mid0"]; fair = so["fair"]
        nbuy += int((d == 1).sum()); nsell += int((d == -1).sum()); nnone += int((d == 0).sum())
        lbl = {1: "buy", -1: "sell", 0: "none"}
        for i in range(len(d) - 1):
            if d[i] == 0:
                continue
            cnt[(lbl[d[i]], lbl[d[i + 1]])] += 1
            # offset at the NEXT block open (mid0[i+1]) relative to fair[i+1]
            if d[i + 1] != 0 and np.isfinite(mid0[i + 1]) and fair[i + 1] > 0:
                off_by_lastdir[lbl[d[i]]].append(1e4 * (mid0[i + 1] - fair[i + 1]) / fair[i + 1])
        # lag-1 autocorr of signed dir over consecutive arb blocks
        for i in range(len(d) - 1):
            if d[i] != 0 and d[i + 1] != 0:
                signed.append((d[i], d[i + 1]))
    signed = np.array(signed)
    if len(signed):
        a, b = signed[:, 0].astype(float), signed[:, 1].astype(float)
        ac = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    else:
        ac = float("nan")
    tot = nbuy + nsell + nnone
    buy_then = cnt[("buy", "buy")] + cnt[("buy", "sell")] + cnt[("buy", "none")]
    p_buy_after_buy = cnt[("buy", "buy")] / buy_then if buy_then else float("nan")
    p_sell_after_buy = cnt[("buy", "sell")] / buy_then if buy_then else float("nan")
    p_none_after_buy = cnt[("buy", "none")] / buy_then if buy_then else float("nan")
    # among NEXT-blocks that HAD an arb, share that continued the same side:
    bb, bs = cnt[("buy", "buy")], cnt[("buy", "sell")]
    p_cont_given_next_arb = bb / (bb + bs) if (bb + bs) else float("nan")
    return dict(
        n_buy=nbuy, n_sell=nsell, n_none=nnone, n_total=tot,
        uncond_p_buy=nbuy / (nbuy + nsell) if (nbuy + nsell) else float("nan"),
        p_buy_after_buy=p_buy_after_buy, p_sell_after_buy=p_sell_after_buy,
        p_none_after_buy=p_none_after_buy, p_cont_given_next_arb=p_cont_given_next_arb,
        lag1_autocorr_signed=ac,
        offset_after_buy_bps=float(np.mean(off_by_lastdir["buy"])) if off_by_lastdir["buy"] else float("nan"),
        offset_after_sell_bps=float(np.mean(off_by_lastdir["sell"])) if off_by_lastdir["sell"] else float("nan"),
    )


def analyze_skew(seeds_out):
    """For the skew pool: did the resting skew land on the arb's side? magnitude? arb markout?"""
    correct = total = 0
    skew_signed = []   # (ask-bid) signed by arb dir: + means loaded on the arb's side
    arb_mk = []
    for so in seeds_out:
        d, af, bf, mk = so["dir"], so["askf"], so["bidf"], so["arb_mk"]
        for i in range(len(d)):
            if d[i] == 0 or not np.isfinite(af[i]):
                continue
            total += 1
            # buy arb pays ask; skew correct if ask>bid. sell arb pays bid; correct if bid>ask.
            on_arb_side = (af[i] - bf[i]) if d[i] == 1 else (bf[i] - af[i])
            skew_signed.append(on_arb_side)
            if on_arb_side > 1e-9:
                correct += 1
            if np.isfinite(mk[i]):
                arb_mk.append(mk[i])
    return dict(
        n_arb=total, frac_skew_on_arb_side=correct / total if total else float("nan"),
        mean_skew_on_arb_side_bps=float(np.mean(skew_signed)) if skew_signed else float("nan"),
        arb_mk_total_per_seed=float(np.sum(arb_mk) / len(seeds_out)) if arb_mk else float("nan"),
    )


def main():
    print("running FLAT 4.5/4.5 (microstructure baseline) ...")
    flat = run(lambda: FixedFeeStrategy(4.5e-4, 4.5e-4))
    ac = analyze_autocorr(flat)
    print("running SKEW-ONLY Nezlobin (surcharge_on=False, exception_on=False) ...")
    skew = run(lambda: NezlobinDynamicFeeStrategy(surcharge_on=False, exception_on=False))
    sk = analyze_skew(skew)
    # arb markout of the flat pool for comparison (LVR baseline)
    flat_arb_mk = float(np.nansum(np.concatenate([s["arb_mk"] for s in flat])) / len(flat))

    print("\n================ ARB-DIRECTION AUTOCORRELATION (flat 4.5/4.5) ================")
    print(f"  TOB arbs:  buy={ac['n_buy']}  sell={ac['n_sell']}  none(no arb)={ac['n_none']}  (of {ac['n_total']} blocks)")
    print(f"  unconditional P(buy among arbs)          = {ac['uncond_p_buy']:.3f}")
    print(f"  P(next arb BUY  | this BUY arb)          = {ac['p_buy_after_buy']:.3f}")
    print(f"  P(next arb SELL | this BUY arb)          = {ac['p_sell_after_buy']:.3f}")
    print(f"  P(no arb next   | this BUY arb)          = {ac['p_none_after_buy']:.3f}")
    print(f"  P(continue same side | next block HAS arb)= {ac['p_cont_given_next_arb']:.3f}   <- user's claim (>0.5 ?)")
    print(f"  lag-1 autocorr of signed arb direction   = {ac['lag1_autocorr_signed']:+.3f}")
    print(f"  block-open offset (mid-fair) after BUY arb = {ac['offset_after_buy_bps']:+.2f} bps  (expect <0: mid below fair)")
    print(f"  block-open offset (mid-fair) after SELL arb= {ac['offset_after_sell_bps']:+.2f} bps  (expect >0)")

    print("\n================ DOES THE EMA SKEW EXPLOIT IT? (skew-only Nezlobin) ===========")
    print(f"  TOB arbs measured                        = {sk['n_arb']}")
    print(f"  frac of arbs with skew on the ARB's side = {sk['frac_skew_on_arb_side']:.3f}   (0.5 = coin flip)")
    print(f"  mean skew loaded on the arb's side       = {sk['mean_skew_on_arb_side_bps']:+.2f} bps  (+ helps, max +4.5)")
    print(f"  TOB-arb markout: skew-only = {sk['arb_mk_total_per_seed']:+.1f}/seed   flat 4.5/4.5 = {flat_arb_mk:+.1f}/seed")
    print(f"  => skew LVR capture on TOB arb           = {sk['arb_mk_total_per_seed'] - flat_arb_mk:+.1f}/seed")

    out = dict(meta=dict(seeds=len(SEEDS), depth=PRIMARY_DEPTH), autocorr=ac, skew=sk,
               flat_arb_mk_per_seed=flat_arb_mk)
    CACHE.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {CACHE}")


if __name__ == "__main__":
    main()
