"""Why the directional fee-skew barely captures LVR in the sim — the full diagnosis.

Companion to nezlobin_arb_autocorr.py (which establishes the microstructure effect:
arb DIRECTION is autocorrelated under a martingale fair price, because the arb leaves
the pool mid offset by the fee, parking fair at the band edge so the next arb most
likely continues the same side).

This script answers: given the effect is real, why doesn't the skew monetize it?
It sweeps the skew LOAD using an offset-IMMUNE signal (RememberedArbSkew: skew next
block toward the directly observed direction of this block's first/arb swap, via
incoming.is_buy -- so the signal cannot be corrupted by the skew's own offset), plus
the doc's EMA-of-PI skew for reference. For each pool, on PAIRED seeds at $1M / §8
normalizer / bottom-of-block-arb env, it reports the 15s-forward LP markout split into
arb / retail, the realized arb-direction PERSISTENCE, and the fraction of arbs the skew
landed on.

The picture: capture peaks at LIGHT load (+~$3.6/seed, ~0.7%) and BACKFIRES at heavy
load -- the skew widens the offset that creates the persistence, flattening the band and
destroying the autocorrelation it exploits (persistence 0.74 -> 0.44). Self-defeating.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arena_eval.core.types import IncomingSwap, TradeInfo
from arena_eval.exact_simple_amm.simulator import ExactSimpleAMMSimulator
from arena_eval.exact_simple_amm.strategies import FixedFeeStrategy
from arena_eval.exact_simple_amm.nezlobin_dynamic_fee import NezlobinDynamicFeeStrategy
from scripts.calibration.nezlobin_backtest import (
    A, NORM_PHI, MARKOUT_S, STEP_S, PRIMARY_DEPTH, make_cfg,
)

CACHE = A / "nezlobin_skew_fix_cache.json"
SEEDS = tuple(range(40, 56))
_CAP = 0.99
LOADS = [0.0, 1.0 / 6, 1.0 / 3, 1.0 / 2, 2.0 / 3, 5.0 / 6, 1.0]   # 4.5/4.5 ... 9/0


@dataclass
class RememberedArbSkew:
    """Resting skew toward the DIRECTLY OBSERVED direction of the previous block's first
    swap (the top-of-block arb), read from incoming.is_buy. No mid inference => the signal
    is immune to corruption by the skew's own price offset. load=0 -> flat 4.5/4.5;
    load=1 -> 9/0. A buy arb (mid pushed up, now below fair) => load the ASK next block."""
    ts_bps: float = 9.0
    load: float = 1.0
    _ts: float = field(default=0.0, init=False)
    _fa: float = field(default=0.0, init=False)
    _fb: float = field(default=0.0, init=False)
    _block: int | None = field(default=None, init=False)
    _last_first_buy: bool | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._ts = self.ts_bps / 1e4
        self._fa = self._fb = self._ts / 2.0

    def after_initialize(self, ix: float, iy: float) -> tuple[float, float]:
        self._fa = self._fb = self._ts / 2.0
        self._block = None
        self._last_first_buy = None
        return (self._fb, self._fa)

    def after_swap(self, trade: TradeInfo) -> tuple[float, float]:
        return (min(self._fb, _CAP), min(self._fa, _CAP))

    def before_swap(self, inc: IncomingSwap) -> tuple[float, float]:
        if int(inc.block) != self._block:
            if self._last_first_buy is not None:
                half = self._ts / 2.0
                shift = self.load * half * (1.0 if self._last_first_buy else -1.0)
                self._fa = min(max(half + shift, 0.0), self._ts)   # buy => load ask
                self._fb = self._ts - self._fa
            self._block = int(inc.block)
            self._last_first_buy = bool(inc.is_buy)               # this block's first swap = the arb
        return (min(self._fb, _CAP), min(self._fa, _CAP))


def measure(strat_factory):
    """Returns dict: arb_mk, retail_mk, total_mk ($/seed), persistence, skew_on_arb_side."""
    arb = ret = 0.0
    cont = cn = on = ntob = 0
    for s in SEEDS:
        sim = ExactSimpleAMMSimulator(config=make_cfg(PRIMARY_DEPTH), submission_strategy=strat_factory(),
                                      normalizer_strategy=FixedFeeStrategy(NORM_PHI, NORM_PHI), seed=s)
        fair, ev = [], []
        while not sim.done:
            out = sim.step_once()
            fair.append(out["fair_price"])
            seen = False
            for e in out["trade_events"]:
                if e["venue"] != "submission":
                    continue
                tob = (e["source"] == "arb") and (not seen)
                if e["source"] == "arb":
                    seen = True
                ev.append((out["timestamp"], e["source"], e["trader_side"], e["amount_x"], e["amount_y"],
                           e["pre_state"].get("submission_ask_fee", np.nan),
                           e["pre_state"].get("submission_bid_fee", np.nan), tob))
        fair = np.asarray(fair); n = len(fair); off = MARKOUT_S / STEP_S
        prevd = None
        for (t, src, side, ax, ay, af, bf, tob) in ev:
            tf = t + off; i0 = int(np.floor(tf))
            if i0 + 1 >= n or ax <= 0:
                continue
            f15 = fair[i0] + (tf - i0) * (fair[i0 + 1] - fair[i0])
            m = (ay - ax * f15) if side == "buy_x" else (ax * f15 - ay)
            if src == "arb":
                arb += m
            else:
                ret += m
            if tob:
                d = 1 if side == "buy_x" else -1
                if np.isfinite(af) and ((af - bf) if d == 1 else (bf - af)) > 1e-9:
                    on += 1
                ntob += 1
                if prevd is not None:
                    cont += int(d == prevd); cn += 1
                prevd = d
    ns = len(SEEDS)
    return dict(arb_mk=arb / ns, retail_mk=ret / ns, total_mk=(arb + ret) / ns,
                persistence=cont / cn if cn else float("nan"),
                skew_on_arb_side=on / ntob if ntob else float("nan"))


def main():
    out = {"meta": dict(seeds=len(SEEDS), depth=PRIMARY_DEPTH), "loads": LOADS, "sweep": [], "doc_ema": None}
    print(f"{'load':>5} {'fa/fb':>9} {'arb mk':>9} {'ret mk':>8} {'total':>9} {'persist':>8} {'on-side':>8} {'vs flat':>8}")
    base = None
    for ld in LOADS:
        r = measure(lambda ld=ld: RememberedArbSkew(load=ld))
        if base is None:
            base = r["total_mk"]
        r["delta_total_vs_flat"] = r["total_mk"] - base
        fa = 4.5 + ld * 4.5
        print(f"{ld:5.2f} {fa:4.1f}/{9 - fa:3.1f} {r['arb_mk']:9.1f} {r['retail_mk']:8.1f} "
              f"{r['total_mk']:9.1f} {r['persistence']:8.3f} {r['skew_on_arb_side']:8.3f} {r['delta_total_vs_flat']:+8.1f}")
        out["sweep"].append(dict(load=ld, **r))
    # doc EMA skew (isolate skew: no surcharge/exception)
    de = measure(lambda: NezlobinDynamicFeeStrategy(surcharge_on=False, exception_on=False))
    de["delta_total_vs_flat"] = de["total_mk"] - base
    out["doc_ema"] = de
    print(f"\n  doc EMA-of-PI skew: total={de['total_mk']:.1f}  persist={de['persistence']:.3f}  "
          f"on-side={de['skew_on_arb_side']:.3f}  vs flat={de['delta_total_vs_flat']:+.1f}")
    CACHE.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {CACHE}")


if __name__ == "__main__":
    main()
