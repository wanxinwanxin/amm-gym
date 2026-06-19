"""Arb-only LVR lab — isolate the effect of each fee mechanism on LVR protection.

A deliberately minimal world (no retail): a single constant-product pool, a martingale
i.i.d. log-normal fair price, and exactly ONE top-of-block arbitrage per block that
corrects the pool toward fair (stops when its after-fee marginal price = fair). With no
retail there is never a swap n>1, so the intra-block surcharge never fires and no
bottom-of-block arb is needed — the only thing that moves value is the TOB arb, so the
LP's loss to that arb *is* the LVR. We reuse the validated StrategyAMM (fee mechanics)
and Arbitrageur (arb sizing) and only add the v2 fee strategy + the arb-only loop.

v2 top-of-block fee (Volatile Hook Algorithm v2, Mainnet doc). PI = price change from
the prior block's top to this block's top. For PI > 0 (symmetric for PI < 0):
  permanent:  h'_a = min(h_a + min(beta*PI, delta_max), h_max);   h'_b = TS - h'_a
  temporary:  g'_b = (h_b - h'_b) + 1/2*PI*1{PI >= cutoff};       g'_a = 0
  total:      f_a = h'_a + g'_a,   f_b = h'_b + g'_b   ==>   f_a = h'_a,  f_b = h_b + exc
So unlike v1 (which held f_a+f_b=TS, pure redistribution), v2 WIDENS the move-block spread
by the ask increment (the temporary term cancels the permanent bid cut), then relaxes to
TS — with the accumulated permanent skew — in quiet blocks. Mechanisms are toggled
independently so each can be isolated:
  permanent_skew_on  -> the accumulating directional skew (sum TS, capped at h_max)
  temporary_widening -> the (h_b - h'_b) hold-up that widens the move-block spread
  exception_on       -> the 1/2*PI big-move widening on the reverting side (|PI|>=cutoff)
All off  ==>  flat symmetric TS/2 fee.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from arena_eval.core.types import IncomingSwap, TradeInfo
from arena_eval.exact_simple_amm.simulator import Arbitrageur, StrategyAMM

_CAP = 0.99


@dataclass
class VolatileHookV2Strategy:
    """v2 top-of-block fee with independently toggleable mechanisms (see module docstring)."""
    ts_bps: float = 9.0
    beta: float = 0.5              # gain on PI for the permanent skew
    delta_max_bps: float = 2.0     # per-block cap on the permanent increment
    h_max_bps: float = 7.0         # level cap on the skewed permanent side (floors the other at TS-h_max)
    cutoff_bps: float = 10.0       # |PI| threshold for the big-move exception
    # mechanism toggles (all False => flat symmetric TS/2)
    permanent_skew_on: bool = False
    temporary_widening_on: bool = False
    exception_on: bool = False

    _ts: float = field(default=0.0, init=False)
    _dmax: float = field(default=0.0, init=False)
    _hmax: float = field(default=0.0, init=False)
    _cut: float = field(default=0.0, init=False)
    _ha: float = field(default=0.0, init=False)   # permanent ask/bid fees (fractions), sum = TS
    _hb: float = field(default=0.0, init=False)
    _fa: float = field(default=0.0, init=False)   # actual fees this block
    _fb: float = field(default=0.0, init=False)
    _prev_tob: float = field(default=0.0, init=False)
    _block: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._ts = self.ts_bps / 1e4
        self._dmax = self.delta_max_bps / 1e4
        self._hmax = self.h_max_bps / 1e4
        self._cut = self.cutoff_bps / 1e4
        self._ha = self._hb = self._fa = self._fb = self._ts / 2.0

    def after_initialize(self, ix: float, iy: float) -> tuple[float, float]:
        self._ha = self._hb = self._fa = self._fb = self._ts / 2.0
        self._prev_tob = (iy / ix) if ix > 0 else 0.0
        self._block = None
        return (self._fb, self._fa)

    def after_swap(self, trade: TradeInfo) -> tuple[float, float]:
        return (min(self._fb, _CAP), min(self._fa, _CAP))

    def before_swap(self, incoming: IncomingSwap) -> tuple[float, float]:
        rx, ry = incoming.reserve_x, incoming.reserve_y
        spot = (ry / rx) if rx > 0 else self._prev_tob
        block = int(incoming.block)
        if block != self._block:
            self._open_block(spot)
            self._block = block
            self._prev_tob = spot
        return (min(self._fb, _CAP), min(self._fa, _CAP))

    def _open_block(self, p_tob: float) -> None:
        ts, ha, hb = self._ts, self._ha, self._hb
        if not self.permanent_skew_on:
            self._ha = self._hb = self._fa = self._fb = ts / 2.0
            return
        pi = (p_tob - self._prev_tob) / self._prev_tob if self._prev_tob > 0 else 0.0
        # permanent skew toward the side of the recent move (sum stays TS, capped at h_max)
        if pi >= 0.0:
            ha_new = min(ha + min(self.beta * pi, self._dmax), self._hmax)
            hb_new = ts - ha_new
        else:
            hb_new = min(hb + min(self.beta * (-pi), self._dmax), self._hmax)
            ha_new = ts - hb_new
        ga = gb = 0.0
        if self.temporary_widening_on:   # hold the reverting side up: cancel its permanent cut
            if pi >= 0.0:
                gb += hb - hb_new
            else:
                ga += ha - ha_new
        if self.exception_on and abs(pi) >= self._cut:   # widen the reverting side by 1/2*|PI|
            if pi >= 0.0:
                gb += 0.5 * abs(pi)
            else:
                ga += 0.5 * abs(pi)
        self._ha, self._hb = ha_new, hb_new
        self._fa, self._fb = ha_new + ga, hb_new + gb


MECHANISMS = {
    "flat TS/2": dict(permanent_skew_on=False, temporary_widening_on=False, exception_on=False),
    "+ permanent skew": dict(permanent_skew_on=True, temporary_widening_on=False, exception_on=False),
    "+ temporary widening": dict(permanent_skew_on=True, temporary_widening_on=True, exception_on=False),
    "+ big-move exception": dict(permanent_skew_on=True, temporary_widening_on=True, exception_on=True),
}


def iid_lognormal_path(n_blocks: int, sigma_bps: float, seed: int, p0: float = 100.0) -> np.ndarray:
    """Martingale i.i.d. log-normal fair path: r_t = sigma*z - sigma^2/2, fair_t = fair_{t-1} e^{r_t}."""
    rng = np.random.default_rng(seed)
    sig = sigma_bps / 1e4
    r = sig * rng.standard_normal(n_blocks) - 0.5 * sig * sig
    return p0 * np.exp(np.cumsum(r))


def run_arb_only(strategy, fair: np.ndarray, pool_value_usd: float = 1_000_000.0, p0: float = 100.0):
    """One TOB arb per block against `fair`. Returns the list of arb trades:
    (block, trader_side, amount_x, amount_y, fee_bps_paid)."""
    pool = StrategyAMM("pool", strategy, pool_value_usd / p0, pool_value_usd)
    pool.initialize()
    arber = Arbitrageur()
    trades = []
    for t, fp in enumerate(fair):
        pool.before_swap(is_buy=pool.spot_price < fp, size=None, block=t)
        fee_bps = 1e4 * (pool.ask_fee if pool.spot_price < fp else pool.bid_fee)
        arb = arber.execute_arb(pool, float(fp), t)
        if arb is None:
            continue
        side = "buy_x" if arb.side == "sell" else "sell_x"
        trades.append((t, side, arb.amount_x, arb.amount_y, fee_bps))
    return trades


def markout_15s(fair: np.ndarray, trades, block_seconds: float = 12.0, markout_seconds: float = 15.0):
    """LP markout of each arb vs fair `markout_seconds` later (interpolated). Returns array ($)."""
    n = len(fair); off = markout_seconds / block_seconds
    out = []
    for (t, side, ax, ay, _fee) in trades:
        tf = t + off; i0 = int(np.floor(tf))
        if i0 + 1 >= n or ax <= 0:
            continue
        f15 = fair[i0] + (tf - i0) * (fair[i0 + 1] - fair[i0])
        out.append((ay - ax * f15) if side == "buy_x" else (ax * f15 - ay))
    return np.asarray(out)


def run_mechanism(mech_kwargs: dict, sigma_bps: float, seeds, ts_bps: float = 9.0,
                  n_blocks: int = 5000, pool_value_usd: float = 1_000_000.0, **strat_kwargs):
    """Aggregate over seeds: net LP markout $/seed (15s fair), arb count/seed, mean fee paid (bps)."""
    mk_tot, n_arb, fee_paid = [], [], []
    for s in seeds:
        fair = iid_lognormal_path(n_blocks, sigma_bps, seed=s)
        strat = VolatileHookV2Strategy(ts_bps=ts_bps, **mech_kwargs, **strat_kwargs)
        trades = run_arb_only(strat, fair, pool_value_usd=pool_value_usd)
        mk = markout_15s(fair, trades)
        mk_tot.append(float(mk.sum()))
        n_arb.append(len(trades))
        fee_paid.append(float(np.mean([tr[4] for tr in trades])) if trades else 0.0)
    return dict(lp_markout=float(np.mean(mk_tot)), lp_markout_std=float(np.std(mk_tot)),
                n_arb=float(np.mean(n_arb)), fee_paid_bps=float(np.mean(fee_paid)),
                lp_markout_per_seed=mk_tot)
