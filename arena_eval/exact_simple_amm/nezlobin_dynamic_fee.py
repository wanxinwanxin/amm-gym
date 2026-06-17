"""Nezlobin / Volatile-Hook dynamic fee — faithful port of the mainnet algorithm.

Implements the spec in the "Volatile Hook Algorithm (Mainnet)" doc (Alex Nezlobin).
Target spread ``TS = f_a + f_b`` (e.g. 9 bps). Let ``P_TOB`` be the top-of-block AMM
price (mid before the first swap of the block) and ``P_BS`` the AMM price before the
current swap ``n``.

INTRA-BLOCK (swaps n>1). With the move measured as a fraction of the current price,
``m = (P_BS − P_TOB)/P_BS``:
  - price rose (P_BS > P_TOB): the REVERTING (bid/sell) side is taxed by half the
    move and the CONTINUATION (ask/buy) side gets a small bump capped at d:
        f_b(n) = f_b + ½·m
        f_a(n) = f_a + min(α·m, d)        # d ("maximum fading") ≈ 2 bps
  - price fell: mirror (reverting = ask, continuation = bid), with m = (P_TOB−P_BS)/P_BS.
The first swap of a block (the arb) pays the resting f_b, f_a with no surcharge.

TOP-OF-BLOCK resting fees. Normally f_a + f_b = TS, skewed toward the side of recent
price impact via an EMA of the per-block signed price impact ``PI``:
        f_a(t+1) = clip(f_a(t) + β·PI, 0, TS),   f_b = TS − f_a    (PI ≥ 0)
        f_b(t+1) = clip(f_b(t) + β·PI, 0, TS),   f_a = TS − f_b    (PI < 0, |PI| used)
EXCEPTION: if the *previous block's* single-block |price impact| exceeds a cutoff
(say 10 bps), widen the resting spread to TS + ½·|PI_prev| by adding ½·|PI_prev| to
the side opposite the impact (the reverting side), so it captures backruns that did
not happen in-block. If applied in block N it is bypassed in block N+1.

All signals (P_TOB, P_BS, per-block PI) are read from the pool's own reserves in
``before_swap`` — no fair-price input. The strategy relies on the simulator running
the arbitrageur at the top AND bottom of the block, iterated (config
``bottom_of_block_arb`` + ``arb_max_passes``), because the surcharge decays as the
reversing arb trades. Doc-unspecified parameters (α, β, EMA weight) are exposed with
documented defaults — the doc's skew is explicitly work-in-progress.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from arena_eval.core.types import IncomingSwap, TradeInfo

_CAP = 0.99


@dataclass
class NezlobinDynamicFeeStrategy:
    ts_bps: float = 9.0            # target spread TS = f_a + f_b at top of block
    alpha: float = 0.25            # continuation-side scale on the move (doc α; unspecified -> default)
    d_bps: float = 2.0             # continuation-side floor (doc d, "say 2bps")
    half: float = 0.5              # reverting-side share of the move (the "½")
    beta: float = 0.25             # skew gain on EMA price impact (doc β; unspecified -> default)
    ema_weight: float = 0.5        # EMA weight on the newest block's PI (heavy-recent)
    cutoff_bps: float = 10.0       # big-PI exception threshold on the previous block's |PI|
    surcharge_on: bool = True      # intra-block ½·move / min(α·move,d) surcharge (set False to isolate the skew)
    exception_on: bool = True      # big-PI spread-widening exception

    # internal state (fractions)
    _ts: float = field(default=0.0, init=False)
    _d: float = field(default=0.0, init=False)
    _cutoff: float = field(default=0.0, init=False)
    _fa_skew: float = field(default=0.0, init=False)   # recursive EMA-skew fees (sum = TS)
    _fb_skew: float = field(default=0.0, init=False)
    _fa: float = field(default=0.0, init=False)         # resting fees for the current block (skew + exception)
    _fb: float = field(default=0.0, init=False)
    _p0: float = field(default=0.0, init=False)         # P_TOB (block-open mid)
    _block: int | None = field(default=None, init=False)
    _prev_open: float | None = field(default=None, init=False)
    _pi_ema: float = field(default=0.0, init=False)
    _bypass_next: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._ts = self.ts_bps / 1e4
        self._d = self.d_bps / 1e4
        self._cutoff = self.cutoff_bps / 1e4
        self._reset(self._ts / 2.0, self._ts / 2.0)

    def _reset(self, fa: float, fb: float) -> None:
        self._fa_skew = self._fa = fa
        self._fb_skew = self._fb = fb
        self._pi_ema = 0.0
        self._bypass_next = False
        self._block = None

    # ---- strategy interface -------------------------------------------------
    def after_initialize(self, initial_x: float, initial_y: float) -> tuple[float, float]:
        mid = (initial_y / initial_x) if initial_x > 0.0 else 0.0
        self._reset(self._ts / 2.0, self._ts / 2.0)
        self._p0 = mid
        self._prev_open = mid
        return (self._fb, self._fa)

    def after_swap(self, trade: TradeInfo) -> tuple[float, float]:
        return (min(self._fb, _CAP), min(self._fa, _CAP))   # echo resting; before_swap drives each swap

    def before_swap(self, incoming: IncomingSwap) -> tuple[float, float]:
        rx, ry = incoming.reserve_x, incoming.reserve_y
        spot = (ry / rx) if rx > 0.0 else self._p0
        block = int(incoming.block)
        if block != self._block:
            self._open_block(spot, block)
            return (min(self._fb, _CAP), min(self._fa, _CAP))   # first swap of block: resting, no surcharge
        fb, fa = self._fb, self._fa
        p0 = self._p0
        if self.surcharge_on and p0 > 0.0 and spot != p0:
            if spot > p0:                                       # rose: revert = bid, continuation = ask
                m = (spot - p0) / spot
                fb = fb + self.half * m
                fa = fa + min(self.alpha * m, self._d)          # continuation: small, capped at d ("max fading")
            else:                                               # fell: revert = ask, continuation = bid
                m = (p0 - spot) / spot
                fa = fa + self.half * m
                fb = fb + min(self.alpha * m, self._d)
        return (min(fb, _CAP), min(fa, _CAP))

    # ---- internals ----------------------------------------------------------
    def _open_block(self, m_open: float, block: int) -> None:
        # 1. signed price impact of the just-finished block (block-open to block-open)
        if self._prev_open is not None and self._prev_open > 0.0:
            pi1 = (m_open - self._prev_open) / self._prev_open
        else:
            pi1 = 0.0
        # 2. EMA of price impact (heavy weight on the newest block)
        self._pi_ema = self.ema_weight * pi1 + (1.0 - self.ema_weight) * self._pi_ema
        # 3. recursive resting skew toward the side of recent impact (sum = TS)
        pi = self._pi_ema
        if pi >= 0.0:
            fa = min(max(self._fa_skew + self.beta * pi, 0.0), self._ts)
            fb = self._ts - fa
        else:
            fb = min(max(self._fb_skew + self.beta * (-pi), 0.0), self._ts)
            fa = self._ts - fb
        self._fa_skew, self._fb_skew = fa, fb
        # 4. big-PI exception: widen the spread to capture an un-backrun large move,
        #    loading the extra onto the reverting (opposite-to-impact) side. Bypass
        #    the block after one is applied.
        if self.exception_on and abs(pi1) > self._cutoff and not self._bypass_next:
            extra = self.half * abs(pi1)
            if pi1 > 0.0:        # up impact -> reverting side is the bid
                fb = fb + extra
            else:                # down impact -> reverting side is the ask
                fa = fa + extra
            self._bypass_next = True
        else:
            self._bypass_next = False
        self._fa, self._fb = fa, fb
        self._p0 = m_open
        self._prev_open = m_open
        self._block = block
