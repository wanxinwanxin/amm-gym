"""Nezlobin dynamic-fee strategy (Alex Nezlobin's intra-block directional fee).

Proposed in the WETH/USDC strategy thread. Two layers:

1. **Intra-block directional surcharge** (the core idea). Let ``p0`` be the pool mid
   at the top of the block (before the first swap) and ``p1`` the pool mid before
   the current swap. If the pool has moved UP within the block (``p1 > p0``), add
   ``(p1 − p0)/2`` to the BID (sell-into-pool) fee and leave the ASK fee; symmetric
   if it moved down. This taxes the *reversing* trade by half the intra-block move,
   so it (i) largely defangs sandwiches — the backrun leg pays it — and (ii) lets a
   same-block backrun/arb still execute while handing the LP ≥½ of the move. The
   surcharge is applied as a fractional fee ``(p1 − p0)/(2·p0)`` on top of the resting
   fee. ``p0`` is the mid *before the top-of-block arb*, so the inter-block move is
   folded into the surcharge (the top arb itself is always on the un-taxed side).

2. **Resting-fee skew** (a second-order refinement). The resting fees sum to
   ``base_total`` (9 bps). Normally split symmetrically (4.5/4.5). When the previous
   block's realized move took the new block-open mid *outside the previous block's
   resting bid-ask range*, skew the resting split to (3, 6) bps in the move's
   direction — a big UP move makes the ASK fee the heavy one (6 bps), a big DOWN move
   makes the BID fee heavy. The trigger is self-referential (the previous block's own
   quotes), so it auto-scales with the fee and needs no absolute threshold.

The driving signal (block-open mid, previous mid, previous range) is read from the
pool's own reserves in ``before_swap`` — no external/fair-price input. This strategy
relies on the simulator running the arbitrageur at BOTH the top and the bottom of
the block (config ``bottom_of_block_arb``) and iterating it (config
``arb_max_passes``), because the surcharge decays as the reversing arb trades.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from arena_eval.core.types import IncomingSwap, TradeInfo

_CAP = 0.99  # hard cap on either side's total fee (fraction)


@dataclass
class NezlobinDynamicFeeStrategy:
    base_total_bps: float = 9.0     # fa + fb resting (sum)
    skew_lo_bps: float = 3.0        # light side when skewed
    skew_hi_bps: float = 6.0        # heavy side when skewed
    half_decay: float = 0.5         # the "/2" in (p1-p0)/2 — fraction of the move kept

    # resting state (fractions): _fb = bid fee, _fa = ask fee
    _fb: float = field(default=0.0, init=False)
    _fa: float = field(default=0.0, init=False)
    _p0: float = field(default=0.0, init=False)         # block-open mid (surcharge reference)
    _block: int | None = field(default=None, init=False)
    _prev_open: float | None = field(default=None, init=False)
    _prev_fb: float = field(default=0.0, init=False)
    _prev_fa: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self._half = (self.base_total_bps / 2.0) / 1e4
        self._lo = self.skew_lo_bps / 1e4
        self._hi = self.skew_hi_bps / 1e4
        self._fb = self._fa = self._half

    # ---- strategy interface -------------------------------------------------
    def after_initialize(self, initial_x: float, initial_y: float) -> tuple[float, float]:
        mid = (initial_y / initial_x) if initial_x > 0.0 else 0.0
        self._fb = self._fa = self._half
        self._p0 = mid
        self._prev_open = mid
        self._prev_fb = self._prev_fa = self._half
        self._block = None
        return (self._fb, self._fa)

    def after_swap(self, trade: TradeInfo) -> tuple[float, float]:
        # before_swap drives fees before every swap; echo the resting quote here.
        return (min(self._fb, _CAP), min(self._fa, _CAP))

    def before_swap(self, incoming: IncomingSwap) -> tuple[float, float]:
        rx, ry = incoming.reserve_x, incoming.reserve_y
        spot = (ry / rx) if rx > 0.0 else self._p0
        block = int(incoming.block)
        if self._block is None or block != self._block:
            self._open_block(spot, block)
        fb, fa = self._fb, self._fa
        p0 = self._p0
        if p0 > 0.0:
            if spot > p0:
                fb = fb + self.half_decay * (spot - p0) / p0       # tax the reversing SELL
            elif spot < p0:
                fa = fa + self.half_decay * (p0 - spot) / p0       # tax the reversing BUY
        return (min(fb, _CAP), min(fa, _CAP))

    # ---- internals ----------------------------------------------------------
    def _open_block(self, m_open: float, block: int) -> None:
        """First swap of a new block: set the resting skew from the previous block's
        move (block-open mid vs the previous block's resting bid-ask range) and reset
        the surcharge reference p0 to this block's opening mid."""
        if self._prev_open is not None and self._prev_open > 0.0:
            ask_lim = self._prev_open * (1.0 + self._prev_fa)
            bid_lim = self._prev_open * (1.0 - self._prev_fb)
            if m_open > ask_lim:          # big up move -> heavy ASK
                self._fb, self._fa = self._lo, self._hi
            elif m_open < bid_lim:        # big down move -> heavy BID
                self._fb, self._fa = self._hi, self._lo
            else:
                self._fb, self._fa = self._half, self._half
        else:
            self._fb, self._fa = self._half, self._half
        self._p0 = m_open
        self._prev_open = m_open
        self._prev_fb, self._prev_fa = self._fb, self._fa
        self._block = block
