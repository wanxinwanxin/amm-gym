"""Flow-aware Guidestar — a unifying dynamic-fee strategy.

The §9/§10 backtests surface two opposing forces on a shallow pool's LP P&L:

  * Benign RETAIL flow earns the LP spread. The pool's share of it scales with
    √((1−fee)·x·y), so a *low* fee captures more — and that profit scales with the
    retail arrival rate. ⇒ frequent / small orders favour a LOW, quiet fee.
  * Toxic ARB / LVR flow loses the LP money. It scales with volatility and order
    size and is mitigated by a directional, defensive fee. ⇒ sparse / large orders
    favour AGGRESSIVE defense.

Guidestar's volatile hook already adapts its *directional* fee to VOLATILITY (the
realized per-block price move), but it does so unconditionally — its anti-backrun
spike fires on every volatile block regardless of how much benign retail is around,
so in a frequent/small regime that spike leaks onto the abundant retail and a quiet
flat fee beats it.

This strategy unifies the two regimes by BLENDING, per block, between the two
policies that each regime favours:

    quote = (1 − d)·flat_floor  +  d·guidestar_quote

where ``d`` ∈ (0,1) is a flow-derived "defensiveness":

    d = σ( a·ln(size/ref) − b·ln(arrival/ref) )
        d → 0   frequent &/or small flow  ⇒ ≈ a quiet flat floor (win the retail)
        d → 1   sparse &/or large flow    ⇒ ≈ full Guidestar defense (cut the LVR)

So in the busy/small corner the directional spike is blended away (no retail
leakage); in the sparse/large corner the full defense is on; and the middle ground
— high volatility *with* heavy retail — sits at a partial blend, defending on the
spike in proportion to how toxic the recent flow looks. ``d`` is built from lagged,
per-block EWMAs of the retail arrival count and order size, both read in
``before_swap`` (which sees the incoming order size); the Guidestar machinery runs
underneath at its standard parameters so its quote is always ready to blend in.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from arena_eval.core.types import IncomingSwap
from arena_eval.exact_simple_amm.guidestar_volatile import GuidestarVolatileStrategy


@dataclass
class FlowAwareGuidestarStrategy(GuidestarVolatileStrategy):
    """Guidestar volatile hook blended with a quiet flat floor by an online
    order-flow controller (see module docstring)."""

    # --- blend endpoints ----------------------------------------------------
    cheap_fee_bps: float = 3.5             # flat floor quoted in the frequent/small regime
                                           #   (= feeInit, i.e. Guidestar with its dynamics off)
    # --- regime signal ------------------------------------------------------
    ref_arrival: float = 98_676.0 / 216_000.0   # calibrated base arrival (orders/block)
    ref_size_usd: float = 275.94                # calibrated median order size ($)
    size_sensitivity: float = 1.0               # weight a on ln(size/ref)
    busy_sensitivity: float = 1.0               # weight b on ln(arrival/ref)
    ewma_halflife_blocks: float = 60.0          # EWMA memory for the flow stats

    # --- controller state (not constructor args) -----------------------------
    _ewma_count: float = field(default=0.0, init=False)
    _ewma_logsize: float = field(default=0.0, init=False)
    _ctl_block: int | None = field(default=None, init=False)
    _cur_count: int = field(default=0, init=False)
    _cur_logsize_sum: float = field(default=0.0, init=False)
    _ewma_decay: float = field(default=0.0, init=False)
    _d: float = field(default=0.5, init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._ewma_count = float(self.ref_arrival)
        self._ewma_logsize = math.log(max(self.ref_size_usd, 1e-9))
        self._ewma_decay = 0.5 ** (1.0 / max(self.ewma_halflife_blocks, 1e-9))
        self._d = self.defensiveness()

    # ---- controller ---------------------------------------------------------
    def defensiveness(self) -> float:
        """σ( a·ln(size/ref) − b·ln(arrival/ref) ) ∈ (0,1): high when large and/or sparse."""
        busy = self._ewma_count / max(self.ref_arrival, 1e-9)
        big = math.exp(self._ewma_logsize) / max(self.ref_size_usd, 1e-9)
        z = self.size_sensitivity * math.log(max(big, 1e-12)) - self.busy_sensitivity * math.log(max(busy, 1e-12))
        return 1.0 / (1.0 + math.exp(-z))

    # ---- strategy interface -------------------------------------------------
    def before_swap(self, incoming: IncomingSwap) -> tuple[float, float]:
        block = int(incoming.block)
        if self._ctl_block is None:
            self._ctl_block = block
        # On a new block, fold the PREVIOUS block's flow into the EWMAs (causal,
        # lagged one block) and refresh the blend weight.
        if block != self._ctl_block:
            c = self._cur_count
            self._ewma_count = self._ewma_decay * self._ewma_count + (1.0 - self._ewma_decay) * c
            if c > 0:
                self._ewma_logsize = self._ewma_decay * self._ewma_logsize + (1.0 - self._ewma_decay) * (self._cur_logsize_sum / c)
            self._d = self.defensiveness()
            self._ctl_block = block
            self._cur_count = 0
            self._cur_logsize_sum = 0.0
        if incoming.size is not None and incoming.size > 0.0:
            self._cur_count += 1
            self._cur_logsize_sum += math.log(incoming.size)
        # Guidestar machinery runs underneath (state keeps tracking the price path);
        # blend its quote with the quiet flat floor by the flow-derived weight.
        gs_bid, gs_ask = super().before_swap(incoming)
        cheap = self.cheap_fee_bps / 1e4
        d = self._d
        return ((1.0 - d) * cheap + d * gs_bid, (1.0 - d) * cheap + d * gs_ask)


@dataclass
class SizeAwareGuidestarStrategy(GuidestarVolatileStrategy):
    """Per-order size-conditioned Guidestar. The two regimes separate by ORDER SIZE
    (small ⇒ benign, large ⇒ toxic), and ``before_swap`` sees the incoming order
    size — so instead of a block-level average (FlowAware), condition the blend on
    the *current* order: small retail is quoted ~the quiet flat floor, large retail
    gets the full Guidestar defensive quote, and the arb leg (size None) is always
    defended. This separates benign from toxic flow within a block, so heavy small
    retail is never taxed by the spike that a large order or arb triggers.

        d(size) = clip( ln(size/size_lo) / ln(size_hi/size_lo), 0, 1 ),  arb ⇒ d = 1
        quote   = (1 − d)·flat_floor + d·guidestar_quote
    """

    cheap_fee_bps: float = 3.5       # flat floor quoted for the smallest orders
    size_lo_usd: float = 300.0       # at/below ⇒ ~flat floor (d→0); ~ the calibrated median
    size_hi_usd: float = 30_000.0    # at/above ⇒ ~full Guidestar defense (d→1)

    def before_swap(self, incoming: IncomingSwap) -> tuple[float, float]:
        gs_bid, gs_ask = super().before_swap(incoming)   # keep the machinery live
        cheap = self.cheap_fee_bps / 1e4
        if incoming.size is None:
            d = 1.0                                       # arb / unknown size ⇒ defend fully
        else:
            lo, hi = math.log(self.size_lo_usd), math.log(self.size_hi_usd)
            t = (math.log(max(incoming.size, 1e-9)) - lo) / (hi - lo)
            d = min(1.0, max(0.0, t))
        return ((1.0 - d) * cheap + d * gs_bid, (1.0 - d) * cheap + d * gs_ask)


@dataclass
class UnifiedGuidestarStrategy(GuidestarVolatileStrategy):
    """Unified controller combining the two lessons of §11:

      * RETAIL is charged by the incoming ORDER SIZE (small ⇒ quiet flat floor,
        large ⇒ full Guidestar defense) — separating benign small flow from toxic
        large flow within a block (the SizeAware idea).
      * The ARB leg's defense is FLOW-GATED by recent busyness: defend it when retail
        is sparse (LVR dominates), but let it realign the pool cheaply when retail is
        heavy — because the arb is also the mechanism that keeps the pool at fair, a
        public good for the abundant retail (the §11 decomposition finding).

        retail:  d(size) = clip( ln(size/size_lo)/ln(size_hi/size_lo), 0, 1 )
        arb:     d(busy) = 1 / (1 + (busy/arb_busy0)^arb_slope)   (busy = arrival/ref)
        quote  = (1−d)·flat_floor + d·guidestar_quote
    """

    cheap_fee_bps: float = 3.5         # flat floor (small retail, busy-regime arb)
    size_lo_usd: float = 300.0         # retail size ramp lower knee
    size_hi_usd: float = 30_000.0      # retail size ramp upper knee
    ref_arrival: float = 98_676.0 / 216_000.0
    arb_busy0: float = 5.0             # arrival multiple at which arb defense is half-on
    arb_slope: float = 2.0             # steepness of the arb gate in busyness
    ewma_halflife_blocks: float = 60.0

    _ewma_count: float = field(default=0.0, init=False)
    _ctl_block: int | None = field(default=None, init=False)
    _cur_count: int = field(default=0, init=False)
    _ewma_decay: float = field(default=0.0, init=False)
    _arb_d: float = field(default=1.0, init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._ewma_count = float(self.ref_arrival)
        self._ewma_decay = 0.5 ** (1.0 / max(self.ewma_halflife_blocks, 1e-9))
        self._arb_d = self._compute_arb_d()

    def _compute_arb_d(self) -> float:
        busy = self._ewma_count / max(self.ref_arrival, 1e-9)
        return 1.0 / (1.0 + (busy / max(self.arb_busy0, 1e-9)) ** self.arb_slope)

    def _retail_d(self, size: float) -> float:
        lo, hi = math.log(self.size_lo_usd), math.log(self.size_hi_usd)
        return min(1.0, max(0.0, (math.log(max(size, 1e-9)) - lo) / (hi - lo)))

    def before_swap(self, incoming: IncomingSwap) -> tuple[float, float]:
        block = int(incoming.block)
        if self._ctl_block is None:
            self._ctl_block = block
        if block != self._ctl_block:
            self._ewma_count = self._ewma_decay * self._ewma_count + (1.0 - self._ewma_decay) * self._cur_count
            self._arb_d = self._compute_arb_d()
            self._ctl_block = block
            self._cur_count = 0
        if incoming.size is not None and incoming.size > 0.0:
            self._cur_count += 1
        gs_bid, gs_ask = super().before_swap(incoming)
        cheap = self.cheap_fee_bps / 1e4
        d = self._arb_d if incoming.size is None else self._retail_d(incoming.size)
        return ((1.0 - d) * cheap + d * gs_bid, (1.0 - d) * cheap + d * gs_ask)
