"""Guidestar volatile-pair hook, ported to the exact-simple-AMM strategy interface.

Faithful Python port of Uniswap's ``Guidestar4Volatile.beforeSwap``
(github.com/Uniswap/guidestar-unichain, src/Guidestar4Volatile.sol) — a
directional, time-decaying, sandwich-aware dynamic fee. Each side (buy/sell of the
base token x) carries a *permanent* (sticky) and *transitory* (fast-decaying)
component. At the first swap of a new block the fee state is recomputed from the
*previous* block's net price move (the realized order-flow impact): the side in the
direction of the move gets a small permanent skew, the opposite side of the book is
held in place with a transitory fee spike ≈ the price impact (anti-backrun), and
both components decay toward a ``2*feeInit`` floor. Later swaps in the same block
take only a sandwich-protection floor (no re-skew).

Design notes for this port:
- Driven via ``before_swap`` (the v4 ``beforeSwap`` analog). The simulator calls it
  before the arb (the block's first swap) and before each routed retail order.
- 1 simulator step == 1 block. The heavy update fires once per block, gated on
  ``block > last_block``; subsequent same-block swaps hit the sandwich branch.
- "Price impact" is read from the submission pool's own spot move since top-of-block
  (``reserve_y/reserve_x``), exactly like the contract reads ``sqrtPriceX96``.
- Fees are tracked internally in MILLIONTHS (1e6 = 100%, 1 bp = 100); ``before_swap``
  returns fractions. The contract's buy fee -> the AMM ``ask_fee`` (a retail buy of x
  pays ask); the sell fee -> the AMM ``bid_fee``.
- Labelling of buy/sell follows the realized price-move direction (matches the spec
  prose and the validated dynamics); the contract's packed ``zeroForOne`` storage
  bookkeeping is rendered in that economic form.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from arena_eval.core.types import IncomingSwap, TradeInfo

_ONE_M = 1_000_000.0
_CAP = 990_000.0  # contract hard cap on a side's total fee


@dataclass
class GuidestarVolatileStrategy:
    """Guidestar volatile dynamic-fee strategy. Parameters in friendly units;
    see the contract's ``HookParams`` for the on-chain equivalents."""

    # Defaults = the mainnet Guidestar4 VOLATILE deployment
    # (guidestar-hooks/script/DeployGuidestar4.s.sol::defaultHookParams):
    #   feeInit=350, alpha=90*100, maxPermGrowth=1<<4, transDecline=(10<<24)/100,
    #   k=(99<<24)/100, logKPerm=(10050*1e12)>>40 (=-ln(0.99)), maxPriceImprovement=1000.
    fee_init_bps: float = 3.5              # minimum fee per side (bps); floor total = 2*feeInit
    alpha: float = 9000.0                  # contract alpha: perm increase = alpha*impact*100 (millionths)
    max_perm_growth: float = 16.0          # cap: max perm increase/block = (buyP+sellP)*max_perm_growth/16
    trans_decline: float = 0.10            # transitory linear decline per block (fraction in [0,1])
    k_perm: float = 0.99                   # permanent decay factor/block toward floor, gaps <=4 blocks (in (0,1))
    logk_perm: float | None = None         # per-block perm-decay exponent for gaps >4 blocks (factor=exp(-logk_perm)).
                                           #   None => derive from k_perm (self-consistent single rate). The contract
                                           #   stores k and logKPerm separately and they can differ slightly.
    max_price_improvement_bps: float = 10.0  # sandwich-protection slack (bps)
    enable_sandwich_protection: bool = True
    # On-chain HookParams -> these friendly units (for ingesting a real deployment):
    #   fee_init_bps  = feeInit / 100                  (feeInit is millionths)
    #   alpha         = alpha                          (same scalar)
    #   max_perm_growth = maxPermGrowth                (same integer; cap uses >>4 == /16)
    #   trans_decline = transDecline / 2**24           (X24 fraction -> [0,1] per block)
    #   k_perm        = k / 2**24                       (X24 fixed point -> factor in (0,1))
    #   logk_perm     = logKPerm * 2**40 / 1e18         (wad exponent the contract feeds expWad)
    #   max_price_improvement_bps = maxPriceImprovement / 100

    # internal state (millionths), not constructor args
    _feeInit: float = field(default=0.0, init=False)
    _buyPerm: float = field(default=0.0, init=False)
    _sellPerm: float = field(default=0.0, init=False)
    _buyTrans: float = field(default=0.0, init=False)
    _sellTrans: float = field(default=0.0, init=False)
    _top: float | None = field(default=None, init=False)        # top-of-block spot price
    _last_block: int | None = field(default=None, init=False)
    _logk: float = field(default=0.0, init=False)
    _maxPI: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self._feeInit = float(self.fee_init_bps) * 100.0
        self._buyPerm = self._sellPerm = self._feeInit
        self._buyTrans = self._sellTrans = 0.0
        if self.logk_perm is not None:
            self._logk = float(self.logk_perm)
        elif 0.0 < self.k_perm < 1.0:
            self._logk = -math.log(self.k_perm)
        else:
            self._logk = 0.0
        self._maxPI = float(self.max_price_improvement_bps) * 100.0

    # ---- strategy interface -------------------------------------------------
    def after_initialize(self, initial_x: float, initial_y: float) -> tuple[float, float]:
        self._top = (initial_y / initial_x) if initial_x > 0.0 else None
        self._last_block = None
        return (self._sell_fee() / _ONE_M, self._buy_fee() / _ONE_M)

    def after_swap(self, trade: TradeInfo) -> tuple[float, float]:
        # before_swap drives the fees; after_swap just echoes the current quote.
        return (self._sell_fee() / _ONE_M, self._buy_fee() / _ONE_M)

    def before_swap(self, incoming: IncomingSwap) -> tuple[float, float]:
        spot = (incoming.reserve_y / incoming.reserve_x) if incoming.reserve_x > 0.0 else self._top
        if spot is None or spot <= 0.0:
            return (self._sell_fee() / _ONE_M, self._buy_fee() / _ONE_M)
        if self._top is None:
            self._top = spot
        block = int(incoming.block)

        minimal_fee = 0.0
        if self._last_block is None or block > self._last_block:
            blocks_passed = 1 if self._last_block is None else (block - self._last_block)
            self._block_update(spot, blocks_passed)
            self._top = spot
            self._last_block = block
        elif self.enable_sandwich_protection:
            minimal_fee = self._sandwich_floor(spot, incoming.is_buy)

        buy, sell = self._buy_fee(), self._sell_fee()
        if incoming.is_buy:
            ask, bid = max(buy, minimal_fee), sell
        else:
            bid, ask = max(sell, minimal_fee), buy
        return (min(bid, _CAP) / _ONE_M, min(ask, _CAP) / _ONE_M)

    # ---- internals ----------------------------------------------------------
    def _buy_fee(self) -> float:
        return self._buyPerm + self._buyTrans

    def _sell_fee(self) -> float:
        return self._sellPerm + self._sellTrans

    def _block_update(self, spot: float, blocks_passed: int) -> None:
        """Port of the ``blocksPassed > 0`` branch of beforeSwap."""
        top = self._top
        price_up = top is not None and spot > top
        if top is None or top <= 0.0:
            ratio = 1.0
        else:
            ratio = (top / spot) if price_up else (spot / top)   # = priceChange / 2**96, in (0, 1]
        impact = max(0.0, 1.0 - ratio)

        if price_up:
            thisP, otherP, thisT, otherT = self._buyPerm, self._sellPerm, self._buyTrans, self._sellTrans
        else:
            thisP, otherP, thisT, otherT = self._sellPerm, self._buyPerm, self._sellTrans, self._buyTrans

        inc = self.alpha * impact * 100.0                                  # step 1
        inc = min(inc, (thisP + otherP) * self.max_perm_growth / 16.0)     # cap by maxPermGrowth
        other_fee_new = _ONE_M - ratio * (_ONE_M - (otherP + otherT))      # step 5 (other side total)
        inc = min(inc, _CAP - (thisP + thisT))
        thisP += inc                                                       # step 2: skew the moved side
        dec = min(otherP, inc)
        otherP -= dec                                                      # step 4
        otherT = other_fee_new - otherP                                    # step 5
        if other_fee_new > _CAP:
            if otherT >= 10_000.0:
                otherT -= 10_000.0
            else:
                otherP -= (10_000.0 - otherT)
                otherT = 0.0

        f = max(0.0, 1.0 - self.trans_decline * blocks_passed)             # transitory: linear decay
        thisT *= f
        otherT *= f

        perm = thisP + otherP                                              # permanent: exp decay -> floor
        floor = 2.0 * self._feeInit
        if perm >= floor:
            f24 = (self.k_perm ** blocks_passed) if blocks_passed <= 4 else math.exp(-self._logk * blocks_passed)
            this_lb = floor * thisP / perm if perm > 0.0 else floor / 2.0
            thisP = this_lb + f24 * (thisP - this_lb)
            other_lb = floor - this_lb
            otherP = other_lb + f24 * (otherP - other_lb)
        else:
            thisP = min(self._feeInit, _CAP)
            otherP = min(self._feeInit, _CAP)
            thisT = min(thisT, _CAP - thisP)
            otherT = min(otherT, _CAP - otherP)

        if price_up:
            self._buyPerm, self._sellPerm, self._buyTrans, self._sellTrans = thisP, otherP, thisT, otherT
        else:
            self._sellPerm, self._buyPerm, self._sellTrans, self._buyTrans = thisP, otherP, thisT, otherT

    def _sandwich_floor(self, spot: float, is_buy: bool) -> float:
        """Port of the ``blocksPassed == 0`` branch: a minimum fee for a swap in the
        same direction as the intra-block move so far, so it cannot execute better
        than the top-of-block book (back-run protection). Rarely binds at our
        arrival rate (needs a 2nd+ same-direction swap in one step)."""
        top = self._top
        if top is None or top <= 0.0:
            return 0.0
        price_up = spot > top
        user_sells = not is_buy                       # zeroForOne == selling base x
        if user_sells != price_up:                    # contract: userSellsZeroForOne == priceIncreased
            return 0.0
        ratio = (top / spot) if price_up else (spot / top)
        if price_up:
            this_fee = self._buy_fee()
            other_fee = self._sell_fee()
        else:
            this_fee = self._sell_fee()
            other_fee = self._buy_fee()
        lower = -this_fee * _ONE_M / (_ONE_M - this_fee) if this_fee < _ONE_M else -1e18
        f = max(other_fee - self._maxPI, lower)
        ff = ratio * (_ONE_M - f)
        if ff > _ONE_M:
            ff = _ONE_M
        return max(0.0, _ONE_M - ff)
