"""Submission for the V2 pmamm-sim competition (600+ markets, median scoring, YES/NO AMM).

Submitted as `VolTimeFastVol5`. Hosted overall median = +1.0307 (rank ~18).
Leader (Nick_Quartz) at +1.866. Gap remains -83pp.

V2 changes vs V1:
- Backtest expanded from 12 → 683 markets
- Scoring switched from MEAN to MEDIAN (overall) of per-market returns
- AMM switched from YES/USDC to YES/NO constant product
- Fees paid in YES (on sell_yes) or NO (on buy_yes); winning-token fees survive

What I tested on hosted (all pruned by top-2 rule unless marked WIN):
  - VolAwareKappa (V1 leader port): +0.9809  (starting baseline)
  - VolAwareMaxFloor20: +0.9898  (WIN — small fee floor causes useful skips)
  - HighBaseVickreySym (300bps base): worse
  - VolAwarePow01Base75 (pow=0.1 + 75bps base): worse
  - VolAwareNoAsym (no sqrt asym): worse
  - KitchenSinkV1 (inventory skew + guard + time): worse
  - HighKappaMaxFloor (κ=0.85): worse
  - TimeRampKappa (linear time ramp standalone): worse
  - FlippedSqrtAsym (preserve-winning-side direction): worse
  - DurationEnsemble (short-vs-long market branch): same as baseline
  - AdditiveSpotSkew: worse
  - MfAwareKappa (κ rises with mf): worse
  - PureHighKappa (κ=0.95 no asym): worse
  - HighKNoAsym (κ=0.85 no asym): worse
  - TimeInterpFixed (fee = 0.005 + 0.05*t): worse
  - VolTimeMaxFloor (time_coef=0.20 on top of baseline): WIN +0.9978
  - VolTimeFastVol (alpha=0.10): WIN +1.0202
  - VolTimeFastVol2 (alpha=0.20): WIN +1.0258
  - VolTimeFastVol3 (alpha=0.30): WIN +1.0271
  - VolTimeFastVol5 (alpha=0.50): WIN +1.0307  ← current best
  - VolTimeFastVol6 (alpha=0.60): pruned
  - VolTimeFastVol7 (alpha=0.70): pruned
  - VolTimeMult200 (vol_mult=200): pruned
  - VolSigmoidTime (sigmoid late-game ramp): pruned
  - VolTime20Floor30 (30bps floor): pruned
  - VolTime25/30/40MaxFloor (time_coef > 0.20): all worse
  - VolTimeExtremeAmp (extreme-spot multiplier): pruned
  - RegimeInitProb (initial-prob regime branch): pruned
  - VolTimeTradeCount (trade-count amplifier): pruned

What I learned:
  - Local 12-market data is unreliable: multiple +1pp local lifts → -0.03pp hosted
  - TIME-rising kappa lifts politics/price/pop_culture (lifts adverse-selection
    extraction on long markets)
  - FAST vol EWMA (alpha=0.5 vs 0.02) helps because short markets never warm up
    the slow EWMA — fast one captures volatility bursts in time
  - Higher kappa alone doesn't help (multiple tests of 0.67, 0.85, 0.95 all worse)
  - Removing sqrt asym hurts (my sqrt-asym IS helping; leaders' asym shape unknown)
  - Inventory-aware skew (linear in reserve drain) HURT — counter to AS theory
  - Hard guard at extreme spots (skip when s>0.9 or <0.1) HURT
  - Adding a base_fee (300bps tested) HURT badly

Where I lose to Nick_Quartz (per-market analysis):
  - 89-98% of all non-sports markets I lose to Nick
  - Biggest gaps: long-shot markets resolving against prior
    (p<0.10 outcome=1: gap +3.7pp;  p>0.90 outcome=1: gap +5.1pp)
  - Politics gap -0.78pp, price gap -0.88pp, sports gap -0.31pp, pop_culture -0.18pp
  - I'm slightly BETTER on `other` category (8 markets only)
  - Gap rises with num_trades (more trades = more compounding of leader advantage)
"""

import math

try:
    from pmamm_sim.types import FeeQuote, PendingTrade, TradeInfo
except ImportError:
    FeeQuote = None
    PendingTrade = None
    TradeInfo = None


STRATEGY_NAME = "VolTimeFastVol5"
AUTHOR = "xin"
DESCRIPTION = "Vol-aware kappa with fast EWMA (alpha=0.50) + time ramp + sqrt asym + 20bps max-floor."


class Strategy:
    def __init__(self):
        self.cap = 0.95
        self.floor = 0.002
        self.base_kappa = 0.50
        self.vol_mult = 100.0
        self.time_coef = 0.20
        self.alpha = 0.50
        self.ewma_vol = 0.0
        self.last_post_spot = None

    def _kappa(self, t):
        return min(0.95, self.base_kappa + self.vol_mult * self.ewma_vol + self.time_coef * t)

    def before_swap(self, p):
        s = max(0.001, min(0.999, p.current_spot))
        f = max(0.001, min(0.999, p.fair_price))
        t = max(0.0, min(1.0, p.normalized_time))
        kappa = self._kappa(t)
        if f > s:
            mf = (f - s) / f
            base = kappa * mf * math.sqrt(1.0 - s)
            fee = min(self.cap, max(self.floor, base))
            return FeeQuote(bid_fee=0.0, ask_fee=fee)
        elif f < s:
            mf = (s - f) / s
            base = kappa * mf * math.sqrt(s)
            fee = min(self.cap, max(self.floor, base))
            return FeeQuote(bid_fee=fee, ask_fee=0.0)
        return FeeQuote(bid_fee=0.0, ask_fee=0.0)

    def after_swap(self, t):
        if t is None:
            return
        if self.last_post_spot is not None:
            d = abs(t.post_spot - self.last_post_spot)
            self.ewma_vol = (1.0 - self.alpha) * self.ewma_vol + self.alpha * d
        self.last_post_spot = t.post_spot
