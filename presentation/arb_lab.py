"""Experiment runners + plotters for the arb-only LVR lab (presentation/arb_only_lvr_lab.ipynb).

Thin layer over arena_eval.exact_simple_amm.arb_only_lab: runs the mechanism matrix and a
couple of sweeps on PAIRED seeds (same fair path across mechanisms) and renders the
figures. Metric throughout: net LP markout vs fair 15s later, $ per 5000-block run on a
$1M pool (more negative = more LVR). Lower |markout| = better LVR protection.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena_eval.core.types import IncomingSwap, TradeInfo  # noqa: E402
from arena_eval.exact_simple_amm.arb_only_lab import (  # noqa: E402
    MECHANISMS, VolatileHookV2Strategy, iid_lognormal_path, markout_15s, run_arb_only)
from arena_eval.exact_simple_amm.simulator import Arbitrageur, StrategyAMM  # noqa: E402

SEEDS = tuple(range(40, 80))
SIGMA_BPS = 5.0
TS_BPS = 9.0
DELTA_MAX_BPS = 6.0
_GREEN, _RED, _BLUE, _PURPLE = "#16a085", "#c0392b", "#2980b9", "#8e44ad"


def _bare(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, axis="y")
    ax.tick_params(labelsize=9)


# ----------------------------------------------------------------------------- matrix
def run_matrix(seeds=SEEDS, sigma_bps=SIGMA_BPS, ts_bps=TS_BPS, delta_max_bps=DELTA_MAX_BPS, n_blocks=5000):
    """Paired per-mechanism total LP markout. Returns {name: np.array over seeds}."""
    per = {n: [] for n in MECHANISMS}
    for s in seeds:
        fair = iid_lognormal_path(n_blocks, sigma_bps, seed=s)
        for n, kw in MECHANISMS.items():
            tr = run_arb_only(VolatileHookV2Strategy(ts_bps=ts_bps, delta_max_bps=delta_max_bps, **kw), fair)
            per[n].append(markout_15s(fair, tr).sum())
    return {n: np.asarray(v) for n, v in per.items()}


def plot_matrix(per: dict):
    """(A) cumulative LP markout per mechanism; (B) each mechanism's marginal effect."""
    names = list(per)
    means = np.array([per[n].mean() for n in names])
    flat = means[0]
    marg = np.diff(means)                                  # marginal effect of perm/temp/exc
    marg_se = np.array([(per[names[i + 1]] - per[names[i]]).std() / np.sqrt(len(per[names[0]]))
                        for i in range(len(names) - 1)])
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.9))
    # (A) cumulative
    cols = [_BLUE, _GREEN, _GREEN, _GREEN]
    ax[0].bar(range(len(names)), means, 0.6, color=cols, edgecolor="black", lw=0.5)
    ax[0].axhline(flat, color="grey", ls=":", lw=1.2)
    for i, v in enumerate(means):
        ax[0].annotate(f"{v:.0f}", (i, v), ha="center", va="top", fontsize=9, xytext=(0, -3),
                       textcoords="offset points")
    ax[0].set_xticks(range(len(names))); ax[0].set_xticklabels(names, fontsize=8.5, rotation=12, ha="right")
    ax[0].set_ylabel("net LP markout  ($/run, 15s fair)")
    lo = min(means.min(), flat) - 12; hi = max(means.max(), flat) + 8
    ax[0].set_ylim(lo, hi)   # zoom: the ~$45 spread is tiny against the −700 level
    ax[0].set_title("(A) LVR vs cumulative mechanisms  (y-axis zoomed)\n(less negative = better LVR protection)",
                    fontsize=10.5, fontweight="bold")
    _bare(ax[0])
    # (B) marginal
    labels = ["permanent\nskew", "temporary\nwidening", "big-move\nexception"]
    cols = [_GREEN if m > 0 else _RED for m in marg]
    ax[1].bar(range(len(marg)), marg, 0.6, color=cols, edgecolor="black", lw=0.5,
              yerr=marg_se, capsize=4)
    ax[1].axhline(0, color="grey", lw=0.9)
    for i, m in enumerate(marg):
        ax[1].annotate(f"{m:+.1f}", (i, m), ha="center", va="bottom" if m > 0 else "top",
                       fontsize=10, fontweight="bold", xytext=(0, 4 if m > 0 else -4),
                       textcoords="offset points")
    ax[1].set_xticks(range(len(marg))); ax[1].set_xticklabels(labels, fontsize=9)
    ax[1].set_ylabel("marginal Δ LP markout  ($/run)")
    ax[1].set_title("(B) isolated marginal effect of each mechanism\n(green = protects LVR, red = hurts)",
                    fontsize=10.5, fontweight="bold")
    _bare(ax[1])
    fig.suptitle(f"Arb-only LVR lab — v2 mechanisms, isolated  (σ={SIGMA_BPS:.0f}bp/block, TS={TS_BPS:.0f}bp, "
                 f"{len(per[names[0]])} paired seeds)", fontweight="bold", fontsize=11.5)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- h_max sweep
def run_hmax_sweep(hmaxes, seeds=SEEDS, sigma_bps=SIGMA_BPS, ts_bps=TS_BPS, delta_max_bps=DELTA_MAX_BPS, n_blocks=5000):
    out = []
    for hm in hmaxes:
        mk, na = [], []
        for s in seeds:
            fair = iid_lognormal_path(n_blocks, sigma_bps, seed=s)
            tr = run_arb_only(VolatileHookV2Strategy(ts_bps=ts_bps, delta_max_bps=delta_max_bps,
                                                     permanent_skew_on=True, h_max_bps=hm), fair)
            mk.append(markout_15s(fair, tr).sum()); na.append(len(tr))
        out.append(dict(h_max=hm, markout=float(np.mean(mk)), n_arb=float(np.mean(na))))
    return out


def plot_hmax_sweep(rows):
    hm = np.array([r["h_max"] for r in rows])
    mk = np.array([r["markout"] for r in rows])
    d = mk - mk[0]
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.plot(hm, d, "o-", color=_GREEN, lw=2.4, ms=6)
    ax.axhline(0, color="grey", lw=0.9)
    ax.annotate("more skew → more LVR protection,\nmonotonically (no self-defeating collapse here)",
                (hm[len(hm) // 2], d[len(hm) // 2]), fontsize=9, ha="left", xytext=(8, -34),
                textcoords="offset points", color="#0e6b5a")
    ax.annotate("h_max = TS/2\n(no skew = flat)", (hm[0], d[0]), fontsize=8.5, xytext=(8, 8),
                textcoords="offset points")
    ax.annotate("h_max = TS  (bid floor = 0, v1-style full skew)", (hm[-1], d[-1]), fontsize=8.5,
                ha="right", xytext=(-8, -16), textcoords="offset points")
    ax.set_ylim(top=d.max() * 1.18)
    ax.set_xlabel("permanent-skew level cap  h_max  (bp)   [bid floors at TS − h_max]")
    ax.set_ylabel("Δ LP markout vs flat  ($/run)")
    ax.set_title("Permanent skew (isolated): LVR protection grows monotonically with the skew cap\n"
                 "— in pure arb-only there is no self-defeating collapse (contrast the retail world, §13)",
                 fontsize=10, fontweight="bold")
    _bare(ax)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- exception vs vol
def run_vol_sweep(sigmas, seeds=SEEDS, ts_bps=TS_BPS, delta_max_bps=DELTA_MAX_BPS, n_blocks=5000):
    base_kw = dict(permanent_skew_on=True, temporary_widening_on=True, exception_on=False)
    exc_kw = dict(permanent_skew_on=True, temporary_widening_on=True, exception_on=True)
    out = []
    for sig in sigmas:
        db, de, pf = [], [], []
        for s in seeds:
            fair = iid_lognormal_path(n_blocks, sig, seed=s)
            db.append(markout_15s(fair, run_arb_only(
                VolatileHookV2Strategy(ts_bps=ts_bps, delta_max_bps=delta_max_bps, **base_kw), fair)).sum())
            de.append(markout_15s(fair, run_arb_only(
                VolatileHookV2Strategy(ts_bps=ts_bps, delta_max_bps=delta_max_bps, **exc_kw), fair)).sum())
            pi = np.diff(fair) / fair[:-1]; pf.append(np.mean(np.abs(pi) >= 10e-4))
        db, de = np.array(db), np.array(de)
        out.append(dict(sigma=sig, d_exc=float((de - db).mean()),
                        d_exc_se=float((de - db).std() / np.sqrt(len(db))), p_fire=float(np.mean(pf))))
    return out


def plot_exception_vol(rows):
    sig = np.array([r["sigma"] for r in rows])
    de = np.array([r["d_exc"] for r in rows])
    se = np.array([r["d_exc_se"] for r in rows])
    cols = [_GREEN if x > 0 else _RED for x in de]
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.bar(sig, de, width=0.9, color=cols, edgecolor="black", lw=0.5, yerr=se, capsize=3)
    ax.axhline(0, color="grey", lw=0.9)
    ax.axvspan(7, 10, color="#f1c40f", alpha=0.18, zorder=0)
    ax.annotate("realistic WETH\n12s vol (~7–10bp):\nexception ≈ neutral/slightly hurts", (8.5, ax.get_ylim()[0] * 0.5),
                fontsize=8.5, ha="center", color="#7d6608")
    ax.annotate("a TAIL mechanism:\nbig win only when moves\nare genuinely large", (sig[-1], de[-1]),
                fontsize=8.5, ha="right", va="top", xytext=(-6, -6), textcoords="offset points", color="#0e6b5a")
    ax.set_xlabel("per-block volatility σ  (bp)")
    ax.set_ylabel("marginal Δ LP markout from the exception  ($/run)")
    ax.set_title("Big-move exception (isolated): a tail mechanism — it only protects LVR when\n"
                 "moves are genuinely large; at realistic vol its 10bp trigger fires on ordinary moves",
                 fontsize=10, fontweight="bold")
    _bare(ax)
    fig.tight_layout()
    return fig


# --------------------------------------------------- exception: why it can hurt (decomposition)
def run_exception_decomp(sigma_bps=10.0, seeds=SEEDS, ts_bps=TS_BPS, delta_max_bps=DELTA_MAX_BPS, n_blocks=5000):
    """Per-block contemporaneous-LVR effect of the exception (vs permanent+temporary), split into
    the blocks where it FIRES (direct effect of the wider bid) vs all OTHER blocks (the knock-on:
    corrections it deters get displaced to later blocks and, by LVR's convexity, cost more)."""
    base = dict(permanent_skew_on=True, temporary_widening_on=True, exception_on=False)
    full = dict(permanent_skew_on=True, temporary_widening_on=True, exception_on=True)

    def perblock(kw, fair):
        pool = StrategyAMM("pool", VolatileHookV2Strategy(ts_bps=ts_bps, delta_max_bps=delta_max_bps, **kw),
                           1_000_000.0 / 100.0, 1_000_000.0)
        pool.initialize(); arber = Arbitrageur()
        mk = np.zeros(len(fair)); fired = np.zeros(len(fair), bool); prev = pool.spot_price
        for t, fp in enumerate(fair):
            tob = pool.spot_price
            pi = (tob - prev) / prev if prev > 0 else 0.0
            fired[t] = abs(pi) >= 10e-4                                                 # 10bp cutoff
            pool.before_swap(is_buy=pool.spot_price < fp, size=None, block=t)
            arb = arber.execute_arb(pool, float(fp), t)
            if arb is not None:
                side = "buy_x" if arb.side == "sell" else "sell_x"
                mk[t] = (arb.amount_y - arb.amount_x * fp) if side == "buy_x" else (arb.amount_x * fp - arb.amount_y)
            prev = tob
        return mk, fired

    fire, nofire, total = [], [], []
    for s in seeds:
        fair = iid_lognormal_path(n_blocks, sigma_bps, seed=s)
        mb, fb = perblock(base, fair); mf, _ = perblock(full, fair)
        d = mf - mb
        fire.append(d[fb].sum()); nofire.append(d[~fb].sum()); total.append(d.sum())
    return dict(sigma=sigma_bps, direct=float(np.mean(fire)), knock_on=float(np.mean(nofire)),
                net=float(np.mean(total)))


def plot_exception_decomp(d: dict):
    """Direct (on firing blocks) vs knock-on (displaced corrections) vs net."""
    vals = [d["direct"], d["knock_on"], d["net"]]
    labels = ["direct\n(blocks it fires:\nwider bid)", "knock-on\n(later blocks:\ndisplaced corrections)", "net"]
    cols = [_GREEN, _RED, _BLUE]
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.bar(range(3), vals, 0.6, color=cols, edgecolor="black", lw=0.5)
    ax.axhline(0, color="grey", lw=0.9)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:+.1f}", (i, v), ha="center", va="bottom" if v > 0 else "top",
                    fontsize=11, fontweight="bold", xytext=(0, 5 if v > 0 else -5), textcoords="offset points")
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("exception's Δ LP markout  ($/run, contemporaneous)")
    ax.set_title(f"Why the exception can hurt (σ={d['sigma']:.0f}bp): the wider bid DOES help where it fires (+),\n"
                 "but it displaces deterred corrections to later blocks where, by LVR's convexity, they cost more (−)",
                 fontsize=9.5, fontweight="bold")
    _bare(ax)
    fig.tight_layout()
    return fig


# ============================================================================
# Worked micro-example (notebook): one episode, block by block.
# DEMO_PATH: quiet, a +30bp up-jump (block 2), then it falls back to 100.
# ============================================================================
DEMO_PATH = [100.0, 100.0, 100.30, 100.15, 100.00, 100.00]
_CAP = 0.99


@dataclass
class FlatPlusException:
    """Flat TS/2 fee + ONLY the big-move exception (widen the reverting side by 1/2*|PI|
    when |PI| >= cutoff). Isolates the exception from the permanent/temporary skew so the
    displacement mechanism is visible on its own."""
    ts_bps: float = 9.0
    cutoff_bps: float = 10.0
    on: bool = True
    _ts: float = field(default=0.0, init=False)
    _fa: float = field(default=0.0, init=False)
    _fb: float = field(default=0.0, init=False)
    _cut: float = field(default=0.0, init=False)
    _prev: float = field(default=0.0, init=False)
    _blk: object = field(default=None, init=False)

    def __post_init__(self):
        self._ts = self.ts_bps / 1e4; self._cut = self.cutoff_bps / 1e4; self._fa = self._fb = self._ts / 2

    def after_initialize(self, ix, iy):
        self._fa = self._fb = self._ts / 2; self._prev = iy / ix; self._blk = None; return (self._fb, self._fa)

    def after_swap(self, t):
        return (min(self._fb, _CAP), min(self._fa, _CAP))

    def before_swap(self, inc):
        spot = inc.reserve_y / inc.reserve_x
        if int(inc.block) != self._blk:
            pi = (spot - self._prev) / self._prev if self._prev > 0 else 0.0
            self._fa = self._fb = self._ts / 2
            if self.on and abs(pi) >= self._cut:
                if pi >= 0: self._fb += 0.5 * abs(pi)
                else: self._fa += 0.5 * abs(pi)
            self._blk = int(inc.block); self._prev = spot
        return (min(self._fb, _CAP), min(self._fa, _CAP))


def show_v2_decomposition(path=DEMO_PATH, ts_bps=9.0, beta=0.5, delta_max_bps=6.0, h_max_bps=7.0, cutoff_bps=10.0):
    """Q1: how the doc's PERMANENT (h) and TEMPORARY (g) fee components evolve, f = h + g (bp)."""
    s = VolatileHookV2Strategy(ts_bps=ts_bps, beta=beta, delta_max_bps=delta_max_bps, h_max_bps=h_max_bps,
                               cutoff_bps=cutoff_bps, permanent_skew_on=True, temporary_widening_on=True,
                               exception_on=True)
    pool = StrategyAMM("p", s, 10_000.0, 1_000_000.0); pool.initialize(); arber = Arbitrageur()
    print("FULL v2 fee decomposition (h=permanent, g=temporary, f=h+g; all bp). PI = inter-block mid change.")
    print(f"{'blk':>3} {'fair':>7} {'PI':>7} | {'h_a->h_a*':>10} {'h_b->h_b*':>10} | {'g_a':>4} {'g_b':>5} | "
          f"{'f_a':>4} {'f_b':>5} | {'midA':>9} {'arb':>5}")
    prevmid = pool.spot_price
    for t, fp in enumerate(path):
        midb = pool.spot_price; pi = (midb - prevmid) / prevmid * 1e4
        ha0, hb0 = s._ha * 1e4, s._hb * 1e4
        pool.before_swap(is_buy=pool.spot_price < fp, size=None, block=t)
        ha1, hb1 = s._ha * 1e4, s._hb * 1e4; fa, fb = s._fa * 1e4, s._fb * 1e4
        arb = arber.execute_arb(pool, float(fp), t)
        ad = "-" if arb is None else ("BUY" if arb.side == "sell" else "SELL")
        print(f"{t:>3} {fp:>7.2f} {pi:>6.1f}b | {ha0:4.1f}->{ha1:4.1f} {hb0:4.1f}->{hb1:4.1f} | "
              f"{fa - ha1:4.1f} {fb - hb1:5.1f} | {fa:4.1f} {fb:5.1f} | {pool.spot_price:9.4f} {ad:>5}")
        prevmid = midb


def _trace_markout(path, on, label):
    pool = StrategyAMM("p", FlatPlusException(on=on), 10_000.0, 1_000_000.0); pool.initialize(); arber = Arbitrageur()
    print(f"\n----- {label} -----")
    tot = 0.0
    for t, fp in enumerate(path):
        rxb, ryb = pool.reserves(); midb = pool.spot_price
        pool.before_swap(is_buy=pool.spot_price < fp, size=None, block=t); fb = pool.bid_fee * 1e4
        arb = arber.execute_arb(pool, float(fp), t)
        if arb is None:
            print(f" b{t}: fair={fp:.2f}  f_b={fb:4.1f}bp  midB={midb:.4f}  -> NO ARB")
            continue
        buy = arb.side == "sell"
        mk = (arb.amount_y - arb.amount_x * fp) if buy else (arb.amount_x * fp - arb.amount_y); tot += mk
        rxa, rya = pool.reserves()
        eq = (f"dy - dx*fair = {arb.amount_y:.2f} - {arb.amount_x:.3f}*{fp:.2f}" if buy
              else f"dx*fair - dy = {arb.amount_x:.3f}*{fp:.2f} - {arb.amount_y:.2f}")
        print(f" b{t}: fair={fp:.2f}  f_b={fb:4.1f}bp  {'BUY ' if buy else 'SELL'}  "
              f"x:{rxb:.2f}->{rxa:.2f}  mid:{midb:.4f}->{pool.spot_price:.4f}  markout = {eq} = {mk:+.4f}")
    print(f"   TOTAL markout = {tot:+.4f}")
    return tot


def show_displacement_example(path=DEMO_PATH):
    """Q-displacement: flat vs flat+exception on DEMO_PATH, with pool state + markout per block."""
    a = _trace_markout(path, False, "FLAT (exception OFF)")
    b = _trace_markout(path, True, "FLAT + EXCEPTION ON")
    print(f"\n  net exception effect on this path = {b - a:+.4f}")
    return a, b


def _path_effect(path):
    def tot(on):
        pool = StrategyAMM("p", FlatPlusException(on=on), 10_000.0, 1_000_000.0); pool.initialize(); arber = Arbitrageur(); s = 0.0
        for t, fp in enumerate(path):
            pool.before_swap(is_buy=pool.spot_price < fp, size=None, block=t)
            arb = arber.execute_arb(pool, float(fp), t)
            if arb: s += (arb.amount_y - arb.amount_x * fp) if arb.side == "sell" else (arb.amount_x * fp - arb.amount_y)
        return s
    return tot(True) - tot(False)


def show_branch_analysis():
    """Q2 (not bad luck): after the +30bp jump fires the exception, branch on what happens next."""
    c = _path_effect([100, 100, 100.30, 100.45, 100.45])              # continuation
    d = _path_effect([100, 100, 100.30, 100.15, 100.00, 100.00])      # reversion, then down
    u = _path_effect([100, 100, 100.30, 100.15, 100.30, 100.30])      # reversion, then back up
    print("exception effect by branch after the +30bp up-jump (which fires the exception):")
    print(f"  continuation (price keeps rising)        = {c:+.4f}   (INERT: continuation arb pays the ask)")
    print(f"  reversion then b4 DOWN                    = {d:+.4f}   (displacement cost)")
    print(f"  reversion then b4 UP                      = {u:+.4f}   (round-trip avoided)")
    print(f"  reversion average (b4 50/50)             = {(d + u) / 2:+.4f}")
    print(f"  full average (1/2 cont + 1/2 reversion)  = {0.5 * c + 0.5 * (d + u) / 2:+.4f}   <- still negative")
    return dict(continuation=c, rev_down=d, rev_up=u)
