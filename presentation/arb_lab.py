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
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena_eval.core.types import IncomingSwap, TradeInfo  # noqa: E402
from arena_eval.exact_simple_amm.arb_only_lab import (  # noqa: E402
    MECHANISMS, VolatileHookV2Strategy, historical_paths, iid_lognormal_path, load_historical_mids,
    markout_15s, regime_path, run_arb_only)
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
def run_matrix(seeds=SEEDS, sigma_bps=SIGMA_BPS, ts_bps=TS_BPS, delta_max_bps=DELTA_MAX_BPS, n_blocks=5000,
               path_fn=None):
    """Paired per-mechanism total LP markout. Returns {name: np.array over seeds}.
    path_fn(seed) -> fair path overrides the default i.i.d.-lognormal GBM (e.g. regime_path)."""
    per = {n: [] for n in MECHANISMS}
    for s in seeds:
        fair = path_fn(s) if path_fn is not None else iid_lognormal_path(n_blocks, sigma_bps, seed=s)
        for n, kw in MECHANISMS.items():
            tr = run_arb_only(VolatileHookV2Strategy(ts_bps=ts_bps, delta_max_bps=delta_max_bps, **kw), fair)
            per[n].append(markout_15s(fair, tr).sum())
    return {n: np.asarray(v) for n, v in per.items()}


def plot_matrix(per: dict, subtitle: str | None = None, n_unit: str = "paired seeds"):
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
    sub = subtitle if subtitle is not None else f"σ={SIGMA_BPS:.0f}bp/block GBM"
    fig.suptitle(f"Arb-only LVR lab — v2 mechanisms, isolated  ({sub}, TS={TS_BPS:.0f}bp, "
                 f"{len(per[names[0]])} {n_unit})", fontweight="bold", fontsize=11.5)
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
    """Q1: the doc's PERMANENT (h) and TEMPORARY (g) fee components per block, f = h + g (bp).
    Returns a DataFrame (renders as a clean table in Jupyter)."""
    s = VolatileHookV2Strategy(ts_bps=ts_bps, beta=beta, delta_max_bps=delta_max_bps, h_max_bps=h_max_bps,
                               cutoff_bps=cutoff_bps, permanent_skew_on=True, temporary_widening_on=True,
                               exception_on=True)
    pool = StrategyAMM("p", s, 10_000.0, 1_000_000.0); pool.initialize(); arber = Arbitrageur()
    rows = []
    prevmid = pool.spot_price
    for t, fp in enumerate(path):
        midb = pool.spot_price; pi = (midb - prevmid) / prevmid * 1e4
        ha0, hb0 = s._ha * 1e4, s._hb * 1e4
        pool.before_swap(is_buy=pool.spot_price < fp, size=None, block=t)
        ha1, hb1 = s._ha * 1e4, s._hb * 1e4; fa, fb = s._fa * 1e4, s._fb * 1e4
        arb = arber.execute_arb(pool, float(fp), t)
        ad = "–" if arb is None else ("BUY" if arb.side == "sell" else "SELL")
        rows.append({"blk": t, "fair": f"{fp:.2f}", "PI (bp)": f"{pi:+.1f}",
                     "h_a→h_a*": f"{ha0:.1f}→{ha1:.1f}", "h_b→h_b*": f"{hb0:.1f}→{hb1:.1f}",
                     "g_a": f"{fa - ha1:.1f}", "g_b": f"{fb - hb1:.1f}", "f_a": f"{fa:.1f}", "f_b": f"{fb:.1f}",
                     "mid after": f"{pool.spot_price:.4f}", "arb": ad})
        prevmid = midb
    return pd.DataFrame(rows).set_index("blk")


def _trace_rows(path, on, run):
    pool = StrategyAMM("p", FlatPlusException(on=on), 10_000.0, 1_000_000.0); pool.initialize(); arber = Arbitrageur()
    rows = []; tot = 0.0
    for t, fp in enumerate(path):
        midb = pool.spot_price
        pool.before_swap(is_buy=pool.spot_price < fp, size=None, block=t); fb = pool.bid_fee * 1e4
        arb = arber.execute_arb(pool, float(fp), t)
        if arb is None:
            rows.append({"run": run, "blk": t, "fair": f"{fp:.2f}", "f_b (bp)": f"{fb:.1f}", "arb": "–",
                         "dx": "", "dy": "", "mid after": f"{midb:.4f}", "markout": ""})
            continue
        buy = arb.side == "sell"
        mk = (arb.amount_y - arb.amount_x * fp) if buy else (arb.amount_x * fp - arb.amount_y); tot += mk
        rows.append({"run": run, "blk": t, "fair": f"{fp:.2f}", "f_b (bp)": f"{fb:.1f}",
                     "arb": "BUY" if buy else "SELL", "dx": f"{arb.amount_x:.3f}", "dy": f"{arb.amount_y:.2f}",
                     "mid after": f"{pool.spot_price:.4f}", "markout": f"{mk:+.4f}"})
    return rows, tot


def show_displacement_example(path=DEMO_PATH):
    """Q-displacement: flat vs flat+exception on DEMO_PATH, pool state + markout per block.
    markout = dy − dx·fair (LP sells X) or dx·fair − dy (LP buys X). Returns a DataFrame."""
    rf, tf = _trace_rows(path, False, "flat")
    re, te = _trace_rows(path, True, "flat+exc")
    df = pd.DataFrame(rf + re).set_index(["run", "blk"])
    df.attrs["totals"] = (tf, te)
    print(f"flat total = {tf:+.4f}   flat+exception total = {te:+.4f}   net exception effect = {te - tf:+.4f}")
    return df


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
    """Q2 (not bad luck): after the +30bp up-jump fires the exception, branch on what happens next.
    Returns a DataFrame of the exception's effect on each branch + the martingale averages."""
    c = _path_effect([100, 100, 100.30, 100.45, 100.45])              # continuation
    d = _path_effect([100, 100, 100.30, 100.15, 100.00, 100.00])      # reversion, then down
    u = _path_effect([100, 100, 100.30, 100.15, 100.30, 100.30])      # reversion, then back up
    rows = [
        {"branch": "continuation (price keeps rising)", "exception effect ($)": f"{c:+.4f}",
         "note": "inert — continuation arb pays the ask"},
        {"branch": "reversion → b4 down", "exception effect ($)": f"{d:+.4f}", "note": "displacement cost"},
        {"branch": "reversion → b4 up", "exception effect ($)": f"{u:+.4f}", "note": "round-trip avoided"},
        {"branch": "reversion average (50/50)", "exception effect ($)": f"{(d + u) / 2:+.4f}", "note": ""},
        {"branch": "full average (½ cont + ½ reversion)", "exception effect ($)": f"{0.5 * c + 0.5 * (d + u) / 2:+.4f}",
         "note": "still negative → not bad luck"},
    ]
    return pd.DataFrame(rows).set_index("branch")


# ============================================================================
# Realistic return distribution: the calibrated regime-switching process.
# ============================================================================
def regime_returns(seeds=SEEDS, n_blocks=5000):
    """Per-block returns (bp) of the calibrated regime process, pooled over seeds."""
    rr = []
    for s in seeds:
        p = regime_path(n_blocks, s)
        rr.append(np.diff(p) / p[:-1] * 1e4)
    return np.concatenate(rr)


def plot_return_distribution(seeds=SEEDS, n_blocks=5000, half_spread_bp=4.5, cutoff_bp=10.0):
    """Calibrated regime returns vs a Gaussian at the SAME std: a tight middle (most moves
    inside the no-arb band) with fat tails (which dominate LVR, since LVR ~ move^2)."""
    rr = regime_returns(seeds, n_blocks); std = rr.std()
    p_fire = float(np.mean(np.abs(rr) >= cutoff_bp))
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 4.8))
    # (A) body, linear
    b = np.linspace(-25, 25, 121); x = 0.5 * (b[:-1] + b[1:])
    g = np.exp(-x ** 2 / (2 * std ** 2)) / (std * np.sqrt(2 * np.pi))
    ax[0].hist(rr, bins=b, density=True, color=_BLUE, alpha=0.55, label="calibrated regime")
    ax[0].plot(x, g, color=_RED, lw=2, label=f"Gaussian (σ={std:.1f}bp, matched)")
    ax[0].axvspan(-half_spread_bp, half_spread_bp, color="grey", alpha=0.18, label="no-arb band (±TS/2)")
    ax[0].set_xlabel("per-block fair return (bp)"); ax[0].set_ylabel("density")
    ax[0].set_title("(A) body — tight peak: most moves stay inside the no-arb band\n(so most blocks have no arb)",
                    fontsize=10, fontweight="bold")
    ax[0].legend(fontsize=8.5); _bare(ax[0]); ax[0].grid(alpha=0.2)
    # (B) tails, log-y
    b2 = np.linspace(-60, 60, 121); x2 = 0.5 * (b2[:-1] + b2[1:])
    g2 = np.exp(-x2 ** 2 / (2 * std ** 2)) / (std * np.sqrt(2 * np.pi))
    ax[1].hist(rr, bins=b2, density=True, color=_BLUE, alpha=0.55, label="calibrated regime")
    ax[1].plot(x2, g2, color=_RED, lw=2, label="Gaussian (matched σ)")
    for c in (-cutoff_bp, cutoff_bp):
        ax[1].axvline(c, color="#7d6608", ls="--", lw=1)
    ax[1].set_yscale("log"); ax[1].set_xlabel("per-block fair return (bp)"); ax[1].set_ylabel("density (log)")
    ax[1].set_title(f"(B) tails (log) — far fatter than Gaussian; max|r|≈{np.abs(rr).max():.0f}bp\n"
                    f"exception cutoff ±10bp fires {p_fire:.0%} (Gaussian would: "
                    f"{2 * (1 - 0.5 * (1 + _erf(cutoff_bp / std / 2 ** .5))):.0%})", fontsize=10, fontweight="bold")
    ax[1].legend(fontsize=8.5); _bare(ax[1])
    fig.suptitle("Calibrated WETH/USD 12s return distribution (the realistic process) — martingale, "
                 f"std={std:.1f}bp, but fat-tailed", fontweight="bold", fontsize=11.5)
    fig.tight_layout()
    return fig


def _erf(z):
    import math
    return math.erf(z)


# ============================================================================
# Historical replay: a THIRD way to run the lab — over the actual Binance
# 12s series, sliced into real ~17h windows. Same marginal CDF as the regime
# process, but the real serial structure (Moallemi: "correct correlations").
# ============================================================================
import json  # noqa: E402

_HIST_CACHE = Path(__file__).resolve().parents[1] / "analysis" / "weth_usdc_90d" / "arb_historical_cache.json"


def run_matrix_historical(n_blocks=5000, stride=None, ts_bps=TS_BPS, delta_max_bps=DELTA_MAX_BPS,
                          cache=_HIST_CACHE, rebuild=False):
    """4-mechanism matrix replayed over EVERY real ~17h window. Same metric as run_matrix
    (net 15s-fair LP markout, $/window). Returns {per:{mech:array over windows}, vol_bp:array,
    meta:{...}}. Cached to JSON (full run ~90s over ~1037 windows)."""
    key = dict(n_blocks=n_blocks, stride=stride or n_blocks, ts_bps=ts_bps, delta_max_bps=delta_max_bps)
    if cache and Path(cache).exists() and not rebuild:
        d = json.loads(Path(cache).read_text())
        if all(d["meta"].get(k) == v for k, v in key.items()):
            return dict(per={k: np.asarray(v) for k, v in d["per"].items()},
                        vol_bp=np.asarray(d["vol_bp"]), meta=d["meta"])
    W = historical_paths(n_blocks, stride)
    per = {n: [] for n in MECHANISMS}
    for w in W:
        for n, kw in MECHANISMS.items():
            tr = run_arb_only(VolatileHookV2Strategy(ts_bps=ts_bps, delta_max_bps=delta_max_bps, **kw), w)
            per[n].append(float(markout_15s(w, tr).sum()))
    vol_bp = [float(np.diff(np.log(w)).std() * 1e4) for w in W]
    meta = dict(n_windows=len(W), **key)
    if cache:
        Path(cache).write_text(json.dumps(dict(meta=meta, per=per, vol_bp=vol_bp)))
    return dict(per={n: np.asarray(v) for n, v in per.items()}, vol_bp=np.asarray(vol_bp), meta=meta)


def _protection(per, name="+ permanent skew"):
    """% LVR reduction of mechanism `name` vs flat, paired across windows (LVR = -markout)."""
    flat = per["flat TS/2"]; mech = per[name]
    return float((mech.sum() - flat.sum()) / (-flat.sum()) * 100.0)


def _boot_protection_ci(per, name="+ permanent skew", n_boot=2000, seed=0):
    """Block-bootstrap CI for %protection over windows (windows ~ pseudo-replicated, so
    resample whole windows to get an honest band on the mechanism contrast)."""
    rng = np.random.default_rng(seed)
    flat, mech = per["flat TS/2"], per[name]
    n = len(flat); idx = rng.integers(0, n, size=(n_boot, n))
    f = flat[idx].sum(1); m = mech[idx].sum(1)
    pcts = (m - f) / (-f) * 100.0
    return float(np.percentile(pcts, 2.5)), float(np.percentile(pcts, 97.5))


def _process_returns_bp(kind, seeds=SEEDS, n_blocks=5000, sigma_bps=None):
    """Per-block returns (bp) for serial-structure diagnostics. Historical uses the full
    CONTIGUOUS series (returns computed within gap-free runs, then concatenated) so real
    autocorrelation is not broken at window edges."""
    if kind == "historical":
        ts, mid = load_historical_mids()
        runs = np.split(np.arange(len(ts)), np.flatnonzero(np.diff(ts) != 12) + 1)
        return np.concatenate([np.diff(np.log(mid[r])) for r in runs if len(r) > 1]) * 1e4
    if kind == "regime":
        return regime_returns(seeds, n_blocks)
    if kind == "gbm":
        return np.concatenate([np.diff(np.log(iid_lognormal_path(n_blocks, sigma_bps, seed=s)))
                               for s in seeds]) * 1e4
    raise ValueError(kind)


def _ac1(x):
    x = np.asarray(x, float); x = x - x.mean()
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def serial_diagnostics(seeds=SEEDS, n_blocks=5000, cutoff_bp=10.0):
    """The Moallemi test: GBM and the regime process have the right marginal but the wrong
    correlations. Compare, per process: return std; lag-1 autocorr of returns (≈0 = martingale,
    all three); lag-1 autocorr of |returns| (vol clustering); and jump follow-through lift =
    P(|r_{t+1}|≥cut | |r_t|≥cut) / P(|r|≥cut) (autocorrelated jumps — >1 means jumps cluster)."""
    hist = _process_returns_bp("historical")
    hstd = float(hist.std())
    rows = []
    for name, r in [(f"GBM (σ={hstd:.1f}bp, matched)", _process_returns_bp("gbm", seeds, n_blocks, hstd)),
                    ("regime-switching", _process_returns_bp("regime", seeds, n_blocks)),
                    ("historical replay", hist)]:
        a = np.abs(r); big = a >= cutoff_bp; p = float(big.mean())
        lift = float((big[:-1] & big[1:]).sum() / max(big[:-1].sum(), 1) / p) if p > 0 else float("nan")
        rows.append(dict(process=name, std_bp=round(r.std(), 2), ac1_return=round(_ac1(r), 4),
                         ac1_absreturn=round(_ac1(a), 4), p_big_pct=round(p * 100, 2),
                         jump_followthrough_lift=round(lift, 2)))
    return pd.DataFrame(rows)


def compare_processes(seeds=SEEDS, n_blocks=5000, ts_bps=TS_BPS, delta_max_bps=DELTA_MAX_BPS, hist=None):
    """Marginal mechanism effects ($/run) under the three processes, side by side. GBM is run
    at σ matched to the historical std so the only differences are tails (GBM→regime) and
    correlations (regime→historical). Returns a tidy DataFrame."""
    hist = hist if hist is not None else run_matrix_historical(n_blocks, ts_bps=ts_bps, delta_max_bps=delta_max_bps)
    hstd = float(_process_returns_bp("historical").std())
    gbm = run_matrix(seeds, hstd, ts_bps, delta_max_bps, n_blocks)
    reg = run_matrix(seeds, None, ts_bps, delta_max_bps, n_blocks, path_fn=lambda s: regime_path(n_blocks, s))
    names = list(MECHANISMS)
    rows = []
    for label, per in [(f"GBM (σ={hstd:.1f}bp)", gbm), ("regime-switching", reg), ("historical replay", hist["per"])]:
        m = {n: per[n].mean() for n in names}
        rows.append(dict(process=label, flat=round(m[names[0]], 1),
                         d_perm=round(m[names[1]] - m[names[0]], 1),
                         d_temp=round(m[names[2]] - m[names[1]], 1),
                         d_exc=round(m[names[3]] - m[names[2]], 1),
                         perm_protect_pct=round(_protection(per), 2)))
    return pd.DataFrame(rows)


def plot_historical_matrix(hist):
    """Headline: the 4-mechanism matrix replayed over the real series (reuses plot_matrix)."""
    return plot_matrix(hist["per"], subtitle="actual Binance ETHUSDT 12s, ~17h windows",
                       n_unit="real windows")


def plot_historical_vol_protection(hist, n_bins=6):
    """%LVR-protection from the permanent skew vs the window's realized vol, recovered from
    REAL windows — the analogue of the synthetic σ-sweep (tests the ∝1/σ finding on real data)."""
    per, vol = hist["per"], hist["vol_bp"]
    flat, perm = per["flat TS/2"], per["+ permanent skew"]
    edges = np.quantile(vol, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    binid = np.clip(np.digitize(vol, edges) - 1, 0, n_bins - 1)
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        m = binid == b
        if m.sum() < 5:
            continue
        xs.append(float(np.median(vol[m])))
        ys.append(float((perm[m].sum() - flat[m].sum()) / (-flat[m].sum()) * 100.0))
        ns.append(int(m.sum()))
    xs, ys = np.array(xs), np.array(ys)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(xs, ys, "o-", color=_GREEN, lw=2, ms=7)
    for x, y, k in zip(xs, ys, ns):
        ax.annotate(f"{y:.1f}%\nn={k}", (x, y), fontsize=8, ha="center", va="bottom", xytext=(0, 5),
                    textcoords="offset points")
    # 1/sigma reference, anchored at the lowest-vol bin
    ref = ys[0] * xs[0] / xs
    ax.plot(xs, ref, ":", color=_RED, lw=1.5, label="∝ 1/σ  (anchored at first bin)")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("window realized vol  (bp/block)"); ax.set_ylabel("permanent-skew LVR protection  (%)")
    ax.set_title("Permanent-skew protection vs realized vol — recovered from REAL windows\n"
                 "(does the synthetic ∝1/σ law survive real correlations?)", fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=9); _bare(ax)
    fig.tight_layout()
    return fig
