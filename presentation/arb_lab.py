"""Experiment runners + plotters for the arb-only LVR lab (presentation/arb_only_lvr_lab.ipynb).

Thin layer over arena_eval.exact_simple_amm.arb_only_lab: runs the mechanism matrix and a
couple of sweeps on PAIRED seeds (same fair path across mechanisms) and renders the
figures. Metric throughout: net LP markout vs fair 15s later, $ per 5000-block run on a
$1M pool (more negative = more LVR). Lower |markout| = better LVR protection.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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
