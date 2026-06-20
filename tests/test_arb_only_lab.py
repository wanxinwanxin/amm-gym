"""Tests for the arb-only LVR lab: the v2 top-of-block fee mechanics + the arb invariant."""

from __future__ import annotations

import math

from arena_eval.core.types import IncomingSwap
from arena_eval.exact_simple_amm.arb_only_lab import (
    VolatileHookV2Strategy, iid_lognormal_path, regime_path, run_arb_only)
from arena_eval.exact_simple_amm.simulator import Arbitrageur, StrategyAMM


def _bs(s, block, spot, x=10_000.0):
    return s.before_swap(IncomingSwap(is_buy=True, size=None, reserve_x=x, reserve_y=x * spot, block=block))


def test_flat_when_all_mechanisms_off():
    s = VolatileHookV2Strategy(ts_bps=9.0)
    s.after_initialize(10_000.0, 1_000_000.0)        # mid 100
    bid, ask = _bs(s, 0, 100.0)
    assert math.isclose(bid, 4.5e-4, abs_tol=1e-9) and math.isclose(ask, 4.5e-4, abs_tol=1e-9)
    bid, ask = _bs(s, 1, 101.0)                       # up move, but skew off -> still flat
    assert math.isclose(bid, 4.5e-4, abs_tol=1e-9) and math.isclose(ask, 4.5e-4, abs_tol=1e-9)


def test_permanent_skew_leans_ask_after_up_and_sums_to_TS():
    s = VolatileHookV2Strategy(ts_bps=9.0, permanent_skew_on=True)
    s.after_initialize(10_000.0, 1_000_000.0)
    _bs(s, 0, 100.0)
    bid, ask = _bs(s, 1, 101.0)                       # PI>0 -> ask up, bid down, sum=TS (no temporary)
    assert ask > 4.5e-4 > bid
    assert math.isclose(bid + ask, 9e-4, abs_tol=1e-9)


def test_permanent_skew_capped_at_h_max():
    s = VolatileHookV2Strategy(ts_bps=9.0, permanent_skew_on=True, h_max_bps=7.0, delta_max_bps=9.0, beta=10.0)
    s.after_initialize(10_000.0, 1_000_000.0)
    spot = 100.0
    for b in range(1, 30):                            # sustained big up-moves saturate the cap
        spot *= 1.01
        bid, ask = _bs(s, b, spot)
    assert math.isclose(ask, 7e-4, abs_tol=1e-7)      # ask capped at h_max
    assert math.isclose(bid, 2e-4, abs_tol=1e-7)      # bid floors at TS - h_max


def test_temporary_widening_holds_bid_and_widens_spread():
    s = VolatileHookV2Strategy(ts_bps=9.0, permanent_skew_on=True, temporary_widening_on=True)
    s.after_initialize(10_000.0, 1_000_000.0)
    _bs(s, 0, 100.0)
    bid, ask = _bs(s, 1, 101.0)                       # move block: bid held at prior permanent, ask raised
    assert math.isclose(bid, 4.5e-4, abs_tol=1e-7)    # bid = prior h_b (not cut)
    assert bid + ask > 9e-4 + 1e-9                    # spread widened beyond TS


def test_big_move_exception_widens_reverting_side():
    s = VolatileHookV2Strategy(ts_bps=9.0, permanent_skew_on=True, exception_on=True, cutoff_bps=10.0)
    s.after_initialize(10_000.0, 1_000_000.0)
    _bs(s, 0, 100.0)
    bid, ask = _bs(s, 1, 102.0)                       # +200bp >= 10bp cutoff -> +1/2*PI on the bid
    assert bid > 9e-4                                 # bid strongly widened by the exception
    no_exc = VolatileHookV2Strategy(ts_bps=9.0, permanent_skew_on=True, exception_on=False)
    no_exc.after_initialize(10_000.0, 1_000_000.0)
    _bs(no_exc, 0, 100.0)
    b2, _ = _bs(no_exc, 1, 102.0)
    assert bid > b2                                   # exception adds widening vs no-exception


def test_arb_leaves_mid_at_fee_offset_from_fair():
    # reused validated arb: after a buy arb the pool mid == (1 - ask_fee) * fair
    s = VolatileHookV2Strategy(ts_bps=9.0)
    pool = StrategyAMM("p", s, 10_000.0, 1_000_000.0)
    pool.initialize()
    pool.before_swap(is_buy=pool.spot_price < 101.0, size=None, block=0)
    fa = pool.ask_fee
    Arbitrageur().execute_arb(pool, 101.0, 0)
    assert math.isclose(pool.spot_price, (1.0 - fa) * 101.0, rel_tol=1e-6)


def test_run_arb_only_produces_trades_and_is_deterministic():
    fair = iid_lognormal_path(500, sigma_bps=10.0, seed=1)
    t1 = run_arb_only(VolatileHookV2Strategy(ts_bps=9.0), fair)
    t2 = run_arb_only(VolatileHookV2Strategy(ts_bps=9.0), fair)
    assert len(t1) > 0 and len(t1) == len(t2)


def test_regime_path_shape_and_determinism():
    p1 = regime_path(500, seed=1)
    p2 = regime_path(500, seed=1)
    p3 = regime_path(500, seed=2)
    assert len(p1) == 500 and (p1 > 0).all()
    assert (p1 == p2).all()        # deterministic per seed
    assert not (p1 == p3).all()    # different seed -> different path


def test_regime_path_drives_the_arb_loop():
    fair = regime_path(500, seed=1)
    trades = run_arb_only(VolatileHookV2Strategy(ts_bps=9.0), fair)
    assert len(trades) > 0
