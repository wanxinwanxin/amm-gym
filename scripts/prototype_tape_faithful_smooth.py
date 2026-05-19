"""Prototype: tape-faithful smooth replay of the exact challenge simulator.

Replays an exact challenge tape (deterministic discrete realizations of price
shocks, order counts, sizes, and side uniforms) with smooth approximations
applied **only** at the measure-zero branches of the simulator:

    D. arb side selection (spot < fair  vs  spot > fair)
    E. router MIN_AMOUNT gate
    F. arb profit positivity gate
    G. fee clamp at [0, MAX_FEE]
    H. AMM quote positivity gates

The smoothing primitives are
    smooth_gate(x, k)     = sigmoid(k * x)
    smooth_positive(x, k) = softplus(k * x) / k
    smooth_clip(x, lo, hi, k) = lo + smooth_positive(x - lo, k) - smooth_positive(x - hi, k)

This prototype uses the EXACT tape's discrete arrivals, sizes, and side
uniforms (the side u_i is a fixed scalar per order; we still smooth the
categorical step `1[u < buy_prob]` so the executor stays differentiable
in policy params, but at high k it converges to the exact discrete choice).

Only fixed-fee policies are exercised, on both submission and normalizer,
to keep the prototype focused on the simulator faithfulness — not on policy
state evolution.

Run:
    .venv/bin/python scripts/prototype_tape_faithful_smooth.py

Expected: smooth_score(k) -> exact_score as k -> infinity.
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arena_eval.diff_simple_amm import build_challenge_tape
from arena_eval.exact_simple_amm import ExactSimpleAMMConfig, FixedFeeStrategy, run_seed


BASELINE_FEE = 0.003
MAX_FEE = 0.1


# ----- smooth primitives (numpy, numerically stable) -----
def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def smooth_gate(x: float, k: float, threshold: float = 0.0) -> float:
    return float(sigmoid(k * (x - threshold)))


def smooth_positive(x: float, k: float) -> float:
    z = k * x
    if z > 30.0:
        return float(x)
    if z < -30.0:
        return 0.0
    return float(math.log1p(math.exp(z)) / k)


def smooth_clip(x: float, lo: float, hi: float, k: float) -> float:
    return lo + smooth_positive(x - lo, k) - smooth_positive(x - hi, k)


def smooth_trade_amount(amount: float, k: float, minimum: float = 0.0001) -> float:
    return smooth_positive(amount, k) * smooth_gate(amount, k, threshold=minimum)


# ----- smooth AMM executors -----
def execute_buy_x_smooth(rx: float, ry: float, bid_fee: float, amount_x: float, k: float):
    """AMM buys X, pays Y. Trader sells X."""
    amount_x_eff = smooth_trade_amount(amount_x, k)
    gamma = max(0.0, min(1.0, 1.0 - bid_fee))
    net_x = amount_x_eff * gamma
    new_rx = rx + net_x
    new_ry = (rx * ry) / max(new_rx, 1e-12)
    amount_y = smooth_positive(ry - new_ry, k)
    fee_x = amount_x_eff * bid_fee
    return new_rx, ry - amount_y, amount_x_eff, amount_y, fee_x


def execute_sell_x_smooth(rx: float, ry: float, ask_fee: float, amount_x: float, k: float):
    """AMM sells X, receives Y. Trader buys X."""
    capped = smooth_clip(amount_x, 0.0, 0.99 * rx, k)
    amount_x_eff = smooth_trade_amount(capped, k)
    gamma = max(0.0, min(1.0, 1.0 - ask_fee))
    new_rx = max(rx - amount_x_eff, 1e-12)
    new_ry = (rx * ry) / new_rx
    net_y = smooth_positive(new_ry - ry, k)
    total_y = net_y / max(gamma, 1e-12)
    fee_y = total_y - net_y
    return new_rx, ry + net_y, amount_x_eff, total_y, fee_y


def execute_buy_x_with_y_smooth(rx: float, ry: float, ask_fee: float, amount_y: float, k: float):
    """AMM sells X for incoming Y. Trader buys X with Y."""
    amount_y_eff = smooth_trade_amount(amount_y, k)
    gamma = max(0.0, min(1.0, 1.0 - ask_fee))
    net_y = amount_y_eff * gamma
    new_ry = ry + net_y
    new_rx = (rx * ry) / max(new_ry, 1e-12)
    amount_x = smooth_positive(rx - new_rx, k)
    fee_y = amount_y_eff * ask_fee
    return rx - amount_x, new_ry, amount_x, amount_y_eff, fee_y


# ----- exact router splits (closed-form, already smooth on the interior) -----
def split_buy(rx1, ry1, askfee1, rx2, ry2, askfee2, total_y, k):
    g1, g2 = 1.0 - askfee1, 1.0 - askfee2
    a1 = math.sqrt(rx1 * g1 * ry1)
    a2 = math.sqrt(rx2 * g2 * ry2)
    if a2 == 0.0:
        return total_y, 0.0
    r = a1 / a2
    num = r * (ry2 + g2 * total_y) - ry1
    den = g1 + r * g2
    raw = total_y / 2.0 if den == 0.0 else num / den
    y1 = smooth_clip(raw, 0.0, total_y, k)
    return y1, total_y - y1


def split_sell(rx1, ry1, bidfee1, rx2, ry2, bidfee2, total_x, k):
    g1, g2 = 1.0 - bidfee1, 1.0 - bidfee2
    b1 = math.sqrt(ry1 * g1 * rx1)
    b2 = math.sqrt(ry2 * g2 * rx2)
    if b2 == 0.0:
        return total_x, 0.0
    r = b1 / b2
    num = r * (rx2 + g2 * total_x) - rx1
    den = g1 + r * g2
    raw = total_x / 2.0 if den == 0.0 else num / den
    x1 = smooth_clip(raw, 0.0, total_x, k)
    return x1, total_x - x1


# ----- smooth arb (only the side branch is smoothed; profit gate via smooth_positive) -----
def smooth_arb(rx, ry, bid_fee, ask_fee, fair_price, k):
    spot = ry / max(rx, 1e-12)
    buy_gate = smooth_gate(fair_price - spot, k)   # 1 if spot < fair
    sell_gate = smooth_gate(spot - fair_price, k)  # 1 if spot > fair

    # buy-arb branch: AMM sells X (execute_sell_x), arb buys X cheap
    gamma_ask = max(1.0 - ask_fee, 1e-12)
    buy_raw = rx - math.sqrt(rx * ry / max(gamma_ask * fair_price, 1e-12))
    buy_amount_x = smooth_clip(smooth_positive(buy_raw, k), 0.0, 0.99 * rx, k) * buy_gate
    buy_rx, buy_ry, buy_ax, buy_ay, _ = execute_sell_x_smooth(rx, ry, ask_fee, buy_amount_x, k)
    buy_profit = buy_ax * fair_price - buy_ay

    # sell-arb branch: AMM buys X (execute_buy_x), arb sells X expensive
    gamma_bid = max(1.0 - bid_fee, 1e-12)
    sell_raw = (math.sqrt(rx * ry * gamma_bid / max(fair_price, 1e-12)) - rx) / max(gamma_bid, 1e-12)
    sell_amount_x = smooth_positive(sell_raw, k) * sell_gate
    sell_rx, sell_ry, sell_ax, sell_ay, _ = execute_buy_x_smooth(rx, ry, bid_fee, sell_amount_x, k)
    sell_profit = sell_ay - sell_ax * fair_price

    # mix branches: pick whichever side gate dominates
    use_sell_weight = smooth_gate(sell_gate - buy_gate, k)
    next_rx = (1.0 - use_sell_weight) * buy_rx + use_sell_weight * sell_rx
    next_ry = (1.0 - use_sell_weight) * buy_ry + use_sell_weight * sell_ry
    profit = (1.0 - use_sell_weight) * buy_profit + use_sell_weight * sell_profit
    return smooth_positive(profit, k), next_rx, next_ry


# ----- tape-faithful smooth rollout (FixedFee, both venues) -----
def replay_smooth(config: ExactSimpleAMMConfig, tape, k: float) -> float:
    rx1, ry1 = config.submission_initial_x, config.submission_initial_y
    rx2, ry2 = config.normalizer_initial_x, config.normalizer_initial_y
    bid1 = ask1 = smooth_clip(BASELINE_FEE, 0.0, MAX_FEE, k)
    bid2 = ask2 = smooth_clip(BASELINE_FEE, 0.0, MAX_FEE, k)

    drift = (config.gbm_mu - 0.5 * config.gbm_sigma**2) * config.gbm_dt
    vol = config.gbm_sigma * math.sqrt(config.gbm_dt)
    fair = config.initial_price

    edge_sub = 0.0
    edge_norm = 0.0

    for t in range(config.n_steps):
        fair = fair * math.exp(drift + vol * tape.gbm_normals[t])

        # arb
        prof_sub, rx1, ry1 = smooth_arb(rx1, ry1, bid1, ask1, fair, k)
        edge_sub -= prof_sub
        prof_norm, rx2, ry2 = smooth_arb(rx2, ry2, bid2, ask2, fair, k)
        edge_norm -= prof_norm

        # retail orders from the EXACT tape
        for i in range(tape.order_counts[t]):
            size = tape.order_sizes[t][i]
            side_u = tape.order_side_uniforms[t][i]
            buy_weight = smooth_gate(config.retail_buy_prob - side_u, k)

            buy_y = buy_weight * size
            sell_y_notional = (1.0 - buy_weight) * size

            # buy route — smooth positive sub-trade on each AMM
            y_sub, y_norm = split_buy(rx1, ry1, ask1, rx2, ry2, ask2, buy_y, k)
            rx1, ry1, ax1, ay1, _ = execute_buy_x_with_y_smooth(rx1, ry1, ask1, y_sub, k)
            edge_sub += ay1 - ax1 * fair
            rx2, ry2, ax2, ay2, _ = execute_buy_x_with_y_smooth(rx2, ry2, ask2, y_norm, k)
            edge_norm += ay2 - ax2 * fair

            # sell route
            sell_total_x = sell_y_notional / max(fair, 1e-12)
            x_sub, x_norm = split_sell(rx1, ry1, bid1, rx2, ry2, bid2, sell_total_x, k)
            rx1, ry1, ax1, ay1, _ = execute_buy_x_smooth(rx1, ry1, bid1, x_sub, k)
            edge_sub += ax1 * fair - ay1
            rx2, ry2, ax2, ay2, _ = execute_buy_x_smooth(rx2, ry2, bid2, x_norm, k)
            edge_norm += ax2 * fair - ay2

    return edge_sub  # match exact score = edge_submission


def main() -> None:
    seeds = (0, 1, 2, 3)
    n_steps = 256
    sharpness_grid = (10.0, 50.0, 200.0, 1000.0, 5000.0, 25000.0)

    print(
        f"{'seed':>4} | {'exact':>10} |"
        + " | ".join(f"smooth(k={int(k)})".rjust(15) for k in sharpness_grid)
    )
    print("-" * (8 + 13 + 17 * len(sharpness_grid)))
    for seed in seeds:
        cfg = replace(ExactSimpleAMMConfig.from_seed(seed), n_steps=n_steps)
        tape = build_challenge_tape(config=cfg, seed=seed)
        exact = float(
            run_seed(
                FixedFeeStrategy(),
                seed,
                config=cfg,
                normalizer_strategy=FixedFeeStrategy(),
            ).edge_submission
        )
        smooths = [replay_smooth(cfg, tape, k) for k in sharpness_grid]
        cells = " | ".join(f"{s:>15.6f}" for s in smooths)
        print(f"{seed:>4} | {exact:>10.6f} | {cells}")

    print()
    print("Convergence per seed (|smooth - exact|):")
    print(
        f"{'seed':>4} | "
        + " | ".join(f"k={int(k)}".rjust(12) for k in sharpness_grid)
    )
    print("-" * (8 + 14 * len(sharpness_grid)))
    for seed in seeds:
        cfg = replace(ExactSimpleAMMConfig.from_seed(seed), n_steps=n_steps)
        tape = build_challenge_tape(config=cfg, seed=seed)
        exact = float(
            run_seed(
                FixedFeeStrategy(),
                seed,
                config=cfg,
                normalizer_strategy=FixedFeeStrategy(),
            ).edge_submission
        )
        gaps = [abs(replay_smooth(cfg, tape, k) - exact) for k in sharpness_grid]
        cells = " | ".join(f"{g:>12.6f}" for g in gaps)
        print(f"{seed:>4} | {cells}")


if __name__ == "__main__":
    main()
