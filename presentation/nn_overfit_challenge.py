"""GRU-cell `after_event` policy for the diff sim, plus helpers.

The policy is a small recurrent network:

  feat = MLP(trade_features)           # input projection
  h'   = GRUCell(feat, h)              # recurrent update
  ln_h = LayerNorm(h')                 # stabilizes BPTT
  fees = base_fee + amp * tanh(W ln_h + b)   # bounded fee head

It conforms to the diff-sim's `after_event(params, state, trade) -> (state, bid, ask)`
contract: `params` is a flat pytree of arrays; `state` is the recurrent hidden
vector (a single jnp.ndarray).

Designed for the capability-check experiment: can backprop through `tape_smooth`
overfit a free-form policy on a fixed set of challenge-mode training tapes?
The leaderboard top (~+522 mean edge per 10k-step simulation) is the lower
bound; the structured-retail clairvoyant on the same tapes is the upper bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp


# --------------------------------------------------------------------- config
@dataclass(frozen=True)
class NNPolicyConfig:
    hidden_dim: int = 64
    feature_dim: int = 32       # MLP projection of raw trade features
    base_fee: float = 0.003      # center of the tanh-bounded fee head
    fee_amplitude: float = 0.02  # half-range of the fee head (so fee in [base-amp, base+amp])
    min_fee: float = 0.0001
    max_fee: float = 0.05        # MAX_FEE in the sim is 0.10; cap below
    init_scale: float = 0.1      # init std of weight matrices


# --------------------------------------------------------------------- params
def init_params(cfg: NNPolicyConfig, *, seed: int = 0) -> dict:
    """Random init of all NN params. Returns a flat dict of jnp arrays."""
    key = jax.random.PRNGKey(int(seed))
    keys = jax.random.split(key, 8)
    H = cfg.hidden_dim
    F = cfg.feature_dim
    s = cfg.init_scale

    # Trade-feature embedder: raw 8-dim trade vec → F hidden
    raw_dim = 8
    Wf = s * jax.random.normal(keys[0], (raw_dim, F), dtype=jnp.float64)
    bf = jnp.zeros((F,), dtype=jnp.float64)

    # GRU cell weights: input F, hidden H. Three gates (reset/update/candidate).
    # Init recurrent kernels orthogonal-ish via small Gaussian (good enough at H=64).
    Wz_i = s * jax.random.normal(keys[1], (F, H), dtype=jnp.float64)
    Wz_h = s * jax.random.normal(keys[2], (H, H), dtype=jnp.float64)
    bz   = jnp.zeros((H,), dtype=jnp.float64)
    Wr_i = s * jax.random.normal(keys[3], (F, H), dtype=jnp.float64)
    Wr_h = s * jax.random.normal(keys[4], (H, H), dtype=jnp.float64)
    br   = jnp.zeros((H,), dtype=jnp.float64)
    Wh_i = s * jax.random.normal(keys[5], (F, H), dtype=jnp.float64)
    Wh_h = s * jax.random.normal(keys[6], (H, H), dtype=jnp.float64)
    bh   = jnp.zeros((H,), dtype=jnp.float64)

    # LayerNorm params on hidden
    ln_gamma = jnp.ones((H,), dtype=jnp.float64)
    ln_beta = jnp.zeros((H,), dtype=jnp.float64)

    # Fee head: H → 2 (bid, ask) → tanh → bounded
    Wo = s * jax.random.normal(keys[7], (H, 2), dtype=jnp.float64)
    bo = jnp.zeros((2,), dtype=jnp.float64)

    return {
        "Wf": Wf, "bf": bf,
        "Wz_i": Wz_i, "Wz_h": Wz_h, "bz": bz,
        "Wr_i": Wr_i, "Wr_h": Wr_h, "br": br,
        "Wh_i": Wh_i, "Wh_h": Wh_h, "bh": bh,
        "ln_gamma": ln_gamma, "ln_beta": ln_beta,
        "Wo": Wo, "bo": bo,
    }


def param_count(params: dict) -> int:
    return int(sum(int(jnp.size(v)) for v in jax.tree_util.tree_leaves(params)))


def initial_hidden_state(cfg: NNPolicyConfig) -> jnp.ndarray:
    return jnp.zeros((cfg.hidden_dim,), dtype=jnp.float64)


# ------------------------------------------------------------------ features
def _trade_features(trade: dict) -> jnp.ndarray:
    """Stable normalized features extracted from a single trade event.

    The diff sim passes a dict with keys: amount_x, amount_y, is_buy, timestamp,
    reserve_x, reserve_y. We turn these into 8 features the NN sees.
    """
    rx = jnp.maximum(trade["reserve_x"], 1e-9)
    ry = jnp.maximum(trade["reserve_y"], 1e-9)
    spot = ry / rx
    size_y_ratio = trade["amount_y"] / ry
    size_x_ratio = trade["amount_x"] / rx
    is_buy = trade["is_buy"]  # 0.0 or 1.0
    side_signed = 2.0 * is_buy - 1.0
    # log reserves recentered (initial pool is (100, 10_000) at spot 100)
    log_rx = jnp.log(rx / 100.0)
    log_ry = jnp.log(ry / 10000.0)
    log_spot = jnp.log(spot / 100.0)
    # timestamp normalized into [0, 1] for a 10k-step tape (loose; we just want O(1))
    t_norm = trade["timestamp"] / 10000.0
    return jnp.stack(
        [
            size_y_ratio,
            size_x_ratio,
            is_buy,
            side_signed,
            log_rx,
            log_ry,
            log_spot,
            t_norm,
        ]
    )


# --------------------------------------------------------------- GRU forward
def _layernorm(x: jnp.ndarray, gamma: jnp.ndarray, beta: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    mu = jnp.mean(x)
    var = jnp.mean((x - mu) ** 2)
    return gamma * (x - mu) / jnp.sqrt(var + eps) + beta


def make_nn_after_event(cfg: NNPolicyConfig) -> Callable:
    """Build an `after_event(params, state, trade)` closure for the given config."""
    base = jnp.asarray(cfg.base_fee, dtype=jnp.float64)
    amp = jnp.asarray(cfg.fee_amplitude, dtype=jnp.float64)
    fmin = jnp.asarray(cfg.min_fee, dtype=jnp.float64)
    fmax = jnp.asarray(cfg.max_fee, dtype=jnp.float64)

    def after_event(params: dict, state: jnp.ndarray, trade: dict):
        x = _trade_features(trade)
        # Feature embedder
        feat = jnp.tanh(x @ params["Wf"] + params["bf"])
        # GRU cell
        z = jax.nn.sigmoid(feat @ params["Wz_i"] + state @ params["Wz_h"] + params["bz"])
        r = jax.nn.sigmoid(feat @ params["Wr_i"] + state @ params["Wr_h"] + params["br"])
        h_tilde = jnp.tanh(feat @ params["Wh_i"] + (r * state) @ params["Wh_h"] + params["bh"])
        h_new = (1.0 - z) * state + z * h_tilde
        # LayerNorm for BPTT stability
        h_ln = _layernorm(h_new, params["ln_gamma"], params["ln_beta"])
        # Bounded fee head
        raw = h_ln @ params["Wo"] + params["bo"]
        fees = base + amp * jnp.tanh(raw)
        bid = jnp.clip(fees[0], fmin, fmax)
        ask = jnp.clip(fees[1], fmin, fmax)
        # Important: return the *un-normed* h_new as carry; LN is only on the
        # head's input. Carrying the LN'd vector would couple state magnitude
        # across timesteps in unhelpful ways.
        return h_new, bid, ask

    return after_event


def initial_fees(cfg: NNPolicyConfig) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Initial bid/ask BEFORE the first after_event call."""
    b = jnp.asarray(cfg.base_fee, dtype=jnp.float64)
    return b, b
