# Environment Training Contract

This document defines the trainer-facing contract for `amm_gym.AMMFeeEnv`.
Training code must treat this contract as the stable interface and should not
import simulator internals directly.

## Current Environment

- Env class: `amm_gym.AMMFeeEnv`
- Reset API: `reset(seed: int | None = None, options: dict | None = None)`
- Step API: `step(action: np.ndarray)`
- Episode end: `terminated` becomes `True` when `current_step >= n_steps`
- Truncation: currently always `False`

## Action Contract

- Action shape: `(2,)`
- Action meaning:
  - `action[0]`: bid fee in decimal units
  - `action[1]`: ask fee in decimal units
- Action bounds:
  - minimum fee: `0.0001`
  - maximum fee: `0.10`
- Out-of-range actions are clipped by the environment before application.

## Observation Contract

Observation dtype is `np.float32`. Shape is `(window_size + 11,)`.

For `ws = window_size`, the observation layout is:

- `obs[0:ws]`: recent log returns, left-padded with zeros
- `obs[ws]`: agent reserve X normalized by initial X
- `obs[ws + 1]`: agent reserve Y normalized by initial Y
- `obs[ws + 2]`: public reserve imbalance in `[-1, 1]`
- `obs[ws + 3]`: cumulative agent edge normalized by initial portfolio value, lagged by one step
- `obs[ws + 4]`: EMA of executed volume normalized by initial portfolio value
- `obs[ws + 5]`: EMA of execution count normalized by arrival-rate proxy
- `obs[ws + 6]`: EMA of signed net Y flow normalized by initial portfolio value
- `obs[ws + 7]`: current bid fee
- `obs[ws + 8]`: current ask fee
- `obs[ws + 9]`: rolling volatility estimate
- `obs[ws + 10]`: episode progress in `[0, 1]`

The observation does not expose hidden retail labels such as routed retail
count, retail volume, or buy-ratio proxies. Any price-dependent signal is
derived from the last completed step only.

## Reward Contract

- Reward is returned one step after the markout that generated it:
  `reward_t` is the realized edge change from the previous completed step.
- The first post-reset step returns zero reward.
- The final step flushes the last pending markout so episode reward still sums
  to final agent edge.

## Info Contract

Training and evaluation code may rely on these `info` keys for logging and
analysis only. They are diagnostic, not policy inputs.

- `edge`
- `edge_normalizer`
- `pnl`
- `pnl_normalizer`
- `spot_price`
- `step`
- `bid_fee`
- `ask_fee`
- `execution_count`
- `execution_volume_y`
- `net_flow_y`

## Seeding Contract

- `reset(seed=<int>)` must be reproducible.
- `reset()` without a seed must start a fresh episode and must not replay the
  last explicitly seeded trajectory by accident.

## Boundary Rules

- Training code may import:
  - `amm_gym.AMMFeeEnv`
  - `amm_gym.sim.engine.SimConfig`
  - general public helpers such as `amm_gym.baselines`
- Training code must not depend on private env state or simulator internals for
  core control flow.
- Any future breaking change to observation layout, reward semantics, or action
  meaning should update this document before the trainer is adapted.
