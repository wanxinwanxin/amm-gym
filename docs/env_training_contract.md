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
- `obs[ws]`: agent reserve X value normalized by initial portfolio value
- `obs[ws + 1]`: agent reserve Y normalized by initial portfolio value
- `obs[ws + 2]`: inventory imbalance in `[-1, 1]`
- `obs[ws + 3]`: cumulative agent edge normalized by initial portfolio value
- `obs[ws + 4]`: EMA of routed retail volume normalized by initial value
- `obs[ws + 5]`: EMA of retail order count normalized by arrival rate proxy
- `obs[ws + 6]`: EMA buy-ratio proxy
- `obs[ws + 7]`: current bid fee
- `obs[ws + 8]`: current ask fee
- `obs[ws + 9]`: rolling volatility estimate
- `obs[ws + 10]`: episode progress in `[0, 1]`

## Reward Contract

- Reward is the per-step change in agent edge:
  `reward_t = edge_t(submission) - edge_{t-1}(submission)`
- Episode reward sums to final agent edge.

## Info Contract

Training and evaluation code may rely on these `info` keys:

- `edge`
- `edge_normalizer`
- `pnl`
- `pnl_normalizer`
- `fair_price`
- `spot_price`
- `step`
- `bid_fee`
- `ask_fee`

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

