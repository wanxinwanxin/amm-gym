# Environment Training Contract

This document defines the trainer-facing public contract for `amm_gym.AMMFeeEnv`.
Training code should depend on this contract, not simulator internals.

## Current Environment

- Env class: `amm_gym.AMMFeeEnv`
- Reset API: `reset(seed: int | None = None, options: dict | None = None)`
- Step API: `step(action: np.ndarray)`
- Episode end: `terminated` becomes `True` when `current_step >= n_steps`
- Truncation: currently always `False`

## Venue Model

- Submission venue: depth-only ladder with fixed internal bands
- Normalizer venue: fixed `30 bps` constant-product AMM
- Submission bands are fixed at:
  `2, 4, 8, 16, 32, 64, 128 bps` from the lagged step reference price
- The submission ladder is posted once per step and then used by both
  arbitrage and retail routing during that step
- Volatility can remain constant for the full episode or change according to a
  schedule supplied through `SimConfig.volatility_schedule`
- The active volatility regime is hidden from the policy

## Action Contract

- Action shape: `(6,)`
- Action range: each component is clipped to `[-1.0, 1.0]`
- Action meaning:
  - `action[0]`: bid-side scale
  - `action[1]`: bid-side decay
  - `action[2]`: bid-side tilt
  - `action[3]`: ask-side scale
  - `action[4]`: ask-side decay
  - `action[5]`: ask-side tilt
- The environment maps this 6D control to positive bid/ask band depths
- The action does not directly set submission fees
- The normalizer is not controlled by the action

## Volatility Schedule Contract

- `SimConfig.gbm_sigma` remains the default constant-volatility input
- `SimConfig.volatility_schedule` may be set to:
  `((start_step_0, sigma_0), (start_step_1, sigma_1), ...)`
- If no schedule is provided, the environment behaves as constant-volatility GBM
- If a schedule is provided, the step uses the last schedule entry whose
  `start_step <= current_step`
- The schedule must be deterministic from config and seed
- The observation and `info` do not expose the current regime label or sigma directly

## Observation Contract

Observation dtype is `np.float32`. Shape is `(window_size + 15,)`.

For `ws = window_size`, the observation layout is:

- `obs[0:ws]`: lagged log returns, left-padded with zeros
- `obs[ws]`: submission reserve X normalized by initial X
- `obs[ws + 1]`: submission reserve Y normalized by initial Y
- `obs[ws + 2]`: submission reserve imbalance
- `obs[ws + 3]`: lagged submission edge normalized by initial value
- `obs[ws + 4]`: EMA of realized execution volume normalized by initial value
- `obs[ws + 5]`: EMA of realized execution count normalized by arrival-rate proxy
- `obs[ws + 6]`: EMA of realized signed net flow normalized by initial value
- `obs[ws + 7:ws + 13]`: last posted action
- `obs[ws + 13]`: rolling volatility estimate from lagged returns
- `obs[ws + 14]`: episode progress in `[0, 1]`

Builder-facing guarantees:

- the policy does not receive the current hidden fair price directly
- the policy does not receive the current hidden volatility regime directly
- price-derived quantities are lagged by one step
- privileged retail labels are not part of the observation

## Reward Contract

- Reward is a one-step-delayed change in submission edge
- The delay avoids exposing same-step markout information at decision time
- Episode reward still sums to final submission edge

## Info Contract

Training and evaluation code may rely on these `info` keys:

- `edge`
- `edge_normalizer`
- `pnl`
- `pnl_normalizer`
- `spot_price`
- `spot_price_normalizer`
- `step`
- `execution_count`
- `execution_count_normalizer`
- `execution_volume_y`
- `execution_volume_y_normalizer`
- `retail_volume_y`
- `retail_volume_y_normalizer`
- `arb_volume_y`
- `arb_volume_y_normalizer`
- `net_flow_y`
- `ask_near_depth_y`
- `ask_far_depth_y`
- `bid_near_depth_y`
- `bid_far_depth_y`

`info` is for logging and diagnostics, not hidden-state leakage. It does not
include the current hidden fair price or current active sigma.

## Seeding Contract

- `reset(seed=<int>)` must be reproducible
- `reset()` without a seed must start a fresh episode and must not replay the
  last explicitly seeded trajectory by accident

## Boundary Rules

- Training code may import:
  - `amm_gym.AMMFeeEnv`
  - `amm_gym.sim.engine.SimConfig`
  - public helpers such as `amm_gym.baselines`
- Training code must not depend on private env state or simulator internals for
  core control flow
- Any future breaking change to action meaning, observation layout, or reward
  timing should update this document before downstream trainer code is adapted
