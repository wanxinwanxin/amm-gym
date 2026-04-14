# Project Coordination

This document is the coordination record for the active `amm-gym` worktrees.
It tracks shared goals, branch responsibilities, integration order, and the
current contract between implementation streams.

## Goals

1. Improve environment realism without leaking hidden or current-period price
   information to the policy.
2. Build a lightweight training client that can benchmark and eventually train
   baseline agents against the environment.
3. Keep `amm_gym/` as the reusable environment package and keep training code
   outside that package.

## Active Worktrees

### `main`

- Path: `/Users/xinwan/Github/amm-gym`
- Role: integration branch and shared documentation
- Responsibilities:
  - maintain the shared contract
  - receive reviewed changes from side branches
  - hold top-level coordination docs

### `env-realism`

- Path: `/Users/xinwan/Github/amm-gym-env-realism`
- Branch: `env-realism`
- Responsibilities:
  - environment timing semantics
  - observation/reward contract
  - simulator-facing realism improvements
  - environment and simulator tests

### `baseline-training`

- Path: `/Users/xinwan/Github/amm-gym-baseline-training`
- Branch: `baseline-training`
- Responsibilities:
  - training-side benchmark runners
  - training-side evaluation harnesses
  - eventual baseline learning code
  - training smoke tests

## Shared Contract

- Public environment class: `amm_gym.AMMFeeEnv`
- Gym API remains stable:
  - `reset(seed=..., options=...) -> (obs, info)`
  - `step(action) -> (obs, reward, terminated, truncated, info)`
- Training code may treat `amm_gym` as an external dependency but must not
  import private env internals for core control flow.
- The trainer-facing environment contract is documented in
  [`docs/env_training_contract.md`](/Users/xinwan/Github/amm-gym/docs/env_training_contract.md).
- Environment correctness/property tests belong in `env-realism`.
- Training worktree focuses on client behavior, benchmarks, and learner
  diagnostics rather than proving environment invariants.

## Integration Order

1. Land environment timing/observation fixes from `env-realism` once reviewed.
2. Rebase `baseline-training` onto the updated `main` if the env contract
   changes materially.
3. Land the fixed-fee benchmark harness from `baseline-training`.
4. Start the first learning pass only after the benchmark harness is stable.

## Current Risks

- The current `main` contract still differs from the in-progress lagged-timing
  environment worktree.
- Fee sweep benchmarks currently target the imperfect environment by design, so
  their results are diagnostic rather than economically final.
- A requested `0` basis-point policy is clipped to the environment minimum fee
  unless the env contract changes.

## Coordination Practice

- Shared docs track coordination.
- Per-worktree status docs track branch-local scope and progress.
- `git diff`, `git status`, and `git log` remain the source of truth for
  implementation detail.
