# Worktree Status: env-realism

## Branch

- Branch: `env-realism`
- Path: `/Users/xinwan/Github/amm-gym-env-realism`

## Scope

Implement environment timing and observation changes so the policy cannot use
current-period true fair price or hidden retail labels when making decisions.

## Contract Assumptions

- Gym `reset` and `step` signatures remain unchanged.
- At decision time `t`, the policy may only use fair-price-dependent
  information through `t-1`.
- Observation fields should be based on observable execution summaries and
  lagged price-dependent quantities only.
- Training-side invariant/property tests remain outside this branch unless they
  directly validate the environment contract.

## Progress

- [x] Remove current-fair-price leakage from observation timing
- [x] Delay markout/fair-price-dependent reward by one step
- [x] Replace retail-label observation fields with observable execution summaries
- [x] Update trainer-facing contract documentation
- [x] Update environment and cross-validation tests

## Blockers

- None currently reported

## Expected Changed Files

- `amm_gym/env.py`
- `amm_gym/sim/engine.py`
- `docs/env_training_contract.md`
- `tests/test_env.py`
- `tests/test_cross_validation.py`

## Handoff Notes

- Worker reported targeted tests passed with `47 passed, 1 skipped`.
- Worker reported full suite passed with `66 passed, 1 skipped`.
- This branch should be reviewed for exact timing semantics before merge to
  `main`.
