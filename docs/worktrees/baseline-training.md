# Worktree Status: baseline-training

## Branch

- Branch: `baseline-training`
- Path: `/Users/xinwan/Github/amm-gym-baseline-training`

## Scope

Implement training-client benchmark and evaluation code only. The first pass is
a fixed symmetric fee sweep used to diagnose how outcomes vary with fee in the
current environment.

## Contract Assumptions

- Public environment class: `amm_gym.AMMFeeEnv`
- Action remains `(bid_fee, ask_fee)`
- Required evaluation info keys currently include:
  - `edge`
  - `pnl`
- Benchmark code stays outside `amm_gym/`
- This branch intentionally benchmarks the current imperfect environment; it is
  not the place to prove environment invariants

## Progress

- [x] Add fixed-fee sweep benchmark helper
- [x] Add CLI entrypoint for running the benchmark
- [x] Add training-side smoke test for the benchmark harness
- [x] Report requested and applied fee levels because `0` bps is clipped by the
      current env minimum fee

## Blockers

- None currently reported

## Expected Changed Files

- `training/fixed_fee_benchmarks.py`
- `training/run_fixed_fee_benchmarks.py`
- `tests/test_training.py`

## Handoff Notes

- Worker reported `tests/test_training.py` passed.
- Worker reported full suite passed with `65 passed, 1 skipped`.
- If the environment contract changes after `env-realism` lands, rebase this
  branch and confirm benchmark outputs still make sense.
