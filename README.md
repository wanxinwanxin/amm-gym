# amm-gym

`amm-gym` is a Gymnasium environment for AMM market-making research with a fixed
`30 bps` constant-product normalizer and a ladder-based submission venue that
posts depth across price-impact bands.

## What It Simulates

- A submission venue that chooses a 6D depth-control action each step
- A fixed `30 bps` normalizer AMM competing for the same retail flow
- Arbitrage against both venues
- Hidden fair-price dynamics with configurable within-episode volatility shifts
- Lagged, non-leaky observations intended for RL and policy benchmarking

## Install

```bash
python -m pip install -e .
python -m pip install -e .[demo]
```

The `demo` extra is only needed for plot output.

## Quickstart

```python
import numpy as np

from amm_gym import AMMFeeEnv
from amm_gym.sim.engine import SimConfig

env = AMMFeeEnv(config=SimConfig(n_steps=20))
obs, info = env.reset(seed=7)
action = np.zeros(6, dtype=np.float32)

for _ in range(20):
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

## Public Contract

The trainer-facing source of truth is [`docs/env_training_contract.md`](docs/env_training_contract.md).

Current action semantics:
- `action[0]`: bid scale
- `action[1]`: bid decay
- `action[2]`: bid tilt
- `action[3]`: ask scale
- `action[4]`: ask decay
- `action[5]`: ask tilt

The observation is lagged by one step for price-derived information. The policy
does not receive the current hidden fair price or trader-type labels directly.

## Run The Benchmark

Compare hand-authored ladder policies across constant and regime-shift volatility:

```bash
python demo/run_policy_benchmark.py --steps 120 --seeds 5
```

## Run The Demo

Run one short episode and optionally save plots:

```bash
python demo/depth_ladder_demo.py --policy aggressive_near_mid --schedule regime_shift
python demo/depth_ladder_demo.py --policy aggressive_near_mid --schedule regime_shift --plot demo/episode.png
```

## Builder Notes

- The submission venue uses fixed internal bands at `2, 4, 8, 16, 32, 64, 128` bps.
- The normalizer remains a fixed-fee constant-product benchmark.
- Volatility regimes are configurable through `SimConfig.volatility_schedule`.
- Training code should depend on `AMMFeeEnv` and `SimConfig`, not simulator internals.
