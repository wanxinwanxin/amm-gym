# amm-gym

`amm-gym` is a Gymnasium-compatible environment for AMM market-making research.
It models a learnable submission venue competing with a benchmark venue for
retail flow while both venues are exposed to arbitrage.

## What It Simulates

- A submission venue controlled by a `6D` ladder action each step
- A configurable benchmark venue, defaulting to a fixed `30 bps` constant-product AMM
- Retail routing across venues
- Arbitrage against both venues
- Hidden fair-price dynamics with optional within-episode volatility regimes
- Lagged observations designed for RL and policy benchmarking

## Start Here

- Setup and first run: [docs/getting_started.md](docs/getting_started.md)
- Train a baseline policy: [docs/training_quickstart.md](docs/training_quickstart.md)
- Generate demo visuals and animation: [docs/demo_guide.md](docs/demo_guide.md)
- Trainer-facing env API: [docs/env_training_contract.md](docs/env_training_contract.md)

## Install

```bash
python -m pip install -e .
python -m pip install -e .[demo,dev]
```

Use:
- base install for the environment package only
- `demo` for plots and animation output
- `dev` for tests

## Minimal Env Loop

```python
import numpy as np

from amm_gym import AMMFeeEnv
from amm_gym.sim.engine import SimConfig

env = AMMFeeEnv(config=SimConfig(n_steps=20))
obs, info = env.reset(seed=7)

while True:
    action = np.zeros(6, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

## Training In One Command

The repo includes a default CEM baseline trainer with a canonical demo config:

```bash
python -m training.run_baseline
```

Optional outputs:

```bash
python -m training.run_baseline \
  --objective edge_advantage \
  --summary-json demo/artifacts/baseline_summary.json \
  --plot demo/artifacts/training_curve.png \
  --comparison-plot demo/artifacts/trained_vs_baseline.png \
  --output demo/artifacts/baseline_run.npz
```

Research benchmark over heuristic policies:

```bash
python -m demo.run_policy_benchmark \
  --scenario regime_shift \
  --split test \
  --output demo/artifacts/policy_benchmark.json
```

Compare first-pass research candidates:

```bash
python -m training.run_first_pass --output demo/artifacts/first_pass.json
```

## Demo Artifacts

Generate the main demo visuals:

```bash
python -m demo.depth_ladder_demo --plot demo/artifacts/mechanism.png
python -m demo.strategy_flexibility_demo --output demo/artifacts/strategy_flexibility.png
python -m demo.episode_animation --output demo/artifacts/trained_episode.gif
```

## Core Interface

Current action semantics:
- `action[0]`: bid scale
- `action[1]`: bid decay
- `action[2]`: bid tilt
- `action[3]`: ask scale
- `action[4]`: ask decay
- `action[5]`: ask tilt

The observation is lagged by one step for price-derived information. The policy
does not receive the current hidden fair price or the current active volatility
regime directly.

## Notes

- The submission venue uses fixed internal bands at `2, 4, 8, 16, 32, 64, 128` bps.
- The default benchmark remains a fixed-fee constant-product AMM, but `SimConfig`
  can now swap in a depth-ladder benchmark via `benchmark_venue=VenueSpec(...)`.
- Training code should treat `AMMFeeEnv` and `SimConfig` as the public surface.
