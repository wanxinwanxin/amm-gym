# Getting Started

This repo is usable in two modes:

- as a Gymnasium-compatible environment package
- as a small baseline-training and demo harness around that package

## Install

Base environment install:

```bash
python -m pip install -e .
```

With demo and test extras:

```bash
python -m pip install -e .[demo,dev]
```

What the extras are for:

- `demo`: matplotlib-based figures and GIF animation
- `dev`: `pytest` and test tooling

## First Environment Run

```python
import numpy as np

from amm_gym import AMMFeeEnv
from amm_gym.sim.engine import SimConfig

env = AMMFeeEnv(config=SimConfig(n_steps=20), window_size=10)
obs, info = env.reset(seed=7)

for _ in range(20):
    action = np.zeros(6, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

What you need to know:

- action shape is `(6,)`
- each action component is clipped to `[-1, 1]`
- `terminated=True` when the episode reaches `n_steps`

## Repo Entry Points

- Environment API: [env_training_contract.md](env_training_contract.md)
- Baseline training CLI: [`training/run_baseline.py`](/Users/xinwan/Github/amm-gym/training/run_baseline.py:1)
- Mechanism demo: [`demo/depth_ladder_demo.py`](/Users/xinwan/Github/amm-gym/demo/depth_ladder_demo.py:1)
- Strategy flexibility visual: [`demo/strategy_flexibility_demo.py`](/Users/xinwan/Github/amm-gym/demo/strategy_flexibility_demo.py:1)
- Episode animation: [`demo/episode_animation.py`](/Users/xinwan/Github/amm-gym/demo/episode_animation.py:1)

## Verify The Install

Run tests:

```bash
python -m pytest -q
```

Generate a mechanism plot:

```bash
python -m demo.depth_ladder_demo --plot demo/artifacts/mechanism.png
```

If that works, the repo is in a usable state.
