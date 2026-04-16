# Training Quickstart

This repo already includes a working baseline trainer. A new user does not need
to supply a custom config, custom policy, or custom dataset to get started.

## Fastest Path

Install:

```bash
python -m pip install -e .[demo,dev]
```

Run the default baseline trainer:

```bash
python training/run_baseline.py
```

That command:

- builds the canonical hackathon scenario
- trains a small linear-tanh policy with CEM
- evaluates the learned policy on a seeded rollout
- prints a JSON summary including score history and final metrics

## Save Training Outputs

```bash
python training/run_baseline.py \
  --plot demo/artifacts/training_curve.png \
  --comparison-plot demo/artifacts/trained_vs_baseline.png \
  --output demo/artifacts/baseline_run.npz
```

Artifacts produced:

- `training_curve.png`: reward by training iteration
- `trained_vs_baseline.png`: one seeded comparison rollout
- `baseline_run.npz`: learned parameter vector and best score

## What A User Must Supply

If they use the built-in trainer:

- nothing mandatory beyond installing dependencies

If they bring their own RL loop:

- an `AMMFeeEnv`
- optionally a `SimConfig`
- a policy that maps observation to a `np.ndarray` action of shape `(6,)`

Action semantics:

- `action[0]`: bid scale
- `action[1]`: bid decay
- `action[2]`: bid tilt
- `action[3]`: ask scale
- `action[4]`: ask decay
- `action[5]`: ask tilt

## What The Trainer Uses By Default

The baseline trainer defaults are defined in
[`training/run_baseline.py`](/Users/xinwan/Github/amm-gym/training/run_baseline.py:1)
and the shared demo preset in
[`demo/presets.py`](/Users/xinwan/Github/amm-gym/demo/presets.py:1).

Default knobs you can override from the CLI:

- `--steps`
- `--window-size`
- `--population-size`
- `--elite-frac`
- `--iterations`
- `--eval-episodes`
- `--seed`

## Minimal Custom Trainer Shape

The public pattern is:

```python
import numpy as np

from amm_gym import AMMFeeEnv
from amm_gym.sim.engine import SimConfig

env = AMMFeeEnv(config=SimConfig(n_steps=120), window_size=10)
obs, info = env.reset(seed=7)

done = False
while not done:
    action = np.zeros(6, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

For the full trainer-facing contract, read [env_training_contract.md](env_training_contract.md).
