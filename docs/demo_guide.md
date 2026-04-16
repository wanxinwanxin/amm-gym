# Demo Guide

This repo includes three demo-facing artifacts:

- a mechanism explainer figure
- a strategy-flexibility comparison figure
- an animated single-episode rollout

## Install

```bash
python -m pip install -e .[demo]
```

## Mechanism Figure

```bash
python -m demo.depth_ladder_demo --plot demo/artifacts/mechanism.png
```

What it shows:

- fair price and venue spots
- the submission ladder at selected timestamps
- cumulative flow won by the submission venue versus the normalizer

## Strategy Flexibility Figure

```bash
python -m demo.strategy_flexibility_demo --output demo/artifacts/strategy_flexibility.png
```

What it shows:

- simple `2D` fee control on one side
- current `6D` ladder control on the other

The purpose is to show that this environment supports richer liquidity shaping
than a simple `(bid fee, ask fee)` control.

## Episode Animation

```bash
python -m demo.episode_animation --output demo/artifacts/trained_episode.gif
```

By default this:

- trains a small baseline policy
- rolls out one seeded evaluation episode
- writes a GIF showing market context, ladder state, cumulative edge, and
  cumulative retail/arbitrage volume by venue

Useful overrides:

- `--policy trained`
- `--policy balanced`
- `--policy aggressive_near_mid`
- `--frame-step 3`
- `--fps 6`
- `--iterations 5`

## Suggested Demo Sequence

1. Show `mechanism.png` to explain the simulator.
2. Show `strategy_flexibility.png` to explain the action space.
3. Show `trained_episode.gif` to explain how one trained rollout behaves.
