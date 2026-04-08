# RL Agent for Dynamic AMM Fee Setting

## Goal

Train an RL agent to set bid/ask fees on a constant-product AMM, maximizing edge (profit from retail flow minus losses to arbitrageurs).

## Environment

Use the simulation engine from [benedictbrady/amm-challenge](https://github.com/benedictbrady/amm-challenge). It already implements:

- Constant-product AMM with fee-on-input
- Fair price following geometric Brownian motion (10,000 steps per episode)
- Rational arbitrageurs that trade the AMM back to fair price
- Retail order flow (Poisson arrivals, log-normal sizes) routed optimally across pools based on fees
- A "normalizer" AMM running fixed 30 bps fees that the agent competes against for retail flow
- Edge calculation (P&L measured against fair price at time of each trade)

Write a Gym wrapper around the existing Rust/Python simulation so the RL agent can interact with it step-by-step.

## Agent

Implement PPO from scratch in PyTorch (no stable-baselines or other RL libraries).

- **Observation space**: recent price returns, current reserves, inventory imbalance, realized edge so far, recent trade flow statistics (size, direction, frequency)
- **Action space**: continuous bid fee and ask fee (clipped to valid range)
- **Reward**: per-step change in edge

## Baselines

- Static 30 bps (the normalizer)
- Static 5 bps and static 100 bps (to show the fee tradeoff)
- A simple heuristic: widen fees after large trades, decay toward 30 bps

## Deliverables

1. Gym wrapper around the amm-challenge simulation engine
2. PPO implementation in PyTorch
3. Training script that runs on GPU
4. Evaluation script comparing the learned policy against baselines
5. Plots: training curves, learned fee dynamics over an episode, edge comparison table across baselines
6. README explaining the project, how to run it, and results
