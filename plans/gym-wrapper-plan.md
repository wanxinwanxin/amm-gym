# Plan: Gym Wrapper for AMM Fee-Setting RL Agent

## Executive Summary

Wrap the amm-challenge simulation as a standard Gymnasium environment so an RL agent can set bid/ask fees step-by-step. The upstream Rust engine runs full episodes atomically (fees come from compiled EVM bytecode), so we **reimplement the core simulation loop in Python** — the math is straightforward (CPMM, GBM, Poisson arrivals, optimal arbitrage) and this gives us full step-by-step control.

**Complexity**: Medium  
**Estimated effort**: 2-3 days  

---

## Key Architectural Decision

### Why reimplement in Python instead of wrapping the Rust engine?

The Rust `run_single()` takes compiled Solidity bytecode and runs all 10,000 steps internally — there's no way to pause mid-episode and ask Python for fees. Our options:

| Approach | Pros | Cons |
|----------|------|------|
| **A. Reimplement sim loop in Python** | Full control, no fork needed, easy to debug | Slower than Rust, must match math exactly |
| B. Fork & modify Rust engine | Fast execution | Heavy lift, must maintain fork, complex FFI for step-by-step |
| C. Hack via EVM callback | Uses existing engine | Insane complexity, latency per step |

**Decision: Approach A.** The simulation math is simple (CPMM formula, GBM, Poisson). Python perf is fine for training — a 10k-step episode is ~50ms in NumPy. We validate correctness by comparing outputs against the Rust engine on identical seeds.

---

## Task Breakdown

### Phase 1: Core Simulation Engine (Python)

#### 1.1 Constant-Product AMM (`amm_gym/sim/amm.py`)
- Implement CPMM with fee-on-input (bid fee for buys, ask fee for sells)
- Track reserves (x, y), collected fees, and k invariant
- Methods: `swap_x_in(amount, fee)`, `swap_y_in(amount, fee)`, `spot_price()`, `edge_vs_fair(fair_price)`
- Use the same math as `amm_sim_rs/src/amm/cfmm.rs`

#### 1.2 Price Process (`amm_gym/sim/price.py`)
- Geometric Brownian Motion: `S(t+1) = S(t) * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z)`
- Seeded RNG for reproducibility
- Parameters: mu, sigma, dt (match upstream defaults)

#### 1.3 Market Actors (`amm_gym/sim/actors.py`)
- **Arbitrageur**: Given fair price and AMM state, compute optimal trade to push spot → fair price. Closed-form: `delta_x = x - sqrt(k / (gamma * p))` where gamma = 1 - fee
- **Retail flow**: Poisson arrivals (rate λ), log-normal sizes (mean, sigma), 50/50 buy/sell
- **Router**: Split retail orders optimally between agent AMM and normalizer AMM based on fees (equalize marginal prices)

#### 1.4 Simulation Step (`amm_gym/sim/engine.py`)
- One step = (1) update fair price, (2) arb trades on both AMMs, (3) generate retail orders, (4) route retail across AMMs
- Track per-step: fair_price, spot_prices, reserves, fees, PnL, edge, trade counts/volumes
- The agent's AMM uses dynamic fees (from RL action); normalizer uses fixed 30 bps

### Phase 2: Gymnasium Environment

#### 2.1 Environment Class (`amm_gym/env.py`)
```python
class AMMFeeEnv(gymnasium.Env):
    observation_space = Box(...)  # ~15-20 dims
    action_space = Box(low=0.0001, high=0.10, shape=(2,))  # (bid_fee, ask_fee)
```

**Observation vector** (all normalized):
| Feature | Description |
|---------|-------------|
| `price_returns[0:5]` | Last 5 log-returns of fair price |
| `reserve_x` | Agent AMM X reserves (normalized) |
| `reserve_y` | Agent AMM Y reserves (normalized) |
| `inventory_imbalance` | `(reserve_x * fair_price - reserve_y) / (reserve_x * fair_price + reserve_y)` |
| `edge_so_far` | Cumulative edge (normalized by initial value) |
| `recent_trade_volume` | EMA of total trade volume (last N steps) |
| `recent_trade_count` | EMA of trade count |
| `recent_buy_ratio` | Fraction of recent trades that were buys |
| `current_bid_fee` | Current bid fee |
| `current_ask_fee` | Current ask fee |
| `volatility_estimate` | Rolling std of recent returns |
| `step_fraction` | Current step / total steps |

**Action**: `[bid_fee, ask_fee]` clipped to `[1 bps, 1000 bps]` (0.0001 to 0.10)

**Reward**: `edge(t) - edge(t-1)` — per-step change in edge

**Episode**: 10,000 steps. Terminates at end. No early termination.

#### 2.2 Config & Reset (`amm_gym/env.py`)
- `__init__` accepts `SimConfig` dataclass with all simulation parameters
- `reset()` re-initializes both AMMs, price process, RNG seed
- Support deterministic seeding for reproducibility
- Optional config randomization per episode (like `HyperparameterVariance` upstream)

#### 2.3 Registration
- Register as `AMMFee-v0` with Gymnasium

### Phase 3: Validation & Testing

#### 3.1 Unit Tests (`tests/`)
- `test_amm.py`: CPMM math (swap amounts, fee collection, k invariant)
- `test_price.py`: GBM distribution properties, seeding
- `test_actors.py`: Arbitrageur pushes spot to fair price, retail distribution
- `test_env.py`: Env API compliance (reset, step, observation/action spaces)

#### 3.2 Cross-Validation Against Rust Engine
- Run identical configs + seed through both Python sim and Rust `run_single()`
- Compare: fair prices, spot prices, PnL, edge at each step
- Accept < 1e-6 relative error (floating point differences)

### Phase 4: Baselines (for quick smoke-testing)

#### 4.1 Static Fee Baselines (`amm_gym/baselines.py`)
- Wrap static fee policies (5 bps, 30 bps, 100 bps) as callables compatible with env
- Simple heuristic: widen fees after large trades, decay toward 30 bps

#### 4.2 Evaluation Script (`scripts/evaluate_baselines.py`)
- Run each baseline for N episodes, report mean/std edge
- Confirms the env produces sensible results before RL training

---

## File Structure

```
amm-gym/
├── amm_gym/
│   ├── __init__.py
│   ├── env.py                  # AMMFeeEnv (Gymnasium)
│   ├── baselines.py            # Static & heuristic fee policies
│   └── sim/
│       ├── __init__.py
│       ├── amm.py              # Constant-product AMM
│       ├── price.py            # GBM price process
│       ├── actors.py           # Arbitrageur, retail flow, router
│       └── engine.py           # SimulationEngine (step-by-step)
├── tests/
│   ├── test_amm.py
│   ├── test_price.py
│   ├── test_actors.py
│   └── test_env.py
├── scripts/
│   └── evaluate_baselines.py
├── pyproject.toml              # Dependencies: gymnasium, numpy, torch
└── project_description.md
```

---

## Dependencies

```
gymnasium >= 0.29
numpy >= 1.24
```

Optional (for cross-validation):
```
amm_sim_rs  # pip install from amm-challenge repo
```

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Python sim doesn't match Rust math | Medium | High | Cross-validate on identical seeds; unit test each component |
| Router logic (optimal splitting) is complex | Medium | Medium | Start with simpler 50/50 split, upgrade to optimal later |
| Observation normalization affects training | Medium | Medium | Use running statistics (like VecNormalize) or fixed scaling |
| 10k steps/episode is slow in Python | Low | Medium | NumPy vectorization; profile if needed. Still ~50ms/episode |

---

## Implementation Order

1. **`sim/amm.py`** — core AMM math (can unit test immediately)
2. **`sim/price.py`** — GBM (independent, test immediately)
3. **`sim/actors.py`** — arbitrageur + retail (depends on AMM)
4. **`sim/engine.py`** — tie it together into step-by-step sim
5. **`env.py`** — Gymnasium wrapper around engine
6. **`tests/`** — unit + integration tests
7. **`baselines.py`** + **`scripts/evaluate_baselines.py`** — smoke test
8. Cross-validate against Rust engine (if `amm_sim_rs` is installable)

---

## Success Criteria

- [x] `env.reset()` and `env.step(action)` work with correct spaces
- [x] Static 30 bps policy produces ~0 edge (competing evenly with normalizer)
- [x] Lower fees capture more retail flow but lose more to arb (visible in baselines)
- [x] Higher fees capture less retail flow (visible in baselines)
- [x] Episode PnL/edge distributions are reasonable and reproducible with seeds
- [x] All unit tests pass
