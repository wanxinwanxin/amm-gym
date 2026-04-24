# Simple AMM Mathematical Primer

This note formalizes the simulator and training stack implemented under
`arena_eval/diff_simple_amm` and the related trainer modules.

It covers:

- the exact rollout law shared by the challenge and realistic evaluators
- the reward and score definitions
- the compact submission policy as a parameterized control law
- the smooth differentiable surrogate used for gradient-based optimization
- the gradient-free and PPO training objectives used elsewhere in the repo

The code references for the definitions below are:

- rollout and accounting: `arena_eval/diff_simple_amm/simulator.py`
- AMM mechanics: `arena_eval/diff_simple_amm/amm.py`
- arbitrage: `arena_eval/diff_simple_amm/arb.py`
- retail routing: `arena_eval/diff_simple_amm/router.py`
- challenge dynamics tape: `arena_eval/diff_simple_amm/challenge_dynamics.py`
- realistic dynamics tape: `arena_eval/diff_simple_amm/realistic_dynamics.py`
- smooth differentiable objective: `arena_eval/diff_simple_amm/objectives.py`
- compact policy: `arena_eval/diff_simple_amm/policies.py`
- trainer-side RL/search objectives: `training/algorithms/cem.py`,
  `training/algorithms/ppo.py`, and `training/eval/metrics.py`

## 1. State and Venues

At each discrete time step $t = 0, 1, \ldots, T-1$, the simulator maintains:

- a hidden fair price $F_t > 0$
- a submission AMM state
  $S_t^{sub} = (x_t^{sub}, y_t^{sub}, f_{b,t}^{sub}, f_{a,t}^{sub}, \phi_{x,t}^{sub}, \phi_{y,t}^{sub})$
- a normalizer AMM state
  $S_t^{nor} = (x_t^{nor}, y_t^{nor}, f_{b,t}^{nor}, f_{a,t}^{nor}, \phi_{x,t}^{nor}, \phi_{y,t}^{nor})$

where:

- $x_t$ is reserve of asset $X$
- $y_t$ is reserve of asset $Y$
- $f_b$ is the bid-side fee, used when the AMM buys $X$
- $f_a$ is the ask-side fee, used when the AMM sells $X$
- $\phi_x, \phi_y$ are accumulated fees

Each venue is a constant-product AMM with invariant

$k_t = x_t y_t$.

Its spot price is

$P_t = y_t / x_t$.

The simulator evaluates two venues against the same exogenous tape. The
submission venue is the one being optimized. The normalizer is a fixed
reference policy unless otherwise specified.

## 2. Exact AMM Trade Map

Let $\gamma_b = 1 - f_b$ and $\gamma_a = 1 - f_a$.

### 2.1 AMM buys $X$ and pays out $Y$

This corresponds to a trader selling $X$ into the venue. If gross input is
$\Delta x > 0$, the net amount that reaches reserves is $\gamma_b \Delta x$.

The exact state update is:

- $x' = x + \gamma_b \Delta x$
- $y' = k / x'$
- $\Delta y = y - y'$
- fee in $X$: $f_b \Delta x$

So the venue pays out $\Delta y$ units of $Y$.

### 2.2 AMM sells $X$ and receives $Y$

This corresponds to a trader buying $X$ from the venue. If the trader wants
$\Delta x > 0$ units of $X$, then

- $x' = x - \Delta x$
- $y' = k / x'$
- net $Y$ added to reserves: $y' - y$
- gross $Y$ paid by trader:
  $\Delta y = (y' - y) / \gamma_a$
- fee in $Y$: $\Delta y - (y' - y)$

### 2.3 AMM sells $X$ against a gross $Y$ input

If the trader submits gross $Y$ amount $\Delta y > 0$, then the reserve sees
net input $\gamma_a \Delta y$:

- $y' = y + \gamma_a \Delta y$
- $x' = k / y'$
- $\Delta x = x - x'$

These are exactly the maps in `amm.py`.

## 3. One-Step Event Order

Each time step applies events in this order:

1. Exogenous fair price update: $F_t \to F_{t+1}$
2. Arbitrage against submission venue
3. Arbitrage against normalizer venue
4. Decode exogenous retail orders for the step
5. Route each retail order across the two venues
6. Update cumulative edge, volumes, fees, and terminal mark-to-market stats

This ordering matters. Retail interacts with post-arbitrage pools, so the AMMs
are first re-anchored toward the new fair price and only then compete for
flow.

## 4. Fair-Price Dynamics

## 4.1 Challenge evaluator

The challenge evaluator uses geometric Brownian motion increments:

$F_{t+1} = F_t \exp((\mu - \tfrac12 \sigma^2)\Delta t + \sigma \sqrt{\Delta t} Z_t)$

with $Z_t \sim \mathcal{N}(0,1)$ from an explicit tape.

The code uses:

- $\mu = \text{gbm\_mu}$
- $\sigma = \text{gbm\_sigma}$
- $\Delta t = \text{gbm\_dt}$

The randomness is materialized in `ChallengeTape.gbm_normals`.

## 4.2 Realistic evaluator

The realistic evaluator replaces GBM with an empirical regime-switching model.

Let $R_t$ be a latent regime in $\{1, \ldots, K\}$ with transition matrix $Q$. Given
the previous regime, the next regime is sampled by

$R_{t+1} \sim Q(R_t, \cdot)$.

Conditional on regime $r$, a percentile draw $U_t \sim \mathrm{Uniform}(0,100)$ is mapped
through that regime's empirical inverse CDF:

$\ell_t = G_r^{-1}(U_t)$.

The fair price then evolves as

$F_{t+1} = F_t \exp(\ell_t)$.

Here $G_r^{-1}$ is loaded from `analysis/weth_usdc_90d/regimes_invcdf.csv`.

## 5. Retail Flow Model

## 5.1 Challenge evaluator

At each step, the number of retail orders is

$N_t \sim \mathrm{Poisson}(\lambda)$.

For each order $j = 1, \ldots, N_t$:

- size $Q_{t,j} \sim \mathrm{LogNormal}(\mu_{LN}, \sigma_{LN}^2)$
- side $B_{t,j} \sim \mathrm{Bernoulli}(p_{buy})$

where $\mu_{LN}$ is chosen so that the lognormal has mean
`retail_mean_size`.

If $B_{t,j} = 1$, the order is a buy with gross $Y$ size $Q_{t,j}$.
If $B_{t,j} = 0$, the order is a sell with notional $Y$ size $Q_{t,j}$, which
the router converts into $X$ size using the current fair price.

## 5.2 Realistic evaluator

Again $N_t \sim \mathrm{Poisson}(\lambda)$, but each order is generated from an empirical
impact distribution instead of a direct size distribution.

For each order, draw an impact log-return $I_{t,j}$ from an empirical inverse
CDF. The sign of $I_{t,j}$ determines side:

- $I_{t,j} > 0$: buy order
- $I_{t,j} < 0$: sell order

The magnitude is converted into an order size by solving the constant-product
impact formula against a reference venue state.

For buys:

$Q^Y_{t,j} = y^{ref}_t \frac{\exp(|I_{t,j}|/2)-1}{1-f_a^{ref}}$

For sells:

$Q^X_{t,j} = x^{ref}_t \frac{\exp(|I_{t,j}|/2)-1}{1-f_b^{ref}}$

and the sell order is represented in $Y$ notional as

$Q^Y_{t,j} = F_t Q^X_{t,j}$.

Depending on config, the reference venue is either the normalizer, the
submission venue, or the initial state.

## 6. Arbitrage

Each venue is arbitraged independently against the current fair price.

## 6.1 If the venue is cheap: $P_t < F_t$

The arbitrageur buys $X$ from the AMM until the post-trade spot equals the fair
price. Solving

$k / (x - \Delta x)^2 = \gamma_a F_t$

gives

$\Delta x^* = x - \sqrt{k / (\gamma_a F_t)}$

whenever this is positive.

The arbitrageur pays gross $Y$ amount $\Delta y$, then realizes profit

$\Pi^{arb,buy} = \Delta x^* F_t - \Delta y$.

## 6.2 If the venue is rich: $P_t > F_t$

The arbitrageur sells $X$ into the AMM until the post-trade spot equals the
fair price. Solving

$k / (x + \gamma_b \Delta x)^2 = F_t / \gamma_b$

gives

$\Delta x^* = \frac{\sqrt{k \gamma_b / F_t} - x}{\gamma_b}$

whenever this is positive.

The AMM pays out $\Delta y$, and arbitrage profit is

$\Pi^{arb,sell} = \Delta y - \Delta x^* F_t$.

Submission and normalizer edges are both reduced by the arbitrage profit
extracted from them.

## 7. Retail Routing Between Two AMMs

Retail orders are routed to maximize execution quality across the submission
venue and the normalizer.

## 7.1 Buy order split

Suppose a buy order brings total gross $Y$ amount $Q^Y$. Let

- $\gamma_1 = 1 - f_{a,1}$
- $\gamma_2 = 1 - f_{a,2}$
- $a_i = \sqrt{x_i \gamma_i y_i}$

Then the exact split to venue 1 is

$Q_1^Y = \frac{(a_1/a_2)(y_2 + \gamma_2 Q^Y) - y_1}{\gamma_1 + (a_1/a_2)\gamma_2}$

clipped to $[0, Q^Y]$, and $Q_2^Y = Q^Y - Q_1^Y$.

## 7.2 Sell order split

For a sell order with total $X$ amount $Q^X$, let

- $\gamma_1 = 1 - f_{b,1}$
- $\gamma_2 = 1 - f_{b,2}$
- $b_i = \sqrt{y_i \gamma_i x_i}$

Then the exact split to venue 1 is

$Q_1^X = \frac{(b_1/b_2)(x_2 + \gamma_2 Q^X) - x_1}{\gamma_1 + (b_1/b_2)\gamma_2}$

clipped to $[0, Q^X]$, and $Q_2^X = Q^X - Q_1^X$.

These formulas are the exact optimizer for the two-venue constant-product
problem encoded in `router.py`.

## 8. Edge, PnL, Score, and Reward

## 8.1 Trade-level edge

For a trade executed at fair price $F_t$:

- if the AMM buys $X$ from the trader, edge is
  $\Delta x F_t - \Delta y$
- if the AMM sells $X$ to the trader, edge is
  $\Delta y - \Delta x F_t$

This is the venue's instantaneous economic surplus relative to fair value.

## 8.2 Venue edge process

For venue $v \in \{\mathrm{sub}, \mathrm{nor}\}$, cumulative edge is

$$E_T^v = \sum_{\text{retail trades}} e_{t,j}^v - \sum_{\text{arb trades}} \Pi_{t}^{\mathrm{arb},v}.$$

Retail adds edge. Arbitrage removes it.

## 8.3 Terminal PnL

Let final fair price be $F_T$. Terminal marked value is

$V_T^v = x_T^v F_T + y_T^v + \phi_{x,T}^v F_T + \phi_{y,T}^v$.

If initial marked value is

$V_0 = x_0 F_0 + y_0$,

then terminal PnL is

$\mathrm{PnL}_T^v = V_T^v - V_0$.

## 8.4 Primary score

The differentiable simulator uses

$\text{score} = E_T^{sub}$.

It also reports:

- edge advantage: $E_T^{sub} - E_T^{nor}$
- PnL advantage: $\mathrm{PnL}_T^{sub} - \mathrm{PnL}_T^{nor}$

## 8.5 Gym reward

In `amm_gym.env`, the step reward is a one-step delayed increment in submission
edge:

$r_t = E_{t-1}^{sub} - E_{t-2}^{sub}$

with the last pending increment paid at episode termination, so the total
episode reward telescopes to

$\sum_t r_t = E_T^{sub}$.

That means the environment reward is aligned with the simulator's submission
edge score.

## 9. Compact Submission Policy

The differentiable stack exposes a compact parameter vector

$\theta \in \mathbb{R}^{20}$

with named components

$$
(\text{base\_fee}, \text{min\_fee}, \text{max\_fee}, \text{flow\_fast\_decay}, \text{flow\_slow\_decay}, \text{size\_fast\_decay}, \text{size\_slow\_decay}, \text{gap\_fast\_decay}, \text{gap\_slow\_decay}, \text{toxicity\_decay}, \text{toxicity\_weight}, \text{base\_spread}, \text{flow\_mid\_weight}, \text{size\_mid\_weight}, \text{gap\_mid\_weight}, \text{skew\_weight}, \text{toxicity\_side\_weight}, \text{hot\_gap\_threshold}, \text{big\_trade\_threshold}, \text{hot\_fee\_bump}).
$$


The policy keeps a state of exponentially decayed statistics:

- fast and slow buy/sell flow
- fast and slow trade size
- fast and slow inter-trade gap
- bid-side and ask-side toxicity proxies
- lagged trade-side metadata

After each event, it updates those statistics and emits new bid/ask fees.

Ignoring clamps for a moment, the fee law is:

$$
\begin{aligned}
\mathrm{mid}_t &= \theta_0 + \theta_{12}(\text{buy\_flow\_fast} + \text{sell\_flow\_fast}) \\
&\quad + \theta_{13}\max(\text{size\_fast} - 0.5\,\text{size\_slow},\, 0) \\
&\quad + \theta_{14}\max(\theta_{17} - \text{gap\_fast},\, 0) + \theta_{19}\,\mathbb{1}_{\text{hot or big}} \\
\mathrm{spread}_t &= \theta_{11} + 0.5\,\theta_{19}\,\mathbb{1}_{\text{hot}} \\
\mathrm{skew}_t &= \theta_{15}\bigl((\text{buy\_flow\_fast} - \text{sell\_flow\_fast}) + 0.5(\text{buy\_flow\_slow} - \text{sell\_flow\_slow})\bigr) \\
\mathrm{bid}_t &= \mathrm{clip}\bigl(\mathrm{mid}_t + 0.5\,\mathrm{spread}_t - \mathrm{skew}_t + \theta_{16}\,\text{tox\_bid},\,\theta_1,\,\theta_2\bigr) \\
\mathrm{ask}_t &= \mathrm{clip}\bigl(\mathrm{mid}_t + 0.5\,\mathrm{spread}_t + \mathrm{skew}_t + \theta_{16}\,\text{tox\_ask},\,\theta_1,\,\theta_2\bigr)\text{.}
\end{aligned}
$$

So optimization means choosing $\theta$ so that this adaptive fee controller
maximizes expected submission performance.

## 10. Optimization Problems

## 10.1 Exact-path policy search

For a fixed evaluator law and random seed $\omega$, the exact simulator induces
an objective

$J_{\text{exact}}(\theta; \omega) = E_T^{sub}(\theta; \omega)$.

Over a seed set $\Omega$, the finite-sample training objective is

$\hat{J}(\theta) = |\Omega|^{-1} \sum_{\omega \in \Omega} J_{\text{exact}}(\theta; \omega)$.

Because the exact simulator contains discrete events, clipping, routing
decisions, and piecewise branches, it is treated as non-differentiable in
practice. The repo therefore uses gradient-free search for exact-path training.

## 10.2 CEM

The cross-entropy method in `training/algorithms/cem.py` optimizes a policy
parameter vector $\theta$ by maintaining a factorized Gaussian search
distribution:

$\theta \sim \mathcal{N}\bigl(m_k,\, \operatorname{diag}(s_k^2)\bigr)$.

At iteration $k$:

1. sample a population ${\theta_i}$
2. score each candidate by mean episode objective over a set of seeds
3. keep the top $\rho$ elite fraction
4. set the next search mean and std to the elite sample mean and std

Formally, if $\mathcal{E}_k$ is the elite set,

$m_{k+1} = |\mathcal{E}_k|^{-1} \sum_{\theta \in \mathcal{E}_k} \theta$

$s_{k+1}^2 = |\mathcal{E}_k|^{-1} \sum_{\theta \in \mathcal{E}_k} (\theta - m_{k+1})^2$

with a small floor on $s$.

This is gradient-free.

## 10.3 PPO on the Gym environment

The PPO trainer does not optimize the compact diff policy directly. It trains a
stochastic policy

$\pi_\psi(a_t | o_t)$

over the public Gym observation $o_t$ and action $a_t$.

The actor emits Gaussian pre-activations:

$u_t \sim \mathcal{N}\bigl(\mu_\psi(o_t),\, \operatorname{diag}(\sigma_\psi^2)\bigr)$

and actions are squashed by

$a_t = \tanh(u_t)$.

The PPO objective is the clipped surrogate

$L^{\mathrm{PPO}}(\psi) = \mathbb{E}\bigl[\min\bigl(r_t(\psi) A_t,\, \mathrm{clip}(r_t(\psi),\, 1-\epsilon,\, 1+\epsilon)\, A_t\bigr)\bigr]$

with

$r_t(\psi) = \pi_\psi(a_t \mid o_t) / \pi_{\psi_{\text{old}}}(a_t \mid o_t)$.

The full optimized loss is

$L(\psi) = -L^{\mathrm{PPO}}(\psi) + c_v\,\mathbb{E}\bigl[(V_\psi(o_t) - \hat{R}_t)^2\bigr] - c_e\,\mathbb{E}\bigl[H(\pi_\psi(\cdot \mid o_t))\bigr]$.

Advantages are computed with GAE:

$\delta_t = r_t + \gamma V(o_{t+1}) - V(o_t)$

$A_t = \delta_t + \gamma \lambda \delta_{t+1} + \gamma^2 \lambda^2 \delta_{t+2} + \cdots$

The reward fed into PPO can be:

- raw env reward $\Delta\,\text{edge}$
- delta edge advantage versus benchmark
- delta PnL advantage versus benchmark
- a balanced objective with inventory and action-smoothness penalties

So PPO is a model-free RL layer on top of the public environment, not on top of
the exact differentiable compact-policy simulator.

## 11. Smooth Differentiable Surrogate

`arena_eval/diff_simple_amm/objectives.py` defines a smooth surrogate

$J_{\text{smooth}}(\theta; \omega)$

for the compact submission policy.

The key idea is to keep the same overall simulator logic while replacing hard
non-differentiabilities with smooth relaxations:

- hard event activation -> sigmoid gates
- $max(x, 0)$ -> smooth positive part
- clipping -> smooth clip
- branch combinations -> smooth gates / smooth OR
- minimum trade threshold -> smooth trade amount

This produces a differentiable computational graph in JAX.

## 11.1 Smooth challenge objective

For the challenge evaluator, the smooth objective is

$J_{\text{smooth}}^{\text{chal}}(\theta; \omega) = E_{T,\text{smooth}}^{\text{sub}}(\theta; \omega)$.

The tape still supplies latent normals and uniforms, but order arrivals and
sides are softly activated. For a per-step slot width $W$, the per-slot
activation probability is

$p_{\mathrm{slot}} = 1 - \exp(-\lambda / W)$.

The slot activity gate is

$g^{\mathrm{arr}}_{t,j} = \mathrm{sigmoid}\bigl(\alpha_{\mathrm{arr}} (p_{\mathrm{slot}} - U^{\mathrm{arr}}_{t,j})\bigr)$.

The buy-side gate is

$g^{\mathrm{buy}}_{t,j} = \mathrm{sigmoid}\bigl(\alpha_{\mathrm{side}} (p_{\mathrm{buy}} - U^{\mathrm{side}}_{t,j})\bigr)$.

Order sizes use the same lognormal transform:

$Q_{t,j} = \exp(\mu_{\mathrm{LN}} + \sigma_{\mathrm{LN}}\, Z^{\mathrm{size}}_{t,j})$.

Then the effective smooth buy/sell order masses are

$Q^{buy}_{t,j} = g^{arr}_{t,j} g^{buy}_{t,j} Q_{t,j}$

$Q^{sell}_{t,j} = g^{arr}_{t,j} (1 - g^{buy}_{t,j}) Q_{t,j}$.

## 11.2 Smooth realistic objective

For the realistic evaluator, the regime is relaxed from a hard latent state to a
probability vector $p_t$ over regimes:

$p_{t+1} = p_t Q$.

For a sampled percentile $U_t$, the regime-conditioned log returns
$\ell_t^{(r)} = G_r^{-1}(U_t)$ are blended as

$\ell_t = \sum_r p_{t+1,r} \ell_t^{(r)}$.

Then

$F_{t+1} = F_t \exp(\ell_t)$.

Retail impact percentiles are also passed through smooth side decompositions
using positive-part relaxations.

## 11.3 Gradient

Because $J_{\text{smooth}}$ is implemented in JAX from differentiable primitives, its
gradient exists almost everywhere and is computed by reverse-mode automatic
differentiation:

$\nabla_{\theta} J_{\text{smooth}}(\theta; \omega)$ is returned by JAX autodiff applied to $J_{\text{smooth}}(\cdot; \omega)$.

For a batch of tapes $\Omega$,

$\bar{J}_{\text{smooth}}(\theta) = |\Omega|^{-1} \sum_{\omega \in \Omega} J_{\text{smooth}}(\theta; \omega)$

and

$\nabla_{\theta} \bar{J}_{\text{smooth}}(\theta) = |\Omega|^{-1} \sum_{\omega \in \Omega} \nabla_{\theta} J_{\text{smooth}}(\theta; \omega)$.

The repo currently exposes these differentiable objectives and batch aggregators,
but does not yet ship a full optimizer loop around them in the same way that it
ships CEM and PPO loops.

## 12. What Is Being Optimized In Each Stack

There are really three related optimization problems in this repo:

1. Exact compact-policy optimization:
   maximize expected submission edge under the exact simulator. This is the most
   faithful objective, but generally non-smooth.

2. Smooth compact-policy optimization:
   maximize expected submission edge under the smooth surrogate. This is the
   same conceptual objective with differentiable approximations so gradients are
   available through JAX.

3. Public-env policy optimization:
   maximize an RL objective over the Gym observation/action interface. This does
   not directly optimize the compact submission policy parameters; it optimizes a
   neural or linear controller over the public observation stream.

## 13. Practical Interpretation

- `edge_submission` is the core economic objective.
- `edge_advantage` measures whether submission beats the normalizer on the same
  tape.
- `pnl_submission` and `pnl_advantage` matter because fee capture can still be
  offset by inventory risk.
- Challenge mode is a stylized GBM plus lognormal-flow world.
- Realistic mode swaps those exogenous laws for empirical regime and impact
  distributions derived from real data artifacts.
- Exact-path evaluation is the canonical benchmark.
- Smooth training is the differentiable approximation layer.
- PPO and CEM are trainer-side optimization methods over public interfaces, not
  the same object as the compact-policy differentiable program.
