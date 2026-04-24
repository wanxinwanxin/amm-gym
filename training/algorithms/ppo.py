"""Minimal PPO trainer for continuous-control AMM policies."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import cycle

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from training.eval.metrics import aggregate_episode_metrics, evaluate_episode


EnvFactory = Callable[[], object]
Policy = Callable[[np.ndarray], np.ndarray]


LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0


@dataclass
class PPOConfig:
    total_timesteps: int = 12_288
    rollout_steps: int = 1_024
    epochs: int = 4
    minibatch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 0
    objective: str = "delta_edge_advantage"
    inventory_penalty_coef: float = 0.02
    action_penalty_coef: float = 0.005
    hidden_sizes: tuple[int, ...] = (64, 64)
    validation_interval: int = 2
    train_seeds: tuple[int, ...] | None = None
    validation_seeds: tuple[int, ...] | None = None
    device: str = "cpu"


@dataclass
class PPOTrainingResult:
    best_validation_score: float
    history: list[dict[str, float]]
    policy: "TorchTanhGaussianPolicy"


class TanhGaussianActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(obs_dim)]
        in_dim = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.Tanh())
            in_dim = hidden_size
        self.trunk = nn.Sequential(*layers)
        self.actor_mean = nn.Linear(in_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        self.critic = nn.Linear(in_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.trunk(obs)
        mean = self.actor_mean(hidden)
        value = self.critic(hidden).squeeze(-1)
        log_std = self.actor_log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).expand_as(mean)
        return mean, log_std, value


class TorchTanhGaussianPolicy:
    def __init__(self, model: TanhGaussianActorCritic, device: torch.device) -> None:
        self.model = model
        self.device = device

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mean, _, _ = self.model(obs_tensor)
        action = torch.tanh(mean).squeeze(0)
        return action.cpu().numpy().astype(np.float32)


def _atanh(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def _tanh_normal_log_prob(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    action: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    clipped_action = torch.clamp(action, -0.999999, 0.999999)
    pre_tanh = _atanh(clipped_action)
    std = torch.exp(log_std)
    base_dist = Normal(mean, std)
    log_prob = base_dist.log_prob(pre_tanh) - torch.log(1.0 - clipped_action.square() + 1e-6)
    entropy = base_dist.entropy()
    return log_prob.sum(dim=-1), entropy.sum(dim=-1)


class PPOTrainer:
    """Train a small actor-critic policy on the public env interface."""

    def __init__(self, env_factory: EnvFactory, config: PPOConfig) -> None:
        self.env_factory = env_factory
        self.config = config

        env = env_factory()
        obs, _ = env.reset(seed=0)
        obs_dim = int(obs.shape[0])
        action_dim = int(np.asarray(env.action_space.low).shape[0])

        self.device = torch.device(config.device)
        self.model = TanhGaussianActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=config.hidden_sizes,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.train_seed_cycle = cycle(config.train_seeds or tuple(range(32)))

    def train(self) -> PPOTrainingResult:
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        env = self.env_factory()
        obs, _ = env.reset(seed=next(self.train_seed_cycle))
        prev_info = {
            "edge": 0.0,
            "edge_benchmark": 0.0,
            "pnl": 0.0,
            "pnl_benchmark": 0.0,
        }
        prev_action = np.zeros_like(env.action_space.low, dtype=np.float32)
        current_episode_train_reward = 0.0
        current_episode_objective = 0.0
        train_episode_rewards: list[float] = []
        train_episode_objectives: list[float] = []

        n_updates = max(1, self.config.total_timesteps // self.config.rollout_steps)
        history: list[dict[str, float]] = []
        best_validation_score = -np.inf
        best_state = {name: value.detach().cpu().clone() for name, value in self.model.state_dict().items()}

        for update_idx in range(n_updates):
            batch = self._collect_rollout(
                env=env,
                initial_obs=obs,
                prev_info=prev_info,
                prev_action=prev_action,
                current_episode_train_reward=current_episode_train_reward,
                current_episode_objective=current_episode_objective,
                train_episode_rewards=train_episode_rewards,
                train_episode_objectives=train_episode_objectives,
            )

            obs = batch["next_obs"]
            prev_info = batch["next_prev_info"]
            prev_action = batch["next_prev_action"]
            current_episode_train_reward = batch["current_episode_train_reward"]
            current_episode_objective = batch["current_episode_objective"]

            advantages = self._compute_gae(
                rewards=batch["rewards"],
                values=batch["values"],
                dones=batch["dones"],
                last_value=batch["last_value"],
            )
            returns = advantages + batch["values"]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            self._update_model(
                obs=batch["obs"],
                actions=batch["actions"],
                log_probs=batch["log_probs"],
                returns=returns,
                advantages=advantages,
            )

            row = {
                "update": float(update_idx),
                "train_objective_mean": float(batch["rewards"].mean()),
                "train_episode_reward_mean": float(np.mean(train_episode_rewards) if train_episode_rewards else 0.0),
                "train_episode_objective_mean": float(
                    np.mean(train_episode_objectives) if train_episode_objectives else 0.0
                ),
            }

            should_validate = (
                self.config.validation_seeds
                and (
                    (update_idx + 1) % self.config.validation_interval == 0
                    or update_idx == n_updates - 1
                )
            )
            if should_validate:
                policy = TorchTanhGaussianPolicy(self.model, self.device)
                metrics = [
                    evaluate_episode(self.env_factory, policy.act, seed=int(seed))
                    for seed in self.config.validation_seeds or ()
                ]
                summary = aggregate_episode_metrics(metrics)
                validation_score = float(summary["edge_advantage_mean"])
                row["validation_edge_advantage_mean"] = validation_score
                row["validation_edge_advantage_win_rate"] = float(summary["edge_advantage_win_rate"])
                row["validation_pnl_advantage_mean"] = float(summary["pnl_advantage"]["mean"])
                if validation_score > best_validation_score:
                    best_validation_score = validation_score
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in self.model.state_dict().items()
                    }
            history.append(row)

        self.model.load_state_dict(best_state)
        return PPOTrainingResult(
            best_validation_score=float(best_validation_score),
            history=history,
            policy=TorchTanhGaussianPolicy(self.model, self.device),
        )

    def _collect_rollout(
        self,
        *,
        env,
        initial_obs: np.ndarray,
        prev_info: dict[str, float],
        prev_action: np.ndarray,
        current_episode_train_reward: float,
        current_episode_objective: float,
        train_episode_rewards: list[float],
        train_episode_objectives: list[float],
    ) -> dict[str, object]:
        obs_buf: list[np.ndarray] = []
        actions_buf: list[np.ndarray] = []
        log_probs_buf: list[float] = []
        rewards_buf: list[float] = []
        values_buf: list[float] = []
        dones_buf: list[float] = []

        obs = initial_obs
        step_prev_info = dict(prev_info)
        step_prev_action = prev_action.astype(np.float32, copy=True)

        for _ in range(self.config.rollout_steps):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                mean, log_std, value = self.model(obs_tensor)
                std = torch.exp(log_std)
                base_dist = Normal(mean, std)
                pre_tanh = base_dist.rsample()
                action_tensor = torch.tanh(pre_tanh)
                log_prob = (
                    base_dist.log_prob(pre_tanh) - torch.log(1.0 - action_tensor.square() + 1e-6)
                ).sum(dim=-1)

            action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
            next_obs, env_reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            objective_reward = self._objective_reward(
                objective=self.config.objective,
                env_reward=float(env_reward),
                info=info,
                prev_info=step_prev_info,
                next_obs=next_obs,
                action=action,
                prev_action=step_prev_action,
                window_size=int(env.window_size),
            )

            obs_buf.append(obs.astype(np.float32, copy=True))
            actions_buf.append(action)
            log_probs_buf.append(float(log_prob.item()))
            rewards_buf.append(objective_reward)
            values_buf.append(float(value.item()))
            dones_buf.append(float(done))

            current_episode_train_reward += float(env_reward)
            current_episode_objective += objective_reward
            step_prev_info = {
                "edge": float(info["edge"]),
                "edge_benchmark": float(info.get("edge_benchmark", info["edge_normalizer"])),
                "pnl": float(info["pnl"]),
                "pnl_benchmark": float(info.get("pnl_benchmark", info["pnl_normalizer"])),
            }
            step_prev_action = action
            obs = next_obs

            if done:
                train_episode_rewards.append(current_episode_train_reward)
                train_episode_objectives.append(current_episode_objective)
                current_episode_train_reward = 0.0
                current_episode_objective = 0.0
                obs, _ = env.reset(seed=next(self.train_seed_cycle))
                step_prev_info = {
                    "edge": 0.0,
                    "edge_benchmark": 0.0,
                    "pnl": 0.0,
                    "pnl_benchmark": 0.0,
                }
                step_prev_action = np.zeros_like(step_prev_action, dtype=np.float32)

        last_obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            _, _, last_value = self.model(last_obs_tensor)

        return {
            "obs": np.asarray(obs_buf, dtype=np.float32),
            "actions": np.asarray(actions_buf, dtype=np.float32),
            "log_probs": np.asarray(log_probs_buf, dtype=np.float32),
            "rewards": np.asarray(rewards_buf, dtype=np.float32),
            "values": np.asarray(values_buf, dtype=np.float32),
            "dones": np.asarray(dones_buf, dtype=np.float32),
            "last_value": float(last_value.item()),
            "next_obs": obs,
            "next_prev_info": step_prev_info,
            "next_prev_action": step_prev_action,
            "current_episode_train_reward": current_episode_train_reward,
            "current_episode_objective": current_episode_objective,
        }

    def _objective_reward(
        self,
        *,
        objective: str,
        env_reward: float,
        info: dict[str, float],
        prev_info: dict[str, float],
        next_obs: np.ndarray,
        action: np.ndarray,
        prev_action: np.ndarray,
        window_size: int,
    ) -> float:
        delta_edge = float(info["edge"]) - float(prev_info["edge"])
        benchmark_edge = float(info.get("edge_benchmark", info["edge_normalizer"]))
        delta_benchmark_edge = benchmark_edge - float(prev_info["edge_benchmark"])
        delta_pnl = float(info["pnl"]) - float(prev_info["pnl"])
        benchmark_pnl = float(info.get("pnl_benchmark", info["pnl_normalizer"]))
        delta_benchmark_pnl = benchmark_pnl - float(prev_info["pnl_benchmark"])
        imbalance = abs(float(next_obs[window_size + 2]))
        action_penalty = float(np.mean((action - prev_action) ** 2))

        if objective == "reward":
            return env_reward
        if objective == "delta_edge_advantage":
            return delta_edge - delta_benchmark_edge
        if objective == "pnl_advantage":
            return delta_pnl - delta_benchmark_pnl
        if objective == "balanced":
            return (
                delta_edge
                - delta_benchmark_edge
                - self.config.inventory_penalty_coef * imbalance
                - self.config.action_penalty_coef * action_penalty
            )
        raise ValueError(f"unknown PPO objective `{objective}`")

    def _compute_gae(
        self,
        *,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: float,
    ) -> np.ndarray:
        advantages = np.zeros_like(rewards, dtype=np.float32)
        next_advantage = 0.0
        next_value = float(last_value)

        for idx in range(len(rewards) - 1, -1, -1):
            not_done = 1.0 - dones[idx]
            delta = rewards[idx] + self.config.gamma * next_value * not_done - values[idx]
            next_advantage = delta + self.config.gamma * self.config.gae_lambda * not_done * next_advantage
            advantages[idx] = next_advantage
            next_value = float(values[idx])

        return advantages

    def _update_model(
        self,
        *,
        obs: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        returns: np.ndarray,
        advantages: np.ndarray,
    ) -> None:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions_tensor = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(log_probs, dtype=torch.float32, device=self.device)
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)

        indices = np.arange(obs.shape[0])
        for _ in range(self.config.epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), self.config.minibatch_size):
                batch_idx = indices[start : start + self.config.minibatch_size]
                batch_obs = obs_tensor[batch_idx]
                batch_actions = actions_tensor[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns_tensor[batch_idx]
                batch_advantages = advantages_tensor[batch_idx]

                mean, log_std, values = self.model(batch_obs)
                new_log_probs, entropy = _tanh_normal_log_prob(mean, log_std, batch_actions)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_eps,
                    1.0 + self.config.clip_eps,
                )
                policy_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()
                value_loss = 0.5 * (batch_returns - values).square().mean()
                entropy_bonus = entropy.mean()

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy_bonus
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

