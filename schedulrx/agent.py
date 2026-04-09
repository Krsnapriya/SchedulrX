"""
SchedulrX PyTorch Actor-Critic Agent
======================================
Shared-trunk PPO agent with action masking for the SchedulrX
Gymnasium environment.

The agent learns to:
  - Read profiles strategically (information foraging)
  - Schedule meetings in constraint-satisfying slots
  - Handle cancellations via replanning

Action masking prevents invalid moves during both training and eval.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from typing import Optional, Tuple


class SchedulrXActorCritic(nn.Module):
    """
    Shared-trunk actor-critic for PPO training on SchedulrXGymEnv.

    Architecture:
        obs → [Linear→LN→GELU]×2 → trunk
        trunk → actor head  → action logits (masked)
        trunk → critic head → state value V(s)
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

        # Initialize weights for stable training
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Smaller init for policy and value heads
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(
        self, obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            obs: observation tensor [batch, obs_dim]
            action_mask: boolean mask [batch, n_actions], True = valid

        Returns:
            logits: masked action logits [batch, n_actions]
            value: state value [batch, 1]
        """
        z = self.trunk(obs)
        logits = self.actor(z)
        value = self.critic(z)

        # Apply action mask: set invalid actions to -inf
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float("-inf"))

        return logits, value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample or evaluate an action.

        Args:
            obs: observation [batch, obs_dim]
            action_mask: boolean mask [batch, n_actions]
            action: if provided, evaluate this action (for PPO update)
            deterministic: if True, take argmax instead of sampling

        Returns:
            action, log_prob, entropy, value
        """
        logits, value = self(obs, action_mask)
        dist = Categorical(logits=logits)

        if action is None:
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value.squeeze(-1)


class RolloutBuffer:
    """
    Fixed-size rollout buffer for PPO with GAE advantage estimation.
    """

    def __init__(self, buffer_size: int, obs_dim: int, n_actions: int):
        self.buffer_size = buffer_size
        self.obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.masks = np.zeros((buffer_size, n_actions), dtype=np.bool_)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        self.ptr = 0

    def store(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
        mask: np.ndarray,
    ):
        idx = self.ptr % self.buffer_size
        self.obs[idx] = obs
        self.actions[idx] = action
        self.log_probs[idx] = log_prob
        self.rewards[idx] = reward
        self.dones[idx] = float(done)
        self.values[idx] = value
        self.masks[idx] = mask
        self.ptr += 1

    def compute_gae(self, last_value: float, gamma: float = 0.99, lam: float = 0.95):
        """Compute Generalized Advantage Estimation."""
        n = min(self.ptr, self.buffer_size)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_done = 0.0
            else:
                next_value = self.values[t + 1]
                next_done = self.dones[t + 1]

            delta = (
                self.rewards[t]
                + gamma * next_value * (1 - self.dones[t])
                - self.values[t]
            )
            last_gae = delta + gamma * lam * (1 - self.dones[t]) * last_gae
            self.advantages[t] = last_gae

        self.returns[:n] = self.advantages[:n] + self.values[:n]

    def get_batches(self, batch_size: int):
        """Yield minibatch indices for PPO update."""
        n = min(self.ptr, self.buffer_size)
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            yield indices[start:end]

    def reset(self):
        self.ptr = 0
