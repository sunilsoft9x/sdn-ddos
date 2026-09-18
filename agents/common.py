"""Shared agent machinery (plan 4.9, 7.2).

Common to all three learners: Double DQN with Huber loss, Adam at 5e-4, gamma
0.95, replay 50k, batch 64, a hard target update every 1000 steps, epsilon
annealed 1.0 -> 0.05 over 30k steps, gradient clip 10.

The three agents differ only in what their Q-network reads:
  A1  the 18 features
  A2  the 18 features plus an 8-d context from a GRU over recent
      reward-inclusive transitions (our method)
  A3  the 18 features plus a one-hot regime label (the DeepAir-style oracle)

Parameter counts are held within 2% across the three (~26k) so the comparison
is fair; check_param_parity enforces it.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class Hyper:
    lr: float = 5e-4
    gamma: float = 0.95
    replay_size: int = 50000
    batch_size: int = 64
    target_update_steps: int = 1000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 30000
    grad_clip: float = 10.0
    obs_dim: int = 18
    n_actions: int = 5

    @classmethod
    def from_cfg(cls, cfg: dict) -> "Hyper":
        c = cfg["agents"]["common"]
        return cls(
            lr=c["optimizer"]["lr"], gamma=c["gamma"],
            replay_size=c["replay_size"], batch_size=c["batch_size"],
            target_update_steps=c["target_update_steps"],
            eps_start=c["epsilon"]["start"], eps_end=c["epsilon"]["end"],
            eps_decay_steps=c["epsilon"]["decay_steps"], grad_clip=c["grad_clip"],
            obs_dim=cfg["observation"]["dim"], n_actions=len(cfg["actions"]),
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def epsilon_at(step: int, h: Hyper) -> float:
    if step >= h.eps_decay_steps:
        return h.eps_end
    frac = step / h.eps_decay_steps
    return h.eps_start + frac * (h.eps_end - h.eps_start)


def param_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def check_param_parity(counts: dict[str, int], tol: float = 0.02) -> None:
    """Assert every agent's parameter count is within tol of the mean."""
    mean = sum(counts.values()) / len(counts)
    for name, n in counts.items():
        if abs(n - mean) / mean > tol:
            raise AssertionError(
                f"{name} has {n} params, {abs(n-mean)/mean:.1%} from the mean "
                f"{mean:.0f} (> {tol:.0%}); adjust hidden sizes")


class ReplayBuffer:
    """Uniform replay. Stores optional per-sample context and label arrays so
    the same buffer serves A1 (neither), A2 (context) and A3 (label)."""

    def __init__(self, capacity: int):
        self.buf: deque = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buf)

    def push(self, **transition) -> None:
        self.buf.append(transition)

    def sample(self, batch_size: int) -> dict:
        batch = random.sample(self.buf, batch_size)
        keys = batch[0].keys()
        out = {}
        for k in keys:
            vals = [b[k] for b in batch]
            out[k] = torch.as_tensor(np.asarray(vals), dtype=torch.float32)
        return out


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], out_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DoubleDQN:
    """Double-DQN update shared by every agent. The agent supplies an online
    and a target network with identical signatures; this class owns the
    optimizer, the Huber loss, gradient clipping and the hard target sync."""

    def __init__(self, online: nn.Module, target: nn.Module, h: Hyper):
        self.online = online
        self.target = target
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.h = h
        self.opt = torch.optim.Adam(self.online.parameters(), lr=h.lr)
        self.loss_fn = nn.SmoothL1Loss()
        self._updates = 0

    def update(self, q_pred: torch.Tensor, q_next_online: torch.Tensor,
               q_next_target: torch.Tensor, actions: torch.Tensor,
               rewards: torch.Tensor, dones: torch.Tensor) -> float:
        """One gradient step given precomputed Q outputs.

        q_pred:         Q(s, .)      from the online net           [B, A]
        q_next_online:  Q(s', .)     online net, for argmax        [B, A]
        q_next_target:  Q(s', .)     target net, for the value     [B, A]
        """
        taken = q_pred.gather(1, actions.long().unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_a = q_next_online.argmax(1, keepdim=True)
            next_q = q_next_target.gather(1, next_a).squeeze(1)
            y = rewards + self.h.gamma * next_q * (1.0 - dones)
        loss = self.loss_fn(taken, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.h.grad_clip)
        self.opt.step()
        self._updates += 1
        if self._updates % self.h.target_update_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())
