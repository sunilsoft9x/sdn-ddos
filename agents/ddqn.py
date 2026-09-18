"""A1: plain Double-DQN over the 18 features (plan 4.9).

The memoryless baseline. It sees only the current window, so it cannot tell R4's
silent gap from benign traffic, nor detect a regime switch from history. Its
gap against A2 is the paper's argument for context.
"""

from __future__ import annotations

import numpy as np
import torch

from agents.common import MLP, DoubleDQN, Hyper, ReplayBuffer, epsilon_at


class DDQNAgent:
    label = "A1"

    def __init__(self, h: Hyper, hidden=(150, 150)):
        self.h = h
        self.online = MLP(h.obs_dim, list(hidden), h.n_actions)
        self.target = MLP(h.obs_dim, list(hidden), h.n_actions)
        self.dqn = DoubleDQN(self.online, self.target, h)
        self.buffer = ReplayBuffer(h.replay_size)

    def act(self, obs: np.ndarray, step: int, greedy: bool = False) -> int:
        if not greedy and np.random.random() < epsilon_at(step, self.h):
            return int(np.random.randint(self.h.n_actions))
        with torch.no_grad():
            q = self.online(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        return int(q.argmax(1).item())

    def remember(self, obs, action, reward, next_obs, done) -> None:
        self.buffer.push(obs=obs, action=action, reward=reward,
                         next_obs=next_obs, done=done)

    def learn(self) -> float | None:
        if len(self.buffer) < self.h.batch_size:
            return None
        b = self.buffer.sample(self.h.batch_size)
        q_pred = self.online(b["obs"])
        q_next_online = self.online(b["next_obs"])
        q_next_target = self.target(b["next_obs"])
        return self.dqn.update(q_pred, q_next_online, q_next_target,
                               b["action"], b["reward"], b["done"])

    def n_params(self) -> int:
        from agents.common import param_count
        return param_count(self.online)
