"""A2: context Double-DQN, the proposed method (plan 4.9).

A GRU reads the last K reward-inclusive transitions -- each the observation, the
one-hot action taken, and the reward received -- and projects to an 8-d context
appended to the current 18 features. Because the history carries the reward, the
agent can infer the hidden regime from how its own actions paid off, without any
external label. This is what lets it hold mitigation across R4's silent gaps and
react to a regime switch.

The GRU trains end to end through the Q-loss; the target network holds its own
GRU copy. The context dimension (8) and history length (K=10) are design
choices, stated in Section IV.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from agents.common import DoubleDQN, Hyper, ReplayBuffer, epsilon_at


class ContextQNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, context_dim: int = 8,
                 gru_hidden: int = 32, hidden=(128, 128)):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        gru_in = obs_dim + n_actions + 1        # obs + one-hot action + reward
        self.gru = nn.GRU(gru_in, gru_hidden, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(gru_hidden, context_dim), nn.Tanh())
        d = obs_dim + context_dim
        layers: list[nn.Module] = []
        for hdim in hidden:
            layers += [nn.Linear(d, hdim), nn.ReLU()]
            d = hdim
        layers.append(nn.Linear(d, n_actions))
        self.head = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        # history: [B, K, obs+action+reward]; masked-zero padded at episode start.
        _, hn = self.gru(history)
        context = self.proj(hn.squeeze(0))
        return self.head(torch.cat([obs, context], dim=1))


class ContextDDQNAgent:
    label = "A2"

    def __init__(self, h: Hyper, context_dim: int = 8, k: int = 10,
                 gru_hidden: int = 32, hidden=(128, 128)):
        self.h = h
        self.k = k
        self.hist_dim = h.obs_dim + h.n_actions + 1
        self.online = ContextQNet(h.obs_dim, h.n_actions, context_dim,
                                  gru_hidden, hidden)
        self.target = ContextQNet(h.obs_dim, h.n_actions, context_dim,
                                  gru_hidden, hidden)
        self.dqn = DoubleDQN(self.online, self.target, h)
        self.buffer = ReplayBuffer(h.replay_size)
        self._history: list[np.ndarray] = []

    # -- history bookkeeping ---------------------------------------------

    def reset_history(self) -> None:
        self._history = []

    def _padded(self, history: list[np.ndarray]) -> np.ndarray:
        """Left-pad the history to length K with zeros (masked at episode start)."""
        h = history[-self.k:]
        pad = self.k - len(h)
        block = np.zeros((self.k, self.hist_dim), dtype=np.float32)
        if h:
            block[pad:] = np.asarray(h, dtype=np.float32)
        return block

    def _transition_vec(self, obs, action, reward) -> np.ndarray:
        onehot = np.zeros(self.h.n_actions, dtype=np.float32)
        onehot[action] = 1.0
        return np.concatenate([obs, onehot, [reward]]).astype(np.float32)

    # -- act / remember / learn ------------------------------------------

    def act(self, obs: np.ndarray, step: int, greedy: bool = False) -> int:
        if not greedy and np.random.random() < epsilon_at(step, self.h):
            return int(np.random.randint(self.h.n_actions))
        hist = torch.as_tensor(self._padded(self._history)).unsqueeze(0)
        with torch.no_grad():
            q = self.online(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0),
                            hist)
        return int(q.argmax(1).item())

    def remember(self, obs, action, reward, next_obs, done) -> None:
        hist_before = self._padded(self._history)
        self._history.append(self._transition_vec(obs, action, reward))
        hist_after = self._padded(self._history)
        self.buffer.push(obs=obs, action=action, reward=reward,
                         next_obs=next_obs, done=done,
                         hist=hist_before, next_hist=hist_after)
        if done:
            self.reset_history()

    def learn(self) -> float | None:
        if len(self.buffer) < self.h.batch_size:
            return None
        b = self.buffer.sample(self.h.batch_size)
        q_pred = self.online(b["obs"], b["hist"])
        q_next_online = self.online(b["next_obs"], b["next_hist"])
        q_next_target = self.target(b["next_obs"], b["next_hist"])
        return self.dqn.update(q_pred, q_next_online, q_next_target,
                               b["action"], b["reward"], b["done"])

    def n_params(self) -> int:
        from agents.common import param_count
        return param_count(self.online)
