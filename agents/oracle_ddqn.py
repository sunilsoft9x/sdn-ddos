"""A3: DeepAir-style oracle Double-DQN (plan 4.9).

Reads the 18 features plus a one-hot regime label tau in {none, syn, udp}. This
is the baseline the paper argues against: it depends on an external detector
label. When the label is wrong, missing or delayed (E2), or the regime is one it
has no class for (E3, R4), it mis-acts. tau = none forces a no-op and stores
reward 0, matching DeepAir's gating.

The one-hot tau encoding is a design choice (Section IV). Parameter count is
tuned to sit within 2% of A1 and A2 by the 150-150 hidden layers over 21 inputs.
"""

from __future__ import annotations

import numpy as np
import torch

from agents.common import MLP, DoubleDQN, Hyper, ReplayBuffer, epsilon_at

TAU = {"none": 0, "syn": 1, "udp": 2}
TAU_DIM = 3
# Which real regime each label corresponds to, for building the oracle's input.
REGIME_TO_TAU = {"R0": "none", "R1": "syn", "R2": "udp"}


def tau_onehot(label: str) -> np.ndarray:
    v = np.zeros(TAU_DIM, dtype=np.float32)
    v[TAU[label]] = 1.0
    return v


class OracleDDQNAgent:
    label = "A3"

    def __init__(self, h: Hyper, hidden=(150, 150)):
        self.h = h
        in_dim = h.obs_dim + TAU_DIM
        self.online = MLP(in_dim, list(hidden), h.n_actions)
        self.target = MLP(in_dim, list(hidden), h.n_actions)
        self.dqn = DoubleDQN(self.online, self.target, h)
        self.buffer = ReplayBuffer(h.replay_size)

    def _input(self, obs: np.ndarray, tau_label: str) -> np.ndarray:
        return np.concatenate([obs, tau_onehot(tau_label)]).astype(np.float32)

    def act(self, obs: np.ndarray, tau_label: str, step: int,
            greedy: bool = False) -> int:
        # tau = none gates to a forced no-op, as in DeepAir.
        if tau_label == "none":
            return 0
        if not greedy and np.random.random() < epsilon_at(step, self.h):
            return int(np.random.randint(self.h.n_actions))
        x = torch.as_tensor(self._input(obs, tau_label)).unsqueeze(0)
        with torch.no_grad():
            q = self.online(x)
        return int(q.argmax(1).item())

    def remember(self, obs, tau_label, action, reward, next_obs,
                 next_tau_label, done) -> None:
        # tau = none stores reward 0 (the gated no-op contributes no learning
        # signal), matching the oracle's design.
        r = 0.0 if tau_label == "none" else reward
        self.buffer.push(x=self._input(obs, tau_label), action=action, reward=r,
                         next_x=self._input(next_obs, next_tau_label), done=done)

    def learn(self) -> float | None:
        if len(self.buffer) < self.h.batch_size:
            return None
        b = self.buffer.sample(self.h.batch_size)
        q_pred = self.online(b["x"])
        q_next_online = self.online(b["next_x"])
        q_next_target = self.target(b["next_x"])
        return self.dqn.update(q_pred, q_next_online, q_next_target,
                               b["action"], b["reward"], b["done"])

    def n_params(self) -> int:
        from agents.common import param_count
        return param_count(self.online)
