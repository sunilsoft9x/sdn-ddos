"""Controlled study 1: does reward-inclusive history resolve a hidden mode?

Paper: Section VII, Table IV. This is a CONTROLLED STUDY, not a testbed result,
and it is trained with REINFORCE rather than DDQN. The question is what
information is available to a policy, not which learner exploits it, so the
algorithm choice does not bear on the conclusion. State both in the paper.

Task: each episode has a hidden mode in {ATTACK, LEGIT}, fixed for the episode
(it persists, as in the hidden-mode MDP of Section III). Two actions, ALLOW and
DROP. Reward is +1 when the action matches the mode's correct action
(ATTACK->DROP, LEGIT->ALLOW) and -1 otherwise. The correct action is OPPOSITE in
the two modes, so a policy that cannot tell them apart can do no better than a
coin flip at 0.0. The only signal to the mode is the reward a past action earned,
which is exactly A2's input and exactly what A1 discards.

Two observation conditions:
  revealed : the observation carries the mode. A control. Both should reach 1.0,
             which validates the harness rather than either architecture.
  hidden   : the observation is pure noise. The real test.

    python3 scripts/study_mechanism.py               # 5 seeds
    python3 scripts/study_mechanism.py --seeds 3
"""

from __future__ import annotations
import argparse
import numpy as np, torch, torch.nn as nn

OBS_DIM, N_ACT, EP_LEN = 4, 2, 12
DROP, ALLOW = 1, 0


def rollout(policy, cond, B=128):
    modes = torch.randint(0, 2, (B,))
    correct = torch.where(modes == 0, torch.tensor(DROP), torch.tensor(ALLOW))
    F = OBS_DIM + N_ACT + 1
    hist, logps, rews = [], [], []
    for _ in range(EP_LEN):
        if cond == "revealed":
            obs = torch.randn(B, OBS_DIM) * 0.1
            obs[:, 0] = obs[:, 0] + torch.where(modes == 0, 1.0, -1.0)
        else:
            obs = torch.randn(B, OBS_DIM)
        hseq = torch.stack(hist, 1) if hist else torch.zeros(B, 0, F)
        dist = torch.distributions.Categorical(logits=policy(obs, hseq))
        a = dist.sample()
        r = torch.where(a == correct, 1.0, -1.0)
        logps.append(dist.log_prob(a)); rews.append(r)
        oh = torch.zeros(B, N_ACT); oh[torch.arange(B), a] = 1.0
        hist.append(torch.cat([obs, oh, r.unsqueeze(1)], 1).detach())
    logps, rews = torch.stack(logps, 1), torch.stack(rews, 1)
    G = rews.flip(1).cumsum(1).flip(1)
    Gz = (G - G.mean()) / (G.std() + 1e-6)
    return -(logps * Gz.detach()).mean(), rews.mean().item()


class Memoryless(nn.Module):        # as A1: observation only
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(OBS_DIM, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, N_ACT))
    def forward(self, obs, hist): return self.net(obs)


class RewardHistory(nn.Module):     # as A2: GRU over (obs, action, reward)
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(OBS_DIM + N_ACT + 1, 32, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(32, 8), nn.Tanh())
        self.head = nn.Sequential(nn.Linear(OBS_DIM + 8, 64), nn.ReLU(),
                                  nn.Linear(64, N_ACT))
    def forward(self, obs, hist):
        if hist.shape[1] == 0:
            ctx = torch.zeros(obs.shape[0], 8)
        else:
            _, hn = self.gru(hist); ctx = self.proj(hn.squeeze(0))
        return self.head(torch.cat([obs, ctx], 1))


def train_once(mk, cond, seed, iters=700, lr=5e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    pol = mk(); opt = torch.optim.Adam(pol.parameters(), lr=lr); last = []
    for it in range(iters):
        loss, mr = rollout(pol, cond)
        opt.zero_grad(); loss.backward(); opt.step()
        if it >= iters - 40: last.append(mr)
    return float(np.mean(last))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--iters", type=int, default=700)
    args = p.parse_args()

    print(f"Controlled study 1: hidden-mode inference, {args.seeds} seeds\n")
    print(f"{'policy':<26}{'revealed obs':>18}{'hidden obs':>18}")
    print(f"{'':<26}{'(control)':>18}{'(the test)':>18}")
    for name, mk in [("memoryless (as A1)", Memoryless),
                     ("reward history (as A2)", RewardHistory)]:
        cells = []
        for cond in ("revealed", "hidden"):
            vals = [train_once(mk, cond, s, args.iters) for s in range(args.seeds)]
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            cells.append((float(np.mean(vals)), sd))
        a, b = cells
        print(f"{name:<26}{a[0]:>11.3f} +-{a[1]:<5.3f}{b[0]:>11.3f} +-{b[1]:<5.3f}",
              flush=True)

    print(f"\nmean reward per step in [-1, +1]; coin flip 0.0, perfect +1.0; "
          f"n = {args.seeds} seeds.")
    print("Both policies near 1.0 on the control validates the harness.")
    print("On the test the memoryless policy is pinned at the coin flip its")
    print("information allows, while reward history resolves the mode.")


if __name__ == "__main__":
    main()
