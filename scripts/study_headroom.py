"""Controlled study 2: how much headroom does regime inference have here?

Paper: Section VII, Table IV. A CONTROLLED STUDY on this paper's own physics,
not a testbed result. Part 1 is the figure the paper quotes and it is
DETERMINISTIC arithmetic over the payoff structure, so it has no seed variance
and must not be reported with an error bar. Part 2 trains policies and does take
seeds, and it is reported only for completeness.

Faithful where it matters: 10 Mbps victim link, 15 hosts with 3 probes, the
3x-median suspect rule, the five actions, and the reward
r = S + wG*G - 0.5L - 0.5U - 0.1C, where G is SERVED APPLICATION GOODPUT at the
victim, a label-free server-side counter (Section III: a reward term computed
from attacker identity would smuggle the label back in through A2's history).

Two hidden regimes, fixed per episode, engineered to produce IDENTICAL
observations:
  DDOS  : 3 high-rate sources whose traffic does no useful work.
  FLASH : 3 high-rate sources that are legitimate users (a flash crowd).
Offered traffic, utilisation, probe success and every observed feature are the
same in both. Only the served goodput differs once you act: dropping a flood
costs no useful work, dropping real users destroys it. G enters the REWARD and
not the OBSERVATION, matching env/features.py, which carries no aggregate
served-goodput feature.

Part 1 reports the reward of each action under each regime, the best action under
each, and the gap between a mode-omniscient policy and the best single fixed
action. That gap bounds what ANY inference method could win in this setting.

    python3 scripts/study_headroom.py              # Part 1 only (instant)
    python3 scripts/study_headroom.py --train --seeds 3
"""

from __future__ import annotations
import argparse
import numpy as np, torch, torch.nn as nn

CAP = 1.25e6              # 10 Mbps in bytes/s
N_BG, N_HIGH = 9, 3       # benign background hosts, high-rate sources
BG_BPS = 0.35e6 / 8       # ~0.35 Mbps each
HIGH_BPS = 3.2e6 / 8      # ~3.2 Mbps each, so 3 of them saturate the link
W_G = 1.0                 # weight on served goodput
EP_LEN, N_ACT, OBS_DIM = 20, 5, 7
DDOS, FLASH = 0, 1
# actions: 0 noop, 1 drop suspects, 2 meter 50%, 3 meter 10%, 4 clear
FACTOR = {0: None, 1: 0.0, 2: 0.5, 3: 0.1, 4: None}


def physics(mode: int, mit: float | None):
    """mit = multiplier on the high-rate sources (None = unmitigated)."""
    f = 1.0 if mit is None else mit
    high = N_HIGH * HIGH_BPS * f
    bg = N_BG * BG_BPS
    offered = high + bg
    util = offered / CAP
    pass_f = min(1.0, 1.0 / max(util, 1e-9))          # fair loss under congestion
    S = 1.0 / (1.0 + np.exp(6.0 * (util - 1.0)))      # probe success
    L = min(util, 1.5) / 1.5 * 0.6
    U = min(max(0.0, util - 0.8) / 0.2, 1.0)
    legit_nominal = bg + (N_HIGH * HIGH_BPS if mode == FLASH else 0.0)
    legit_passed = bg * pass_f + (high * pass_f if mode == FLASH else 0.0)
    G = legit_passed / legit_nominal
    obs = np.array([min(offered / (4 * CAP), 1.0), min(util, 1.5) / 1.5,
                    (HIGH_BPS * f) / max(offered, 1.0), (N_BG + N_HIGH) / 15.0,
                    S, L, 0.0 if mit is None else 1.0 - f], dtype=np.float32)
    return obs, S, G, L, U


def step_reward(mode, mit, churn=0):
    _, S, G, L, U = physics(mode, mit)
    return 1.0 * S + W_G * G - 0.5 * L - 0.5 * U - 0.1 * min(churn / 15.0, 1.0)


def part1():
    """Deterministic. The paper's headroom figure comes from here."""
    acts = [("noop/clear", None), ("drop", 0.0), ("meter 50%", 0.5),
            ("meter 10%", 0.1)]
    print("Part 1 - reward by action per hidden regime (deterministic, no seeds)\n")
    print(f"{'action':<14}{'DDOS':>10}{'FLASH':>10}{'mean':>10}")
    table = {}
    for name, a in acts:
        rd, rf = step_reward(DDOS, a), step_reward(FLASH, a)
        table[name] = (rd, rf)
        print(f"{name:<14}{rd:>10.3f}{rf:>10.3f}{(rd + rf) / 2:>10.3f}")
    bd = max(table, key=lambda k: table[k][0])
    bf = max(table, key=lambda k: table[k][1])
    omniscient = (table[bd][0] + table[bf][1]) / 2
    best_fixed_name = max(table, key=lambda k: (table[k][0] + table[k][1]) / 2)
    best_fixed = (table[best_fixed_name][0] + table[best_fixed_name][1]) / 2
    print(f"\nbest under DDOS : {bd}")
    print(f"best under FLASH: {bf}")
    print(f"{'FLIP PRESENT' if bd != bf else 'NO FLIP - design broken'}")
    print(f"\nmode-omniscient policy : {omniscient:.3f}")
    print(f"best single fixed action: {best_fixed:.3f}  ({best_fixed_name})")
    gap = omniscient - best_fixed
    print(f"HEADROOM for any inference method: {gap:.3f} "
          f"({100 * gap / best_fixed:.1f}% of the best fixed action)")
    print("\nThis is the number that decides whether a setting is worth building.")
    print("Screen a candidate design here before committing testbed time to it.")
    return gap


# ------------------------- Part 2 (optional, trained) --------------------

def rollout(policy, B=128):
    modes = torch.randint(0, 2, (B,))
    F = OBS_DIM + N_ACT + 1
    hist, logps, rews = [], [], []
    mit = [None] * B
    for _ in range(EP_LEN):
        obs = torch.zeros(B, OBS_DIM)
        for i in range(B):
            obs[i] = torch.from_numpy(physics(int(modes[i]), mit[i])[0])
        hseq = torch.stack(hist, 1) if hist else torch.zeros(B, 0, F)
        dist = torch.distributions.Categorical(logits=policy(obs, hseq))
        a = dist.sample()
        r = torch.zeros(B)
        for i in range(B):
            act = int(a[i]); churn = 0
            if act == 4:
                if mit[i] is not None: churn = N_HIGH
                mit[i] = None
            elif act != 0:
                new = FACTOR[act]
                if mit[i] != new: churn = N_HIGH
                mit[i] = new
            r[i] = step_reward(int(modes[i]), mit[i], churn)
        logps.append(dist.log_prob(a)); rews.append(r)
        oh = torch.zeros(B, N_ACT); oh[torch.arange(B), a] = 1.0
        hist.append(torch.cat([obs, oh, r.unsqueeze(1)], 1).detach())
    logps, rews = torch.stack(logps, 1), torch.stack(rews, 1)
    G = rews.flip(1).cumsum(1).flip(1)
    Gz = (G - G.mean()) / (G.std() + 1e-6)
    return -(logps * Gz.detach()).mean(), rews.mean().item()


class Memoryless(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(OBS_DIM, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, N_ACT))
    def forward(self, obs, hist): return self.net(obs)


class RewardHistory(nn.Module):
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


def train_once(mk, seed, iters=600, lr=5e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    pol = mk(); opt = torch.optim.Adam(pol.parameters(), lr=lr); last = []
    for it in range(iters):
        loss, mr = rollout(pol)
        opt.zero_grad(); loss.backward(); opt.step()
        if it >= iters - 40: last.append(mr)
    return float(np.mean(last))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true",
                   help="also run Part 2 (trained policies; slow)")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--iters", type=int, default=600)
    args = p.parse_args()

    part1()
    if not args.train:
        print("\n(Part 2 skipped. Add --train to run the trained comparison.)")
        return
    print("\n" + "-" * 50)
    print(f"\nPart 2 - trained policies, {args.seeds} seeds "
          f"(reported for completeness)\n")
    for name, mk in [("memoryless (as A1)", Memoryless),
                     ("reward history (as A2)", RewardHistory)]:
        vals = [train_once(mk, s, args.iters) for s in range(args.seeds)]
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        print(f"  {name:<26}{np.mean(vals):>8.3f} +-{sd:<5.3f}", flush=True)
    print("\nBoth should land near the best fixed action, because Part 1 shows")
    print("there is little more than that available in this setting.")


if __name__ == "__main__":
    main()
