"""Offline tests for scripts/evaluate_iipem.py (no testbed).

    ~/venv/bin/python scripts/test_evaluate_iipem.py

1. The job grid: 400 episodes, the right counts, pairing by environment seed,
   the 4 arms of an episode back to back.
2. Sharding by seed: 200 + 200, disjoint, and every seed (so every arm of a
   seed) on one VM.
3. The episode loop on the fake network from test_wp1_events.py. With PyTorch
   available (the WSL venv) it uses the real frozen policies from runs/models
   and ISASF's own loading and labelling code; without it, stand-ins.
4. Restarting a shard skips only episodes that are complete.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile
import types

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
failures: list[str] = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t = load("t", REPO / "scripts" / "test_wp1_events.py")          # stubs mininet/requests
ev = load("ev", REPO / "scripts" / "evaluate_iipem.py")

print("1. Job grid")
jobs = ev.build_jobs()
check(len(jobs) == 400, f"400 episodes (got {len(jobs)})")
from collections import Counter  # noqa: E402
check(set(Counter((j["cond"], j["agent"]) for j in jobs).values()) == {50},
      "50 episodes per condition and arm (10 seeds x 5)")
check(all(j["env_seed"] == j["seed"] * 1000 + j["ep"] and j["w_seed"] == j["seed"] - 1000 for j in jobs),
      "environment seed = seed*1000 + ep; weights seed = seed - 1000 (0..9)")
check(sorted({j["w_seed"] for j in jobs}) == list(range(10)), "training seeds 0 to 9 all used")
blocks = [jobs[k:k + 4] for k in range(0, 400, 4)]
check(all([b["agent"] for b in blk] == list(ev.ARMS) and len({(b["cond"], b["env_seed"]) for b in blk}) == 1
          for blk in blocks), "each episode's 4 arms run back to back on one environment seed")
check(ev.CONDITIONS["C1_surge"]["surge"] == {"n_hosts": 2, "rate_mbps": 4.0, "at": 50},
      "C1 uses the pre-registered WP2 choice (2 hosts, 4.0 Mbps, step 50)")

print("2. Sharding by seed")
s0, s1 = ev.shard_jobs(jobs, 0, 2), ev.shard_jobs(jobs, 1, 2)
check(len(s0) == len(s1) == 200, "200 episodes per VM")
k = lambda j: (j["cond"], j["agent"], j["seed"], j["ep"])  # noqa: E731
check(not ({k(j) for j in s0} & {k(j) for j in s1}) and len({k(j) for j in s0 + s1}) == 400,
      "disjoint, and together they cover all 400")
check(not ({j["seed"] for j in s0} & {j["seed"] for j in s1}), "no seed is split across VMs")
check(all(set(Counter(j["agent"] for j in s).values()) == {50} for s in (s0, s1)),
      "each VM runs every arm equally: 5 seeds x 5 episodes x 2 conditions = 50 each")

print("3. Episode loop on the fake network")
try:
    import torch  # noqa: F401
    REAL = True
except ImportError:
    REAL = False
if REAL:
    isasf = ev._isasf()
    import yaml
    from agents.common import Hyper
    h = Hyper.from_cfg(yaml.safe_load(open(REPO / "configs" / "base.yaml")))
    agents = {a: isasf.load_agent(a, 0, h) for a in ev.ARMS}
    print("  (real policies, seed 0, from runs/models)")
else:
    names = ["f%d" % i for i in range(18)]

    class Stub:
        def __init__(self, arm): self.arm, self.seen = arm, 0
        def reset_history(self): self.seen = 0
        def reset(self): pass
        def act(self, obs, *a, **k): return 0 if (a and a[0] == "none") else 1
        def act_from_suspects(self, s): return 1 if s else 4
        def observe(self, *a): self.seen += 1
    isasf = types.SimpleNamespace(FEATURE_NAMES=names,
                                  oracle_label=lambda regime, *a: "none" if regime == "R0" else "udp")
    agents = {a: Stub(a) for a in ev.ARMS}
    print("  (no PyTorch here: stand-in policies)")

new = t.load(REPO / "env" / "testbed_env.py", "new_env_ev")
ACCT = "".join(f" 10 1000 all -- * * 10.0.0.{i} 0.0.0.0/0\n" for i in range(1, 16))


def fake_env():
    env = t.make_env(new)
    env._iipem_log, env._acct_ready, env._acct_pre, env._ctl, env._last_apply_ms = False, False, None, None, None
    vname = env.cfg["topology"]["victim"]["name"]
    env.net._by.setdefault(vname, t.FakeHost(vname, env.log))
    env.net._by[vname].cmd = lambda c: ACCT if "-L IIPEM_ACCT" in c else ""
    env._ctl = types.SimpleNamespace(pid=1, sample=lambda: (5.0, 800.0))
    env._src_ip = lambda h_: f"10.0.0.{int(h_.lstrip('h'))}"

    def obs(prev, curr, probe_window, churn, first):
        info = {"reward": 0.5, "reward_terms": {"S": 1.0, "L": 0.0, "U": 0.0, "C": 0.0},
                "rule_count": 0, "collateral": 0.0, "probe_n": 5, "t_wall": float(env._t), "regime": env._regime}
        info.update(env._event_info()); info.update(env._iipem_info())
        return np.zeros(18, dtype=np.float32), info
    env._observe = obs
    return env


env = fake_env()
rng = np.random.default_rng(0)
results = {}
for cond in ev.CONDITIONS:
    for arm in ev.ARMS:
        job = {"cond": cond, "agent": arm, "seed": 1000, "ep": 0, "w_seed": 0, "env_seed": 1000000}
        results[(cond, arm)] = ev.run_episode(env, agents[arm], job, isasf, rng)
check(all(len(r) == 100 for r, _ in results.values()), "100 step rows per episode")
need = {"infer_ms", "apply_ms", "ctl_cpu_pct", "flagged", "mitigated", "attackers", "surge_hosts",
        "former_attackers", "event_fired", "f_" + isasf.FEATURE_NAMES[0]}
check(all(need <= set(r[0]) for r, _ in results.values()), "every row carries the WP3 fields and features")
check(all(isinstance(x["infer_ms"], float) and x["infer_ms"] >= 0 for r, _ in results.values() for x in r),
      "inference time recorded every step")
for cond in ev.CONDITIONS:
    roles = {arm: [(x["attackers"], x["surge_hosts"], x["former_attackers"]) for x in results[(cond, arm)][0]]
             for arm in ev.ARMS}
    check(len({tuple(v) for v in roles.values()}) == 1, f"{cond}: all 4 arms see identical roles at every step")
check(all(len(d) == 30 for _, d in results.values()), "delivered: 2 phases x 15 sources per episode")
a3 = [x["action"] for x in results[("C1_surge", "A3")][0]]
check(set(a3) == {0}, "A3 in R0 receives 'none' and takes no action")
if REAL:
    check(len({x["action"] for x in results[("C2_swap", "A2")][0]}) >= 1, "real A2 runs with its history")

print("4. Restarting a shard")
import pandas as pd  # noqa: E402
with tempfile.TemporaryDirectory() as d:
    p = pathlib.Path(d) / "s.parquet"
    full = [{"cond": "C1_surge", "agent": "A1", "seed": 1000, "ep": 0, "step": s} for s in range(100)]
    part = [{"cond": "C1_surge", "agent": "A2", "seed": 1000, "ep": 0, "step": s} for s in range(37)]
    pd.DataFrame(full + part).to_parquet(p, index=False)
    done = ev.completed(p, 100)
    check(done == {("C1_surge", "A1", 1000, 0)}, "only the complete episode counts as done; the partial one reruns")

print(f"\n{'ALL PASSED' if not failures else f'{len(failures)} FAILED'}")
sys.exit(1 if failures else 0)
