"""Evaluation runner on the testbed (plan 4.11, 7.7).

Loads each frozen policy and runs it on the OpenDaylight + Mininet testbed --
epsilon = 0, weights frozen, A2's GRU still running -- across the E0..E3 grid.
Every number reported in the paper comes from here, not the simulator. E4 is
derived from the E1 and E3 logs in analysis, so it schedules no episodes.

The full grid is 470 episodes (~6 h over one VM at W=0.5). Use --shard i/n to
split it across VM-1 and VM-2; each writes its own Parquet, combined in analysis.

  VM-1:  sudo ~/venv/bin/python scripts/evaluate.py --shard 0/2
  VM-2:  sudo ~/venv/bin/python scripts/evaluate.py --shard 1/2

Models are expected in runs/models/<AGENT>_s<seed>_final.pt (copied from WSL).
Eval seeds start at 1000, disjoint from the training seeds 0..4.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agents.common import Hyper
from agents.context_ddqn import ContextDDQNAgent
from agents.ddqn import DDQNAgent
from agents.oracle_ddqn import REGIME_TO_TAU, OracleDDQNAgent
from agents.static_rule import StaticRuleAgent
from env.testbed_env import TestbedEnv

REPO = pathlib.Path(__file__).resolve().parents[1]
MODELS = REPO / "runs" / "models"
EVAL_SEED_BASE = 1000
BIG = 10 ** 9                     # no-switch schedule sentinel
FEATURE_NAMES = [
    "pkt_rate", "byte_rate", "mean_pkt_size", "tcp_share", "udp_share",
    "icmp_share", "src_entropy", "active_sources", "top_source_share",
    "victim_link_util", "victim_tx_dropped_rate", "edge_table_occupancy",
    "probe_success", "probe_latency_median", "probe_goodput",
    "d_pkt_rate", "d_udp_share", "mitigated_fraction",
]
# which trained weights each evaluation agent loads
WEIGHTS = {"A1": "A1", "A2": "A2", "A3": "A3",
           "A3miss": "A3", "A3nearest": "A3", "static": None}


def cell(n: int, n_seeds: int = 5) -> list[tuple[int, int]]:
    """(eval_seed, episode-in-cell) pairs: n episodes spread over 5 seeds."""
    per = max(1, n // n_seeds)
    return [(EVAL_SEED_BASE + s, j) for s in range(n_seeds) for j in range(per)]


def build_jobs() -> list[dict]:
    jobs: list[dict] = []

    # E0: each regime alone, no switch, sanity. 3 regimes x 3 agents x 10.
    for regime in ("R0", "R1", "R2"):
        for agent in ("A1", "A2", "A3"):
            for seed, ep in cell(10):
                jobs.append(dict(exp="E0", agent=agent, cond=regime, seed=seed,
                                 ep=ep, schedule=[[regime, BIG]], label="true"))

    # E1: R1 for 50 then R2 for 50. RL agents 25 each + static 10.
    for agent in ("A1", "A2", "A3"):
        for seed, ep in cell(25):
            jobs.append(dict(exp="E1", agent=agent, cond="R1->R2", seed=seed,
                             ep=ep, schedule=[[R := "R1", 50], ["R2", 50]],
                             label="true"))
    for seed, ep in cell(10):
        jobs.append(dict(exp="E1", agent="static", cond="R1->R2", seed=seed,
                         ep=ep, schedule=[["R1", 50], ["R2", 50]], label="true"))

    # E2: E1 schedule, A3 with degraded labels. 3 modes x 25.
    for mode in ("wrong", "missing", "delayed"):
        for seed, ep in cell(25):
            jobs.append(dict(exp="E2", agent="A3", cond=mode, seed=seed, ep=ep,
                             schedule=[["R1", 50], ["R2", 50]], label=mode))

    # E3: R4 alone and R1->R4. A1,A2,A3miss,A3nearest 25 each + static 10, x2 conds.
    conds = {"R4": [["R4", BIG]], "R1->R4": [["R1", 50], ["R4", 50]]}
    for cond, sched in conds.items():
        for agent in ("A1", "A2", "A3miss", "A3nearest"):
            for seed, ep in cell(25):
                jobs.append(dict(exp="E3", agent=agent, cond=cond, seed=seed,
                                 ep=ep, schedule=sched, label="true"))
        for seed, ep in cell(10):
            jobs.append(dict(exp="E3", agent="static", cond=cond, seed=seed,
                             ep=ep, schedule=sched, label="true"))
    return jobs


def load_agent(eval_agent: str, seed_for_weights: int, h: Hyper):
    """Load the frozen policy. A3miss/A3nearest reuse A3's weights."""
    if eval_agent == "static":
        return StaticRuleAgent()
    base = WEIGHTS[eval_agent]
    ctor = {"A1": DDQNAgent, "A2": ContextDDQNAgent, "A3": OracleDDQNAgent,
            "A3miss": OracleDDQNAgent, "A3nearest": OracleDDQNAgent}[eval_agent]
    agent = ctor(h)
    path = MODELS / f"{base}_s{seed_for_weights}_final.pt"
    agent.online.load_state_dict(torch.load(path, map_location="cpu"))
    agent.online.eval()
    return agent


def oracle_label(true_regime: str, eval_agent: str, mode: str, step: int,
                 regime_history: list[str], rng: np.random.Generator) -> str:
    """The tau label the oracle receives, per experiment condition."""
    true = REGIME_TO_TAU.get(true_regime, "none")
    if eval_agent == "A3miss":
        # R4 has no label -> none (forced no-op); known regimes labelled truly.
        return "none" if true_regime == "R4" else true
    if eval_agent == "A3nearest":
        return "udp" if true_regime == "R4" else true
    if mode == "wrong":
        if rng.random() < 0.3:
            wrong = [t for t in ("none", "syn", "udp") if t != true]
            return str(rng.choice(wrong))
        return true
    if mode == "missing":
        return "none" if rng.random() < 0.3 else true
    if mode == "delayed":
        past = regime_history[max(0, step - 10)] if regime_history else true_regime
        return REGIME_TO_TAU.get(past, "none")
    return true


def run_episode(env: TestbedEnv, agent, eval_agent: str, job: dict,
                h: Hyper, rng: np.random.Generator) -> list[dict]:
    obs, info = env.reset(seed=job["seed"] * 1000 + job["ep"],
                          options={"schedule": job["schedule"], "scale": 1.0,
                                   "n_attackers": 3})
    if eval_agent == "A2":
        agent.reset_history()
    if eval_agent == "static":
        agent.reset()
    rows = []
    regime_hist: list[str] = []
    for step in range(env.steps_per_episode):
        regime = info["regime"]
        regime_hist.append(regime)
        if eval_agent == "static":
            action = agent.act_from_suspects(len(env.detector.suspects()) > 0)
        elif eval_agent in ("A3", "A3miss", "A3nearest"):
            tau = oracle_label(regime, eval_agent, job["label"], step,
                               regime_hist, rng)
            action = agent.act(obs, tau, step=10 ** 9, greedy=True)
        else:
            action = agent.act(obs, step=10 ** 9, greedy=True)

        next_obs, reward, term, trunc, info = env.step(action)
        terms = info["reward_terms"]
        row = {"exp": job["exp"], "agent": eval_agent, "cond": job["cond"],
               "seed": job["seed"], "ep": job["ep"], "step": step,
               "regime": regime, "action": action, "reward": reward,
               "S": terms["S"], "L": terms["L"], "U": terms["U"], "C": terms["C"],
               "rule_count": info["rule_count"], "collateral": info["collateral"],
               "t_wall": info.get("t_wall", time.time())}
        for name, val in zip(FEATURE_NAMES, obs):
            row[f"f_{name}"] = float(val)
        rows.append(row)
        if eval_agent == "A2":
            agent.observe(obs, action, reward, 1.0 if trunc else 0.0)
        obs = next_obs
        if trunc:
            break
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--shard", default="0/1", help="i/n, this VM runs shard i of n")
    p.add_argument("--out", default=None)
    p.add_argument("--smoke", action="store_true",
                   help="run 3 quick episodes to check the switch and logging")
    args = p.parse_args()

    cfg = yaml.safe_load(open(REPO / "configs" / "base.yaml"))
    h = Hyper.from_cfg(cfg)

    if args.smoke:
        return _smoke(cfg, h)

    i, n = (int(x) for x in args.shard.split("/"))
    jobs = [j for k, j in enumerate(build_jobs()) if k % n == i]
    print(f"shard {i}/{n}: {len(jobs)} of {len(build_jobs())} episodes")

    out = pathlib.Path(args.out or (REPO / "runs" / f"eval_shard{i}.parquet"))
    out.parent.mkdir(parents=True, exist_ok=True)

    env = TestbedEnv(run_id=f"eval{i}")
    rng = np.random.default_rng(1234 + i)
    all_rows: list[dict] = []
    cache: dict = {}
    started = time.time()
    try:
        for n_done, job in enumerate(jobs):
            key = (job["agent"], job["seed"])
            if key not in cache:
                # seed used for weights: eval seed 1000+s -> train seed s (0..4)
                w_seed = (job["seed"] - EVAL_SEED_BASE) % 5
                cache.clear()          # one policy in memory at a time
                cache[key] = load_agent(job["agent"], w_seed, h)
            rows = run_episode(env, cache[key], job["agent"], job, h, rng)
            all_rows.extend(rows)
            if (n_done + 1) % 5 == 0:
                _write(all_rows, out)
                el = (time.time() - started) / 60
                print(f"  [{n_done+1}/{len(jobs)}] {job['exp']} {job['agent']} "
                      f"{job['cond']} — {el:.1f} min elapsed", flush=True)
    finally:
        _write(all_rows, out)
        env.close()
    print(f"shard {i}/{n} done: {len(all_rows)} rows -> {out} "
          f"({(time.time()-started)/60:.1f} min)")
    return 0


def _write(rows: list[dict], out: pathlib.Path) -> None:
    if not rows:
        return
    import pandas as pd
    pd.DataFrame(rows).to_parquet(out, index=False)


def _smoke(cfg: dict, h: Hyper) -> int:
    """Three quick episodes to exercise the untested wrapper paths before the
    long queue: a plain episode, a mid-episode regime switch, and R4."""
    env = TestbedEnv(run_id="evalsmoke")
    rng = np.random.default_rng(0)
    checks = []
    try:
        probes = [
            dict(exp="E0", agent="A2", cond="R1", seed=1000, ep=0,
                 schedule=[["R1", BIG]], label="true"),
            dict(exp="E1", agent="A2", cond="R1->R2", seed=1000, ep=0,
                 schedule=[["R1", 50], ["R2", 50]], label="true"),
            dict(exp="E3", agent="A3miss", cond="R4", seed=1000, ep=0,
                 schedule=[["R4", BIG]], label="true"),
        ]
        for job in probes:
            agent = load_agent(job["agent"], 0, h)
            rows = run_episode(env, agent, job["agent"], job, h, rng)
            regimes = {r["regime"] for r in rows}
            switched = len(regimes) > 1
            print(f"{job['exp']} {job['agent']} {job['cond']}: "
                  f"{len(rows)} rows, regimes seen {sorted(regimes)}, "
                  f"mean S {np.mean([r['S'] for r in rows]):.2f}")
            if job["exp"] == "E1" and not switched:
                checks.append("E1 did NOT switch regime mid-episode")
            if len(rows) != env.steps_per_episode:
                checks.append(f"{job['exp']} logged {len(rows)} rows, "
                              f"expected {env.steps_per_episode}")
    finally:
        env.close()
    if checks:
        print("SMOKE FAILED:", checks)
        return 1
    print("SMOKE PASS: switch fires, R4 runs, logging complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
