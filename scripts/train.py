"""Training runner (plan 7.3 pilot, 7.4 the fifteen runs).

Trains each agent in the calibrated simulator to 100k steps, five seeds, three
agents = fifteen runs. CPU only, one thread per process, twelve processes in
parallel, so the whole set finishes well inside the 12:30-15:00 window. Each run
checkpoints every 50 episodes and keeps the FINAL model (never a checkpoint
picked on evaluation data), and logs the per-episode reward for the convergence
figure.

  Pilot (7.3):   ~/venv/bin/python scripts/train.py --pilot
  Full (7.4):    ~/venv/bin/python scripts/train.py --all
  One run:       ~/venv/bin/python scripts/train.py --agent A2 --seed 0

Frozen weights and epsilon=0 at evaluation are the deployment claim; here we
only train and save.
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

from agents.common import Hyper, set_seed
from agents.context_ddqn import ContextDDQNAgent
from agents.ddqn import DDQNAgent
from agents.oracle_ddqn import OracleDDQNAgent
from agents.oracle_ddqn import REGIME_TO_TAU
from sim.model import ServiceModel
from sim.sim_env import SimEnv

REPO = pathlib.Path(__file__).resolve().parents[1]
AGENTS = {"A1": DDQNAgent, "A2": ContextDDQNAgent, "A3": OracleDDQNAgent}


def make_agent(agent_type: str, h: Hyper):
    return AGENTS[agent_type](h)


def train_one(agent_type: str, seed: int, steps: int, out_dir: pathlib.Path,
              cfg: dict, ckpt_every_ep: int = 50) -> dict:
    torch.set_num_threads(1)                 # one thread per process
    set_seed(seed)
    h = Hyper.from_cfg(cfg)
    env = SimEnv(cfg=cfg, service_model=ServiceModel.load(
        REPO / "sim" / "service_model.json"), run_id=f"{agent_type}_s{seed}")
    agent = make_agent(agent_type, h)

    out_dir.mkdir(parents=True, exist_ok=True)
    steps_per_ep = cfg["training"]["steps_per_episode"]
    global_step = 0
    episode = 0
    log: list[dict] = []
    started = time.time()

    while global_step < steps:
        obs, info = env.reset(seed=seed * 100000 + episode)
        if agent_type == "A2":
            agent.reset_history()
        regime = info["regime"]
        ep_reward = 0.0
        ep_S = 0.0
        for _ in range(steps_per_ep):
            if agent_type == "A3":
                tau = REGIME_TO_TAU.get(regime, "none")
                action = agent.act(obs, tau, global_step)
            else:
                action = agent.act(obs, global_step)
            next_obs, reward, term, trunc, info = env.step(action)
            next_regime = info["regime"]
            done = 1.0 if (term or trunc) else 0.0

            if agent_type == "A3":
                next_tau = REGIME_TO_TAU.get(next_regime, "none")
                agent.remember(obs, tau, action, reward, next_obs, next_tau, done)
            else:
                agent.remember(obs, action, reward, next_obs, done)
            agent.learn()

            obs, regime = next_obs, next_regime
            ep_reward += reward
            ep_S += info["reward_terms"]["S"]
            global_step += 1
            if trunc:
                break

        episode += 1
        log.append({"episode": episode, "global_step": global_step,
                    "ep_reward": ep_reward, "mean_S": ep_S / steps_per_ep})
        if episode % ckpt_every_ep == 0:
            torch.save(agent.online.state_dict(),
                       out_dir / f"{agent_type}_s{seed}_ep{episode}.pt")

    # Keep the FINAL model, not a best-on-eval checkpoint.
    torch.save(agent.online.state_dict(), out_dir / f"{agent_type}_s{seed}_final.pt")
    (out_dir / f"{agent_type}_s{seed}_log.json").write_text(json.dumps(log))
    dur = time.time() - started
    recent = np.mean([r["ep_reward"] for r in log[-20:]])
    return {"agent": agent_type, "seed": seed, "episodes": episode,
            "steps": global_step, "minutes": round(dur / 60, 1),
            "final20_reward": round(float(recent), 3)}


def _worker(args):
    agent_type, seed, steps, cfg = args
    out = REPO / "runs" / "models"
    return train_one(agent_type, seed, steps, out, cfg)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", action="store_true", help="one short A2 run")
    p.add_argument("--all", action="store_true", help="15 runs in parallel")
    p.add_argument("--agent", choices=list(AGENTS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--procs", type=int, default=12)
    args = p.parse_args()

    cfg = yaml.safe_load(open(REPO / "configs" / "base.yaml"))
    full_steps = (cfg["training"]["episodes"] * cfg["training"]["steps_per_episode"])

    if args.pilot:
        print("PILOT: A2, seed 0, 3000 steps")
        r = train_one("A2", 0, 3000, REPO / "runs" / "pilot", cfg, ckpt_every_ep=999)
        print(r)
        print("Check: final20_reward should be clearly above an early-episode "
              "reward, i.e. the loop learns. Then run --all.")
        return 0

    if args.agent:
        r = train_one(args.agent, args.seed, full_steps,
                      REPO / "runs" / "models", cfg)
        print(r)
        return 0

    if args.all:
        import multiprocessing as mp
        seeds = (cfg.get("meta", {}).get("seed_namespace", {}).get("train")
                 or cfg.get("training", {}).get("seed_namespace", {}).get("train")
                 or [0, 1, 2, 3, 4])
        tasks = [(a, s, full_steps, cfg) for a in AGENTS for s in seeds]
        print(f"training {len(tasks)} runs x {full_steps} steps on "
              f"{args.procs} processes")
        started = time.time()
        with mp.Pool(args.procs) as pool:
            for res in pool.imap_unordered(_worker, tasks):
                print(f"  done {res['agent']} seed{res['seed']}: "
                      f"{res['episodes']} ep, {res['minutes']} min, "
                      f"final20 reward {res['final20_reward']}")
        print(f"all runs done in {(time.time()-started)/60:.1f} min "
              f"-> runs/models/")
        return 0

    p.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
