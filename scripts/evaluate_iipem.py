"""IIPEM evaluation runner (WP5): the two validation conditions on the testbed.

Same frozen policies, same loading and labelling code as ISASF's
scripts/evaluate.py (imported from it, not copied), with:

  conditions  C1 surge: R0, 2 benign hosts jump to 4.0 Mbps at step 50
              (fixed by the pre-registered WP2 calibration)
              C2 swap:  R2 flood, at step 50 the 3 attackers turn benign and
              3 benign hosts take over the flood
  arms        A1, A2, A3 (true label: "none" in R0, "udp" in R2), static rule
  seeds       eval seeds 1000..1009 -> training seeds 0..9, 5 episodes each
  grid        2 conditions x 4 arms x 10 seeds x 5 episodes = 400 episodes
  pairing     every arm of an episode uses the same environment seed
              (seed * 1000 + ep), so it sees the same attackers, surge hosts
              and benign rates; the 4 arms of an episode run back to back
  sharding    by seed: shard i of n runs the seeds with (seed - 1000) % n == i,
              so all arms of a seed run on the same VM
  logging     iipem_log on: flags, rules, action time, controller load per
              step; delivered bytes per source per phase per episode;
              inference time per step measured here

  VM1:  sudo ~/venv/bin/python scripts/evaluate_iipem.py --shard 0/2
  VM2:  sudo ~/venv/bin/python scripts/evaluate_iipem.py --shard 1/2
  quick end-to-end check first (8 episodes, ~8 min):  --smoke

Outputs (per shard): runs/iipem_shard<i>.parquet (one row per step) and
runs/iipem_shard<i>_delivered.parquet (one row per episode, phase and source).
A shard that stops part-way can be restarted with the same command: episodes
already complete in its output are skipped.
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys
import time
from time import perf_counter

import numpy as np
import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

EVAL_SEED_BASE = 1000
N_SEEDS = 10
EPISODES_PER_SEED = 5
ARMS = ("A1", "A2", "A3", "static")
SURGE = {"n_hosts": 2, "rate_mbps": 4.0, "at": 50}     # WP2, pre-registered choice
CONDITIONS = {
    "C1_surge": {"schedule": [["R0", 10 ** 9]], "surge": SURGE},
    "C2_swap": {"schedule": [["R2", 10 ** 9]], "swap": {"at": 50}},
}
KEY = ["cond", "agent", "seed", "ep"]


def _isasf():
    """ISASF's evaluate.py as a module, for load_agent, oracle_label and FEATURE_NAMES."""
    spec = importlib.util.spec_from_file_location("isasf_evaluate", REPO / "scripts" / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_jobs() -> list[dict]:
    """Episode order: seed, episode, condition, then the 4 arms back to back."""
    jobs = []
    for s in range(N_SEEDS):
        seed = EVAL_SEED_BASE + s
        for ep in range(EPISODES_PER_SEED):
            for cond in CONDITIONS:
                for agent in ARMS:
                    jobs.append({"cond": cond, "agent": agent, "seed": seed, "ep": ep,
                                 "w_seed": s, "env_seed": seed * 1000 + ep})
    return jobs


def shard_jobs(jobs: list[dict], i: int, n: int) -> list[dict]:
    return [j for j in jobs if (j["seed"] - EVAL_SEED_BASE) % n == i]


def run_episode(env, agent, job: dict, isasf, rng) -> tuple[list[dict], list[dict]]:
    options = dict(CONDITIONS[job["cond"]], scale=1.0, n_attackers=3, iipem_log=True)
    obs, info = env.reset(seed=job["env_seed"], options=options)
    arm = job["agent"]
    if arm == "A2":
        agent.reset_history()
    if arm == "static":
        agent.reset()
    rows, regime_hist = [], []
    for step in range(env.steps_per_episode):
        regime = info["regime"]
        regime_hist.append(regime)
        t0 = perf_counter()
        if arm == "static":
            action = agent.act_from_suspects(len(env.detector.suspects()) > 0)
        elif arm == "A3":
            tau = isasf.oracle_label(regime, "A3", "true", step, regime_hist, rng)
            action = agent.act(obs, tau, step=10 ** 9, greedy=True)
        else:
            action = agent.act(obs, step=10 ** 9, greedy=True)
        infer_ms = (perf_counter() - t0) * 1000.0

        next_obs, reward, term, trunc, info = env.step(action)
        terms = info["reward_terms"]
        row = {**{k: job[k] for k in KEY}, "step": step, "regime": regime,
               "action": int(action), "reward": reward,
               "S": terms["S"], "L": terms["L"], "U": terms["U"], "C": terms["C"],
               "rule_count": info["rule_count"], "collateral": info["collateral"],
               "probe_n": info["probe_n"], "t_wall": info["t_wall"],
               "infer_ms": infer_ms, "apply_ms": info["apply_ms"],
               "ctl_cpu_pct": info["ctl_cpu_pct"], "ctl_rss_mb": info["ctl_rss_mb"],
               "event_fired": info["event_fired"],
               "attackers": ",".join(info["attackers"]),
               "former_attackers": ",".join(info["former_attackers"]),
               "surge_hosts": ",".join(info["surge_hosts"]),
               "flagged": ",".join(info["flagged"]),
               "mitigated": ",".join(info["mitigated"])}
        for name, val in zip(isasf.FEATURE_NAMES, obs):
            row[f"f_{name}"] = float(val)
        rows.append(row)
        if arm == "A2":
            agent.observe(obs, action, reward, 1.0 if trunc else 0.0)
        obs = next_obs
        if trunc:
            break
    delivered = []
    for phase in ("pre", "post"):
        for src, c in (info.get(f"delivered_{phase}") or {}).items():
            delivered.append({**{k: job[k] for k in KEY}, "phase": phase, "src": src,
                              "pkts": c["pkts"], "bytes": c["bytes"]})
    return rows, delivered


def completed(out_steps: pathlib.Path, steps_per_episode: int) -> set[tuple]:
    """Episodes already complete in an earlier run of this shard."""
    if not out_steps.exists():
        return set()
    import pandas as pd
    df = pd.read_parquet(out_steps, columns=KEY + ["step"])
    n = df.groupby(KEY).step.count()
    return {k for k, v in n.items() if v == steps_per_episode}


def _write(rows, path):
    if rows:
        import pandas as pd
        pd.DataFrame(rows).to_parquet(path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1", help="i/n: this VM runs shard i of n (split by seed)")
    ap.add_argument("--smoke", action="store_true", help="8 episodes: each condition x arm, seed 1000, episode 0")
    args = ap.parse_args()

    import torch  # noqa: F401  (fail early if the venv is wrong)
    from agents.common import Hyper
    from env.testbed_env import TestbedEnv
    isasf = _isasf()
    cfg = yaml.safe_load(open(REPO / "configs" / "base.yaml"))
    h = Hyper.from_cfg(cfg)

    i, n = (int(x) for x in args.shard.split("/"))
    if args.smoke:
        jobs = [j for j in build_jobs() if j["seed"] == EVAL_SEED_BASE and j["ep"] == 0]
        tag = "smoke"
    else:
        jobs = shard_jobs(build_jobs(), i, n)
        tag = f"shard{i}"
    out_steps = REPO / "runs" / f"iipem_{tag}.parquet"
    out_deliv = REPO / "runs" / f"iipem_{tag}_delivered.parquet"
    out_steps.parent.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    rows, deliv = [], []
    if not args.smoke and out_steps.exists():
        rows = pd.read_parquet(out_steps).to_dict("records")
        if out_deliv.exists():
            deliv = pd.read_parquet(out_deliv).to_dict("records")
    env = TestbedEnv(run_id=f"iipem_env_{tag}")      # differs from both output names
    done = completed(out_steps, env.steps_per_episode) if not args.smoke else set()
    rows = [r for r in rows if tuple(r[k] for k in KEY) in done]
    deliv = [r for r in deliv if tuple(r[k] for k in KEY) in done]
    todo = [j for j in jobs if tuple(j[k] for k in KEY) not in done]
    print(f"{tag}: {len(jobs)} episodes, {len(done)} already complete, {len(todo)} to run", flush=True)

    rng = np.random.default_rng(1234 + i)            # used only by degraded labels; A3 here is "true"
    started = time.time()
    try:
        for k, job in enumerate(todo):
            agent = isasf.load_agent(job["agent"], job["w_seed"], h)
            r, d = run_episode(env, agent, job, isasf, rng)
            rows += r
            deliv += d
            _write(rows, out_steps)
            _write(deliv, out_deliv)
            print(f"  [{k + 1}/{len(todo)}] {job['cond']} {job['agent']:6s} seed {job['seed']} ep {job['ep']}"
                  f" - {len(r)} steps, {(time.time() - started) / 60:.1f} min", flush=True)
    finally:
        env.close()
        _write(rows, out_steps)
        _write(deliv, out_deliv)
    print(f"{tag} done: {len(rows)} step rows -> {out_steps}; {len(deliv)} delivered rows -> {out_deliv}")
    if args.smoke:
        return smoke_checks(pd.DataFrame(rows), pd.DataFrame(deliv), env.steps_per_episode)
    return 0


def smoke_checks(df, dv, steps_per_episode) -> int:
    fails = []
    n = df.groupby(KEY).step.count()
    if len(n) != 8 or (n != steps_per_episode).any():
        fails.append(f"expected 8 episodes of {steps_per_episode} steps, got {n.to_dict()}")
    for cond, g in df.groupby("cond"):
        roles = g[g.step == steps_per_episode - 1].groupby("agent")[["attackers", "former_attackers", "surge_hosts"]].first()
        if len(roles.drop_duplicates()) != 1:
            fails.append(f"{cond}: arms saw different roles")
    if df.infer_ms.isna().any() or df.apply_ms.isna().any() or df.ctl_cpu_pct.isna().mean() > 0.1:
        fails.append("timing or controller fields missing")
    a3r0 = df[(df.agent == "A3") & (df.cond == "C1_surge")]
    if (a3r0.action != 0).any():
        fails.append("A3 acted in R0 although its label is 'none'")
    if dv.groupby(KEY + ["phase"]).src.count().ne(15).any() or len(dv) != 8 * 2 * 15:
        fails.append(f"delivered rows: expected 240 (8 episodes x 2 phases x 15 sources), got {len(dv)}")
    print("SMOKE PASS" if not fails else "SMOKE FAILED:\n  " + "\n  ".join(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
