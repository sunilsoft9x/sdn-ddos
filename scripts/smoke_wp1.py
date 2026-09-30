"""Live smoke run for the IIPEM WP1 events (run on a VM, before any main run).

Checks on the real testbed what the offline test cannot:
  - the probes keep reporting after the step-50 restart: the share of steps
    with at least one probe result after the event is at least half the share
    before it (a flood can leave single windows empty, before and after alike);
  - the event fires once, in the window after step 50;
  - two arms on the same episode seed get the same roles;
  - after a swap, collateral is measured against the new roles.
It also reports, without asserting, how often the surge hosts are flagged as
suspects at the placeholder rate. That is a preview of WP2, not the calibration.

Run from the repo root, as root, with ODL up and no other Mininet running:

    sudo mn -c
    sudo ~/venv/bin/python scripts/smoke_wp1.py

12 episodes (2 events x 3 seeds x 2 arms), about 11 minutes. Per-step rows go
to runs/smoke_wp1.parquet.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agents.static_rule import StaticRuleAgent  # noqa: E402
from env.testbed_env import TestbedEnv  # noqa: E402

PLACEHOLDER_RATE_MBPS = 2.0      # WP2 fixes the real surge rate
SEEDS = (1000001, 1000002, 1000003)
EVENTS = {
    "surge": {"schedule": [["R0", 100]],
              "surge": {"n_hosts": 3, "rate_mbps": PLACEHOLDER_RATE_MBPS, "at": 50}},
    "swap": {"schedule": [["R2", 100]], "swap": {"at": 50}},
}


def run_episode(env, options, seed, arm):
    static = StaticRuleAgent() if arm == "static" else None
    obs, info = env.reset(seed=seed, options=options)
    rows = []
    for t in range(env.steps_per_episode):
        action = (static.act_from_suspects(len(env.detector.suspects()) > 0)
                  if static else 0)
        obs, r, term, trunc, info = env.step(action)
        suspects = env.detector.suspects()          # flags for the window just observed
        surge_ips = {env._src_ip(h) for h in info["surge_hosts"]}
        rows.append({
            "arm": arm, "seed": seed, "step": t + 1, "action": action,
            "probe_n": info["probe_n"], "event_fired": info["event_fired"],
            "attackers": ",".join(info["attackers"]),
            "former_attackers": ",".join(info["former_attackers"]),
            "surge_hosts": ",".join(info["surge_hosts"]),
            "surge_flagged": len(surge_ips & suspects) if surge_ips else 0,
            "collateral": info["collateral"], "rule_count": info["rule_count"],
            "S": info["reward_terms"]["S"],
        })
        if trunc:
            break
    return rows


def main() -> int:
    import pandas as pd
    env = TestbedEnv(run_id="smoke_wp1")
    rows, failures = [], []
    try:
        for name, options in EVENTS.items():
            for seed in SEEDS:
                per_arm = {}
                for arm in ("noop", "static"):
                    ep = run_episode(env, options, seed, arm)
                    for r in ep:
                        r["event"] = name
                    rows += ep
                    per_arm[arm] = ep
                    fired = [r["step"] for r in ep if r["event_fired"]]
                    first = fired[0] if fired else None
                    pre = [r["probe_n"] > 0 for r in ep if 3 <= r["step"] <= 50]
                    post = [r["probe_n"] > 0 for r in ep if r["step"] >= 52]
                    pre_share, post_share = sum(pre) / len(pre), sum(post) / len(post)
                    print(f"{name:5s} seed {seed} {arm:6s}: fired from step {first}, "
                          f"steps with probe results: before {pre_share:.3f}, after {post_share:.3f}")
                    if first != 51:
                        failures.append(f"{name} {seed} {arm}: event first seen at {first}, expected 51")
                    if post_share == 0 or post_share < 0.5 * pre_share:
                        failures.append(f"{name} {seed} {arm}: probes stopped reporting after the event")
                roles = lambda ep: [(r["attackers"], r["surge_hosts"], r["former_attackers"]) for r in ep]
                if roles(per_arm["noop"]) != roles(per_arm["static"]):
                    failures.append(f"{name} {seed}: the two arms saw different roles")
    finally:
        env.close()

    df = pd.DataFrame(rows)
    out = pathlib.Path("runs/smoke_wp1.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    post = df[(df.event == "surge") & (df.step >= 52)]
    print(f"\npreview for WP2 (not a pass/fail): at {PLACEHOLDER_RATE_MBPS} Mbps, all 3 surge hosts "
          f"flagged in {100 * (post.surge_flagged == 3).mean():.1f}% of post-surge steps "
          f"(noop arm {100 * (post[post.arm == 'noop'].surge_flagged == 3).mean():.1f}%)")
    sw = df[(df.event == "swap") & (df.arm == "static")]
    print(f"swap, static arm: mean collateral before step 50 = {sw[sw.step <= 50].collateral.mean():.3f}, "
          f"after step 51 = {sw[sw.step >= 52].collateral.mean():.3f}")
    print(f"\nrows written to {out}")
    print("ALL CHECKS PASSED" if not failures else "FAILED:\n  " + "\n  ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
