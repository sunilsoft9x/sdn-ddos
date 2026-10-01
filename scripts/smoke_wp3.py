"""Live smoke run for IIPEM WP3 logging (run on a VM before any main run).

Checks on the real testbed what the offline test cannot:
  - the victim's iptables accounting works inside its namespace: all 15
    sources appear, before and after the event, with traffic counted;
  - the Karaf JVM is found and its CPU and memory are read every step;
  - the action time, flagged hosts and mitigated hosts are logged every step.
It also reports, without asserting, how the delivered bytes compare with the
victim-link utilisation feature, and what the static rule did to the surge
hosts and the former attackers (a preview of C1 and C2, not a result).

Run from the repo root, as root, with ODL up and no other Mininet running:

    sudo mn -c
    sudo ~/venv/bin/python scripts/smoke_wp3.py

4 episodes (surge and swap, noop and static arms, one seed), about 4 minutes.
Rows go to runs/smoke_wp3.parquet and delivered bytes to runs/smoke_wp3_delivered.json.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agents.static_rule import StaticRuleAgent  # noqa: E402
from env.testbed_env import TestbedEnv  # noqa: E402

SEED = 3000001
EVENTS = {
    "surge": {"schedule": [["R0", 100]], "iipem_log": True,
              "surge": {"n_hosts": 2, "rate_mbps": 4.0, "at": 50}},
    "swap": {"schedule": [["R2", 100]], "iipem_log": True, "swap": {"at": 50}},
}
OUT = pathlib.Path("runs/smoke_wp3.parquet")
OUT_DELIVERED = pathlib.Path("runs/smoke_wp3_delivered.json")
ENV_RUN_ID = "smoke_wp3_env"          # must differ from OUT's name (see WP2, step 15)
LINK_BYTES_PER_S = 10e6 / 8           # victim link, 10 Mbps


def run_episode(env, options, arm):
    static = StaticRuleAgent() if arm == "static" else None
    obs, info = env.reset(seed=SEED, options=options)
    rows, last = [], None
    for t in range(env.steps_per_episode):
        action = (static.act_from_suspects(len(env.detector.suspects()) > 0)
                  if static else 0)
        obs, r, term, trunc, info = env.step(action)
        rows.append({"arm": arm, "step": t + 1, "action": action,
                     "link_util": float(obs[9]), "t_wall": info["t_wall"],
                     "event_fired": info["event_fired"],
                     "attackers": ",".join(info["attackers"]),
                     "former_attackers": ",".join(info["former_attackers"]),
                     "surge_hosts": ",".join(info["surge_hosts"]),
                     "flagged": ",".join(info["flagged"]),
                     "mitigated": ",".join(info["mitigated"]),
                     "apply_ms": info["apply_ms"], "ctl_cpu_pct": info["ctl_cpu_pct"],
                     "ctl_rss_mb": info["ctl_rss_mb"], "ctl_pid": info["ctl_pid"],
                     "collateral": info["collateral"], "S": info["reward_terms"]["S"]})
        last = info
        if trunc:
            break
    return rows, {"pre": last.get("delivered_pre"), "post": last.get("delivered_post")}


def main() -> int:
    import pandas as pd
    env = TestbedEnv(run_id=ENV_RUN_ID)
    rows, delivered, failures = [], {}, []
    try:
        for name, options in EVENTS.items():
            for arm in ("noop", "static"):
                ep, dv = run_episode(env, options, arm)
                for r in ep:
                    r["event"] = name
                rows += ep
                delivered[f"{name}/{arm}"] = dv
                tag = f"{name} {arm}"
                cpu = [r["ctl_cpu_pct"] for r in ep]
                if ep[0]["ctl_pid"] is None:
                    failures.append(f"{tag}: Karaf JVM not found")
                elif sum(c is not None and not math.isnan(c) for c in cpu) < 0.9 * len(cpu):
                    failures.append(f"{tag}: controller CPU missing in more than 10% of steps")
                if any(r["apply_ms"] is None for r in ep):
                    failures.append(f"{tag}: action time missing")
                for phase in ("pre", "post"):
                    d = dv[phase] or {}
                    total = sum(v["bytes"] for v in d.values())
                    if len(d) != 15 or total <= 0:
                        failures.append(f"{tag}: {phase} accounting has {len(d)} sources, {total} bytes")
                print(f"{tag:12s}: Karaf pid {ep[0]['ctl_pid']}, "
                      f"delivered {sum(v['bytes'] for v in (dv['pre'] or {}).values()) / 1e6:.2f} MB before, "
                      f"{sum(v['bytes'] for v in (dv['post'] or {}).values()) / 1e6:.2f} MB after the event",
                      flush=True)
    finally:
        env.close()
        if rows:
            OUT.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(OUT, index=False)
            OUT_DELIVERED.write_text(json.dumps(delivered, indent=1))

    df = pd.DataFrame(rows)
    print("\nreported, not asserted:")
    for (name, arm), g in df.groupby(["event", "arm"]):
        dv = delivered[f"{name}/{arm}"]
        got = sum(v["bytes"] for p in ("pre", "post") for v in (dv[p] or {}).values())
        est = (g.link_util * LINK_BYTES_PER_S * g.t_wall.diff().fillna(0.5)).sum()
        print(f"  {name:5s} {arm:6s}: delivered / (link utilisation x capacity x time) = {got / est:.2f}; "
              f"mean apply {g.apply_ms.mean():.1f} ms; Karaf CPU {g.ctl_cpu_pct.mean():.1f}% of a core, "
              f"RSS {g.ctl_rss_mb.mean():.0f} MB")
    st = df[(df.event == "surge") & (df.arm == "static") & (df.step >= 52)]
    if len(st):
        surge_ips = {f"10.0.0.{int(h.lstrip('h'))}" for h in st.surge_hosts.iloc[0].split(",") if h}
        share = st.mitigated.apply(lambda m: bool(surge_ips & set(filter(None, m.split(","))))).mean()
        print(f"  surge static: a surge host had a rule in {100 * share:.1f}% of post-surge steps")
    sw = df[(df.event == "swap") & (df.arm == "static") & (df.step >= 52)]
    if len(sw):
        former = {f"10.0.0.{int(h.lstrip('h'))}" for h in sw.former_attackers.iloc[0].split(",") if h}
        share = sw.mitigated.apply(lambda m: bool(former & set(filter(None, m.split(","))))).mean()
        print(f"  swap static: a former attacker still had a rule in {100 * share:.1f}% of post-swap steps")
    print(f"\nrows in {OUT}, delivered bytes in {OUT_DELIVERED}")
    print("ALL CHECKS PASSED" if not failures else "FAILED:\n  " + "\n  ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
