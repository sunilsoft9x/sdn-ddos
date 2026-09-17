"""Diagnose why probes fail under R0 (no attack). One episode, raw failure mode.

R0 success should be ~1.0. When it is not, the aggregate number hides the cause.
This runs one no-attack episode and reports, from the raw probe records, whether
failures are timeouts (latency at the 0.5 s ceiling -> packets dropped or badly
delayed, e.g. benign bufferbloat) or fast failures (connection refused/reset ->
server or accept-queue problem), plus the latency spread and per-window counts.

    sudo mn -c
    sudo ~/venv/bin/python scripts/diag_probe.py
"""

from __future__ import annotations

import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from env.testbed_env import TestbedEnv  # noqa: E402


def summarize(records: list[dict], timeout_s: float) -> None:
    if not records:
        print("  NO probe records at all — clients not writing (path/cwd?)")
        return
    n = len(records)
    ok = [r for r in records if r["ok"]]
    bad = [r for r in records if not r["ok"]]
    timeouts = [r for r in bad if r["latency"] >= timeout_s - 0.02]
    fast_fail = [r for r in bad if r["latency"] < timeout_s - 0.02]
    print(f"  attempts        : {n}")
    print(f"  succeeded       : {len(ok)} ({len(ok)/n:.2%})")
    print(f"  failed          : {len(bad)}")
    print(f"    - timeouts    : {len(timeouts)}  (latency ~{timeout_s}s: dropped/delayed)")
    print(f"    - fast fails  : {len(fast_fail)}  (refused/reset quickly)")
    if ok:
        lat = sorted(r["latency"] for r in ok)
        print(f"  ok latency (ms) : min {lat[0]*1000:.0f}  "
              f"med {statistics.median(lat)*1000:.0f}  max {lat[-1]*1000:.0f}")
    if fast_fail:
        lat = sorted(r["latency"] for r in fast_fail)
        print(f"  fastfail lat(ms): min {lat[0]*1000:.0f}  "
              f"med {statistics.median(lat)*1000:.0f}  max {lat[-1]*1000:.0f}")


def main() -> int:
    env = TestbedEnv(run_id="diag")
    timeout_s = env.cfg["roles"]["probe"]["timeout_s"]
    try:
        print("=== R0 (no attack): one episode ===")
        env.reset(seed=7, options={"schedule": [["R0", 9999]], "n_attackers": 3})
        for i in range(12):
            obs, reward, term, trunc, info = env.step(0)
            print(f"  window {i:2d}: probe_n={info['probe_n']} "
                  f"S={info['reward_terms']['S']:.2f} util={obs[9]:.2f}")

        # Read the raw probe logs the episode just wrote.
        all_records: list[dict] = []
        for name in env.cfg["roles"]["probes"]:
            p = pathlib.Path("logs") / f"probe_{name}.jsonl"
            recs = []
            if p.exists():
                for line in p.read_text().splitlines():
                    if line.strip():
                        try:
                            recs.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            print(f"\nprobe host {name}: {len(recs)} records")
            summarize(recs, timeout_s)
            all_records.extend(recs)

        print("\n=== all probe hosts combined ===")
        summarize(all_records, timeout_s)

        # Cross-check: is the victim server even reachable right now?
        victim = env.net.get(env.cfg["topology"]["victim"]["name"])
        probe0 = env.net.get(env.cfg["roles"]["probes"][0])
        out = probe0.cmd(f"curl -s -o /dev/null -w '%{{http_code}} %{{time_total}}' "
                         f"--max-time 1 http://{env.victim_ip}/ ; echo")
        print(f"\nmanual curl {env.cfg['roles']['probes'][0]} -> victim: {out.strip()}")
        srv = victim.cmd("ss -ltn 2>/dev/null | grep ':80' || echo 'NO LISTENER on :80'")
        print(f"victim listener on :80: {srv.strip()}")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
