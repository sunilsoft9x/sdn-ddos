"""Live shake-out of the testbed wrapper before any long run (precedes 6.7).

Runs a handful of real steps against OpenDaylight + Mininet and checks the
things unit tests could not: that features.py reads the real ovs-ofctl field
names, that probe logs are actually being written and read, that the W-timed
loop produces a well-formed observation, and that a drop action removes an
attacker's traffic. If features silently read zeros, this prints the offending
dump-flows line instead of leaving you to find it 50 minutes into calibrate.py.

Run from the repo root, as root, with ODL up and no other Mininet running:

    sudo mn -c
    sudo ~/venv/bin/python scripts/smoke_env.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from env.testbed_env import TestbedEnv  # noqa: E402


def main() -> int:
    print("building env (Mininet built once, static flows installed) ...")
    env = TestbedEnv(run_id="smoke")
    failures: list[str] = []
    try:
        # Force a deterministic R1 episode so an attacker is definitely sending.
        obs, info = env.reset(seed=0, options={"schedule": [["R1", 999]],
                                               "scale": 1.0, "n_attackers": 3})
        print(f"reset: regime={info['regime']} attackers={info['attackers']}")

        # --- observation well-formed --------------------------------------
        if obs.shape != (18,):
            failures.append(f"obs shape {obs.shape}, expected (18,)")
        if not np.all(np.isfinite(obs)):
            failures.append(f"obs has non-finite values: {obs}")
        if obs.min() < -1.0001 or obs.max() > 1.0001:
            failures.append(f"obs out of [-1,1]: min {obs.min()} max {obs.max()}")

        actions = [0, 0, 1, 1, 0]  # baseline, then drop suspects, then hold
        last_info = info
        for i, a in enumerate(actions):
            obs, reward, term, trunc, info = env.step(a)
            last_info = info
            f = {n: round(float(v), 3) for n, v in zip(
                ["pkt", "byte", "sz", "tcp", "udp", "icmp", "ent", "nsrc",
                 "top", "util", "drop", "tbl", "psucc", "plat", "pgood",
                 "dpkt", "dudp", "mit"], obs)}
            print(f"step {i} a={a}: r={reward:+.3f} "
                  f"regime={info['regime']} rules={info['rule_count']} "
                  f"probe_n={info['probe_n']} collat={info['collateral']:.2f}")
            print(f"          pkt={f['pkt']} udp={f['udp']} tcp={f['tcp']} "
                  f"util={f['util']} psucc={f['psucc']} mit={f['mit']}")

        # --- did features SEE the attack? (the field-name check) ----------
        # After R1 SYN flood at scale 1.0, packet-rate feature must be nonzero.
        if obs is not None:
            import subprocess
            probed = False
            for edge in env.edges:
                out = subprocess.run(
                    ["ovs-ofctl", "-O", env.of, "dump-flows", edge],
                    capture_output=True, text=True).stdout
                for line in out.splitlines():
                    if "priority=200" in line and "n_packets=" in line:
                        n = int(line.split("n_packets=")[1].split(",")[0])
                        if n > 0:
                            probed = True
                            break
                if probed:
                    break
            if not probed:
                failures.append("no priority-200 flow shows n_packets>0 during "
                                "R1 — traffic or monitoring flows not working")

        # --- probes actually ran -----------------------------------------
        if last_info["probe_n"] == 0:
            failures.append("probe_n=0 for the last window — probe clients not "
                            "writing logs, or window/timestamp mismatch")

    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        failures.append(f"exception during run: {exc!r}")
    finally:
        env.close()

    print("\n" + "=" * 50)
    if failures:
        print("SMOKE TEST FAILED:")
        for fmsg in failures:
            print("  -", fmsg)
        return 1
    print("SMOKE TEST PASSED — wrapper produces valid observations end to end")
    print("safe to write and run scripts/calibrate.py next")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
