"""Suspect-rule check (plan 6.8, gate G4 second half).

Runs the volume-only suspect detector live against R1, R2 and R4 with no
mitigation, and measures the two rates G4 cares about: the share of windows in
which every attacker is flagged, and the share in which any benign or probe
host is flagged. The detector reads calib data indirectly through the live env,
because the per-source rates it needs were not logged in calib.parquet.

G4 pass: attackers flagged in >= 90% of windows for R1, R2 and R4, and a benign
or probe host flagged in <= 2%. If only R4 fails, the plan's remedy is to lower
the median multiplier to 2.5 or raise R4's burst rate.

    sudo mn -c
    sudo ~/venv/bin/python scripts/suspect_check.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from env.testbed_env import TestbedEnv  # noqa: E402

REGIMES = ["R1", "R2", "R4"]
WINDOWS = 40
WARMUP = 6          # let the 2 s history fill before judging flags


def check_regime(env: TestbedEnv, regime: str) -> dict:
    env.reset(seed=4321, options={"schedule": [[regime, 999999]],
                                  "scale": 1.0, "n_attackers": 3})
    attacker_ips = {env._src_ip(a) for a in env._attackers}
    probe_ips = {env._src_ip(p) for p in env.cfg["roles"]["probes"]}

    all_flagged = 0
    any_innocent_flagged = 0
    judged = 0
    for step in range(WARMUP + WINDOWS):
        env.step(0)  # no mitigation; detector still updates each window
        if step < WARMUP:
            continue
        suspects = env.detector.suspects()
        judged += 1
        if attacker_ips.issubset(suspects):
            all_flagged += 1
        innocent = suspects - attacker_ips        # benign + any flagged probe
        if innocent:
            any_innocent_flagged += 1

    return {
        "regime": regime,
        "attackers": sorted(attacker_ips),
        "all_attackers_flagged_rate": all_flagged / judged,
        "innocent_flagged_rate": any_innocent_flagged / judged,
        "windows": judged,
    }


def main() -> int:
    env = TestbedEnv(run_id="suspectchk")
    results = []
    try:
        for regime in REGIMES:
            r = check_regime(env, regime)
            results.append(r)
            print(f"{regime}: attackers flagged in "
                  f"{r['all_attackers_flagged_rate']:.0%} of windows, "
                  f"innocent flagged in {r['innocent_flagged_rate']:.0%}  "
                  f"(n={r['windows']}, attackers={r['attackers']})")
    finally:
        env.close()

    print("\n" + "=" * 55)
    ok = all(r["all_attackers_flagged_rate"] >= 0.90
             and r["innocent_flagged_rate"] <= 0.02 for r in results)
    if ok:
        print("PASS G4: attackers flagged >=90%, innocents <=2% for all regimes")
    else:
        for r in results:
            if r["all_attackers_flagged_rate"] < 0.90:
                print(f"  {r['regime']}: attacker flag rate low "
                      f"({r['all_attackers_flagged_rate']:.0%}) - "
                      "raise its rate or lower median_multiplier to 2.5")
            if r["innocent_flagged_rate"] > 0.02:
                print(f"  {r['regime']}: innocent flag rate high "
                      f"({r['innocent_flagged_rate']:.0%}) - raise multiplier")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
