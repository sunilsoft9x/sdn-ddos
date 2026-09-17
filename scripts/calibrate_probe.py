"""Dose-response probe for rate calibration (the tuning part of 6.6).

Runs the real environment under each regime at several intensity scales with NO
mitigation, and reports the resulting probe success, victim utilisation and
packet-rate. Use it to land each regime in its target band before the full
calibrate.py sweep, because the sweep and the simulator are only meaningful once
the attack actually degrades service.

Target bands (plan 6.6):
    R0            success >= 0.95
    R1, R2        success 0.30 - 0.60
    R4            success 0.60 - 0.85, mean utilisation < 0.70

The environment is built once and reused; each setting is one no-op episode of a
few windows, warm-up windows dropped.

    sudo mn -c
    sudo ~/venv/bin/python scripts/calibrate_probe.py
"""

from __future__ import annotations

import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from env.testbed_env import TestbedEnv  # noqa: E402

SETTINGS = [
    ("R0", 1.0),
    ("R1", 0.5), ("R1", 1.0), ("R1", 1.5),
    ("R2", 0.5), ("R2", 1.0), ("R2", 1.5),
    ("R4", 0.5), ("R4", 1.0), ("R4", 1.5),
]
WINDOWS = 12
WARMUP = 4          # drop the first few windows while the attack ramps


def band(regime: str, success: float, util: float) -> str:
    if regime == "R0":
        return "OK" if success >= 0.95 else "LOW (want >=0.95)"
    if regime in ("R1", "R2"):
        if 0.30 <= success <= 0.60:
            return "IN BAND"
        return "too mild" if success > 0.60 else "too harsh"
    if regime == "R4":
        ok = 0.60 <= success <= 0.85 and util < 0.70
        if ok:
            return "IN BAND"
        if success > 0.85:
            return "too mild"
        if success < 0.60:
            return "too harsh"
        return "util too high" if util >= 0.70 else "check"
    return ""


def run_setting(env: TestbedEnv, regime: str, scale: float) -> dict:
    env.reset(seed=1234, options={"schedule": [[regime, 9999]],
                                  "scale": scale, "n_attackers": 3})
    succ, util, pkt = [], [], []
    for i in range(WINDOWS):
        obs, reward, term, trunc, info = env.step(0)  # never mitigate
        if i >= WARMUP:
            succ.append(info["reward_terms"]["S"])
            util.append(float(obs[9]))
            pkt.append(float(obs[0]))
    return {
        "success": statistics.fmean(succ),
        "util": statistics.fmean(util),
        "pkt": statistics.fmean(pkt),
        "success_sd": statistics.pstdev(succ) if len(succ) > 1 else 0.0,
    }


def main() -> int:
    env = TestbedEnv(run_id="dose")
    rows = []
    try:
        for regime, scale in SETTINGS:
            r = run_setting(env, regime, scale)
            verdict = band(regime, r["success"], r["util"])
            rows.append((regime, scale, r, verdict))
            print(f"{regime} x{scale:<4} success={r['success']:.2f}"
                  f"(+-{r['success_sd']:.2f}) util={r['util']:.2f} "
                  f"pkt={r['pkt']:.2f}  -> {verdict}")
    finally:
        env.close()

    print("\n" + "=" * 60)
    print("Pick, per regime, the scale whose success sits in band. Put R1/R2")
    print("nominal_pps and R4 burst rate in base.yaml so scale 1.0 lands there,")
    print("then run the full scripts/calibrate.py sweep.")
    # Flag the learnability check the plan wants: R1/R2 must be reachable.
    for regime in ("R1", "R2", "R4"):
        hits = [s for (rg, s, r, v) in rows if rg == regime and "IN BAND" in v or
                (rg == regime and v == "OK")]
        if not hits:
            print(f"  WARNING: no tested scale put {regime} in band — widen the "
                  "sweep or change its base rate/mechanism before calibrating.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
