"""Calibration sweep (plan 6.7). Produces data/calib.parquet.

For every (regime, intensity scale, fixed mitigation state, attacker draw) it
resets the testbed to that condition, applies the mitigation to the true
attacker set, lets it settle, then logs 20 windows of the 18 features, the
probe metrics, victim utilisation and the ground-truth regime. The simulator
(Friday) is fitted to these rows; nothing here trains anything.

Because it sweeps the four mitigation states it is also the learnability check
the plan asks for: under R1 and R2, the 'drop' rows should show probe success
climbing back above 0.9 while the 'none' rows stay degraded.

Runs unattended for roughly 40 minutes at W = 0.5 s. Do not touch the VM while
it runs.

    sudo mn -c
    sudo ~/venv/bin/python scripts/calibrate.py
"""

from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from env.testbed_env import TestbedEnv  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]

# mitigation label -> action id (0 none, 1 drop, 2 meter50, 3 meter10)
MIT = {"none": 0, "meter50": 2, "meter10": 3, "drop": 1}

WINDOWS = 20
WARMUP_STEPS = 2          # let traffic and (if any) mitigation reach steady state


def build_settings() -> list[dict]:
    settings: list[dict] = []

    # R0: benign only, no mitigation, several draws.
    for draw in range(5):
        settings.append(dict(regime="R0", scale=1.0, mit="none",
                             draw=draw, holdout=False))

    # R1 and R2: full intensity x mitigation grid, 2 draws.
    for regime in ("R1", "R2"):
        for scale in (0.2, 0.4, 0.6, 0.8, 1.0, 1.2):
            for mit in ("none", "meter50", "meter10", "drop"):
                for draw in range(2):
                    settings.append(dict(regime=regime, scale=scale, mit=mit,
                                         draw=draw, holdout=False))
        # Hold-out intensity 0.7, all four mitigation states, never fitted.
        for mit in ("none", "meter50", "meter10", "drop"):
            for draw in range(2):
                settings.append(dict(regime=regime, scale=0.7, mit=mit,
                                     draw=draw, holdout=True))

    # R4: coarser grid, three mitigation states.
    for scale in (0.5, 1.0, 1.5):
        for mit in ("none", "meter50", "drop"):
            for draw in range(2):
                settings.append(dict(regime="R4", scale=scale, mit=mit,
                                     draw=draw, holdout=False))
    return settings


def run_setting(env: TestbedEnv, s: dict) -> list[dict]:
    # Deterministic seed per condition so a rerun reproduces the same draw.
    seed = abs(hash((s["regime"], s["scale"], s["mit"], s["draw"]))) % (2**31)
    env.reset(seed=seed, options={"schedule": [[s["regime"], 999999]],
                                  "scale": s["scale"], "n_attackers": 3})
    attacker_ips = {env._src_ip(a) for a in env._attackers}

    # Warm up, then apply the fixed mitigation to the true attacker set.
    for _ in range(WARMUP_STEPS):
        env.step(0)
    action = MIT[s["mit"]]
    if action != 0:
        env._apply_action(action, attacker_ips, env._last_rates)

    rows = []
    for w in range(WINDOWS):
        obs, reward, term, trunc, info = env.step(0)  # hold the mitigation
        terms = info["reward_terms"]
        row = {
            "regime": s["regime"], "scale": s["scale"], "mit": s["mit"],
            "draw": s["draw"], "holdout": s["holdout"], "window": w,
            "reward": reward, "S": terms["S"], "L": terms["L"],
            "U": terms["U"], "C": terms["C"],
            "probe_success": info["reward_terms"]["S"],
            "rule_count": info["rule_count"], "collateral": info["collateral"],
            "n_attackers": len(attacker_ips),
        }
        for name, val in zip(
            ["pkt_rate", "byte_rate", "mean_pkt_size", "tcp_share", "udp_share",
             "icmp_share", "src_entropy", "active_sources", "top_source_share",
             "victim_link_util", "victim_tx_dropped_rate", "edge_table_occupancy",
             "probe_success_feat", "probe_latency_median", "probe_goodput",
             "d_pkt_rate", "d_udp_share", "mitigated_fraction"], obs):
            row[f"f_{name}"] = float(val)
        rows.append(row)
    return rows


def main() -> int:
    settings = build_settings()
    print(f"calibration sweep: {len(settings)} settings x {WINDOWS} windows")
    est_min = len(settings) * (WARMUP_STEPS + WINDOWS) * 0.5 / 60 + len(settings) * 4 / 60
    print(f"estimated ~{est_min:.0f} min. Do not touch the VM.\n")

    env = TestbedEnv(run_id="calib")
    all_rows: list[dict] = []
    started = time.time()
    try:
        for i, s in enumerate(settings):
            rows = run_setting(env, s)
            all_rows.extend(rows)
            mean_S = sum(r["S"] for r in rows) / len(rows)
            flag = " [holdout]" if s["holdout"] else ""
            print(f"[{i+1:3d}/{len(settings)}] {s['regime']} x{s['scale']} "
                  f"{s['mit']:<8} draw{s['draw']} -> meanS={mean_S:.2f}{flag} "
                  f"({(time.time()-started)/60:.1f} min)")
            # Periodic checkpoint so a late crash does not lose everything.
            if (i + 1) % 20 == 0:
                _write(all_rows)
    finally:
        _write(all_rows)
        env.close()
    print(f"\ndone in {(time.time()-started)/60:.1f} min, "
          f"{len(all_rows)} rows -> data/calib.parquet")
    return 0


def _write(rows: list[dict]) -> None:
    if not rows:
        return
    import pandas as pd
    out = REPO / "data" / "calib.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)


if __name__ == "__main__":
    raise SystemExit(main())
