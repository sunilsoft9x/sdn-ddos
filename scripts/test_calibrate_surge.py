"""Offline tests for scripts/calibrate_surge.py (no testbed).

    ~/venv/bin/python scripts/test_calibrate_surge.py

1. The pre-registered choice rule on constructed results.
2. run_episode against the fake network from test_wp1_events.py: the surge
   fires at step 10, rows carry one line per surge host per step, and the
   post-surge window starts at step 12.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cal = load("cal", REPO / "scripts" / "calibrate_surge.py")
failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def synthetic(shares):
    """shares[(n, rate)] = flag share for every host and seed, or a callable(seed, host)."""
    rows = []
    for (n, rate), share in shares.items():
        for seed in cal.SEEDS:
            for h in range(n):
                s = share(seed, h) if callable(share) else share
                k = round(s * 29)                     # 29 post-surge steps (12..40)
                for i, step in enumerate(range(1, cal.STEPS + 1)):
                    post_i = step - cal.FIRST_POST
                    rows.append({"n_hosts": n, "rate_mbps": rate, "seed": seed, "step": step,
                                 "host": f"h{h}", "flagged": post_i >= 0 and post_i < k,
                                 "current_pps": 100.0, "threshold_pps": 50.0,
                                 "other_flagged": 0, "S": 1.0})
    return pd.DataFrame(rows)


print("1. Pre-registered choice rule")
grid = {(n, r): 0.0 for n in cal.N_HOSTS for r in cal.RATES_MBPS}
_, choice = cal.summarise(synthetic(grid))
check(choice is None, "nothing passes -> no choice (negative calibration)")
g = dict(grid); g[(2, 4.0)] = 1.0; g[(2, 6.0)] = 1.0; g[(1, 2.0)] = 1.0
_, choice = cal.summarise(synthetic(g))
check(choice == (2, 4.0), f"largest passing n, lowest passing rate -> (2, 4.0); got {choice}")
g = dict(grid); g[(3, 6.0)] = lambda seed, h: 0.5 if (seed == cal.SEEDS[2] and h == 1) else 1.0
g[(1, 3.0)] = 1.0
_, choice = cal.summarise(synthetic(g))
check(choice == (1, 3.0), f"one weak host in one seed fails the combination -> (1, 3.0); got {choice}")
g = dict(grid); g[(3, 2.0)] = 27 / 29
_, choice = cal.summarise(synthetic(g))
check(choice == (3, 2.0), f"27 of 29 post-surge steps ({27/29:.3f}) passes")
g = dict(grid); g[(3, 2.0)] = 26 / 29
_, choice = cal.summarise(synthetic(g))
check(choice is None, f"26 of 29 post-surge steps ({26/29:.3f}) fails: 90% needs 27")
df = synthetic({(3, 2.0): 1.0}); df.loc[df.step < cal.FIRST_POST, "flagged"] = False
combo, _ = cal.summarise(df)
check(abs(combo.min_flag_share.iloc[0] - 1.0) < 1e-12, "unflagged steps before 12 do not count against a combination")

print("2. Output file")
check(pathlib.Path("runs") / f"{cal.ENV_RUN_ID}.parquet" != cal.OUT,
      "the environment's own log cannot overwrite the calibration rows")
src = (REPO / "scripts" / "calibrate_surge.py").read_text()
check("TestbedEnv(run_id=ENV_RUN_ID)" in src, "the script opens the environment under ENV_RUN_ID")

print("3. run_episode on the fake network")
t = load("t", REPO / "scripts" / "test_wp1_events.py")
new = t.load(REPO / "env" / "testbed_env.py", "new_env")
env = t.make_env(new)
env._src_ip = lambda h: f"10.0.0.{int(h.lstrip('h'))}"


class FakeDetector:
    def __init__(self):
        self._history = []
    def reset(self):
        self._history = []
    def snapshot_terms(self):
        return {"median_long_rate": 40.0, "threshold": 120.0, "n_active": 12,
                "suspects": sorted({env._src_ip(h) for h in env._surge_hosts} if env._event_fired else [])}
    def _long_window_means(self):
        return {}
    def suspects(self):
        return set(self.snapshot_terms()["suspects"])


env.detector = FakeDetector()
def obs(prev, curr, probe_window, churn, first):
    env.detector._history.append((0.0, {}))
    info = {"reward": 0.0, "reward_terms": {"S": 1.0}, "probe_n": 5}
    info.update(env._event_info())
    return None, info
env._observe = obs
rows = cal.run_episode(env, 3, 2.0, 2000001)
df = pd.DataFrame(rows)
check(len(df) == 3 * cal.STEPS, f"one row per surge host per step ({len(df)} rows)")
first = df[df.event_fired].step.min()
check(first == cal.AT + 1, f"surge first visible at step {cal.AT + 1} (got {first})")
check(df.groupby("step").host.nunique().eq(3).all(), "the same 3 surge hosts every step")
combo, choice = cal.summarise(df.assign(rate_mbps=2.0))
check(choice == (3, 2.0), "fake detector flags all surge hosts after the event -> passes")

print(f"\n{'ALL PASSED' if not failures else f'{len(failures)} FAILED'}")
sys.exit(1 if failures else 0)
