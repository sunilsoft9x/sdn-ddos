"""R4 mechanism diagnostic (pre-eval sanity, not a paper artifact).

The E3 claim lives in the burst windows, not the episode mean: a pulsed
attack leaves the link clear ~80% of the time, so mean S is dominated by the
idle majority and cannot tell an agent that HOLDS mitigation across the silent
gaps from one that tears it down every gap. This script runs a few R4 episodes
for A3miss (do-nothing baseline), A1 (memoryless), and A2 (context/GRU) and
reports burst-conditioned readouts:

  hotS   mean S over the hottest windows (the bursts)         -> is R4 doing damage?
  coldS  mean S over the calm windows (the gaps)              -> both should be ~1
  minS   worst single-window S                                -> depth of the dip
  gapR   mean rule_count during calm windows                  -> is the meter HELD in gaps?
  hotR   mean rule_count during hot windows                   -> is the meter up at burst onset?

Read it like this: if A3miss's hotS/minS barely dip, the pulse is too easy to
coast through and we harden the duty cycle. If A3miss craters in the bursts and
A2's gapR stays high while A1's gapR collapses, the mechanism is real and mean S
was just hiding it. "Hot" windows are defined empirically as the top-40% by
victim_link_util within each episode, so this needs no timing assumptions.

    sudo ~/venv/bin/python scripts/r4_probe.py            # 3 eps each, ~8-10 min
    sudo ~/venv/bin/python scripts/r4_probe.py --eps 2
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agents.common import Hyper
from scripts.evaluate import BIG, load_agent, run_episode
from env.testbed_env import TestbedEnv

HOT_FRACTION = 0.40          # top 40% of windows by link util = "in a burst"


def summarize(rows: list[dict]) -> dict:
    S = np.array([r["S"] for r in rows], dtype=float)
    R = np.array([r["rule_count"] for r in rows], dtype=float)
    util = np.array([r["f_victim_link_util"] for r in rows], dtype=float)
    if len(S) == 0:
        return {}
    # Hot = windows whose util is in the top HOT_FRACTION for this episode.
    thresh = np.quantile(util, 1.0 - HOT_FRACTION)
    hot = util >= thresh
    cold = ~hot
    return {
        "hotS": float(S[hot].mean()) if hot.any() else float("nan"),
        "coldS": float(S[cold].mean()) if cold.any() else float("nan"),
        "minS": float(S.min()),
        "gapR": float(R[cold].mean()) if cold.any() else float("nan"),
        "hotR": float(R[hot].mean()) if hot.any() else float("nan"),
        "meanS": float(S.mean()),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--eps", type=int, default=3, help="episodes per agent")
    args = p.parse_args()

    cfg = yaml.safe_load(open(pathlib.Path(__file__).resolve().parents[1]
                              / "configs" / "base.yaml"))
    h = Hyper.from_cfg(cfg)
    env = TestbedEnv(run_id="r4probe")
    rng = np.random.default_rng(7)

    agents = ["A3miss", "A1", "A2"]
    print(f"R4 mechanism probe: {args.eps} episode(s) each, "
          f"hot = top {int(HOT_FRACTION*100)}% windows by link util\n")
    print(f"{'agent':<8} {'hotS':>6} {'coldS':>6} {'minS':>6} "
          f"{'gapR':>6} {'hotR':>6} {'meanS':>6}")
    try:
        for a in agents:
            agent = load_agent(a, 0, h)
            accum: list[dict] = []
            for ep in range(args.eps):
                job = dict(exp="E3", agent=a, cond="R4", seed=1000 + ep, ep=0,
                           schedule=[["R4", BIG]], label="true")
                accum.append(summarize(run_episode(env, agent, a, job, h, rng)))
            m = {k: np.nanmean([d[k] for d in accum]) for k in accum[0]}
            print(f"{a:<8} {m['hotS']:>6.2f} {m['coldS']:>6.2f} {m['minS']:>6.2f} "
                  f"{m['gapR']:>6.2f} {m['hotR']:>6.2f} {m['meanS']:>6.2f}")
    finally:
        env.close()

    print("\nread: R4 is real if A3miss hotS/minS are clearly below its coldS.")
    print("      A2 wins the mechanism if its gapR stays high while A1's gapR")
    print("      collapses toward 0 (A1 clears the rule in the gaps, A2 holds).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
