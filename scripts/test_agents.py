"""Agent unit tests — gate G6 (plan 7.2).

Checks the two things the gate names: parameter counts within 2% across A1/A2/A3,
and a working train step for each. The R4-never-in-training invariant is enforced
inside sim_env._configure_episode and checked separately in the training run.

    ~/venv/bin/python scripts/test_agents.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agents.common import Hyper, check_param_parity, set_seed
from agents.context_ddqn import ContextDDQNAgent
from agents.ddqn import DDQNAgent
from agents.oracle_ddqn import OracleDDQNAgent
from agents.static_rule import StaticRuleAgent

REPO = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    set_seed(0)
    cfg = yaml.safe_load(open(REPO / "configs" / "base.yaml"))
    h = Hyper.from_cfg(cfg)

    a1, a2, a3 = DDQNAgent(h), ContextDDQNAgent(h), OracleDDQNAgent(h)
    counts = {"A1": a1.n_params(), "A2": a2.n_params(), "A3": a3.n_params()}
    print("parameter counts:", counts)
    check_param_parity(counts, tol=0.02)
    print("param parity within 2%: PASS")

    # one train step each on random transitions
    labels = ["none", "syn", "udp"]
    a2.reset_history()
    for i in range(h.batch_size + 20):
        o = np.random.rand(h.obs_dim).astype(np.float32)
        no = np.random.rand(h.obs_dim).astype(np.float32)
        r = float(np.random.rand())
        a1.remember(o, a1.act(o, i), r, no, 0.0)
        done = 1.0 if (i + 1) % 25 == 0 else 0.0
        a2.remember(o, a2.act(o, i), r, no, done)
        tl, ntl = labels[i % 3], labels[(i + 1) % 3]
        a3.remember(o, tl, a3.act(o, tl, i), r, no, ntl, 0.0)
    losses = {"A1": a1.learn(), "A2": a2.learn(), "A3": a3.learn()}
    print("train-step losses:", {k: round(v, 4) for k, v in losses.items()})
    assert all(v is not None and np.isfinite(v) for v in losses.values())

    assert a3.act(np.random.rand(h.obs_dim).astype(np.float32), "none", 999,
                  greedy=True) == 0, "oracle tau=none must force no-op"
    s = StaticRuleAgent()
    assert (s.act_from_suspects(True), s.act_from_suspects(False),
            s.act_from_suspects(False)) == (1, 4, 0)
    print("oracle gating and static rule: PASS")
    print("\nG6 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
