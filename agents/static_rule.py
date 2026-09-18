"""Static threshold rule (plan 4.11), no training.

Drops any source above three times the median rate -- the suspect rule from 4.5
turned straight into an action. It needs no learning and answers the reviewer
who asks whether reinforcement learning is needed at all: if a fixed rule
matches the agents, the RL machinery is not earning its place; if it does not,
the adaptation is doing real work.

It acts on the environment's own suspect set, so at each step it drops when
there are suspects and clears when there are none.
"""

from __future__ import annotations


class StaticRuleAgent:
    label = "static"

    def __init__(self, *args, **kwargs):
        self._mitigating = False

    def act_from_suspects(self, has_suspects: bool) -> int:
        """Action 1 (drop suspects) when any exist, else action 4 (clear)."""
        if has_suspects:
            self._mitigating = True
            return 1
        if self._mitigating:
            self._mitigating = False
            return 4
        return 0

    def reset(self) -> None:
        self._mitigating = False
