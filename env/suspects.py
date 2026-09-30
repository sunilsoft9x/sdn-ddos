"""Volume-only suspect rule (plan section 4.5), shared by testbed and simulator.

For every source, two packet rates toward the victim are kept: the rate in the
current window W, and the mean rate over a sliding long window (2 s). A source
is a suspect if EITHER rate exceeds `median_multiplier` times the median of the
long-window rates across sources. The current-window rate catches a new flood
at once; the long-window rate keeps a pulsed attacker flagged between bursts.

The rule reads volume only, never protocol or regime, so it hands the agent no
label information. That is the property the whole contribution rests on.

Probe hosts are excluded from the suspect set by name (config
exclude_probes_by_name = true). The probes are the operator's own monitoring,
which an operator would not block; the exemption is the same in every regime
and reveals no attack type. The simulator creates no probe traffic, so the
setting has no effect in training.

Design choice recorded for Section IV: the cross-source median is taken over
sources active in the long window (nonzero rate), not over all fifteen. With a
mix of loud attackers and silent hosts the all-source median collapses to zero
and the threshold degenerates; the active-source median is what the rule
intends by "across sources".
"""

from __future__ import annotations

import statistics
from collections import deque


class SuspectDetector:
    def __init__(self, cfg: dict):
        s = cfg["suspects"]
        self.multiplier = s["median_multiplier"]
        self.long_window_s = s["long_window_s"]
        self.exclude_probes = s.get("exclude_probes_by_name", False)
        self.probes = set(_host_ips(cfg, cfg["roles"]["probes"]))
        # (t_wall, {src_ip: rate_pps}) samples spanning the last long_window_s.
        self._history: deque[tuple[float, dict[str, float]]] = deque()

    def reset(self) -> None:
        self._history.clear()

    def update(self, rates: dict[str, float], t_wall: float) -> None:
        self._history.append((t_wall, dict(rates)))
        cutoff = t_wall - self.long_window_s
        while len(self._history) > 1 and self._history[0][0] < cutoff:
            self._history.popleft()

    def _long_window_means(self) -> dict[str, float]:
        """Mean rate per source across the samples held in the window."""
        if not self._history:
            return {}
        seen: set[str] = set()
        for _, rates in self._history:
            seen.update(rates)
        n = len(self._history)
        return {
            src: sum(rates.get(src, 0.0) for _, rates in self._history) / n
            for src in seen
        }

    def suspects(self) -> set[str]:
        if not self._history:
            return set()
        current = self._history[-1][1]
        long_means = self._long_window_means()

        active = [r for r in long_means.values() if r > 0.0]
        if len(active) < 2:
            # Not enough sources generating traffic to define a median.
            return set()
        median = statistics.median(active)
        if median <= 0.0:
            return set()
        threshold = self.multiplier * median

        flagged = {
            src for src in set(current) | set(long_means)
            if current.get(src, 0.0) > threshold
            or long_means.get(src, 0.0) > threshold
        }
        if self.exclude_probes:
            flagged -= self.probes
        return flagged

    def snapshot_terms(self) -> dict:
        """Diagnostics for the separability check and logging."""
        long_means = self._long_window_means()
        active = [r for r in long_means.values() if r > 0.0]
        median = statistics.median(active) if len(active) >= 2 else 0.0
        return {
            "median_long_rate": median,
            "threshold": self.multiplier * median,
            "n_active": len(active),
            "suspects": sorted(self.suspects()),
        }


def _host_ips(cfg: dict, names: list[str]) -> list[str]:
    out = []
    for name in names:
        idx = int(name.lstrip("h"))
        out.append(f"10.0.0.{idx}")
    return out


if __name__ == "__main__":
    cfg = {
        "suspects": {"median_multiplier": 3.0, "long_window_s": 2.0,
                     "exclude_probes_by_name": False},
        "roles": {"probes": ["h1", "h6", "h11"]},
    }
    d = SuspectDetector(cfg)
    # Benign: everyone near 100 pps -> nobody 3x the median -> no suspects.
    for t in (0.0, 0.5, 1.0, 1.5):
        d.update({f"10.0.0.{i}": 100 + i for i in range(1, 16)}, t)
    assert d.suspects() == set(), d.snapshot_terms()
    print("benign -> no suspects, terms:", d.snapshot_terms())

    # One flooder at 5000 pps among benign -> flagged.
    d.reset()
    for t in (0.0, 0.5, 1.0, 1.5):
        rates = {f"10.0.0.{i}": 100 for i in range(1, 16)}
        rates["10.0.0.7"] = 5000
        d.update(rates, t)
    assert d.suspects() == {"10.0.0.7"}, d.snapshot_terms()
    print("one flooder -> flagged 10.0.0.7, terms:", d.snapshot_terms())
    print("suspects self-check ok")
