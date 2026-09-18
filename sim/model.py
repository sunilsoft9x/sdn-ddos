"""Flow-level simulator physics (plan 7.1).

The simulator generates the SAME raw snapshot dictionary that ovs_stats.collect
produces on the testbed, plus a probe summary, so env/features.py runs unchanged
on both. That is what keeps training features identical to evaluation features.

Three pieces:
  TrafficModel  - per-source offered packets/bytes by regime, scale, protocol.
  LinkModel     - offered load past mitigation -> victim-link utilisation, drops.
  ServiceModel  - fitted from calibration: (passed load, utilisation) -> probe
                  success and latency. Parameters come from sim/fit.py.

Everything above the raw counters (the 18 features, the reward) is computed by
the real env code, not re-implemented here.
"""

from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass, field

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
LINK_CAPACITY_BPS = 1.25e6      # 10 Mbps victim link, bytes/s
PROTO_NUM = {"tcp": 6, "udp": 17, "icmp": 1}

# Per-regime attacker traffic: (protocol, packet bytes, nominal pps/attacker).
# Rates mirror the locked configs/base.yaml; scale and randomisation multiply pps.
REGIME_ATTACK = {
    "R0": None,
    "R1": {"proto": "tcp", "pkt_bytes": 554, "nominal_pps": 640},
    "R2": {"proto": "udp", "pkt_bytes": 1042, "nominal_pps": 520},
    "R4": {"proto": "udp", "pkt_bytes": 1200, "nominal_pps": 1200,
           "burst_on": 0.2, "burst_period": 1.0},
}
BENIGN = {"proto": "tcp", "pkt_bytes": 1400, "mbps_each": (0.2, 0.5)}


def _pps_for_mbps(mbps: float, pkt_bytes: int) -> float:
    return mbps * 1e6 / 8 / pkt_bytes


class TrafficModel:
    """Offered traffic per source, before mitigation and before the link."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.victim_ip = cfg["topology"]["victim"]["ip"]

    def offered(self, regime: str, scale: float, attackers: list[str],
                benign: list[str], rng: np.random.Generator,
                phase: float = 0.0, util_cal: dict | None = None) -> dict[str, dict]:
        """Return {src_ip: {'proto','pps','pkt_bytes'}} offered this window.

        When util_cal has an entry for the regime, the aggregate attacker load
        is set so the victim-link utilisation matches the testbed's measured
        util-versus-scale curve (util = a + b*scale), rather than trusting the
        nominal packet rate. That makes the simulator reproduce the testbed's
        service response by construction. R4 (not in util_cal) keeps its nominal
        pulsed model; it is evaluation-only and never trained on.
        """
        out: dict[str, dict] = {}

        benign_bps = 0.0
        for host in benign:
            ip = self._ip(host)
            mbps = rng.uniform(*BENIGN["mbps_each"])
            pps = _pps_for_mbps(mbps, BENIGN["pkt_bytes"])
            out[ip] = {"proto": BENIGN["proto"], "pps": pps,
                       "pkt_bytes": BENIGN["pkt_bytes"]}
            benign_bps += pps * BENIGN["pkt_bytes"]

        spec = REGIME_ATTACK.get(regime)
        if spec is not None and attackers:
            if util_cal and regime in util_cal:
                a, b = util_cal[regime]
                target_util = max(0.0, a + b * scale)
                target_bps = target_util * LINK_CAPACITY_BPS
                attacker_bps = max(0.0, target_bps - benign_bps)
                pps_each = (attacker_bps / len(attackers)) / spec["pkt_bytes"]
            else:
                pps_each = spec["nominal_pps"] * scale
                if regime == "R4":
                    on = spec["burst_on"] / spec["burst_period"]
                    pps_each = pps_each if phase < on else 0.0
            for host in attackers:
                out[self._ip(host)] = {"proto": spec["proto"], "pps": pps_each,
                                       "pkt_bytes": spec["pkt_bytes"]}
        return out

    def _ip(self, host: str) -> str:
        if host.startswith("10."):
            return host
        return f"10.0.0.{int(host.lstrip('h'))}"


def apply_mitigation(offered: dict[str, dict],
                     mitigation: dict[str, dict]) -> dict[str, dict]:
    """Reduce offered traffic by the mitigation state.

    mitigation maps src_ip -> {'kind': 'drop'|'meter', 'factor': f}. A drop
    zeroes the source; a meter caps its rate at factor x its offered rate
    (the band was sized from the measured rate on the testbed).
    """
    passed: dict[str, dict] = {}
    for ip, o in offered.items():
        m = mitigation.get(ip)
        pps = o["pps"]
        if m is not None:
            if m["kind"] == "drop":
                pps = 0.0
            else:
                pps = pps * m.get("factor", 1.0)
        passed[ip] = {**o, "pps": pps}
    return passed


class LinkModel:
    """Aggregate passed traffic onto the 10 Mbps victim link."""

    def utilisation_and_drops(self, passed: dict[str, dict]) -> tuple[float, float]:
        offered_bps = sum(p["pps"] * p["pkt_bytes"] for p in passed.values())
        util = offered_bps / LINK_CAPACITY_BPS
        # Drops rise once offered load exceeds capacity (piecewise linear).
        drop_frac = max(0.0, (util - 1.0)) / max(util, 1e-6) if util > 1.0 else 0.0
        return util, drop_frac


@dataclass
class ServiceModel:
    """Fitted mapping from passed load / utilisation to probe outcomes.

    success = sigmoid(b0 + b1*util + b2*util^2 + b3*log1p(pps) + b4*tcp_share)
    latency_s follows a monotone rise with utilisation (two params).
    Noise stds are multiplicative, taken from the R0 calibration windows.
    """
    success_coef: list[float] = field(default_factory=lambda: [4.0, -6.0, 0.0, 0.0, 0.0])
    latency: list[float] = field(default_factory=lambda: [0.02, 0.4])
    noise_std: dict[str, float] = field(default_factory=dict)
    util_cal: dict = field(default_factory=dict)   # regime -> [a, b] for util=a+b*scale
    fitted: bool = False

    @staticmethod
    def _features(util: float, pps: float, tcp_share: float) -> list[float]:
        return [1.0, util, util * util, math.log1p(pps), tcp_share]

    def success(self, util: float, pps: float, tcp_share: float) -> float:
        z = sum(c * f for c, f in
                zip(self.success_coef, self._features(util, pps, tcp_share)))
        return 1.0 / (1.0 + math.exp(-z))

    def latency_s(self, util: float) -> float:
        base, slope = self.latency
        return base + slope * max(0.0, util) ** 2

    def save(self, path: pathlib.Path | str) -> None:
        pathlib.Path(path).write_text(json.dumps({
            "success_coef": self.success_coef, "latency": self.latency,
            "noise_std": self.noise_std, "util_cal": self.util_cal,
            "fitted": self.fitted}, indent=2))

    @classmethod
    def load(cls, path: pathlib.Path | str) -> "ServiceModel":
        d = json.loads(pathlib.Path(path).read_text())
        return cls(d["success_coef"], d["latency"], d.get("noise_std", {}),
                   d.get("util_cal", {}), d.get("fitted", True))


def build_snapshot(cumulative: dict[str, dict], victim_port_counts: dict,
                   t_wall: float, victim_ip: str, core: str, edges: list[str],
                   host_edge_fn) -> dict:
    """Assemble an ovs_stats-compatible snapshot from cumulative counters.

    cumulative maps src_ip -> {'tcp','udp','icmp','packets','bytes'} running
    totals. Priority-200 monitoring flows are emitted per source per protocol,
    exactly as the testbed's dump-flows would show, so per_source_packets and
    the feature extractor read it unchanged.
    """
    switches: dict[str, dict] = {core: {"flows": [], "ports": {}},
                                 **{e: {"flows": [], "ports": {}} for e in edges}}
    for ip, c in cumulative.items():
        edge = host_edge_fn(ip)
        for proto in ("tcp", "udp", "icmp"):
            pk = c.get(proto, 0)
            if pk == 0 and proto != "tcp":
                continue
            switches[edge]["flows"].append({
                "priority": 200, "proto": proto,
                "nw_src": ip, "nw_dst": victim_ip,
                "packets": pk,
                "bytes": int(pk * c.get("_bytes_per_pkt", 700)),
            })
    switches[core]["ports"] = {"1": victim_port_counts}
    return {"t_wall": t_wall, "switches": switches}
