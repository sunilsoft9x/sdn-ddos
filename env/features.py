"""The 18-feature observation vector (plan section 4.6).

Every feature is derived from two ovs_stats snapshots a window W apart plus the
probe aggregate for that window. Rates are counter deltas divided by the actual
elapsed time between snapshots, not by nominal W, because the real gap jitters.

Nothing here uses knowledge of who is attacking or which regime is active. That
is the whole point: the observation is label-free, so a policy trained on it
cannot be reading a detector's answer off its own input. State that in Section V.

Normalisation is a design choice (Section IV). Each feature is scaled to sit
roughly in [0,1] by a fixed constant derived from the topology, so the network
sees inputs of comparable magnitude. The constants live in _Scales and are
reported in the paper. The suspects/mitigation bookkeeping (features 12, 18 and
the churn term) is passed in by the env, which owns that state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

FEATURE_NAMES = [
    "pkt_rate", "byte_rate", "mean_pkt_size",
    "tcp_share", "udp_share", "icmp_share",
    "src_entropy", "active_sources", "top_source_share",
    "victim_link_util", "victim_tx_dropped_rate", "edge_table_occupancy",
    "probe_success", "probe_latency_median", "probe_goodput",
    "d_pkt_rate", "d_udp_share", "mitigated_fraction",
]
PROTO_KEYS = {"tcp": 6, "udp": 17, "icmp": 1}


@dataclass
class _Scales:
    """Fixed normalisers. Documented constants, not tuned on data."""
    max_pkt_rate: float = 20000.0      # pps; victim link saturated at min-size frames
    max_byte_rate: float = 2.5e6       # B/s; ~20 Mbps aggregate toward the victim
    max_pkt_size: float = 1500.0       # bytes; Ethernet MTU
    max_sources: float = 15.0          # hosts h1..h15
    link_capacity_bps: float = 1.25e6  # 10 Mbps victim link in bytes/s
    max_dropped_rate: float = 5000.0   # pps; a saturated drop counter
    edge_table_capacity: float = 200.0 # flows an edge switch holds when full
    max_goodput_bps: float = 1.25e6    # 10 Mbps


class FeatureExtractor:
    def __init__(self, cfg: dict, scales: _Scales | None = None):
        self.cfg = cfg
        self.s = scales or _Scales()
        self.victim_ip = cfg["topology"]["victim"]["ip"]
        self.core = cfg["topology"]["core"]
        self.edges = list(cfg["topology"]["edges"])
        self.victim_port = None  # resolved lazily from the port map if provided

    # -- per-source, per-protocol packet counts toward the victim ---------

    def _source_table(self, snapshot: dict) -> dict[str, dict[str, int]]:
        table: dict[str, dict[str, int]] = {}
        for entry in snapshot["switches"].values():
            for flow in entry.get("flows", []):
                if flow.get("priority") != 200:
                    continue
                if flow.get("nw_dst") != self.victim_ip:
                    continue
                src = flow.get("nw_src")
                if not src:
                    continue
                bucket = table.setdefault(src, {"packets": 0, "bytes": 0,
                                                "tcp": 0, "udp": 0, "icmp": 0})
                proto = flow.get("proto", "ip")
                pkts = flow["packets"]
                bucket["packets"] += pkts
                bucket["bytes"] += flow["bytes"]
                if proto in bucket:
                    bucket[proto] += pkts
        return table

    @staticmethod
    def _delta_table(prev: dict, curr: dict) -> dict[str, dict[str, int]]:
        """curr - prev per source, floored at zero (counters can reset on rebuild)."""
        out: dict[str, dict[str, int]] = {}
        for src, now in curr.items():
            before = prev.get(src, {})
            d = {}
            for k, v in now.items():
                d[k] = max(0, v - before.get(k, 0))
            out[src] = d
        return out

    # -- port counters on the victim uplink -------------------------------

    def _victim_port_counts(self, snapshot: dict, victim_port: int) -> dict:
        core = snapshot["switches"].get(self.core, {})
        return core.get("ports", {}).get(str(victim_port), {})

    # -- the vector -------------------------------------------------------

    def extract(self, prev_snapshot: dict | None, curr_snapshot: dict,
                probe: dict, victim_port: int,
                edge_flow_count: int, mitigated_fraction: float,
                prev_derived: dict | None) -> tuple[np.ndarray, dict]:
        """Return (18-vector float32, raw dict for logging and the reward).

        prev_derived carries the previous window's pkt_rate and udp_share so
        features 16-17 can be first differences; pass None on the first window.
        """
        dt = 1e-6
        if prev_snapshot is not None:
            dt = max(curr_snapshot["t_wall"] - prev_snapshot["t_wall"], 1e-6)

        curr_tbl = self._source_table(curr_snapshot)
        if prev_snapshot is not None:
            prev_tbl = self._source_table(prev_snapshot)
            delta = self._delta_table(prev_tbl, curr_tbl)
        else:
            # First window: no delta available, treat counts as the window's own.
            delta = {s: dict(v) for s, v in curr_tbl.items()}

        total_pkts = sum(d.get("packets", 0) for d in delta.values())
        total_bytes = sum(d.get("bytes", 0) for d in delta.values())
        tcp = sum(d.get("tcp", 0) for d in delta.values())
        udp = sum(d.get("udp", 0) for d in delta.values())
        icmp = sum(d.get("icmp", 0) for d in delta.values())

        pkt_rate = total_pkts / dt
        byte_rate = total_bytes / dt
        mean_pkt_size = (total_bytes / total_pkts) if total_pkts else 0.0

        proto_total = tcp + udp + icmp
        tcp_share = tcp / proto_total if proto_total else 0.0
        udp_share = udp / proto_total if proto_total else 0.0
        icmp_share = icmp / proto_total if proto_total else 0.0

        # Per-source distribution over this window's packets.
        active = {s: d.get("packets", 0) for s, d in delta.items() if d.get("packets", 0) > 0}
        n_active = len(active)
        if total_pkts > 0 and n_active > 0:
            probs = [c / total_pkts for c in active.values()]
            entropy = -sum(p * math.log2(p) for p in probs if p > 0)
            max_entropy = math.log2(self.s.max_sources)
            src_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            top_source_share = max(probs)
        else:
            src_entropy = 0.0
            top_source_share = 0.0

        vp = self._victim_port_counts(curr_snapshot, victim_port)
        if prev_snapshot is not None:
            vp_prev = self._victim_port_counts(prev_snapshot, victim_port)
        else:
            vp_prev = {}
        tx_bytes = max(0, vp.get("tx_bytes", 0) - vp_prev.get("tx_bytes", 0))
        tx_drop = max(0, vp.get("tx_drop", 0) - vp_prev.get("tx_drop", 0))
        link_util = (tx_bytes / dt) / self.s.link_capacity_bps
        dropped_rate = (tx_drop / dt) / self.s.max_dropped_rate

        table_occupancy = edge_flow_count / self.s.edge_table_capacity

        # Derived first differences.
        if prev_derived is not None:
            d_pkt_rate = (pkt_rate - prev_derived.get("pkt_rate", 0.0)) / self.s.max_pkt_rate
            d_udp_share = udp_share - prev_derived.get("udp_share", 0.0)
        else:
            d_pkt_rate = 0.0
            d_udp_share = 0.0

        vector = np.array([
            min(pkt_rate / self.s.max_pkt_rate, 1.0),
            min(byte_rate / self.s.max_byte_rate, 1.0),
            min(mean_pkt_size / self.s.max_pkt_size, 1.0),
            tcp_share,
            udp_share,
            icmp_share,
            src_entropy,
            min(n_active / self.s.max_sources, 1.0),
            top_source_share,
            min(link_util, 1.0),
            min(dropped_rate, 1.0),
            min(table_occupancy, 1.0),
            probe["success"],
            probe["latency_scaled"],
            min(probe["goodput_bps"] / self.s.max_goodput_bps, 1.0),
            max(-1.0, min(d_pkt_rate, 1.0)),
            max(-1.0, min(d_udp_share, 1.0)),
            min(max(mitigated_fraction, 0.0), 1.0),
        ], dtype=np.float32)

        raw = {
            "pkt_rate": pkt_rate, "byte_rate": byte_rate,
            "mean_pkt_size": mean_pkt_size, "udp_share": udp_share,
            "tcp_share": tcp_share, "icmp_share": icmp_share,
            "n_active": n_active, "src_entropy": src_entropy,
            "top_source_share": top_source_share, "link_util": link_util,
            "dropped_rate": dropped_rate, "dt": dt,
            "probe_success": probe["success"],
            "probe_latency_s": probe["latency_s"],
            "mitigated_fraction": mitigated_fraction,
        }
        # Carry the values features 16-17 will difference against next window.
        raw["_derived"] = {"pkt_rate": pkt_rate, "udp_share": udp_share}
        return vector, raw


def reward(raw: dict, churn_count: int, cfg: dict) -> tuple[float, dict]:
    """r = 1.0*S - 0.5*L - 0.5*U - 0.1*C  (plan section 4.8).

    S, L are the probe terms; U is victim over-utilisation above 0.8; C is the
    count of priority-300 rules the action added or removed this step over 15.
    All four use only observable signals.
    """
    w = cfg["reward"]["weights"]
    S = raw["probe_success"]
    L = min(raw["probe_latency_s"] / (cfg["reward"]["latency_scale_ms"] / 1000.0), 1.0)
    thr = cfg["reward"]["util_threshold"]
    U = max(0.0, raw["link_util"] - thr) / (1.0 - thr)
    U = min(U, 1.0)
    C = min(churn_count / cfg["reward"]["churn_scale"], 1.0)
    r = w["S"] * S + w["L"] * L + w["U"] * U + w["C"] * C
    return r, {"S": S, "L": L, "U": U, "C": C, "r": r}


if __name__ == "__main__":
    # Self-check with synthetic snapshots (no Mininet needed).
    import yaml
    import pathlib
    cfg = yaml.safe_load(open(pathlib.Path(__file__).resolve().parents[1]
                              / "configs" / "base.yaml"))
    fx = FeatureExtractor(cfg)

    def snap(t, rows, ports):
        return {"t_wall": t, "switches": {"s1": {"ports": ports}, "s2": {"flows": rows}}}

    prev = snap(100.0,
                [{"priority": 200, "proto": "tcp", "nw_src": "10.0.0.1",
                  "nw_dst": "10.0.0.16", "packets": 0, "bytes": 0}],
                {"1": {"tx_bytes": 0, "tx_drop": 0}})
    curr = snap(100.5,
                [{"priority": 200, "proto": "tcp", "nw_src": "10.0.0.1",
                  "nw_dst": "10.0.0.16", "packets": 500, "bytes": 60000},
                 {"priority": 200, "proto": "udp", "nw_src": "10.0.0.7",
                  "nw_dst": "10.0.0.16", "packets": 1000, "bytes": 120000}],
                {"1": {"tx_bytes": 90000, "tx_drop": 200}})
    probe = {"n": 5, "success": 0.6, "latency_s": 0.15,
             "latency_scaled": 0.3, "goodput_bps": 40000.0}
    vec, raw = fx.extract(prev, curr, probe, victim_port=1,
                          edge_flow_count=47, mitigated_fraction=0.1,
                          prev_derived=None)
    assert vec.shape == (18,), vec.shape
    assert abs(vec.sum()) < 100
    r, terms = reward(raw, churn_count=3, cfg=cfg)
    for name, val in zip(FEATURE_NAMES, vec):
        print(f"  {name:24s} {val: .4f}")
    print(f"reward {r:.4f}  terms {terms}")
    print("features self-check ok")
