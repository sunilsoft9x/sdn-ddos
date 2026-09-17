"""Read switch counters straight from Open vSwitch (plan section 4.6).

ODL is the control plane; this is the measurement plane. ovs-ofctl reports
counters current to the millisecond, where ODL's statistics path refreshes far
too slowly for a sub-second window W.

Budget: the whole collect() call must finish inside 60 ms, so all eight
commands (dump-flows and dump-ports on four switches) are launched at once and
then reaped, rather than run one after another.
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import time

import yaml

CONFIG_PATH = pathlib.Path(__file__).resolve().parents[1] / "configs" / "base.yaml"

_FLOW_FIELDS = re.compile(
    r"n_packets=(?P<packets>\d+).*?n_bytes=(?P<bytes>\d+).*?priority=(?P<priority>\d+)"
)
_PORT_HEADER = re.compile(r"^\s*port\s+(?P<port>[\w\d-]+):")
_RX = re.compile(r"rx pkts=(?P<pkts>[\d?]+), bytes=(?P<bytes>[\d?]+), drop=(?P<drop>[\d?]+)")
_TX = re.compile(r"tx pkts=(?P<pkts>[\d?]+), bytes=(?P<bytes>[\d?]+), drop=(?P<drop>[\d?]+)")


_SUDO = [] if os.geteuid() == 0 else ["sudo", "-n"]


def load_config(path: pathlib.Path | str = CONFIG_PATH) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _int(value: str) -> int:
    """ovs-ofctl prints '?' for counters a datapath does not maintain."""
    return 0 if value == "?" else int(value)


def _parse_match(match_blob: str) -> dict:
    """Pull the fields the feature code needs out of a dump-flows match string."""
    out: dict[str, str | int] = {}
    for token in match_blob.split(","):
        token = token.strip()
        if "=" in token:
            key, _, value = token.partition("=")
            # priority is taken from _FLOW_FIELDS as an int; do not shadow it here
            if key in ("nw_src", "nw_dst", "cookie"):
                out[key] = value
            elif key == "table":
                out[key] = int(value)
        elif token in ("tcp", "udp", "icmp", "ip", "arp"):
            out["proto"] = token
    return out


def parse_flows(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        if "n_packets=" not in line:
            continue
        found = _FLOW_FIELDS.search(line)
        if not found:
            continue
        match_blob, _, _ = line.partition(" actions=")
        row = {
            "packets": int(found.group("packets")),
            "bytes": int(found.group("bytes")),
            "priority": int(found.group("priority")),
        }
        row.update(_parse_match(match_blob))
        rows.append(row)
    return rows


def parse_ports(text: str) -> dict[str, dict[str, int]]:
    ports: dict[str, dict[str, int]] = {}
    current = None
    for line in text.splitlines():
        header = _PORT_HEADER.match(line)
        if header:
            current = header.group("port")
            ports[current] = {}
            # No continue: ovs-ofctl puts the rx counters on the header line
            # itself, and skipping it loses rx_pkts, rx_bytes and rx_drop.
        if current is None:
            continue
        rx = _RX.search(line)
        if rx:
            ports[current].update(
                rx_pkts=_int(rx.group("pkts")),
                rx_bytes=_int(rx.group("bytes")),
                rx_drop=_int(rx.group("drop")),
            )
        tx = _TX.search(line)
        if tx:
            ports[current].update(
                tx_pkts=_int(tx.group("pkts")),
                tx_bytes=_int(tx.group("bytes")),
                tx_drop=_int(tx.group("drop")),
            )
    return ports


def collect(switches: list[str] | None = None, cfg: dict | None = None) -> dict:
    """Snapshot every switch. Returns a dict with a single timestamp for the lot.

    One timestamp is taken at dispatch, not per command, because the features
    treat the whole snapshot as one instant and per-command times would imply a
    precision the measurement does not have.
    """
    cfg = cfg or load_config()
    if switches is None:
        switches = [cfg["topology"]["core"]] + list(cfg["topology"]["edges"])
    version = cfg["topology"]["openflow_version"]

    started = time.monotonic()
    wall = time.time()
    procs: dict[tuple[str, str], subprocess.Popen] = {}
    for switch in switches:
        for command in ("dump-flows", "dump-ports"):
            procs[(switch, command)] = subprocess.Popen(
                _SUDO + ["ovs-ofctl", "-O", version, command, switch],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

    snapshot: dict = {"t_monotonic": started, "t_wall": wall, "switches": {}}
    errors = []
    for (switch, command), proc in procs.items():
        stdout, stderr = proc.communicate(timeout=5)
        if proc.returncode != 0:
            errors.append(f"{command} {switch}: {stderr.strip()[:120]}")
            continue
        entry = snapshot["switches"].setdefault(switch, {})
        if command == "dump-flows":
            entry["flows"] = parse_flows(stdout)
        else:
            entry["ports"] = parse_ports(stdout)

    snapshot["elapsed_ms"] = (time.monotonic() - started) * 1000.0
    if errors:
        snapshot["errors"] = errors
    return snapshot


def per_source_packets(snapshot: dict, victim_ip: str) -> dict[str, dict[str, int]]:
    """Packets per source toward the victim, split by protocol.

    Reads the priority-200 monitoring flows, which exist precisely to give these
    counters. Keys are source IPs; values map protocol name to packet count.
    """
    result: dict[str, dict[str, int]] = {}
    for entry in snapshot["switches"].values():
        for flow in entry.get("flows", []):
            if flow.get("priority") != 200:
                continue
            if flow.get("nw_dst") != victim_ip:
                continue
            src = flow.get("nw_src")
            if not src:
                continue
            bucket = result.setdefault(src, {})
            proto = flow.get("proto", "ip")
            bucket[proto] = bucket.get(proto, 0) + flow["packets"]
            bucket["bytes"] = bucket.get("bytes", 0) + flow["bytes"]
    return result


if __name__ == "__main__":
    import json

    config = load_config()
    shot = collect(cfg=config)
    print(f"elapsed {shot['elapsed_ms']:.1f} ms")
    if "errors" in shot:
        print("errors:", shot["errors"])
    for name, data in shot["switches"].items():
        print(name, len(data.get("flows", [])), "flows,",
              len(data.get("ports", {})), "ports")
    print(json.dumps(
        per_source_packets(shot, config["topology"]["victim"]["ip"]), indent=2
    ))
