"""Install the proactive forwarding and monitoring flows (plan section 4.4).

Recent OpenDaylight releases ship no reactive forwarding application, so nothing
is learned on demand. Everything below priority 300 is installed once, up front,
before an episode starts.

Priority 100: destination-based forwarding on every switch.
Priority 200: one flow per (host, victim) pair per protocol, in each direction,
              at the host's own edge switch. These are what give the per-source,
              per-protocol counters features 1-9 read.
Priority 300: not installed here. That is the agent's, through OdlClient.

Deviation from the plan worth noting in Section V: the plan describes
priority-100 rules "for every host pair", which would be 240 directed rules.
Destination-based matching is behaviourally identical on a single-path tree and
needs 64, which keeps flow-table occupancy (feature 12) meaningful rather than
saturated.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from env.odl_client import PROTO_NAMES, OdlClient  # noqa: E402
from env.topology import host_edge, load_config  # noqa: E402


def host_ip(host: str, cfg: dict) -> str:
    if host == cfg["topology"]["victim"]["name"]:
        return cfg["topology"]["victim"]["ip"]
    return f"10.0.0.{int(host.lstrip('h'))}"


def all_hosts(cfg: dict) -> list[str]:
    return [f"h{i}" for i in range(1, cfg["topology"]["host_count"] + 1)]


def install_default_forwarding(client: OdlClient, port_map: dict, cfg: dict) -> int:
    """Priority 100, destination-based, on every switch."""
    core = cfg["topology"]["core"]
    victim = cfg["topology"]["victim"]["name"]
    installed = 0

    for switch, ports in port_map.items():
        for host in all_hosts(cfg) + [victim]:
            ip = host_ip(host, cfg)
            if host in ports:                      # directly attached
                out_port = ports[host]
            elif switch == core:                   # via the edge that owns it
                out_port = ports[host_edge(host, cfg)]
            else:                                  # anything else goes up
                out_port = ports[core]
            client.put_flow(
                switch,
                flow_id=f"f100-{host}",
                priority=cfg["flows"]["priority"]["default_forward"],
                out_port=out_port,
                dst=ip,
            )
            installed += 1
    return installed


def install_monitoring(client: OdlClient, port_map: dict, cfg: dict) -> int:
    """Priority 200, per source, per protocol, both directions, at the source edge."""
    core = cfg["topology"]["core"]
    victim = cfg["topology"]["victim"]["name"]
    victim_ip = host_ip(victim, cfg)
    protocols = cfg["flows"]["per_protocol_split"]
    priority = cfg["flows"]["priority"]["monitor"]
    installed = 0

    for host in all_hosts(cfg):
        edge = host_edge(host, cfg)
        ip = host_ip(host, cfg)
        uplink = port_map[edge][core]
        downlink = port_map[edge][host]
        for proto in protocols:
            name = PROTO_NAMES[proto]
            client.put_flow(
                edge,
                flow_id=f"f200-{host}-{name}-out",
                priority=priority,
                out_port=uplink,
                src=ip,
                dst=victim_ip,
                proto=proto,
            )
            client.put_flow(
                edge,
                flow_id=f"f200-{host}-{name}-in",
                priority=priority,
                out_port=downlink,
                src=victim_ip,
                dst=ip,
                proto=proto,
            )
            installed += 2
    return installed


def clear_all(client: OdlClient, cfg: dict) -> None:
    """Remove everything this script installs, plus any leftover mitigation."""
    switches = [cfg["topology"]["core"]] + list(cfg["topology"]["edges"])
    for priority_name in ("mitigation", "monitor", "default_forward"):
        client.delete_all(switches, cfg["flows"]["priority"][priority_name])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port-map", default=None,
                        help="path to data/port_map.json (written by topology.py)")
    parser.add_argument("--clear", action="store_true",
                        help="delete flows instead of installing them")
    args = parser.parse_args()

    cfg = load_config()
    client = OdlClient(cfg)

    if args.clear:
        clear_all(client, cfg)
        print("cleared")
        return 0

    path = args.port_map or (
        pathlib.Path(__file__).resolve().parents[1] / "data" / "port_map.json"
    )
    with open(path) as fh:
        port_map = json.load(fh)

    started = time.monotonic()
    forwarding = install_default_forwarding(client, port_map, cfg)
    monitoring = install_monitoring(client, port_map, cfg)
    elapsed = time.monotonic() - started

    print(f"priority-100 forwarding rules: {forwarding}")
    print(f"priority-200 monitoring rules: {monitoring}")
    print(f"total {forwarding + monitoring} in {elapsed:.1f}s")
    print("expected: 64 forwarding, 90 monitoring")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
