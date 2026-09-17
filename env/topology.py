"""Mininet topology for the INDICON 2026 testbed (plan section 4.1).

Shape: core s1 holds the victim h16 on a 10 Mbps link. Three edge switches
s2, s3, s4 each carry five hosts on 10 Mbps links with 20 Mbps uplinks to s1.
Mininet's built-in tree cannot produce this, hence the custom class.

build_network() also returns a port map, because every flow rule the project
installs needs the OpenFlow port number of a given link and Mininet is the
only place that knows it.
"""

from __future__ import annotations

import pathlib

import yaml
from mininet.link import TCLink
from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController
from mininet.topo import Topo

CONFIG_PATH = pathlib.Path(__file__).resolve().parents[1] / "configs" / "base.yaml"


def load_config(path: pathlib.Path | str = CONFIG_PATH) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


class DdosTopo(Topo):
    """Three-edge star with a single victim behind the core switch."""

    def build(self, cfg: dict | None = None):
        cfg = cfg or load_config()
        topo = cfg["topology"]
        bw = topo["bandwidth_mbps"]
        of = topo["openflow_version"]

        core = self.addSwitch(topo["core"], protocols=of)

        victim = topo["victim"]
        h_victim = self.addHost(victim["name"], ip=f"{victim['ip']}/24")
        self.addLink(h_victim, core, bw=bw["victim_link"])

        host_index = 1
        for edge_name in topo["edges"]:
            edge = self.addSwitch(edge_name, protocols=of)
            self.addLink(edge, core, bw=bw["uplink"])
            for _ in range(topo["hosts_per_edge"]):
                host = self.addHost(f"h{host_index}", ip=f"10.0.0.{host_index}/24")
                self.addLink(host, edge, bw=bw["host_link"])
                host_index += 1


def build_port_map(net: Mininet) -> dict[str, dict[str, int]]:
    """Map switch name -> peer node name -> OpenFlow port number.

    Read from Mininet's own link and interface tables rather than guessed from
    creation order, because TCLink inserts intermediate interfaces and the
    numbering is not what the addLink sequence suggests.
    """
    port_map: dict[str, dict[str, int]] = {}
    for switch in net.switches:
        ports: dict[str, int] = {}
        for intf in switch.intfList():
            if intf.name == "lo" or intf.link is None:
                continue
            far = intf.link.intf2 if intf.link.intf1 is intf else intf.link.intf1
            ports[far.node.name] = switch.ports[intf]
        port_map[switch.name] = ports
    return port_map


def build_network(cfg: dict | None = None, controller_ip: str = "127.0.0.1",
                  controller_port: int = 6653):
    """Start the network and return (net, port_map).

    Static ARP is applied so no ARP traffic reaches the controller and the
    per-source packet counters measure only the traffic under study.
    """
    cfg = cfg or load_config()
    net = Mininet(
        topo=DdosTopo(cfg=cfg),
        link=TCLink,
        switch=OVSSwitch,
        controller=RemoteController("c0", ip=controller_ip, port=controller_port),
        autoSetMacs=True,
        autoStaticArp=False,
        build=True,
    )
    net.start()
    net.staticArp()
    return net, build_port_map(net)


def victim_uplink_port(port_map: dict, cfg: dict) -> int:
    """Port on the core switch facing the victim, used for feature 10 and 11."""
    return port_map[cfg["topology"]["core"]][cfg["topology"]["victim"]["name"]]


def host_edge(host: str, cfg: dict) -> str:
    """Which edge switch a host hangs off. Mitigation is placed here (4.4)."""
    index = int(host.lstrip("h"))
    per_edge = cfg["topology"]["hosts_per_edge"]
    return cfg["topology"]["edges"][(index - 1) // per_edge]


if __name__ == "__main__":
    import json
    import sys

    from mininet.cli import CLI
    from mininet.log import setLogLevel

    setLogLevel("info")
    config = load_config()
    network, ports = build_network(config)
    print(json.dumps(ports, indent=2))
    out = pathlib.Path(__file__).resolve().parents[1] / "data" / "port_map.json"
    out.write_text(json.dumps(ports, indent=2))
    print(f"port map written to {out}")
    if "--cli" in sys.argv:
        CLI(network)
        network.stop()
