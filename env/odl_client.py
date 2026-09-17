"""OpenDaylight RESTCONF client for flow and meter control (plan section 4.4).

Every control action in this project goes through ODL. Counters do not — they
come from ovs-ofctl, because ODL's statistics path refreshes far too slowly for
a sub-second measurement window. State that split in Section V.

Path form verified against Titanium SR2 (karaf 0.22.2) on 17 Sep 2026: the
node id needs its colon percent-encoded and the table segment needs its module
prefix, i.e. node=openflow%3A2/flow-node-inventory:table=0. A PUT in that form
returned 201. Both are read from configs/base.yaml rather than hard-coded.
"""

from __future__ import annotations

import pathlib
from typing import Any

import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

CONFIG_PATH = pathlib.Path(__file__).resolve().parents[1] / "configs" / "base.yaml"

ETH_TYPE_IPV4 = 2048
PROTO_NAMES = {6: "tcp", 17: "udp", 1: "icmp"}


class OdlError(RuntimeError):
    pass


class OdlClient:
    def __init__(self, cfg: dict | None = None):
        if cfg is None:
            with open(CONFIG_PATH) as fh:
                cfg = yaml.safe_load(fh)
        odl = cfg["runtime"]["odl"]
        self.base = odl["base_url"].rstrip("/")
        self.timeout = odl["timeout_s"]
        self.table_segment = odl["table_path"]
        self.session = requests.Session()
        self.session.auth = tuple(odl["auth"])
        self.session.headers.update(
            {"Content-Type": "application/json", "Accept": "application/json"}
        )
        retry = Retry(
            total=odl["retries"],
            backoff_factor=0.2,
            status_forcelist=(500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "PUT", "DELETE"]),
        )
        self.session.mount("http://", HTTPAdapter(max_retries=retry))

    # -- URL construction -------------------------------------------------

    @staticmethod
    def node_id(switch: str) -> str:
        """s2 -> openflow%3A2. Mininet names switches sN and ODL numbers them N."""
        return f"openflow%3A{int(switch.lstrip('s'))}"

    def _flow_url(self, switch: str, flow_id: str) -> str:
        return (
            f"{self.base}/opendaylight-inventory:nodes/node={self.node_id(switch)}"
            f"/{self.table_segment}/flow={flow_id}"
        )

    def _meter_url(self, switch: str, meter_id: int) -> str:
        return (
            f"{self.base}/opendaylight-inventory:nodes/node={self.node_id(switch)}"
            f"/flow-node-inventory:meter={meter_id}"
        )

    # -- Request plumbing -------------------------------------------------

    def _send(self, method: str, url: str, body: dict | None = None) -> int:
        response = self.session.request(
            method, url, json=body, timeout=self.timeout
        )
        if response.status_code not in (200, 201, 204, 404):
            raise OdlError(
                f"{method} {url} -> {response.status_code}: {response.text[:300]}"
            )
        return response.status_code

    # -- Flow bodies ------------------------------------------------------

    @staticmethod
    def _match(src: str | None, dst: str | None, proto: int | None) -> dict:
        match: dict[str, Any] = {
            "ethernet-match": {"ethernet-type": {"type": ETH_TYPE_IPV4}}
        }
        if proto is not None:
            match["ip-match"] = {"ip-protocol": proto}
        if src is not None:
            match["ipv4-source"] = f"{src}/32"
        if dst is not None:
            match["ipv4-destination"] = f"{dst}/32"
        return match

    @staticmethod
    def _output(port: int, order: int = 0) -> dict:
        return {
            "order": order,
            "output-action": {"output-node-connector": str(port)},
        }

    def _flow_body(self, flow_id: str, priority: int, match: dict,
                   actions: list[dict], meter_id: int | None = None) -> dict:
        instructions: list[dict] = []
        if meter_id is not None:
            instructions.append({"order": 0, "meter": {"meter-id": meter_id}})
            instructions.append(
                {"order": 1, "apply-actions": {"action": actions}}
            )
        elif actions:
            instructions.append({"order": 0, "apply-actions": {"action": actions}})
        else:
            # No apply-actions instruction at all is how OpenFlow expresses drop.
            instructions = []
        return {
            "flow-node-inventory:flow": [
                {
                    "id": flow_id,
                    "table_id": 0,
                    "priority": priority,
                    "match": match,
                    "instructions": {"instruction": instructions},
                }
            ]
        }

    # -- Public operations ------------------------------------------------

    def put_flow(self, switch: str, flow_id: str, priority: int,
                 out_port: int | None = None, src: str | None = None,
                 dst: str | None = None, proto: int | None = None,
                 meter_id: int | None = None) -> int:
        """Install one flow. out_port=None means drop."""
        actions = [] if out_port is None else [self._output(out_port, 0 if meter_id is None else 0)]
        body = self._flow_body(
            flow_id, priority, self._match(src, dst, proto), actions, meter_id
        )
        return self._send("PUT", self._flow_url(switch, flow_id), body)

    def delete_flow(self, switch: str, flow_id: str) -> int:
        return self._send("DELETE", self._flow_url(switch, flow_id))

    def put_meter(self, switch: str, meter_id: int, rate_kbps: int,
                  burst_kb: int | None = None) -> int:
        burst = burst_kb if burst_kb is not None else max(1, rate_kbps // 10)
        body = {
            "flow-node-inventory:meter": [
                {
                    "meter-id": meter_id,
                    "flags": "meter-kbps",
                    "meter-band-headers": {
                        "meter-band-header": [
                            {
                                "band-id": 0,
                                "meter-band-types": {"flags": "ofpmbt-drop"},
                                "band-rate": rate_kbps,
                                "band-burst-size": burst,
                                "drop-rate": rate_kbps,
                                "drop-burst-size": burst,
                            }
                        ]
                    },
                }
            ]
        }
        return self._send("PUT", self._meter_url(switch, meter_id), body)

    def delete_meter(self, switch: str, meter_id: int) -> int:
        return self._send("DELETE", self._meter_url(switch, meter_id))

    def get_flows(self, switch: str) -> dict:
        url = (
            f"{self.base}/opendaylight-inventory:nodes/node={self.node_id(switch)}"
            f"/{self.table_segment}?content=config"
        )
        response = self.session.get(url, timeout=self.timeout)
        return response.json() if response.status_code == 200 else {}

    def delete_all(self, switches: list[str], priority: int,
                   known_ids: list[str] | None = None) -> int:
        """Remove every flow at one priority. Used by action 4, clear mitigation.

        Deleting by id is the only reliable route: RESTCONF has no delete-by-match,
        and 404 is treated as success so the operation is idempotent.
        """
        removed = 0
        for switch in switches:
            ids = known_ids
            if ids is None:
                config = self.get_flows(switch)
                table = config.get("flow-node-inventory:table", [{}])
                ids = [
                    f["id"]
                    for f in table[0].get("flow", [])
                    if f.get("priority") == priority
                ]
            for flow_id in ids:
                if self.delete_flow(switch, flow_id) != 404:
                    removed += 1
        return removed


if __name__ == "__main__":
    client = OdlClient()
    print("node id for s2:", client.node_id("s2"))
    print("flow url:", client._flow_url("s2", "probe"))
    code = client.put_flow(
        "s2", "selftest", 1, out_port=1, src="10.0.0.1", dst="10.0.0.16", proto=6
    )
    print("put selftest ->", code)
    print("delete selftest ->", client.delete_flow("s2", "selftest"))
