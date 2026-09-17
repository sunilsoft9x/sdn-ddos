"""Plan section 6.5: meters, control latency, and the choice of W.

Two jobs.

1. Install a meter through ODL and attach it to a priority-300 flow for one
   host, so the iperf3 test at the mininet> prompt can show whether metering
   actually rate-limits on this datapath. The plan's fallback is to rebuild
   with datapath='user' if it does not; on kernel 5.15 with OVS 2.17 it should.

2. Measure how long a control action takes to reach the datapath: from the
   RESTCONF PUT returning to the rule being visible in dump-flows. This is what
   decides W. The plan's rule: W = 0.5 s if the 95th percentile is under 150 ms,
   otherwise W = 1.0 s.

Results are written to data/timing.json, which Section V quotes.

Run with sudo (ovs-ofctl needs root), from the repo root, with Mininet already
up in another terminal:

    sudo ~/venv/bin/python scripts/meters_and_timing.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from env.odl_client import OdlClient  # noqa: E402
from env.topology import load_config  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
METER_ID = 1
LATENCY_SAMPLES = 100
POLL_INTERVAL_S = 0.005
POLL_TIMEOUT_S = 3.0


def dump_flows(switch: str, of: str) -> str:
    return subprocess.run(
        ["ovs-ofctl", "-O", of, "dump-flows", switch],
        capture_output=True, text=True,
    ).stdout


def dump_meters(switch: str, of: str) -> str:
    return subprocess.run(
        ["ovs-ofctl", "-O", of, "dump-meters", switch],
        capture_output=True, text=True,
    ).stdout


def sweep_datapath(switches: list[str], priority: int, of: str,
                   match_contains: str | None = None) -> int:
    """Force out every flow at one priority, straight on the datapath.

    ODL's config datastore can diverge from the switch, so the switch gets the
    last word. Feature 18 and the reward's C term both read the switch.
    """
    removed = 0
    for switch in switches:
        for line in dump_flows(switch, of).splitlines():
            if f"priority={priority}," not in line:
                continue
            if match_contains is not None and match_contains not in line:
                continue
            match_blob = line.partition(" actions=")[0]
            spec = "priority=" + match_blob.split("priority=", 1)[1].strip()
            subprocess.run(
                ["ovs-ofctl", "-O", of, "--strict", "del-flows", switch, spec],
                capture_output=True,
            )
            removed += 1
    return removed


def setup_meter(client: OdlClient, cfg: dict, host: str = "h1",
                rate_kbps: int = 5000) -> dict:
    """Meter on the host's edge switch, attached by a priority-300 flow."""
    from env.topology import host_edge

    of = cfg["topology"]["openflow_version"]
    edge = host_edge(host, cfg)
    with open(REPO / "data" / "port_map.json") as fh:
        port_map = json.load(fh)
    uplink = port_map[edge][cfg["topology"]["core"]]
    host_ip = f"10.0.0.{int(host.lstrip('h'))}"
    victim_ip = cfg["topology"]["victim"]["ip"]

    meter_code = client.put_meter(edge, METER_ID, rate_kbps)
    time.sleep(1.0)
    flow_code = client.put_flow(
        edge,
        flow_id=f"m300-{host}",
        priority=cfg["flows"]["priority"]["mitigation"],
        out_port=uplink,
        src=host_ip,
        dst=victim_ip,
        proto=17,
        meter_id=METER_ID,
    )
    time.sleep(1.0)

    meters = dump_meters(edge, of)
    flows = dump_flows(edge, of)
    return {
        "edge": edge,
        "host": host,
        "band_kbps": rate_kbps,
        "meter_put": meter_code,
        "flow_put": flow_code,
        "meter_visible_on_switch": "meter=1" in meters or "meter_id=1" in meters,
        "flow_has_meter_instruction": "meter:1" in flows or "meter=1" in flows,
        "dump_meters": meters.strip(),
    }


def measure_control_latency(client: OdlClient, cfg: dict,
                            samples: int = LATENCY_SAMPLES) -> dict:
    """Time PUT-returns to rule-visible-in-dump-flows, `samples` times.

    Synthetic source addresses (10.0.9.x) are used so these probe rules cannot
    match any real traffic while the measurement runs.
    """
    of = cfg["topology"]["openflow_version"]
    edge = cfg["topology"]["edges"][0]
    victim_ip = cfg["topology"]["victim"]["ip"]
    priority = cfg["flows"]["priority"]["mitigation"]

    latencies_ms: list[float] = []
    timeouts = 0
    installed: list[str] = []

    for i in range(samples):
        flow_id = f"lat{i}"
        src = f"10.0.9.{i % 250 + 1}"
        started = time.monotonic()
        client.put_flow(edge, flow_id, priority, out_port=None,
                        src=src, dst=victim_ip, proto=17)
        deadline = started + POLL_TIMEOUT_S
        seen = False
        while time.monotonic() < deadline:
            if f"nw_src={src}" in dump_flows(edge, of):
                latencies_ms.append((time.monotonic() - started) * 1000.0)
                seen = True
                break
            time.sleep(POLL_INTERVAL_S)
        installed.append(flow_id)
        if not seen:
            timeouts += 1
        # Remove immediately so the table does not grow and skew later samples.
        client.delete_flow(edge, flow_id)
        time.sleep(0.05)

    swept = sweep_datapath([edge], priority, of,
                           match_contains="nw_src=10.0.9.")

    if not latencies_ms:
        return {"samples": 0, "timeouts": timeouts, "swept": swept}

    ordered = sorted(latencies_ms)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * len(ordered))) - 1)]
    return {
        "samples": len(ordered),
        "timeouts": timeouts,
        "median_ms": round(statistics.median(ordered), 2),
        "p95_ms": round(p95, 2),
        "min_ms": round(ordered[0], 2),
        "max_ms": round(ordered[-1], 2),
        "mean_ms": round(statistics.fmean(ordered), 2),
        "swept_leftovers": swept,
        "poll_interval_ms": POLL_INTERVAL_S * 1000,
    }


def recommend_w(latency: dict) -> dict:
    """The plan's rule, applied to the measurement rather than to a guess.

    The poll interval bounds how finely latency can be resolved, so it is
    reported alongside: a p95 of 5 ms measured at 5 ms resolution means
    "at or below the resolution", not "exactly 5".
    """
    p95 = latency.get("p95_ms")
    if p95 is None:
        return {"w_s": 1.0, "reason": "no successful latency samples; taking the safe value"}
    if p95 < 150:
        return {
            "w_s": 0.5,
            "reason": f"p95 control latency {p95} ms is under the 150 ms threshold",
            "caveat": "still confirm probe success is stable at a 0.5 s timeout under R0",
        }
    return {
        "w_s": 1.0,
        "reason": f"p95 control latency {p95} ms is at or above the 150 ms threshold",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="h1")
    parser.add_argument("--rate-kbps", type=int, default=5000)
    parser.add_argument("--samples", type=int, default=LATENCY_SAMPLES)
    parser.add_argument("--skip-meter", action="store_true")
    parser.add_argument("--skip-latency", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    client = OdlClient(cfg)
    results: dict = {"measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    if not args.skip_meter:
        print("installing meter and priority-300 flow ...")
        results["meter"] = setup_meter(client, cfg, args.host, args.rate_kbps)
        m = results["meter"]
        print(f"  edge switch        : {m['edge']}")
        print(f"  meter PUT          : {m['meter_put']}")
        print(f"  flow PUT           : {m['flow_put']}")
        print(f"  meter on switch    : {m['meter_visible_on_switch']}")
        print(f"  flow cites meter   : {m['flow_has_meter_instruction']}")
        print("\n  dump-meters output:")
        print("   ", m["dump_meters"].replace("\n", "\n    ") or "(empty)")
        print(f"\n  Now at the mininet> prompt, with 'h16 iperf3 -s -D' running:")
        print(f"    {args.host} iperf3 -c {cfg['topology']['victim']['ip']} -u -b 10M -t 10")
        print(f"  Receiver rate should land near {args.rate_kbps / 1000:.1f} Mbps.")
        print("  If it stays at 10 Mbps, metering is not enforced on this datapath:")
        print("  rebuild with OVSSwitch(datapath='user') and note it in Section V.\n")

    if not args.skip_latency:
        print(f"measuring control latency over {args.samples} installs ...")
        results["control_latency"] = measure_control_latency(client, cfg, args.samples)
        lat = results["control_latency"]
        if lat["samples"]:
            print(f"  n                  : {lat['samples']} ({lat['timeouts']} timed out)")
            print(f"  median             : {lat['median_ms']} ms")
            print(f"  95th percentile    : {lat['p95_ms']} ms")
            print(f"  min / max          : {lat['min_ms']} / {lat['max_ms']} ms")
            if lat["swept_leftovers"]:
                print(f"  datastore/datapath divergence: {lat['swept_leftovers']} "
                      "orphan rules swept off the switch")
        else:
            print("  no samples completed; ODL is not installing flows")

        results["w_recommendation"] = recommend_w(lat)
        rec = results["w_recommendation"]
        print(f"\n  recommended W      : {rec['w_s']} s")
        print(f"  reason             : {rec['reason']}")
        if "caveat" in rec:
            print(f"  before committing  : {rec['caveat']}")
        print(f"\n  To apply: set runtime.window_w_s to {rec['w_s']} in configs/base.yaml")

    out = REPO / "data" / "timing.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
