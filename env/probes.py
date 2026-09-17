"""Service probes and their aggregation (plan sections 4.2, 4.12).

Probe hosts h1, h6, h11 (one per edge switch) each send an HTTP GET to the
victim h16 every 0.1 s with a 0.5 s timeout. The probes are the ground truth
for service quality: features 13-15 and the reward's S and L terms all come
from them.

Two roles in one file:
  --client  runs on a probe host inside Mininet, launched by traffic.py via
            host.popen, and appends one JSON line per attempt to a log file.
  (library) the env wrapper calls aggregate() each window to turn the lines
            that fall inside [t0, t1] into probe_success, median latency and
            goodput.

Wall-clock time is used for the timestamps so the env, running in a different
process, can select a window by the same clock the counters use.
"""

from __future__ import annotations

import argparse
import http.client
import json
import pathlib
import statistics
import time


def client_loop(dst: str, port: int, path: str, interval_s: float,
                timeout_s: float, out_path: str) -> None:
    """Probe until killed. One JSON line per attempt, flushed immediately.

    A failed or timed-out request is logged as success=false with latency set
    to the timeout, so a window full of failures still yields a defined median
    latency (the timeout) rather than an empty set.
    """
    out = open(out_path, "a", buffering=1)
    while True:
        cycle_start = time.monotonic()
        t_wall = time.time()
        ok = False
        latency = timeout_s
        nbytes = 0
        conn = None
        started = time.monotonic()
        try:
            conn = http.client.HTTPConnection(dst, port, timeout=timeout_s)
            conn.request("GET", path)
            resp = conn.getresponse()
            body = resp.read()
            latency = time.monotonic() - started
            nbytes = len(body)
            ok = 200 <= resp.status < 300
        except Exception:
            latency = min(time.monotonic() - started, timeout_s)
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
        out.write(json.dumps({
            "t": round(t_wall, 4),
            "ok": ok,
            "latency": round(latency, 5),
            "bytes": nbytes,
        }) + "\n")
        # Pace to the requested interval, measured from the cycle start.
        sleep_for = interval_s - (time.monotonic() - cycle_start)
        if sleep_for > 0:
            time.sleep(sleep_for)


def read_lines(paths: list[str]) -> list[dict]:
    records: list[dict] = []
    for p in paths:
        fp = pathlib.Path(p)
        if not fp.exists():
            continue
        for line in fp.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A half-written last line during a live read: skip it.
                continue
    return records


def aggregate(records: list[dict], t0: float, t1: float,
              latency_scale_ms: float = 500.0) -> dict:
    """Collapse probe attempts in [t0, t1) into the window's service terms.

    Returns success rate in [0,1], median latency in seconds and its scaled,
    clipped form for the reward, goodput in bytes/s, and the attempt count so
    the env can tell an empty window (no probes ran) from a fully failed one.
    """
    window = [r for r in records if t0 <= r["t"] < t1]
    n = len(window)
    if n == 0:
        return {
            "n": 0, "success": 0.0,
            "latency_s": latency_scale_ms / 1000.0,
            "latency_scaled": 1.0, "goodput_bps": 0.0,
        }

    successes = [r for r in window if r["ok"]]
    success_rate = len(successes) / n
    latencies = [r["latency"] for r in window]
    median_latency = statistics.median(latencies)
    scaled = min(median_latency / (latency_scale_ms / 1000.0), 1.0)
    span = max(t1 - t0, 1e-6)
    goodput = sum(r["bytes"] for r in successes) / span

    return {
        "n": n,
        "success": success_rate,
        "latency_s": median_latency,
        "latency_scaled": scaled,
        "goodput_bps": goodput,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="HTTP service probe (Mininet testbed)")
    p.add_argument("--client", action="store_true")
    p.add_argument("--dst", required=True)
    p.add_argument("--port", type=int, default=80)
    p.add_argument("--path", default="/")
    p.add_argument("--interval", type=float, default=0.1)
    p.add_argument("--timeout", type=float, default=0.5)
    p.add_argument("--out", default="logs/probe.jsonl")
    args = p.parse_args()
    if args.client:
        client_loop(args.dst, args.port, args.path, args.interval,
                    args.timeout, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
