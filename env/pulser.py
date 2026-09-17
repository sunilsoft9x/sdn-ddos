"""Pulsed UDP source for the held-out R4 evaluation regime (plan section 4.3).

Scope: this runs only as a Mininet host process inside the testbed VM. It sends
UDP datagrams to another host on the emulated 10.0.0.0/24 (the victim h16),
which exists only inside Mininet. It is launched by env/traffic.py through
host.popen and terminated by stop_all; it is not meant to be run against
anything outside the emulation.

Why R4 needs its own sender rather than hping3: R4 is a low-rate pulsed regime.
Each source bursts together for `on` seconds out of every `period`, so the
average rate stays moderate while the synchronised bursts still cause loss at
the victim. That on/off duty cycle is awkward to express with hping3's flat
interval, so this is a small socket sender with a duty-cycle gate. Nothing here
is volumetric by design: at the plan's defaults the average is about 1.6 Mbps.

Design choices to record in Section IV: real source address (no spoofing),
fixed payload size, synchronised phase across attackers via a shared wall-clock
epoch so the bursts line up without any coordination channel.
"""

from __future__ import annotations

import argparse
import signal
import socket
import sys
import time

_RUNNING = True


def _stop(signum, frame):  # noqa: ARG001
    global _RUNNING
    _RUNNING = False


def run(dst: str, port: int, pps: int, size: int, on_s: float,
        period_s: float, duration_s: float | None, epoch: float | None) -> None:
    """Send `size`-byte UDP datagrams at `pps` during the on-phase of each period.

    Phase is aligned to `epoch` (a shared wall-clock time) so that several
    senders started a little apart still burst together. Pacing uses the
    monotonic clock and short sleeps rather than a busy loop, so an idle
    attacker costs no CPU — which matters when fifteen of them share the VM.
    """
    off_s = max(0.0, period_s - on_s)
    interval = 1.0 / pps if pps > 0 else 0.0
    payload = b"\x00" * size

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
    target = (dst, port)

    if epoch is None:
        epoch = time.time()
    start_mono = time.monotonic()
    deadline = None if duration_s is None else start_mono + duration_s

    # Registering handlers only works in the main thread. In production the
    # pulser is its own popen subprocess, so this always succeeds; the guard
    # just keeps an import-and-run in a worker thread from crashing.
    try:
        signal.signal(signal.SIGTERM, _stop)
        signal.signal(signal.SIGINT, _stop)
    except ValueError:
        pass

    sent = 0
    while _RUNNING:
        now = time.monotonic()
        if deadline is not None and now >= deadline:
            break

        # Where are we within the current period, measured on the shared epoch?
        phase = (time.time() - epoch) % period_s
        if phase >= on_s:
            # Off-phase: sleep until the next on-phase begins.
            time.sleep(min(off_s - (phase - on_s) if phase < on_s + off_s else off_s,
                           period_s))
            continue

        # On-phase: send at the target rate until the burst window closes.
        burst_end = now + (on_s - phase)
        next_send = now
        while _RUNNING and time.monotonic() < burst_end:
            if deadline is not None and time.monotonic() >= deadline:
                break
            try:
                sock.sendto(payload, target)
                sent += 1
            except OSError:
                # Socket buffer full: yield briefly and keep the rate bounded.
                time.sleep(0.001)
            next_send += interval
            slack = next_send - time.monotonic()
            if slack > 0:
                time.sleep(slack)

    sock.close()
    sys.stderr.write(f"pulser: sent {sent} datagrams to {dst}:{port}\n")


def main() -> int:
    p = argparse.ArgumentParser(description="Pulsed UDP source for R4 (Mininet testbed only)")
    p.add_argument("--dst", required=True, help="victim host IP on the emulated subnet")
    p.add_argument("--port", type=int, default=5353)
    p.add_argument("--pps", type=int, default=833, help="packets/s during the on-phase")
    p.add_argument("--size", type=int, default=1200, help="payload bytes per datagram")
    p.add_argument("--on", dest="on_s", type=float, default=0.2, help="burst length (s)")
    p.add_argument("--period", dest="period_s", type=float, default=1.0, help="burst period (s)")
    p.add_argument("--duration", dest="duration_s", type=float, default=None,
                   help="stop after this many seconds (default: run until signalled)")
    p.add_argument("--epoch", type=float, default=None,
                   help="shared wall-clock epoch for phase alignment across senders")
    args = p.parse_args()

    if not args.dst.startswith("10.0.0."):
        # Guard rail: R4 targets the emulated victim. Refuse anything off-subnet.
        sys.stderr.write(
            f"pulser: refusing dst {args.dst!r}; this sender is for the "
            "emulated 10.0.0.0/24 testbed victim only\n"
        )
        return 2

    run(args.dst, args.port, args.pps, args.size, args.on_s,
        args.period_s, args.duration_s, args.epoch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
