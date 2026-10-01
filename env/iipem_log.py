"""IIPEM WP3 measurement helpers.

Used only when an episode's options set "iipem_log"; the ISASF code path never
calls them.

Delivered traffic. The features count packets with priority-200 flows at the
edge switch, the same switch that holds the mitigation rules, so a mitigated
source vanishes from them. What actually reached the victim is counted here
instead, by one accounting rule per source in the victim's own INPUT chain
(inside its network namespace). A rule with no target only counts.

Controller load. CPU and resident memory of the OpenDaylight Karaf JVM, read
from /proc, so no extra Python package is needed on the VMs.
"""
from __future__ import annotations

import os
import time

CHAIN = "IIPEM_ACCT"
KARAF_MAIN = "org.apache.karaf.main.Main"


def acct_setup_cmd(source_ips: list[str]) -> str:
    """Create the accounting chain (idempotent) with one counting rule per source."""
    parts = [f"iptables -w -N {CHAIN} 2>/dev/null",
             f"iptables -w -F {CHAIN}",
             f"(iptables -w -C INPUT -j {CHAIN} 2>/dev/null || iptables -w -I INPUT 1 -j {CHAIN})"]
    parts += [f"iptables -w -A {CHAIN} -s {ip}" for ip in source_ips]
    return "; ".join(parts)


def acct_zero_cmd() -> str:
    return f"iptables -w -Z {CHAIN}"


def acct_read_cmd(zero: bool = False) -> str:
    """List the counters exactly (-x); with zero=True, reset them in the same call."""
    return f"iptables -w -L {CHAIN} -v -x -n" + (" -Z" if zero else "")


def parse_acct(text: str) -> dict[str, dict[str, int]]:
    """{source_ip: {"pkts": n, "bytes": n}} from `iptables -L CHAIN -v -x -n`.

    Counter lines start with the packet count; the source address is the
    second-to-last column (the last is the destination, 0.0.0.0/0). Headers,
    warnings and blank lines are skipped.
    """
    out: dict[str, dict[str, int]] = {}
    for line in text.splitlines():
        f = line.split()
        if len(f) < 4 or not f[0].isdigit() or not f[1].isdigit():
            continue
        src = f[-2]
        out[src] = {"pkts": int(f[0]), "bytes": int(f[1])}
    return out


class ControllerSampler:
    """CPU (% of one core, since the previous sample) and RSS (MB) of the Karaf JVM."""

    def __init__(self, proc_root: str = "/proc", clock=time.monotonic):
        self.proc = proc_root
        self.clock = clock
        self.tick = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
        self.pid = self.find_pid()
        self._prev = None            # (wall, cpu_ticks)

    def find_pid(self) -> int | None:
        """The process whose command line names Karaf's main class; None if absent."""
        try:
            entries = os.listdir(self.proc)
        except OSError:
            return None
        for name in sorted(entries, key=lambda s: (len(s), s)):
            if not name.isdigit():
                continue
            try:
                with open(os.path.join(self.proc, name, "cmdline"), "rb") as fh:
                    cmd = fh.read().replace(b"\0", b" ").decode(errors="replace")
            except OSError:
                continue
            if KARAF_MAIN in cmd:
                return int(name)
        return None

    def _cpu_ticks(self) -> int:
        with open(os.path.join(self.proc, str(self.pid), "stat")) as fh:
            stat = fh.read()
        # Fields after the command name, which is in parentheses and may contain spaces.
        rest = stat[stat.rindex(")") + 2:].split()
        utime, stime = int(rest[11]), int(rest[12])      # fields 14 and 15 of proc(5)
        return utime + stime

    def _rss_mb(self) -> float:
        with open(os.path.join(self.proc, str(self.pid), "status")) as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0      # kB -> MB
        return float("nan")

    def sample(self) -> tuple[float | None, float | None]:
        """(cpu_pct, rss_mb). cpu_pct is None on the first sample or if Karaf is not found."""
        if self.pid is None:
            return None, None
        try:
            now, ticks, rss = self.clock(), self._cpu_ticks(), self._rss_mb()
        except (OSError, ValueError, IndexError):
            return None, None
        cpu = None
        if self._prev is not None and now > self._prev[0]:
            cpu = 100.0 * (ticks - self._prev[1]) / self.tick / (now - self._prev[0])
        self._prev = (now, ticks)
        return cpu, rss
