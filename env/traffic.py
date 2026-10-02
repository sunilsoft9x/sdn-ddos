"""Regime launcher and service startup (plan sections 4.2, 4.3, 6.6).

This is orchestration only. It launches the standard tools named in
configs/base.yaml (hping3 for R1/R2, env/pulser.py for R4, iperf3 for benign
background, python http.server on the victim) as Mininet host processes, and
stops them again. It defines no traffic behaviour of its own; the command
templates live in the config.

Every generator is started with host.popen, never host.cmd, because host.cmd
blocks until the process exits and these run for the length of an episode.
stop_all pkills the generators on every host and waits briefly for the sockets
to close, so the next episode starts from silence.
"""

from __future__ import annotations

import time

# Non-probe, non-attacker hosts carry benign iperf3. Each gets its own victim
# port because a single iperf3 -s serves one test at a time.
BENIGN_BASE_PORT = 5000


def _host(net, name):
    return net.get(name)


def regime_epoch() -> float:
    """A shared wall-clock instant so synchronised regimes (R4) line up."""
    return time.time()


def start_servers(net, cfg, benign_hosts: list[str]) -> None:
    """HTTP + iperf3 targets on the victim. One iperf3 server per benign port.

    Victim TCP-stack condition (disclose in Section V): SYN cookies are turned
    off on the victim so that R1, a genuine SYN flood, degrades service through
    listen-backlog exhaustion rather than being silently absorbed. This is what
    gives R1 a low-utilisation, high-packet-rate signature distinct from R2's
    volumetric one; the setting is namespaced to the victim host only.
    """
    victim = _host(net, cfg["topology"]["victim"]["name"])
    victim.cmd("sysctl -w net.ipv4.tcp_syncookies=0 2>/dev/null")
    # Moderate half-open queue: benign completed connections never occupy it, so
    # R0 is unaffected, but a SYN flood fills it at once and legit SYNs are lost.
    victim.cmd("sysctl -w net.ipv4.tcp_max_syn_backlog=64 2>/dev/null")
    victim.popen("python3 env/victim_server.py --port 80", shell=True)
    victim.popen("iperf3 -s -p 5201", shell=True)  # kept for manual checks
    for i, _ in enumerate(benign_hosts):
        victim.popen(f"iperf3 -s -p {BENIGN_BASE_PORT + i}", shell=True)
    time.sleep(0.5)


def start_probes(net, cfg, log_dir: str = "logs") -> dict[str, str]:
    """Launch a probe client on each probe host. Returns host -> log path.

    Truncates the previous episode's log first, so aggregate() over a window
    never picks up a stale line, and so a fresh episode starts from an empty
    record.
    """
    import os
    victim_ip = cfg["topology"]["victim"]["ip"]
    probe = cfg["roles"]["probe"]
    logs: dict[str, str] = {}
    os.makedirs(log_dir, exist_ok=True)
    for name in cfg["roles"]["probes"]:
        out = f"{log_dir}/probe_{name}.jsonl"
        try:
            open(out, "w").close()   # truncate any previous episode's log
        except OSError:
            pass
        _host(net, name).popen(
            f"python3 env/probes.py --client --dst {victim_ip} "
            f"--port {probe['target_port']} --interval {probe['interval_s']} "
            f"--timeout {probe['timeout_s']} --out {out}",
            shell=True,
        )
        logs[name] = out
    return logs


def _interval_us(nominal_pps: float, scale: float) -> int:
    return max(1, int(round(1e6 / max(nominal_pps * scale, 1e-6))))


def _scaled_pps_in_template(template: str, scale: float) -> str:
    """Multiply the --pps value in a pulser template by scale, leave the rest.

    Keeps the pulser command entirely defined by the config: only the rate is
    adjusted for domain randomisation, exactly as the interval is for hping3.
    """
    import re

    def repl(m):
        return f"--pps {max(1, int(round(int(m.group(1)) * scale)))}"

    return re.sub(r"--pps\s+(\d+)", repl, template)


def start_regime(net, regime: str, attackers: list[str], cfg: dict,
                 scale: float = 1.0, epoch: float | None = None) -> list:
    """Start `regime` from `attackers`. Returns the popen handles.

    R0 is benign-only, so there is nothing to launch. Both attack regimes are
    defined entirely by their `cmd` template in base.yaml; scale adjusts only
    the rate (hping3's interval, the pulser's pps) and the shared epoch is
    appended to the pulser so several senders burst in phase.
    """
    if regime == "R0":
        return []
    spec = cfg["regimes"][regime]
    victim_ip = cfg["topology"]["victim"]["ip"]
    epoch = epoch if epoch is not None else regime_epoch()
    template = spec["cmd"]
    handles = []

    for name in attackers:
        host = _host(net, name)
        if "pulser" in template:
            cmd = _scaled_pps_in_template(template, scale).format(victim=victim_ip)
            cmd = f"{cmd} --epoch {epoch:.6f}"
        else:
            cmd = template.format(
                victim=victim_ip,
                interval_us=_interval_us(spec.get("nominal_pps", 1000), scale),
            )
        handles.append(host.popen(cmd, shell=True))
    return handles


def start_benign(net, cfg, benign_hosts: list[str], rng, duration_s: int = 120,
                 rate_override: dict[str, float] | None = None) -> list:
    """iperf3 TCP from each benign host at 0.2-0.5 Mbps, its own victim port.

    rate_override maps a host to a fixed rate in Mbps (the IIPEM benign
    surge). The random rate is drawn for every host either way, so the
    generator is consumed identically with or without an override.
    """
    victim_ip = cfg["topology"]["victim"]["ip"]
    lo, hi = cfg["roles"]["benign"]["rate_mbps"]
    handles = []
    for i, name in enumerate(benign_hosts):
        rate = round(rng.uniform(lo, hi), 3)
        if rate_override and name in rate_override:
            rate = round(float(rate_override[name]), 3)
        cmd = (f"iperf3 -c {victim_ip} -p {BENIGN_BASE_PORT + i} "
               f"-b {rate}M -t {duration_s}")
        handles.append(_host(net, name).popen(cmd, shell=True))
    return handles


def restart_benign_servers(net, cfg, n_ports: int) -> None:
    """Kill and relaunch the benign iperf3 servers (ports BENIGN_BASE_PORT+).

    The servers used to live for the whole run while their clients were killed
    and restarted every episode; over a long run they stopped accepting one by
    one and the benign hosts went silent (IIPEM status, step 20). Restarting
    them together with the clients makes every episode start from fresh
    servers. Port 5201 and the HTTP server are not touched.
    """
    victim = _host(net, cfg["topology"]["victim"]["name"])
    pat = "iperf3 -s -p 50[0-9][0-9]"
    victim.cmd(f"pkill -9 -f '{pat}' 2>/dev/null; "
               f"for i in $(seq 40); do pgrep -f '{pat}' >/dev/null || break; "
               f"sleep 0.05; done")
    for i in range(n_ports):
        victim.popen(f"iperf3 -s -p {BENIGN_BASE_PORT + i}", shell=True)
    time.sleep(0.3)   # listening before the clients connect


def stop_all(net) -> None:
    """Kill every generator AND probe client on every host.

    The probe kill matters: reset() launches fresh probe clients each episode,
    so without this they accumulate across episodes, pile load on the victim
    and interleave writes into the logs — which silently corrupts calibration.
    """
    for host in net.hosts:
        host.cmd("pkill -f hping3 2>/dev/null; "
                 "pkill -f 'pulser.py' 2>/dev/null; "
                 "pkill -f 'iperf3 -c' 2>/dev/null; "
                 "pkill -f 'probes.py' 2>/dev/null")
    time.sleep(0.5)


def stop_servers(net) -> None:
    for host in net.hosts:
        host.cmd("pkill -f 'http.server' 2>/dev/null; "
                 "pkill -f 'iperf3 -s' 2>/dev/null")
    time.sleep(0.3)
