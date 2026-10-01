"""Offline tests for IIPEM WP3 logging (no Mininet, no ODL, no switches).

    ~/venv/bin/python scripts/test_wp3_logging.py

1. The iptables counter parser, on legacy and nft-style output.
2. The accounting set-up command.
3. The controller sampler, on a fake /proc.
4. The logging flow in the environment, on the fake network from
   test_wp1_events.py: off by default (no iptables commands at all); when on,
   the chain is set up once, zeroed at every reset, read and zeroed just before
   the event's restart, read once more at the last step; every step's info
   carries the WP3 fields.
Also rerun scripts/test_wp1_events.py: it checks that ISASF's options still
produce exactly the frozen code's commands.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

failures: list[str] = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t = load("t", REPO / "scripts" / "test_wp1_events.py")     # also stubs mininet/requests
from env import iipem_log as L  # noqa: E402

LEGACY = """Chain IIPEM_ACCT (1 references)
    pkts      bytes target     prot opt in     out     source               destination
    1520  2101840            all  --  *      *       10.0.0.4             0.0.0.0/0
       0        0            all  --  *      *       10.0.0.5             0.0.0.0/0
  883301 921744114           all  --  *      *       10.0.0.12            0.0.0.0/0
"""
NFT = """# Warning: iptables-legacy tables present, use iptables-legacy to see them
Chain IIPEM_ACCT (1 references)
    pkts      bytes target     prot opt in     out     source               destination
      17     1428            all  --  *      *       10.0.0.1             0.0.0.0/0
"""

print("1. Counter parser")
p = L.parse_acct(LEGACY)
check(p == {"10.0.0.4": {"pkts": 1520, "bytes": 2101840},
            "10.0.0.5": {"pkts": 0, "bytes": 0},
            "10.0.0.12": {"pkts": 883301, "bytes": 921744114}},
      "legacy output: three sources, exact counts, header skipped")
check(L.parse_acct(NFT) == {"10.0.0.1": {"pkts": 17, "bytes": 1428}},
      "nft output: warning and header lines skipped")
check(L.parse_acct("") == {}, "empty output gives no sources")

print("2. Set-up command")
ips = [f"10.0.0.{i}" for i in range(1, 16)]
cmd = L.acct_setup_cmd(ips)
check(cmd.count("-A IIPEM_ACCT -s ") == 15 and "-j" not in cmd.split("-A IIPEM_ACCT")[1],
      "one counting rule per source, no target (count only)")
check("-C INPUT -j IIPEM_ACCT" in cmd and "-I INPUT 1 -j IIPEM_ACCT" in cmd,
      "INPUT jump added only if missing, so repeated set-up does not duplicate it")
check(L.acct_read_cmd(zero=True).endswith("-x -n -Z") and " -x " in L.acct_read_cmd(),
      "exact counters (-x); read-and-zero in one call")

print("3. Controller sampler on a fake /proc")
with tempfile.TemporaryDirectory() as d:
    root = pathlib.Path(d)

    def proc(pid, cmdline, ticks, rss_kb, comm="java"):
        q = root / str(pid)
        q.mkdir(exist_ok=True)
        (q / "cmdline").write_bytes(b"\0".join(c.encode() for c in cmdline) + b"\0")
        fields = ["S"] + ["0"] * 10 + [str(ticks), "0"] + ["0"] * 30   # utime=ticks, stime=0
        (q / "stat").write_text(f"{pid} ({comm}) " + " ".join(fields))
        (q / "status").write_text(f"Name:\tjava\nVmRSS:\t{rss_kb} kB\n")

    proc(100, ["java", "-jar", "other.jar"], 5, 1000)
    proc(200, ["java", "-Dkaraf.home=/opt/odl", "org.apache.karaf.main.Main"], 1000, 2_097_152,
         comm="java (odl) x")
    (root / "self").mkdir()
    clock = iter([10.0, 10.5, 11.5]).__next__
    s = L.ControllerSampler(proc_root=str(root), clock=clock)
    s.tick = 100
    check(s.pid == 200, "finds the Karaf JVM, not another Java process")
    cpu, rss = s.sample()
    check(cpu is None and abs(rss - 2048.0) < 1e-9, "first sample: no CPU share yet; RSS 2048 MB")
    proc(200, ["java", "org.apache.karaf.main.Main"], 1050, 2_097_152, comm="java (odl) x")
    cpu, _ = s.sample()
    check(abs(cpu - 100.0) < 1e-9, "50 ticks over 0.5 s at 100 ticks/s = 100% of one core")
    proc(200, ["java", "org.apache.karaf.main.Main"], 1075, 2_097_152, comm="java (odl) x")
    cpu, _ = s.sample()
    check(abs(cpu - 25.0) < 1e-9, "25 ticks over 1.0 s = 25%; command name with spaces parsed")
    empty = L.ControllerSampler(proc_root=str(root / "self"))
    check(empty.pid is None and empty.sample() == (None, None), "no Karaf found: (None, None), no error")

print("4. Logging flow in the environment (fake network)")
new = t.load(REPO / "env" / "testbed_env.py", "new_env_wp3")
src = (REPO / "env" / "testbed_env.py").read_text()
check("info.update(self._iipem_info())" in src, "the real _observe adds the WP3 fields to every step's info")

COUNTS = {"pre": LEGACY, "post": NFT}


def make(log_on_reads):
    env = t.make_env(new)
    env._iipem_log, env._acct_ready, env._acct_pre, env._ctl, env._last_apply_ms = False, False, None, None, None
    env.cfg = dict(env.cfg)
    reads = {"n": 0}
    vname = env.cfg["topology"]["victim"]["name"]
    if vname not in env.net._by:                         # the fake net may stop short of the victim
        env.net._by[vname] = t.FakeHost(vname, env.log)
    victim = env.net._by[vname]

    def cmd(c, _orig=victim.cmd):
        _orig(c)
        if "IIPEM_ACCT -v -x -n" in c:
            reads["n"] += 1
            return COUNTS["pre"] if " -Z" in c else COUNTS["post"]
        return ""
    victim.cmd = cmd

    def obs(prev, curr, probe_window, churn, first):
        info = {"reward": 0.0, "step": env._t}
        info.update(env._event_info())
        info.update(env._iipem_info())
        return None, info
    env._observe = obs

    def apply(a, s, r):
        if a == 1:
            env._mitigation["m300-10_0_0_7"] = {"src": "10.0.0.7"}
        return 0
    env._apply_action = apply
    return env, reads


def episode(env, options, action=0):
    _, info0 = env.reset(seed=1004002, options=options)
    infos = [info0]
    for _ in range(env.steps_per_episode):
        _, _, _, trunc, info = env.step(action)
        infos.append(info)
        if trunc:
            break
    return infos


surge = {"schedule": [["R0", 100]], "surge": {"n_hosts": 2, "rate_mbps": 4.0, "at": 50}}

env, reads = make(True)
infos = episode(env, surge)
ipt = [e for e in env.log if e[0] == "cmd" and "iptables" in e[2]]
check(not ipt and "flagged" not in infos[5] and "delivered_pre" not in infos[-1],
      "off by default: no iptables commands and no WP3 fields")

env, reads = make(True)
infos = episode(env, dict(surge, iipem_log=True), action=1)
infos2 = episode(env, dict(surge, iipem_log=True), action=1)
ipt = [(i, e[2]) for i, e in enumerate(env.log) if e[0] == "cmd" and "iptables" in e[2]]
setups = [c for _, c in ipt if "-N IIPEM_ACCT" in c]
zeros = [c for _, c in ipt if c.strip() == L.acct_zero_cmd()]
check(len(setups) == 1 and setups[0].count("-A IIPEM_ACCT -s ") == 15,
      "chain set up once over two episodes, with 15 source rules")
check(len(zeros) == 2, "counters zeroed at each reset")
check(reads["n"] == 4, "two reads per episode: at the event and at the last step")
first_zero_read = next(i for i, c in ipt if c.endswith("-Z"))
pk = [i for i, e in enumerate(env.log) if e[0] == "cmd" and "pkill -f 'probes.py'" in e[2]]
event_stop = pk[len(env.net.hosts)]          # first pkill of episode 1's second stop_all (the event)
check(pk[len(env.net.hosts) - 1] < first_zero_read < event_stop,
      "the event's counters are read after the reset's stop_all and before the event's")
last = infos[-1]
check(last["delivered_pre"] == L.parse_acct(LEGACY) and last["delivered_post"] == L.parse_acct(NFT),
      "last step reports delivered bytes per source before and after the event")
mid = infos[60]
check(all(k in mid for k in ("flagged", "mitigated", "apply_ms", "ctl_cpu_pct", "ctl_rss_mb", "ctl_pid")),
      "every step's info carries the WP3 fields")
check(mid["mitigated"] == ["10.0.0.7"] and isinstance(mid["apply_ms"], float) and mid["apply_ms"] >= 0,
      "mitigated hosts and action time recorded")
check(mid["ctl_pid"] is None and mid["ctl_cpu_pct"] is None,
      "no Karaf on this machine: controller fields are None, not an error")

env, reads = make(True)
infos = episode(env, {"schedule": [["R2", 100]], "iipem_log": True})
check(infos[-1]["delivered_post"] is None and infos[-1]["delivered_pre"] == L.parse_acct(NFT),
      "no event: a single phase, reported as delivered_pre")

print(f"\n{'ALL PASSED' if not failures else f'{len(failures)} FAILED'}")
sys.exit(1 if failures else 0)
