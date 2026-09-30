"""Offline tests for the IIPEM WP1 events (no Mininet, no ODL, no switches).

    python3 scripts/test_wp1_events.py            # compares against tag isasf-2026
    python3 scripts/test_wp1_events.py --orig-dir /path/to/frozen/env

The environment runs with its network replaced by fakes that record every
command a host would have run. The checks:

  1. Regression: with ISASF's own options (E0, E1, E3 style schedules) the
     patched environment issues exactly the same commands, in the same order,
     with the same random rates, as the frozen code.
  2. Pairing: two arms on the same episode seed get the same attackers, surge
     hosts, replacement attackers and benign rates.
  3. Events fire once, at the requested step, through stop/restart with the
     probes restarted, and the roles change as specified.
  4. Invalid options are refused.
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import random
import subprocess
import sys
import tempfile
import types

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Stand-ins for modules the offline test never exercises.
for name in ("mininet", "mininet.link", "mininet.net", "mininet.node", "mininet.topo"):
    mod = types.ModuleType(name)
    for attr in ("TCLink", "Mininet", "OVSSwitch", "RemoteController", "Topo"):
        setattr(mod, attr, object)
    sys.modules.setdefault(name, mod)

try:
    import requests  # noqa: F401
except ImportError:                     # odl_client is imported but never called here
    req = types.ModuleType("requests"); req.Session = object
    ad = types.ModuleType("requests.adapters"); ad.HTTPAdapter = object
    u3 = types.ModuleType("urllib3.util.retry"); u3.Retry = object
    sys.modules.update({"requests": req, "requests.adapters": ad,
                        "urllib3": types.ModuleType("urllib3"),
                        "urllib3.util": types.ModuleType("urllib3.util"),
                        "urllib3.util.retry": u3})

import yaml  # noqa: E402

CFG = yaml.safe_load(open(REPO / "configs" / "base.yaml"))
EPOCH = 1_000_000.0


class FakeHost:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def popen(self, cmd, shell=True):
        self.log.append(("popen", self.name, cmd))

    def cmd(self, c):
        self.log.append(("cmd", self.name, c))
        return ""


class FakeNet:
    def __init__(self, log, n):
        self.hosts = [FakeHost(f"h{i}", log) for i in range(1, n + 1)]
        self._by = {h.name: h for h in self.hosts}

    def get(self, name):
        return self._by[name]


def load(path: pathlib.Path, alias: str):
    """Import a testbed_env.py and its sibling traffic.py under private names."""
    tr_spec = importlib.util.spec_from_file_location(f"{alias}_traffic", path.parent / "traffic.py")
    tr = importlib.util.module_from_spec(tr_spec)
    tr_spec.loader.exec_module(tr)
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.traffic = tr                       # the module under test uses its own traffic.py
    tr.time = types.SimpleNamespace(sleep=lambda s: None, time=lambda: EPOCH)
    tr.regime_epoch = lambda: EPOCH        # fixed, so command logs compare exactly
    mod.time = types.SimpleNamespace(sleep=lambda s: None, time=lambda: 0.0)
    return mod


def make_env(mod):
    env = object.__new__(mod.TestbedEnv)
    env.cfg = CFG
    env.W = float(CFG["runtime"]["window_w_s"])
    env.settle_s = 0.0
    env.steps_per_episode = CFG["training"]["steps_per_episode"]
    all_hosts = [f"h{i}" for i in range(1, CFG["topology"]["host_count"] + 1)]
    env.probe_hosts = list(CFG["roles"]["probes"])
    env.candidate_attackers = [h for h in all_hosts if h not in env.probe_hosts]
    env.log = []
    env.net = FakeNet(env.log, CFG["topology"]["host_count"])
    env.detector = types.SimpleNamespace(reset=lambda: None)
    env._mitigation = {}
    env._episode = -1
    env._t = 0
    env._rows = []
    env._prev_snapshot = None
    env._prev_derived = None
    # Fields the patched __init__ adds; harmless on the frozen class.
    env._event, env._event_step, env._event_fired = None, None, False
    env._surge_hosts, env._surge_rates = [], {}
    env._next_attackers, env._former_attackers = [], []
    env._snapshot = lambda: {"t_wall": 0.0}
    env._clear_mitigation = lambda: 0
    env._apply_action = lambda a, s, r: 0
    env.detector.suspects = lambda: set()

    def fake_observe(prev, curr, probe_window, churn, first):
        info = {"reward": 0.0, "step": env._t}
        if hasattr(env, "_event_info"):
            info.update(env._event_info())
        return None, info
    env._observe = fake_observe
    return env


def run(mod, options, seed, steps=None):
    env = make_env(mod)
    _, info0 = env.reset(seed=seed, options=options)
    infos = [info0]
    for _ in range(steps or env.steps_per_episode):
        _, _, _, trunc, info = env.step(0)
        infos.append(info)
        if trunc:
            break
    return env, infos


def rates_by_host(log):
    """(host, rate) for every iperf3 client start, in order."""
    out = []
    for kind, host, cmd in log:
        if kind == "popen" and cmd.startswith("iperf3 -c"):
            out.append((host, cmd.split("-b ")[1].split("M")[0]))
    return out


def check(cond, msg, failures):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-dir", help="directory holding the frozen testbed_env.py and traffic.py")
    args = ap.parse_args()

    tmp = None
    if args.orig_dir:
        orig_dir = pathlib.Path(args.orig_dir)
    else:
        tmp = tempfile.TemporaryDirectory()
        orig_dir = pathlib.Path(tmp.name)
        for f in ("testbed_env.py", "traffic.py"):
            src = subprocess.run(["git", "show", f"isasf-2026:env/{f}"], cwd=REPO,
                                 capture_output=True, text=True, check=True).stdout
            (orig_dir / f).write_text(src)

    orig = load(orig_dir / "testbed_env.py", "orig_env")
    new = load(REPO / "env" / "testbed_env.py", "new_env")
    failures: list[str] = []

    print("1. Regression against the frozen code (ISASF options)")
    isasf_cases = [({"schedule": [["R0", 100]]}, "E0 R0"),
                   ({"schedule": [["R1", 100]]}, "E0 R1"),
                   ({"schedule": [["R1", 50], ["R2", 50]]}, "E1 R1->R2"),
                   ({"schedule": [["R1", 50], ["R4", 50]]}, "E3 R1->R4")]
    for opts, name in isasf_cases:
        for seed in (1000, 1003007, 1009004):
            eo, _ = run(orig, opts, seed)
            en, _ = run(new, opts, seed)
            same_log = eo.log == en.log
            same_rng = eo._rng.getstate() == en._rng.getstate()
            if not (same_log and same_rng):
                check(False, f"{name} seed {seed}: log equal {same_log}, rng equal {same_rng}", failures)
                break
        else:
            check(True, f"{name}: identical commands and generator state on 3 seeds", failures)

    surge = {"schedule": [["R0", 100]], "surge": {"n_hosts": 3, "rate_mbps": 2.0, "at": 50}}
    swap = {"schedule": [["R2", 100]], "swap": {"at": 50}}

    print("2. Pairing across arms (same episode seed)")
    for opts, name in ((surge, "surge"), (swap, "swap")):
        a, ia = run(new, opts, 1004002)
        b, ib = run(new, opts, 1004002)
        check(a.log == b.log and ia == ib, f"{name}: two arms get identical roles, rates and commands", failures)
        c, ic = run(new, opts, 1004003)
        check(ia[0]["attackers"] != ic[0]["attackers"] or rates_by_host(a.log) != rates_by_host(c.log),
              f"{name}: a different seed gives a different draw", failures)

    print("3. Events")
    env, infos = run(new, surge, 1001001)
    stops = [i for i, e in enumerate(env.log) if e[0] == "cmd" and "pkill -f 'probes.py'" in e[2]]
    starts = [e for e in env.log if e[0] == "popen" and "probes.py --client" in e[2]]
    fired = [i for i, inf in enumerate(infos) if inf.get("event_fired")]
    check(len(stops) == 2 * CFG["topology"]["host_count"], "surge: stop_all ran at reset and once at the event", failures)
    check(len(starts) == 2 * len(CFG["roles"]["probes"]), "surge: probes started at reset and restarted at the event", failures)
    check(fired and fired[0] == 51 and infos[50]["event_fired"] is False,
          "surge: first flagged in the info after step 50's window (index 51), not before", failures)
    sh = infos[-1]["surge_hosts"]
    r = rates_by_host(env.log)
    n_benign = len(env.candidate_attackers) - CFG["roles"]["attackers"]["count"]
    after = dict(r[n_benign:])
    before = dict(r[:n_benign])
    check(len(sh) == 3 and set(sh) <= set(before), "surge: 3 surge hosts, all from the benign set", failures)
    check(all(after[h] == "2.0" for h in sh), "surge: surge hosts restarted at 2.0 Mbps", failures)
    lo, hi = CFG["roles"]["benign"]["rate_mbps"]
    check(all(lo <= float(v) <= hi for h, v in after.items() if h not in sh),
          "surge: other benign hosts keep ordinary rates", failures)
    check(infos[-1]["attackers"] == infos[0]["attackers"], "surge: attacker list unchanged", failures)

    env, infos = run(new, swap, 1002002)
    old, newa = infos[0]["attackers"], infos[-1]["attackers"]
    check(set(old).isdisjoint(newa) and len(newa) == len(old), "swap: new attackers are all former benign hosts", failures)
    check(infos[-1]["former_attackers"] == old, "swap: former attackers recorded", failures)
    hp = [e for e in env.log if e[0] == "popen" and e[2].startswith("hping3")]
    first = {e[1] for e in hp[:len(old)]}
    second = {e[1] for e in hp[len(old):]}
    check(first == set(old) and second == set(newa) and len(hp) == 2 * len(old),
          "swap: the flood restarts from exactly the new attackers", failures)
    benign_after = {h for h, _ in rates_by_host(env.log)[len(env.candidate_attackers) - len(old):]}
    check(set(old) <= benign_after and set(newa).isdisjoint(benign_after),
          "swap: former attackers now send benign traffic", failures)
    check(sum(1 for inf in infos if inf.get("event_fired")) == len(infos) - 51,
          "swap: fires once and stays fired", failures)

    print("4. Invalid options are refused")
    bad = [({"schedule": [["R0", 100]], "surge": {"n_hosts": 3, "rate_mbps": 2, "at": 50}, "swap": {"at": 50}}, "surge + swap"),
           ({"schedule": [["R1", 50], ["R2", 50]], "swap": {"at": 60}}, "event + regime switch"),
           ({"schedule": [["R0", 100]], "swap": {"at": 50}}, "swap in R0"),
           ({"schedule": [["R2", 100]], "swap": {"at": 100}}, "event at the last step"),
           ({"schedule": [["R0", 100]], "surge": {"n_hosts": 99, "rate_mbps": 2, "at": 50}}, "too many surge hosts"),
           ({"schedule": [["R0", 100]], "surge": {"n_hosts": 3, "rate_mbps": 0, "at": 50}}, "zero surge rate")]
    for opts, name in bad:
        try:
            run(new, opts, 7, steps=1)
            check(False, f"{name} accepted", failures)
        except ValueError:
            check(True, f"{name} refused", failures)

    if tmp:
        tmp.cleanup()
    print(f"\n{'ALL PASSED' if not failures else f'{len(failures)} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
