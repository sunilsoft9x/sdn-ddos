"""Gymnasium environment over the live OpenDaylight + Mininet testbed (6.9).

The Mininet network is built once in __init__ and kept alive across every
episode. Rebuilding it forces ODL to reconnect and costs about 20 s, so reset()
only tears down traffic and mitigation, never the network.

One step is one measurement window W: apply the action, wait until W has passed
since the last snapshot, read counters straight from Open vSwitch, turn the two
snapshots into the 18-feature observation and the reward, then apply any
scheduled regime switch. Every number reported in the paper's results comes
from this environment; the simulator only trains the policies.

The environment owns all priority-300 (mitigation) state, because three
quantities depend on it and nothing else knows them: the reward's churn term C,
feature 18 (mitigated fraction) and feature 12 (edge table occupancy). After
every action it reconciles its own record against the datapath, so an action-4
clear that ODL reports as done but that leaves a rule on the switch cannot go
unnoticed — the divergence seen during 6.5 is closed here rather than trusted
away.
"""

from __future__ import annotations

import pathlib
import random
import subprocess
import time
from time import perf_counter

import numpy as np
import yaml

try:
    import gymnasium as gym
    from gymnasium import spaces
    _BASE = gym.Env
except Exception:  # allow import for offline unit tests without gymnasium
    gym = None
    spaces = None
    _BASE = object

from env.features import FeatureExtractor, reward as compute_reward
from env.ovs_stats import collect, load_config
from env.odl_client import OdlClient
from env.probes import aggregate, read_lines
from env.suspects import SuspectDetector
from env import iipem_log, topology, traffic

REPO = pathlib.Path(__file__).resolve().parents[1]


class TestbedEnv(_BASE):
    metadata = {"render_modes": []}

    def __init__(self, cfg: dict | None = None, run_id: str = "run",
                 log_path: str | None = None, settle_s: float = 3.0,
                 controller_ip: str = "127.0.0.1"):
        self.cfg = cfg or load_config()
        self.run_id = run_id
        self.W = float(self.cfg["runtime"]["window_w_s"])
        self.settle_s = settle_s
        self.steps_per_episode = self.cfg["training"]["steps_per_episode"]
        self.victim_ip = self.cfg["topology"]["victim"]["ip"]
        self.core = self.cfg["topology"]["core"]
        self.edges = list(self.cfg["topology"]["edges"])
        self.switches = [self.core] + self.edges
        self.of = self.cfg["topology"]["openflow_version"]

        # Build the network once and install the static flow tables.
        self.net, self.port_map = topology.build_network(
            self.cfg, controller_ip=controller_ip)
        self.victim_port = topology.victim_uplink_port(self.port_map, self.cfg)
        self._install_static_flows()

        self.odl = OdlClient(self.cfg)
        self.fx = FeatureExtractor(self.cfg)
        self.detector = SuspectDetector(self.cfg)

        all_hosts = [f"h{i}" for i in range(1, self.cfg["topology"]["host_count"] + 1)]
        self.probe_hosts = list(self.cfg["roles"]["probes"])
        self.candidate_attackers = [h for h in all_hosts if h not in self.probe_hosts]

        traffic.start_servers(self.net, self.cfg,
                              benign_hosts=self.candidate_attackers)

        # Mitigation bookkeeping: flow id -> {src, edge, kind, meter_id}.
        self._mitigation: dict[str, dict] = {}
        self._next_meter_id = 1

        # Per-episode state, set in reset().
        self._episode = -1
        self._t = 0
        self._prev_snapshot = None
        self._prev_derived = None
        self._regime = "R0"
        self._switch_step = None
        self._switch_to = None
        self._attackers: list[str] = []
        # IIPEM events (C1 surge, C2 swap); inactive unless options ask.
        self._event: str | None = None
        self._event_step: int | None = None
        self._event_fired = False
        self._surge_hosts: list[str] = []
        self._surge_rates: dict[str, float] = {}
        self._next_attackers: list[str] = []
        self._former_attackers: list[str] = []
        # IIPEM WP3 measurement; inactive unless options["iipem_log"].
        self._iipem_log = False
        self._acct_ready = False
        self._acct_pre: dict | None = None
        self._ctl = None
        self._last_apply_ms: float | None = None
        self._scale = 1.0
        self._epoch = 0.0
        self._probe_logs: dict[str, str] = {}
        self._rows: list[dict] = []
        self.log_path = log_path or str(REPO / "runs" / f"{run_id}.parquet")

        if spaces is not None:
            self.action_space = spaces.Discrete(5)
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(18,), dtype=np.float32)

    # -- setup ------------------------------------------------------------

    def _install_static_flows(self) -> None:
        from env import setup_flows
        client = OdlClient(self.cfg)
        setup_flows.install_default_forwarding(client, self.port_map, self.cfg)
        setup_flows.install_monitoring(client, self.port_map, self.cfg)

    # -- counters and rates ----------------------------------------------

    def _snapshot(self) -> dict:
        return collect(self.switches, self.cfg)

    def _per_source_rates(self, prev: dict | None, curr: dict) -> dict[str, float]:
        """Current-window packet rate per source toward the victim, in pps."""
        from env.ovs_stats import per_source_packets
        curr_counts = per_source_packets(curr, self.victim_ip)
        if prev is None:
            return {}
        prev_counts = per_source_packets(prev, self.victim_ip)
        dt = max(curr["t_wall"] - prev["t_wall"], 1e-6)
        rates = {}
        for src, c in curr_counts.items():
            now = sum(c.get(p, 0) for p in ("tcp", "udp", "icmp"))
            before = prev_counts.get(src, {})
            was = sum(before.get(p, 0) for p in ("tcp", "udp", "icmp"))
            rates[src] = max(0, now - was) / dt
        return rates

    def _edge_flow_count(self) -> int:
        """Feature 12: current flow-table occupancy across the source edges."""
        total = 0
        for edge in self.edges:
            out = subprocess.run(
                ["ovs-ofctl", "-O", self.of, "dump-flows", edge],
                capture_output=True, text=True).stdout
            total += out.count("n_packets=")
        return total

    # -- mitigation actions ----------------------------------------------

    def _src_ip(self, host_or_ip: str) -> str:
        if host_or_ip.startswith("10."):
            return host_or_ip
        return f"10.0.0.{int(host_or_ip.lstrip('h'))}"

    def _edge_of_ip(self, ip: str) -> str:
        idx = int(ip.split(".")[-1])
        per = self.cfg["topology"]["hosts_per_edge"]
        return self.edges[(idx - 1) // per]

    def _apply_action(self, action: int, suspect_ips: set[str],
                      rates: dict[str, float]) -> int:
        """Enact one action. Returns the churn count (installs + deletes)."""
        pri = self.cfg["flows"]["priority"]["mitigation"]
        churn = 0

        if action == 0:
            return 0
        if action == 4:
            churn = self._clear_mitigation()
            return churn

        factor = {2: 0.5, 3: 0.1}.get(action)  # None for action 1 (drop)
        for ip in suspect_ips:
            edge = self._edge_of_ip(ip)
            uplink = self.port_map[edge][self.core]
            flow_id = f"m300-{ip.replace('.', '_')}"
            if flow_id in self._mitigation:
                continue  # already mitigated this source
            # proto=None: mitigate ALL of the suspect source's traffic to the
            # victim, not just UDP. The suspect is flagged by volume with no
            # protocol knowledge (label-free), so mitigation must be
            # protocol-agnostic; a hardcoded proto match silently fails to
            # touch a TCP flood like R1.
            if action == 1:
                self.odl.put_flow(edge, flow_id, pri, out_port=None,
                                  src=ip, dst=self.victim_ip, proto=None)
                self._mitigation[flow_id] = {"src": ip, "edge": edge,
                                             "kind": "drop", "meter_id": None}
                churn += 1
            else:
                meter_id = self._next_meter_id
                self._next_meter_id += 1
                rate_pps = rates.get(ip, 0.0)
                # Band from measured rate; convert pps to kbps at 1200 B/pkt.
                band_kbps = max(64, int(rate_pps * factor * 1200 * 8 / 1000))
                self.odl.put_meter(edge, meter_id, band_kbps)
                self.odl.put_flow(edge, flow_id, pri, out_port=uplink,
                                  src=ip, dst=self.victim_ip, proto=None,
                                  meter_id=meter_id)
                self._mitigation[flow_id] = {"src": ip, "edge": edge,
                                             "kind": f"meter{factor}",
                                             "meter_id": meter_id}
                churn += 1
        return churn

    def _clear_mitigation(self) -> int:
        churn = 0
        for flow_id, rec in list(self._mitigation.items()):
            self.odl.delete_flow(rec["edge"], flow_id)
            if rec["meter_id"] is not None:
                self.odl.delete_meter(rec["edge"], rec["meter_id"])
            churn += 1
        self._mitigation.clear()
        self._reconcile_datapath()
        return churn

    def _reconcile_datapath(self) -> None:
        """Force off any priority-300 rule the switch still holds after a clear.

        ODL's datastore can lag the datapath (seen in 6.5). Feature 18 and the
        reward's C term read the switch, so the switch is made authoritative.
        """
        pri = self.cfg["flows"]["priority"]["mitigation"]
        for edge in self.edges:
            out = subprocess.run(
                ["ovs-ofctl", "-O", self.of, "dump-flows", edge],
                capture_output=True, text=True).stdout
            for line in out.splitlines():
                if f"priority={pri}," not in line:
                    continue
                spec = "priority=" + line.partition(" actions=")[0].split(
                    "priority=", 1)[1].strip()
                subprocess.run(
                    ["ovs-ofctl", "-O", self.of, "--strict", "del-flows",
                     edge, spec], capture_output=True)

    def _mitigated_fraction(self, rates: dict[str, float]) -> float:
        """Feature 18: share of victim-bound packets currently under a p300 rule."""
        mitigated_ips = {rec["src"] for rec in self._mitigation.values()}
        total = sum(rates.values())
        if total <= 0:
            return 0.0
        under = sum(r for ip, r in rates.items() if ip in mitigated_ips)
        return under / total

    # -- gym API ----------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = random.Random(seed)
        elif not hasattr(self, "_rng"):
            self._rng = random.Random()
        self._episode += 1
        self._t = 0
        self._prev_snapshot = None
        self._prev_derived = None
        self.detector.reset()

        traffic.stop_all(self.net)
        traffic.restart_benign_servers(self.net, self.cfg, len(self.candidate_attackers))
        self._clear_mitigation()

        options = options or {}
        self._configure_episode(options)
        self._configure_event(options)
        self._configure_iipem_log(options)

        self._epoch = traffic.regime_epoch()
        traffic.start_regime(self.net, self._regime, self._attackers,
                             self.cfg, scale=self._scale, epoch=self._epoch)
        traffic.start_benign(self.net, self.cfg,
                             benign_hosts=self._benign_hosts(), rng=self._rng)
        self._probe_logs = traffic.start_probes(self.net, self.cfg)

        time.sleep(self.settle_s)
        self._prev_snapshot = self._snapshot()
        obs, _ = self._observe(self._prev_snapshot, self._prev_snapshot,
                               probe_window=(self._prev_snapshot["t_wall"] - self.W,
                                             self._prev_snapshot["t_wall"]),
                               churn=0, first=True)
        info = {"regime": self._regime, "episode": self._episode,
                "attackers": list(self._attackers)}
        info.update(self._event_info())
        return obs, info

    def _configure_event(self, options: dict) -> None:
        """Set up an IIPEM event from options["surge"] or options["swap"].

        The hosts involved are drawn here, once, from the episode's own
        generator, so every arm run on the same episode seed gets the
        same surge hosts or the same replacement attackers.
        """
        self._event = None
        self._event_step = None
        self._event_fired = False
        self._surge_hosts = []
        self._surge_rates = {}
        self._next_attackers = []
        self._former_attackers = []
        surge, swap = options.get("surge"), options.get("swap")
        if not surge and not swap:
            return
        if surge and swap:
            raise ValueError("an episode takes one event: surge or swap")
        if self._switch_step is not None:
            raise ValueError("an event cannot be combined with a regime switch")
        spec = surge or swap
        at = int(spec["at"])
        if not 1 <= at < self.steps_per_episode:
            raise ValueError(f"event step {at} outside 1..{self.steps_per_episode - 1}")
        benign = self._benign_hosts()
        if surge:
            n = int(surge["n_hosts"])
            rate = float(surge["rate_mbps"])
            if not 1 <= n <= len(benign):
                raise ValueError(f"surge n_hosts {n} outside 1..{len(benign)}")
            if rate <= 0:
                raise ValueError("surge rate_mbps must be positive")
            self._surge_hosts = self._rng.sample(benign, n)
            self._surge_rates = {h: rate for h in self._surge_hosts}
            self._event = "surge"
        else:
            if self._regime == "R0":
                raise ValueError("a swap needs an attack regime, not R0")
            n = len(self._attackers)
            if n > len(benign):
                raise ValueError("not enough benign hosts to swap in")
            self._next_attackers = self._rng.sample(benign, n)
            self._event = "swap"
        self._event_step = at

    def _fire_event(self) -> None:
        """Enact the event through the same stop/restart path as a regime switch."""
        if self._iipem_log:
            self._acct_pre = self._acct_read(zero=True)
        traffic.stop_all(self.net)
        traffic.restart_benign_servers(self.net, self.cfg, len(self.candidate_attackers))
        if self._event == "swap":
            self._former_attackers = list(self._attackers)
            self._attackers = list(self._next_attackers)
        self._epoch = traffic.regime_epoch()
        traffic.start_regime(self.net, self._regime, self._attackers,
                             self.cfg, scale=self._scale, epoch=self._epoch)
        traffic.start_benign(self.net, self.cfg,
                             benign_hosts=self._benign_hosts(), rng=self._rng,
                             rate_override=self._surge_rates or None)
        self._probe_logs = traffic.start_probes(self.net, self.cfg)
        self._event_fired = True

    def _event_info(self) -> dict:
        """Roles for logging. Surge hosts count as surging only once fired."""
        return {
            "event": self._event, "event_step": self._event_step,
            "event_fired": self._event_fired,
            "attackers": list(self._attackers),
            "former_attackers": list(self._former_attackers),
            "surge_hosts": list(self._surge_hosts) if self._event_fired else [],
        }

    def _victim(self):
        return self.net.get(self.cfg["topology"]["victim"]["name"])

    def _configure_iipem_log(self, options: dict) -> None:
        """Arm WP3 logging for this episode if options["iipem_log"] is set."""
        self._iipem_log = bool(options.get("iipem_log"))
        self._acct_pre = None
        self._last_apply_ms = None
        if not self._iipem_log:
            return
        if not self._acct_ready:
            sources = sorted({self._src_ip(h) for h in self.candidate_attackers + self.probe_hosts},
                             key=lambda ip: int(ip.split(".")[-1]))
            self._victim().cmd(iipem_log.acct_setup_cmd(sources))
            self._acct_ready = True
        self._victim().cmd(iipem_log.acct_zero_cmd())
        if self._ctl is None:
            self._ctl = iipem_log.ControllerSampler()
        self._ctl.sample()             # baseline for the first CPU share

    def _acct_read(self, zero: bool) -> dict:
        return iipem_log.parse_acct(self._victim().cmd(iipem_log.acct_read_cmd(zero=zero)))

    def _iipem_info(self) -> dict:
        """Per-step WP3 fields: flags for the window just observed, rules in
        place during it, the action's install time and controller load."""
        if not self._iipem_log:
            return {}
        cpu, rss = self._ctl.sample() if self._ctl else (None, None)
        return {
            "flagged": sorted(self.detector.suspects()),
            "mitigated": sorted({m["src"] for m in self._mitigation.values()}),
            "apply_ms": self._last_apply_ms,
            "ctl_cpu_pct": cpu, "ctl_rss_mb": rss,
            "ctl_pid": self._ctl.pid if self._ctl else None,
        }

    def _configure_episode(self, options: dict) -> None:
        """Pick attackers, scale, and the regime schedule from seed or options."""
        if "schedule" in options:
            # Evaluation: explicit [(regime, steps), ...]
            sched = options["schedule"]
            self._regime = sched[0][0]
            if len(sched) > 1:
                self._switch_step = sched[0][1]
                self._switch_to = sched[1][0]
            else:
                self._switch_step = None
                self._switch_to = None
            self._scale = options.get("scale", 1.0)
        else:
            # Training: start in {R0,R1,R2}, maybe switch once.
            self._regime = self._rng.choice(self.cfg["training"]["start_regimes"])
            sw = self.cfg["training"]["switch"]
            if self._rng.random() < sw["probability"]:
                lo, hi = sw["step_range"]
                self._switch_step = self._rng.randint(lo, hi)
                choices = [r for r in self.cfg["training"]["start_regimes"]
                           if r != self._regime]
                self._switch_to = self._rng.choice(choices)
            else:
                self._switch_step = None
                self._switch_to = None
            r = self.cfg["training"]["randomisation"]
            self._scale = self._rng.uniform(*r["attack_rate"])
            self._n_attackers = self._rng.randint(*r["attacker_count"])

        forbidden = set(self.cfg["training"].get("forbid_in_training", []))
        if "schedule" not in options:
            assert self._regime not in forbidden and self._switch_to not in forbidden, \
                f"training must not draw a held-out regime: {self._regime}/{self._switch_to}"

        n = getattr(self, "_n_attackers",
                    options.get("n_attackers", self.cfg["roles"]["attackers"]["count"]))
        self._attackers = self._rng.sample(self.candidate_attackers, n)

    def _benign_hosts(self) -> list[str]:
        return [h for h in self.candidate_attackers if h not in self._attackers]

    def step(self, action: int):
        # 1. Suspect set the detector formed from the window that just closed,
        #    plus that window's per-source rates for sizing meter bands.
        suspects = self.detector.suspects()
        rates = getattr(self, "_last_rates", {})

        # 2. Enact the action.
        t0 = perf_counter()
        churn = self._apply_action(int(action), suspects, rates)
        self._last_apply_ms = (perf_counter() - t0) * 1000.0

        # 3. Wait out the window, then read the new snapshot.
        elapsed = time.time() - self._prev_snapshot["t_wall"]
        if elapsed < self.W:
            time.sleep(self.W - elapsed)
        curr = self._snapshot()

        # 4. Features + reward over the window that just closed.
        t1 = curr["t_wall"]
        obs, info = self._observe(self._prev_snapshot, curr,
                                  probe_window=(self._prev_snapshot["t_wall"], t1),
                                  churn=churn, first=False)

        # 5. Scheduled regime switch.
        self._t += 1
        if self._switch_step is not None and self._t == self._switch_step:
            traffic.stop_all(self.net)
            self._regime = self._switch_to
            self._epoch = traffic.regime_epoch()
            traffic.start_regime(self.net, self._regime, self._attackers,
                                 self.cfg, scale=self._scale, epoch=self._epoch)
            traffic.start_benign(self.net, self.cfg,
                                 benign_hosts=self._benign_hosts(), rng=self._rng)
            self._probe_logs = traffic.start_probes(self.net, self.cfg)

        if (self._event_step is not None and not self._event_fired
                and self._t == self._event_step):
            self._fire_event()

        self._prev_snapshot = curr
        terminated = False
        truncated = self._t >= self.steps_per_episode
        if truncated and self._iipem_log:
            final = self._acct_read(zero=False)
            if self._acct_pre is None:          # no event: one phase
                info["delivered_pre"], info["delivered_post"] = final, None
            else:
                info["delivered_pre"], info["delivered_post"] = self._acct_pre, final
        return obs, info["reward"], terminated, truncated, info

    def _observe(self, prev, curr, probe_window, churn, first):
        rates = self._per_source_rates(None if first else prev, curr)
        self._last_rates = rates
        self.detector.update(rates, curr["t_wall"])

        records = read_lines(list(self._probe_logs.values())) if self._probe_logs else []
        probe = aggregate(records, probe_window[0], probe_window[1],
                          latency_scale_ms=self.cfg["reward"]["latency_scale_ms"])
        edge_flows = self._edge_flow_count()
        mfrac = self._mitigated_fraction(rates)

        vec, raw = self.fx.extract(
            None if first else prev, curr, probe,
            victim_port=self.victim_port, edge_flow_count=edge_flows,
            mitigated_fraction=mfrac, prev_derived=self._prev_derived)
        self._prev_derived = raw["_derived"]

        r, terms = compute_reward(raw, churn_count=churn, cfg=self.cfg)
        info = {
            "regime": self._regime, "episode": self._episode, "step": self._t,
            "reward": r, "reward_terms": terms, "rule_count": len(self._mitigation),
            "collateral": self._collateral_fraction(),
            "probe_n": probe["n"], "t_wall": curr["t_wall"],
        }
        info.update(self._event_info())
        info.update(self._iipem_info())
        if not first:
            self._log_row(vec, action_reward=r, terms=terms, info=info)
        return vec, info

    def _collateral_fraction(self) -> float:
        """Fraction of currently-mitigated sources that are benign or probe."""
        if not self._mitigation:
            return 0.0
        attacker_ips = {self._src_ip(a) for a in self._attackers}
        mitigated = [rec["src"] for rec in self._mitigation.values()]
        wrong = [ip for ip in mitigated if ip not in attacker_ips]
        return len(wrong) / len(mitigated)

    # -- logging ----------------------------------------------------------

    def _log_row(self, vec, action_reward, terms, info) -> None:
        row = {"run": self.run_id, "episode": self._episode, "step": self._t,
               "regime": self._regime, "reward": action_reward,
               "S": terms["S"], "L": terms["L"], "U": terms["U"], "C": terms["C"],
               "rule_count": info["rule_count"], "collateral": info["collateral"],
               "probe_n": info["probe_n"], "t_wall": info["t_wall"],
               "event": info.get("event"), "event_fired": info.get("event_fired"),
               "attackers": ",".join(info.get("attackers", [])),
               "surge_hosts": ",".join(info.get("surge_hosts", [])),
               "flagged": ",".join(info.get("flagged", [])),
               "mitigated": ",".join(info.get("mitigated", [])),
               "apply_ms": info.get("apply_ms"),
               "ctl_cpu_pct": info.get("ctl_cpu_pct"),
               "ctl_rss_mb": info.get("ctl_rss_mb")}
        for name, val in zip(
            ["pkt_rate", "byte_rate", "mean_pkt_size", "tcp_share", "udp_share",
             "icmp_share", "src_entropy", "active_sources", "top_source_share",
             "victim_link_util", "victim_tx_dropped_rate", "edge_table_occupancy",
             "probe_success", "probe_latency_median", "probe_goodput",
             "d_pkt_rate", "d_udp_share", "mitigated_fraction"], vec):
            row[f"f_{name}"] = float(val)
        self._rows.append(row)

    def flush(self) -> None:
        if not self._rows:
            return
        import pandas as pd
        pathlib.Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self._rows).to_parquet(self.log_path, index=False)

    def close(self) -> None:
        try:
            self.flush()
            traffic.stop_all(self.net)
            traffic.stop_servers(self.net)
        finally:
            self.net.stop()
