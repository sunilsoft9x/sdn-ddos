"""Calibrated simulator environment (plan 7.1, 4.10).

Same observation, action set, reward and scheduler as env/testbed_env.py, so a
policy trained here deploys on the testbed unchanged. Speed is the point: a step
is arithmetic, not a real measurement window, so 100k training steps take
minutes instead of the 60+ machine-hours the testbed would need.

The features and reward come from env/features.py, running on a raw snapshot the
simulator builds exactly as ovs_stats would. Only the physics (offered traffic,
the link, the fitted service response) lives here.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

try:
    import gymnasium as gym
    from gymnasium import spaces
    _BASE = gym.Env
except Exception:
    gym = None
    spaces = None
    _BASE = object

from env.features import FeatureExtractor, reward as compute_reward
from env.suspects import SuspectDetector
from sim.model import (LINK_CAPACITY_BPS, LinkModel, ServiceModel, TrafficModel,
                       apply_mitigation, build_snapshot)

REPO = pathlib.Path(__file__).resolve().parents[1]


def _on_fraction(t: float, w: float, on: float, period: float) -> float:
    """Fraction of the window [t, t+w) that falls inside a burst-on interval."""
    n = 24
    dt = w / n
    hits = 0
    for i in range(n):
        phase = ((t + (i + 0.5) * dt) % period)
        if phase < on:
            hits += 1
    return hits / n


class SimEnv(_BASE):
    metadata = {"render_modes": []}

    def __init__(self, cfg: dict | None = None,
                 service_model: ServiceModel | None = None,
                 run_id: str = "sim"):
        if cfg is None:
            cfg = yaml.safe_load(open(REPO / "configs" / "base.yaml"))
        self.cfg = cfg
        self.run_id = run_id
        self.W = float(cfg["runtime"]["window_w_s"])
        self.steps_per_episode = cfg["training"]["steps_per_episode"]
        self.victim_ip = cfg["topology"]["victim"]["ip"]
        self.core = cfg["topology"]["core"]
        self.edges = list(cfg["topology"]["edges"])
        self.hosts_per_edge = cfg["topology"]["hosts_per_edge"]

        self.traffic = TrafficModel(cfg)
        self.link = LinkModel()
        self.service = service_model or ServiceModel.load(
            REPO / "sim" / "service_model.json")
        self.fx = FeatureExtractor(cfg)
        self.detector = SuspectDetector(cfg)

        all_hosts = [f"h{i}" for i in range(1, cfg["topology"]["host_count"] + 1)]
        self.probe_hosts = list(cfg["roles"]["probes"])
        self.candidate_attackers = [h for h in all_hosts if h not in self.probe_hosts]

        self._rng = np.random.default_rng()
        self._episode = -1
        if spaces is not None:
            self.action_space = spaces.Discrete(5)
            self.observation_space = spaces.Box(-1.0, 1.0, (18,), np.float32)

    # -- helpers ----------------------------------------------------------

    def _ip(self, host: str) -> str:
        return host if host.startswith("10.") else f"10.0.0.{int(host.lstrip('h'))}"

    def _host_edge(self, ip: str) -> str:
        idx = int(ip.split(".")[-1])
        return self.edges[(idx - 1) // self.hosts_per_edge]

    # -- gym API ----------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._episode += 1
        self._t = 0
        self._sim_t = 0.0
        self._prev_snapshot = None
        self._prev_derived = None
        self._cumulative: dict[str, dict] = {}
        self._victim_port = {"tx_bytes": 0, "tx_drop": 0, "rx_bytes": 0}
        self._mitigation: dict[str, dict] = {}
        self.detector.reset()

        options = options or {}
        self._configure_episode(options)

        # counter-noise coefficient for this episode (domain randomisation)
        rnd = self.cfg["training"]["randomisation"]
        self._noise_k = self._rng.uniform(*rnd["counter_noise"])

        obs = self._advance(action=0, first=True)
        info = {"regime": self._regime, "episode": self._episode,
                "attackers": list(self._attackers)}
        return obs, info

    def _configure_episode(self, options: dict) -> None:
        rnd = self.cfg["training"]["randomisation"]
        if "schedule" in options:
            sched = options["schedule"]
            self._regime = sched[0][0]
            self._switch_step = sched[0][1] if len(sched) > 1 else None
            self._switch_to = sched[1][0] if len(sched) > 1 else None
            self._scale = options.get("scale", 1.0)
            n = options.get("n_attackers", self.cfg["roles"]["attackers"]["count"])
        else:
            self._regime = self._rng.choice(self.cfg["training"]["start_regimes"])
            sw = self.cfg["training"]["switch"]
            if self._rng.random() < sw["probability"]:
                self._switch_step = int(self._rng.integers(*sw["step_range"]))
                choices = [r for r in self.cfg["training"]["start_regimes"]
                           if r != self._regime]
                self._switch_to = str(self._rng.choice(choices))
            else:
                self._switch_step = None
                self._switch_to = None
            self._scale = float(self._rng.uniform(*rnd["attack_rate"]))
            n = int(self._rng.integers(rnd["attacker_count"][0],
                                       rnd["attacker_count"][1] + 1))

        forbidden = set(self.cfg["training"].get("forbid_in_training", []))
        if "schedule" not in options:
            assert self._regime not in forbidden and self._switch_to not in forbidden, \
                f"training drew a held-out regime: {self._regime}/{self._switch_to}"

        idx = self._rng.choice(len(self.candidate_attackers), size=n, replace=False)
        self._attackers = [self.candidate_attackers[i] for i in idx]
        self._benign = [h for h in self.candidate_attackers if h not in self._attackers]

    def _apply_action(self, action: int, suspects: set[str],
                      offered: dict[str, dict]) -> int:
        if action == 0:
            return 0
        if action == 4:
            churn = len(self._mitigation)
            self._mitigation.clear()
            return churn
        factor = {2: 0.5, 3: 0.1}.get(action)
        churn = 0
        for ip in suspects:
            if ip in self._mitigation:
                continue
            if action == 1:
                self._mitigation[ip] = {"kind": "drop", "factor": 0.0}
            else:
                self._mitigation[ip] = {"kind": "meter", "factor": factor}
            churn += 1
        return churn

    def _advance(self, action: int, first: bool):
        # 1. suspects and action (skipped on the reset warm observation)
        churn = 0
        if not first:
            suspects = self.detector.suspects()
            churn = self._apply_action(action, suspects, self._last_offered
                                       if hasattr(self, "_last_offered") else {})

        # 2. offered traffic this window (R4 burst overlap gives the pulse)
        spec_r4 = self._regime == "R4" or self._switch_to == "R4"
        on_frac = 1.0
        if self._regime == "R4":
            on_frac = _on_fraction(self._sim_t, self.W, 0.2, 1.0)
        offered = self.traffic.offered(self._regime, self._scale,
                                       self._attackers, self._benign, self._rng,
                                       util_cal=self.service.util_cal)
        if self._regime == "R4":
            for ip in (self._ip(a) for a in self._attackers):
                if ip in offered:
                    offered[ip] = {**offered[ip], "pps": offered[ip]["pps"] * on_frac}
        self._last_offered = offered

        passed = apply_mitigation(offered, self._mitigation)

        # 3. multiplicative counter noise
        if not first:
            for p in passed.values():
                p["pps"] = max(0.0, p["pps"] * (1.0 + self._noise_k *
                               self._rng.normal(0, 0.1)))

        # 4. link + service
        util_true, drop_frac = self.link.utilisation_and_drops(passed)
        offered_bps = sum(p["pps"] * p["pkt_bytes"] for p in passed.values())
        total_pps = sum(p["pps"] for p in passed.values())
        tcp_pps = sum(p["pps"] for p in passed.values() if p["proto"] == "tcp")
        tcp_share = (tcp_pps / total_pps) if total_pps > 0 else 0.0

        # 5. accumulate cumulative counters (bytes delivered clipped at capacity)
        self._sim_t += self.W
        for ip, p in passed.items():
            c = self._cumulative.setdefault(
                ip, {"tcp": 0, "udp": 0, "icmp": 0, "packets": 0, "bytes": 0,
                     "_bytes_per_pkt": p["pkt_bytes"]})
            n_pk = p["pps"] * self.W
            c[p["proto"]] += n_pk
            c["packets"] += n_pk
            c["bytes"] += n_pk * p["pkt_bytes"]
        delivered_bps = min(offered_bps, LINK_CAPACITY_BPS)
        self._victim_port["tx_bytes"] += delivered_bps * self.W
        self._victim_port["tx_drop"] += drop_frac * total_pps * self.W

        # 6. probe outcome from the fitted service model. Utilisation is capped
        # at 1.0 to match the service model's fitted domain (the testbed feature
        # is likewise capped); passing an over-saturation value would extrapolate.
        succ = self.service.success(min(util_true, 1.0), total_pps, tcp_share)
        succ = float(np.clip(succ + self._rng.normal(0, 0.03), 0.0, 1.0))
        lat_scaled = float(np.clip(self.service.latency[0]
                                   + self.service.latency[1] * util_true ** 2, 0.0, 1.0))
        probe = {"n": 5, "success": succ,
                 "latency_s": lat_scaled * 0.5, "latency_scaled": lat_scaled,
                 "goodput_bps": succ * 1.0e5}

        # 7. snapshot -> features -> reward (real env code)
        curr = build_snapshot(self._cumulative, dict(self._victim_port),
                              self._sim_t, self.victim_ip, self.core, self.edges,
                              self._host_edge)
        # per-source rates for the detector
        rates = self._window_rates(passed)
        self.detector.update(rates, self._sim_t)

        edge_flows = 138 + len(self._mitigation)
        mfrac = self._mitigated_fraction(offered)
        vec, raw = self.fx.extract(
            None if first else self._prev_snapshot, curr, probe,
            victim_port=1, edge_flow_count=edge_flows,
            mitigated_fraction=mfrac, prev_derived=self._prev_derived)
        self._prev_derived = raw["_derived"]
        self._prev_snapshot = curr

        r, terms = compute_reward(raw, churn_count=churn, cfg=self.cfg)
        self._last_reward = r
        self._last_info = {
            "regime": self._regime, "episode": self._episode, "step": self._t,
            "reward": r, "reward_terms": terms, "rule_count": len(self._mitigation),
            "collateral": self._collateral(), "probe_n": 5,
        }
        return vec

    def _window_rates(self, passed: dict[str, dict]) -> dict[str, float]:
        return {ip: p["pps"] for ip, p in passed.items()}

    def _mitigated_fraction(self, offered: dict[str, dict]) -> float:
        total = sum(p["pps"] for p in offered.values())
        if total <= 0:
            return 0.0
        under = sum(p["pps"] for ip, p in offered.items() if ip in self._mitigation)
        return under / total

    def _collateral(self) -> float:
        if not self._mitigation:
            return 0.0
        attacker_ips = {self._ip(a) for a in self._attackers}
        wrong = [ip for ip in self._mitigation if ip not in attacker_ips]
        return len(wrong) / len(self._mitigation)

    def step(self, action: int):
        obs = self._advance(int(action), first=False)
        self._t += 1
        if self._switch_step is not None and self._t == self._switch_step:
            self._regime = self._switch_to
            self._mitigation_valid_switch()
        terminated = False
        truncated = self._t >= self.steps_per_episode
        return obs, self._last_reward, terminated, truncated, self._last_info

    def _mitigation_valid_switch(self) -> None:
        # Regime changed; the attacker set is unchanged, so mitigation persists.
        pass


if __name__ == "__main__":
    # Smoke run once a service_model.json exists (falls back to defaults).
    import pathlib as _p
    sm = None
    p = _p.Path(__file__).resolve().parents[1] / "sim" / "service_model.json"
    if p.exists():
        sm = ServiceModel.load(p)
    env = SimEnv(service_model=sm)
    obs, info = env.reset(seed=0, options={"schedule": [["R1", 999]], "scale": 1.0})
    print("reset ok, obs shape", obs.shape, "regime", info["regime"])
    tot = 0.0
    for i in range(10):
        obs, r, term, trunc, info = env.step(1 if i > 2 else 0)
        tot += r
    print(f"10 steps ok, mean reward {tot/10:.3f}, "
          f"last S {info['reward_terms']['S']:.2f} rules {info['rule_count']}")
