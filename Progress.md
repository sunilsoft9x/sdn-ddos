# INDICON 2026 — Testbed and Paper Progress

Label-Free Regime Inference for Deep Reinforcement Learning-Based DDoS Mitigation in Software-Defined Networks

PhD research, Chitkara University. Academic testbed work for an IEEE INDICON 2026 submission. Everything below runs inside an isolated Mininet emulation on a single lab VM; the attack regimes are synthetic traffic against an emulated victim, used to train and evaluate a defensive reinforcement-learning agent.

Repository: `github.com/sunilsoft9x/sdn-ddos`
Plan window: Thursday 17 to Sunday 20 September 2026 (IST). Submission target Sunday 20 September.

---

## Where things stand

Thursday's block is complete. Every gate through G4 has passed. The testbed is built, calibrated, and its separability is confirmed, which is the foundation the whole label-free claim rests on.

### Infrastructure (VM-1, all recorded for Section V)

- Host: Windows 11 Pro, i7-13800H, 32 GB, RTX 2000 Ada, running Hyper-V.
- VM-1: Ubuntu 22.04.5, kernel 5.15.0-191, static IP 192.168.50.11, 6 vCPU, 10 GB.
- Open vSwitch 2.17.12, Mininet 2.3.0.
- OpenDaylight Titanium SR2 (karaf 0.22.2) on OpenJDK 21, heap capped at 3 GB, RESTCONF confirmed, restarts on reboot via a cron `@reboot` entry.
- Python 3.10.12 venv (`--system-site-packages` for Mininet): torch 2.14.0+cpu, numpy 2.2.6, gymnasium 1.3.0, scikit-learn 1.7.2, pandas, pyarrow, scipy.
- A Hyper-V checkpoint named `base` exists as a clean fallback and as the source for the VM-2 clone.

### Scheme (locked in `configs/base.yaml`)

- Topology: core s1 with victim h16 on a 10 Mbps link; three edge switches s2/s3/s4, five hosts each (h1–h15), 20 Mbps uplinks. OpenFlow 1.3, static ARP.
- Step window W = 0.5 s. Chosen from measured control latency: median 13 ms, 95th percentile 18 ms over 100 flow installs, well under the 150 ms threshold.
- Counters read directly from Open vSwitch (`ovs-ofctl`), current to the millisecond; ODL used for every control action. This split is stated because ODL's statistics path is too slow for a sub-second window.
- Meters enforce on the kernel datapath (OVS 2.17 on kernel 5.15): a 5 Mbps band held a 10 Mbps UDP stream to 5.3 Mbps received. No userspace-datapath fallback needed.
- Attack regimes, calibrated so each degrades victim service and mitigation restores it:
  - R1 — TCP SYN flood to port 80, with payload so it is volumetric (`hping3 -S -p 80 -d 500`), nominal 640 pps/attacker.
  - R2 — UDP flood, 1000-byte payloads to port 5353 (`hping3 --udp -d 1000`), nominal 520 pps/attacker.
  - R4 — low-rate pulsed UDP (`pulser.py`), 1200 pps during a 0.2 s burst each 1.0 s. Held out of training; evaluation only.
- Reward uses only observable terms: `r = S − 0.5 L − 0.5 U − 0.1 C` (probe success, probe latency, victim over-utilisation, rule churn). No term uses attacker identity, which is what keeps the observation label-free.

### Calibration and separability (gates G3, G4)

- R0 baseline clean at 1.00 probe success.
- All three regimes degrade service under no mitigation and recover under `drop` — learnability confirmed for every regime across the full 2700-row sweep in `data/calib.parquet`.
- Regime classifier on the label-free features alone: 100% held-out accuracy, driven by protocol share (TCP vs UDP) and packet rate. The features carry the regime, so an external detector label is redundant — the empirical basis of the contribution.
- Suspect rule flags all three attackers in 100% of windows for R1, R2 and R4, with zero innocent hosts flagged.

### Design decisions to carry into the paper

- With real source addresses (no spoofing), a SYN flood is self-defeating: the victim's SYN-ACK draws a RST from the attacker's own stack and frees the half-open slot. So R1 must be volumetric, and true state-exhaustion needs spoofing — which is the R5 journal extension. Worth one honest sentence in Section IV; it motivates the journal work.
- Mitigation is protocol-agnostic: a flagged source has all its traffic to the victim dropped or metered, not just one protocol. This matches the label-free design (the suspect rule reads volume, not protocol) and is what makes R1's TCP flood mitigable.
- Priority-100 forwarding is destination-based (64 rules) rather than per-host-pair (240); identical behaviour on a single-path tree and keeps flow-table occupancy meaningful as a feature.
- Victim TCP stack left at defaults; all degradation is via link congestion.

### Bugs caught during the shakedown (worth noting — none were visible without testing)

1. Probe clients accumulated across episodes because `stop_all` never killed them; corrupted every multi-setting run until fixed.
2. Single-threaded victim HTTP server backed up under normal probe load, breaking the R0 baseline; replaced with a threaded server.
3. Mitigation flow hardcoded to match UDP, so R1's TCP flood was structurally unmitigable; made protocol-agnostic.
4. `scripts/queue.py` shadowed the standard-library `queue` module; renamed to `eval_queue.py`.

---

## Today (Friday) — the heavy day

Goal: simulator fitted and validated, agents built, fifteen training runs done, VM-2 cloned, and the evaluation queue launched overnight. Training should start by about 12:30 for the overnight evaluation to finish by morning.

### 7.1 Simulator and fidelity check — in WSL2 (target 07:00–10:00, gate G5)

Setup in a WSL2 Ubuntu terminal on the host:

1. Clone the repo, create a venv, install: torch (CPU), numpy, pandas, pyarrow, scipy, scikit-learn, matplotlib, pyyaml, gymnasium.
2. `gh auth login` (WSL is a separate machine and needs its own login).
3. Copy the calibration data over: `scp vm1:~/sdn-ddos/data/calib.parquet ~/sdn-ddos/data/`

Then build:
- `sim/model.py` — per-step, per-source offered traffic from regime and scale; mitigation applied with the delay from `timing.json`; passed traffic summed onto the 10 Mbps victim link.
- `sim/fit.py` — fit the service model on the non-hold-out calibration rows: probe success as a logistic function of passed load and utilisation; latency by a smooth fit; drop rate piecewise-linear; counter noise from the R0 windows.
- `sim/sim_env.py` — same interface and scheduler as `testbed_env.py`, plus the per-episode domain randomization.

Fidelity check on the hold-out (intensity 0.7) rows: absolute error in mean probe success and relative error in each feature mean, plus a cross-domain classifier check. Save to `analysis/fidelity.json` — it becomes a small table in Section V.

### 7.2 Agents (10:00–12:00, gate G6)

- `agents/common.py` — Double DQN, Huber loss, Adam 5e-4, γ 0.95, replay 50k, batch 64, target update every 1000 steps, ε 1.0→0.05 over 30k, gradient clip 10.
- `agents/ddqn.py` (A1, 18 features, MLP 150-150).
- `agents/context_ddqn.py` (A2, ours: 18 features + 8-d GRU context over the last 10 reward-inclusive transitions).
- `agents/oracle_ddqn.py` (A3, DeepAir-style, 18 features + one-hot regime label).
- `agents/static_rule.py` (threshold baseline, no training).
- Unit tests: parameter counts within 2% across A1/A2/A3; the training scheduler can never draw R4.

### 7.3 Pilot run (12:00–12:30, gate G7)
One short run to confirm the loop learns before committing the full set.

### 7.4 Full training (12:30–15:00, gate G8)
Fifteen runs (3 agents × 5 seeds), 100k steps each, on the host CPU with twelve parallel processes. Finish any remaining wrapper polish on VM-1 in parallel.

### 7.5 Clone VM-2 (15:00–15:45)
Clone from the `base` checkpoint for parallel evaluation. It inherits the repo, git identity and gh credentials; just `git pull`.

### 7.6 Transfer check (15:45–17:30, gate G9)
Deploy the frozen policies unchanged on the testbed and confirm they behave.

### 7.7 Evaluation queue (17:30–19:00, then overnight)
Launch the 470-episode queue across VM-1 and VM-2. Before launching, smoke-test the two wrapper paths not yet exercised live: the mid-episode regime switch (R1→R2) and the env's own per-episode Parquet logging.

### 7.8 Writing (19:00–21:00)
Draft Sections II (related work) and V (testbed and methods) while evaluation runs.

---

## Saturday

- 07:00–07:30 (G10): check the overnight evaluation, rerun any failed episodes.
- 07:30–10:30 (G11): analysis, statistics, figures, tables from the evaluation logs.
- 10:30–21:00: write Sections IV, VI, VII, I, and the abstract; draw Fig. 1.
- 21:00: full draft to Dr. Panda and Dr. Badotra (23:30 Singapore). Ask for comments by 09:00 IST Sunday.

## Sunday

- 07:00–11:00: revise from supervisor comments.
- 11:00–14:00 (G12): hardening — originality check, verified references, IEEE formatting, PDF eXpress, AI-use disclosure.
- 14:00–16:00: buffer.
- 16:00: submit.

---

## Resuming on VM-1

```bash
cd ~/sdn-ddos
git pull
sudo mn -c                     # clear any leftover Mininet before a live run
ls -lh data/calib.parquet      # confirm the calibration data is present
```

The Mininet/ODL/OVS work stays on VM-1. The simulator, agents, training and analysis run in WSL2. Only `calib.parquet` needs to cross between them (VM-1 → WSL, via scp). The final evaluation runs back on VM-1 and VM-2, and every number reported in the paper comes from there, not the simulator.

## Open items and risk

- Schedule is tight. The testbed took longer than Thursday allowed, so Friday has no slack. Confirm by Friday midday whether the fifteen training runs are on track; if not, the fallbacks in the plan's gate list come out in order (drop Fig. 5, cut E0 episodes, drop the static rule from E3, cut the Sunday buffer).
- If more than a day slips, raise a late-submission request with INDICON or target the next IEEE venue — decide early, not Saturday night.
- Two wrapper paths (regime switch, episode logging) get their first live run at evaluation; smoke-test them first.
- Report the 100% separability plainly. It reads as almost too clean, but it is the point: the raw features encode the regime, so the label-free agent needs no detector. The real test is R4, held out entirely, which the oracle has no label for.
