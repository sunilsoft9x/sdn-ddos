# IIPEM analysis summary

Inputs: `data/iipem/iipem_rerun.parquet`, `data/iipem/iipem_rerun_delivered.parquet`
Episodes: 400; seeds per arm: C1_surge A2: 10, C1_surge A1: 10, C1_surge A3: 10, C1_surge static: 10, C2_swap A2: 10, C2_swap A1: 10, C2_swap A3: 10, C2_swap static: 10

## Checks

- Steps contiguous, event fired, roles and event step identical across arms for every (cond, seed, episode): passed.
- Probe hosts never mitigated: passed.
- Note: collateral recomputed from mitigated hosts and roles matches the logged collateral at every step

## Reading the results

- The event restarts the probe clients, so S = 0 at the event step in 86% to 100% of episodes per arm and condition. Consequences: `worst_isasf` (ISASF minS over the whole episode) is close to 0 for every arm, as it is for every ISASF E1/E3 episode; `worst_post` skips the event step and the next (2 steps) for that reason. The recovery clock starts 1 step after the event, where ISASF's starts (its switch step is one row after the dip), so the dip alone gives a latency of 0.
- A3 in C1 is a forced no-op: its label is "none" in R0 and `agents/oracle_ddqn.py` (lines 48-49) returns a no-op without calling the network. Its zero blocking in C1 holds by construction, and its C1 inference and rule-installation times measure a branch.
- Collateral is a share of the active rules. Report the count of blocked benign hosts (exploratory table) beside it: the share and the count can move by different amounts.
- `n (non-zero)`: seeds whose difference is not zero. The exact test uses only those.

Worst window and recovery follow ISASF `scripts/analyze.py`: minS from step 2; recovery when the 5-step rolling mean of S returns to 0.95 x the mean of the 20 steps before the start and holds 5 steps; the start is event + 1, ISASF's switch step (censored if never). `worst_post` additionally skips the 2 steps after the event, where every arm loses its probes to the restart.

## Planned parameters (plan Section 7; IQM over seeds, bootstrap 95% interval)

### C1: benign surge

| Parameter | A2 | A1 | A3 | static |
|---|---|---|---|---|
| Mean service after the event | 0.980 [0.980, 0.982] | 0.981 [0.980, 0.983] | 0.980 [0.980, 0.980] | 0.980 [0.980, 0.980] |
| Worst-window service after the event | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| Worst-window service, ISASF minS (whole episode) | 0.000 [0.000, 0.100] | 0.033 [0.000, 0.167] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| Recovery latency, recovered episodes (steps) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| Collateral after the event | 0.629 [0.521, 0.815] | 0.627 [0.445, 0.866] | 0.000 [0.000, 0.000] | 0.603 [0.586, 0.618] |
| Collateral on other benign hosts after the event | 0.290 [0.168, 0.521] | 0.317 [0.145, 0.577] | 0.000 [0.000, 0.000] | 0.313 [0.296, 0.329] |
| Delivery from other benign hosts (post/pre) | 0.882 [0.805, 0.933] | 0.855 [0.732, 0.901] | 0.915 [0.871, 0.950] | 0.900 [0.863, 0.936] |
| Share of post steps with a rule on a surge host | 0.563 [0.478, 0.777] | 0.560 [0.395, 0.820] | 0.000 [0.000, 0.000] | 0.504 [0.481, 0.528] |
| Surge-host delivery (share of 4.0 Mbps offered) | 0.346 [0.213, 0.405] | 0.334 [0.176, 0.479] | 0.848 [0.838, 0.861] | 0.351 [0.340, 0.372] |
| Inference time, median (ms) | 0.688 [0.670, 0.707] | 0.323 [0.317, 0.332] | 0.004 [0.004, 0.004] | 0.023 [0.022, 0.023] |
| Rule installation time, median (ms) | 3.843 [0.883, 6.556] | 5.413 [1.862, 7.447] | 0.002 [0.002, 0.002] | 8.601 [8.282, 8.812] |
| Controller CPU, mean (%) | 3.272 [3.092, 3.408] | 3.262 [2.927, 3.513] | 2.665 [2.558, 2.837] | 3.585 [3.458, 3.673] |

### C2: attacker swap

| Parameter | A2 | A1 | A3 | static |
|---|---|---|---|---|
| Mean service after the event | 0.958 [0.950, 0.969] | 0.962 [0.954, 0.973] | 0.957 [0.928, 0.970] | 0.929 [0.917, 0.941] |
| Worst-window service after the event | 0.367 [0.167, 0.667] | 0.400 [0.200, 0.767] | 0.233 [0.067, 0.433] | 0.100 [0.000, 0.200] |
| Worst-window service, ISASF minS (whole episode) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.100] | 0.000 [0.000, 0.033] | 0.000 [0.000, 0.000] |
| Recovery latency, recovered episodes (steps) | 0.400 [0.000, 1.400] | 1.500 [0.300, 2.400] | 2.533 [1.200, 4.467] | 2.350 [0.800, 4.033] |
| Collateral after the event | 0.253 [0.193, 0.470] | 0.278 [0.164, 0.541] | 0.103 [0.088, 0.116] | 0.307 [0.293, 0.324] |
| Collateral on other benign hosts after the event | 0.136 [0.086, 0.268] | 0.146 [0.073, 0.324] | 0.043 [0.032, 0.052] | 0.193 [0.179, 0.210] |
| Delivery from other benign hosts (post/pre) | 0.862 [0.798, 0.901] | 0.838 [0.680, 0.916] | 0.891 [0.856, 0.931] | 0.884 [0.850, 0.928] |
| Post steps with a rule on a former attacker | 16.000 [12.467, 32.900] | 19.700 [10.700, 37.767] | 6.833 [5.700, 7.333] | 17.167 [15.933, 18.100] |
| Steps until no rule on a former attacker | 5.867 [4.333, 27.933] | 9.933 [2.800, 29.333] | 1.233 [0.867, 2.700] | 4.033 [3.400, 4.533] |
| Former-attacker delivery (vs benign level) | 0.737 [0.374, 0.820] | 0.748 [0.434, 0.891] | 0.867 [0.821, 0.914] | 0.831 [0.800, 0.869] |
| Attack delivery (post/pre) | 0.760 [0.496, 0.855] | 0.727 [0.439, 0.843] | 0.844 [0.821, 0.878] | 0.741 [0.703, 0.780] |
| Inference time, median (ms) | 0.696 [0.680, 0.713] | 0.329 [0.324, 0.336] | 0.351 [0.344, 0.361] | 0.023 [0.023, 0.024] |
| Rule installation time, median (ms) | 4.997 [0.604, 11.536] | 7.402 [2.457, 12.432] | 15.755 [14.634, 17.320] | 6.023 [5.400, 6.771] |
| Controller CPU, mean (%) | 3.605 [3.249, 3.952] | 3.686 [3.223, 4.038] | 3.838 [3.726, 4.082] | 3.528 [3.387, 3.743] |

## Planned paired tests (exact Wilcoxon, Holm within each family of 3)

| Condition | Parameter | Comparison | n (non-zero) | median diff | p | p (Holm) | r |
|---|---|---|---|---|---|---|---|
| C1_surge | service_post | A2 vs A1 | 10 (5) | 0.000 | 0.7500 | 1.0000 | -0.33 |
| C1_surge | service_post | A2 vs A3 | 10 (2) | 0.000 | 0.5000 | 1.0000 | 1.00 |
| C1_surge | service_post | A2 vs static | 10 (2) | 0.000 | 0.5000 | 1.0000 | 1.00 |
| C1_surge | worst_post | A2 vs A1 | 10 (0) | 0.000 | n/a | n/a | n/a |
| C1_surge | worst_post | A2 vs A3 | 10 (0) | 0.000 | n/a | n/a | n/a |
| C1_surge | worst_post | A2 vs static | 10 (0) | 0.000 | n/a | n/a | n/a |
| C1_surge | worst_isasf | A2 vs A1 | 10 (5) | 0.000 | 0.7500 | 1.0000 | -0.33 |
| C1_surge | worst_isasf | A2 vs A3 | 10 (2) | 0.000 | 0.5000 | 1.0000 | 1.00 |
| C1_surge | worst_isasf | A2 vs static | 10 (2) | 0.000 | 0.5000 | 1.0000 | 1.00 |
| C1_surge | adapt_latency | A2 vs A1 | 10 (0) | 0.000 | n/a | n/a | n/a |
| C1_surge | adapt_latency | A2 vs A3 | 10 (0) | 0.000 | n/a | n/a | n/a |
| C1_surge | adapt_latency | A2 vs static | 10 (0) | 0.000 | n/a | n/a | n/a |
| C1_surge | collateral_post | A2 vs A1 | 10 (10) | 0.022 | 1.0000 | 1.0000 | -0.02 |
| C1_surge | collateral_post | A2 vs A3 | 10 (10) | 0.598 | 0.0020 | 0.0059 | 1.00 |
| C1_surge | collateral_post | A2 vs static | 10 (10) | -0.010 | 0.5566 | 1.0000 | 0.24 |
| C1_surge | collateral_other_post | A2 vs A1 | 10 (10) | -0.053 | 0.4316 | 0.8633 | -0.31 |
| C1_surge | collateral_other_post | A2 vs A3 | 10 (10) | 0.261 | 0.0020 | 0.0059 | 1.00 |
| C1_surge | collateral_other_post | A2 vs static | 10 (10) | -0.047 | 1.0000 | 1.0000 | -0.02 |
| C1_surge | benign_delivery_other | A2 vs A1 | 10 (10) | 0.006 | 0.3750 | 0.7500 | 0.35 |
| C1_surge | benign_delivery_other | A2 vs A3 | 10 (10) | -0.009 | 0.0645 | 0.1934 | -0.67 |
| C1_surge | benign_delivery_other | A2 vs static | 10 (10) | 0.002 | 0.9219 | 0.9219 | -0.05 |
| C1_surge | surge_block_share | A2 vs A1 | 10 (10) | 0.022 | 0.7480 | 0.7480 | 0.13 |
| C1_surge | surge_block_share | A2 vs A3 | 10 (10) | 0.562 | 0.0020 | 0.0059 | 1.00 |
| C1_surge | surge_block_share | A2 vs static | 10 (8) | 0.044 | 0.2500 | 0.5000 | 0.50 |
| C1_surge | surge_delivery | A2 vs A1 | 10 (10) | 0.025 | 0.7695 | 1.0000 | 0.13 |
| C1_surge | surge_delivery | A2 vs A3 | 10 (10) | -0.503 | 0.0020 | 0.0059 | -1.00 |
| C1_surge | surge_delivery | A2 vs static | 10 (10) | -0.015 | 0.5566 | 1.0000 | -0.24 |
| C1_surge | infer_ms_median | A2 vs A1 | 10 (10) | 0.359 | 0.0020 | 0.0059 | 1.00 |
| C1_surge | infer_ms_median | A2 vs A3 | 10 (10) | 0.683 | 0.0020 | 0.0059 | 1.00 |
| C1_surge | infer_ms_median | A2 vs static | 10 (10) | 0.664 | 0.0020 | 0.0059 | 1.00 |
| C1_surge | apply_ms_median | A2 vs A1 | 10 (10) | -0.549 | 0.0840 | 0.0840 | -0.64 |
| C1_surge | apply_ms_median | A2 vs A3 | 10 (10) | 4.104 | 0.0020 | 0.0059 | 1.00 |
| C1_surge | apply_ms_median | A2 vs static | 10 (10) | -4.677 | 0.0020 | 0.0059 | -1.00 |
| C1_surge | ctl_cpu_mean | A2 vs A1 | 10 (10) | -0.122 | 0.7695 | 0.7695 | -0.13 |
| C1_surge | ctl_cpu_mean | A2 vs A3 | 10 (10) | 0.594 | 0.0020 | 0.0059 | 1.00 |
| C1_surge | ctl_cpu_mean | A2 vs static | 10 (10) | -0.321 | 0.0039 | 0.0078 | -0.96 |
| C2_swap | service_post | A2 vs A1 | 10 (8) | -0.002 | 0.6562 | 1.0000 | -0.22 |
| C2_swap | service_post | A2 vs A3 | 10 (9) | 0.002 | 0.5469 | 1.0000 | 0.24 |
| C2_swap | service_post | A2 vs static | 10 (9) | 0.028 | 0.0078 | 0.0234 | 0.96 |
| C2_swap | worst_post | A2 vs A1 | 10 (4) | 0.000 | 0.8750 | 0.8750 | -0.20 |
| C2_swap | worst_post | A2 vs A3 | 10 (8) | 0.100 | 0.2812 | 0.5625 | 0.47 |
| C2_swap | worst_post | A2 vs static | 10 (9) | 0.200 | 0.0664 | 0.1992 | 0.73 |
| C2_swap | worst_isasf | A2 vs A1 | 10 (2) | 0.000 | 0.5000 | 1.0000 | -1.00 |
| C2_swap | worst_isasf | A2 vs A3 | 10 (1) | 0.000 | 1.0000 | 1.0000 | -1.00 |
| C2_swap | worst_isasf | A2 vs static | 10 (0) | 0.000 | n/a | n/a | n/a |
| C2_swap | adapt_latency | A2 vs A1 | 10 (7) | -0.600 | 0.1875 | 0.1875 | -0.61 |
| C2_swap | adapt_latency | A2 vs A3 | 10 (9) | -1.800 | 0.0664 | 0.1875 | -0.71 |
| C2_swap | adapt_latency | A2 vs static | 10 (7) | -1.200 | 0.0625 | 0.1875 | -0.86 |
| C2_swap | collateral_post | A2 vs A1 | 10 (10) | 0.004 | 0.6953 | 1.0000 | 0.16 |
| C2_swap | collateral_post | A2 vs A3 | 10 (10) | 0.131 | 0.0020 | 0.0059 | 1.00 |
| C2_swap | collateral_post | A2 vs static | 10 (10) | -0.085 | 0.5566 | 1.0000 | -0.24 |
| C2_swap | collateral_other_post | A2 vs A1 | 10 (10) | -0.011 | 0.6953 | 1.0000 | -0.16 |
| C2_swap | collateral_other_post | A2 vs A3 | 10 (10) | 0.075 | 0.0020 | 0.0059 | 1.00 |
| C2_swap | collateral_other_post | A2 vs static | 10 (10) | -0.085 | 0.5566 | 1.0000 | -0.24 |
| C2_swap | benign_delivery_other | A2 vs A1 | 10 (10) | -0.002 | 1.0000 | 1.0000 | 0.02 |
| C2_swap | benign_delivery_other | A2 vs A3 | 10 (10) | -0.018 | 0.0273 | 0.0820 | -0.78 |
| C2_swap | benign_delivery_other | A2 vs static | 10 (10) | -0.019 | 0.1309 | 0.2617 | -0.56 |
| C2_swap | stale_steps | A2 vs A1 | 10 (10) | -0.500 | 1.0000 | 1.0000 | -0.02 |
| C2_swap | stale_steps | A2 vs A3 | 10 (10) | 9.300 | 0.0039 | 0.0117 | 0.96 |
| C2_swap | stale_steps | A2 vs static | 10 (10) | -2.300 | 0.8457 | 1.0000 | -0.09 |
| C2_swap | stale_clear_latency | A2 vs A1 | 10 (10) | 1.300 | 0.3105 | 0.3105 | 0.38 |
| C2_swap | stale_clear_latency | A2 vs A3 | 10 (10) | 4.700 | 0.0039 | 0.0117 | 0.96 |
| C2_swap | stale_clear_latency | A2 vs static | 10 (10) | 1.900 | 0.0215 | 0.0430 | 0.80 |
| C2_swap | former_attacker_delivery | A2 vs A1 | 10 (10) | -0.017 | 0.4922 | 0.4922 | -0.27 |
| C2_swap | former_attacker_delivery | A2 vs A3 | 10 (10) | -0.103 | 0.0039 | 0.0117 | -0.96 |
| C2_swap | former_attacker_delivery | A2 vs static | 10 (10) | -0.070 | 0.0273 | 0.0547 | -0.78 |
| C2_swap | attack_delivery_ratio | A2 vs A1 | 10 (10) | -0.007 | 1.0000 | 1.0000 | -0.02 |
| C2_swap | attack_delivery_ratio | A2 vs A3 | 10 (10) | -0.099 | 0.0488 | 0.1465 | -0.71 |
| C2_swap | attack_delivery_ratio | A2 vs static | 10 (10) | -0.014 | 0.5566 | 1.0000 | -0.24 |
| C2_swap | infer_ms_median | A2 vs A1 | 10 (10) | 0.370 | 0.0020 | 0.0059 | 1.00 |
| C2_swap | infer_ms_median | A2 vs A3 | 10 (10) | 0.343 | 0.0020 | 0.0059 | 1.00 |
| C2_swap | infer_ms_median | A2 vs static | 10 (10) | 0.677 | 0.0020 | 0.0059 | 1.00 |
| C2_swap | apply_ms_median | A2 vs A1 | 10 (10) | -2.543 | 0.3750 | 0.7500 | -0.35 |
| C2_swap | apply_ms_median | A2 vs A3 | 10 (10) | -10.115 | 0.0098 | 0.0293 | -0.89 |
| C2_swap | apply_ms_median | A2 vs static | 10 (10) | -0.642 | 0.9219 | 0.9219 | -0.05 |
| C2_swap | ctl_cpu_mean | A2 vs A1 | 10 (10) | -0.141 | 0.4922 | 0.9844 | -0.27 |
| C2_swap | ctl_cpu_mean | A2 vs A3 | 10 (10) | -0.159 | 0.1602 | 0.4805 | -0.53 |
| C2_swap | ctl_cpu_mean | A2 vs static | 10 (10) | 0.240 | 0.9219 | 0.9844 | 0.05 |

## Exploratory parameters (added 2 Oct 2026 after the results were seen; not pre-registered)

Pre = steps 2 to the step before the event; post = the event step to the end. Other benign hosts exclude probes, attackers (before and after), former attackers and surge hosts. A held-over rule (C2) is on a former attacker that had a rule at the step before the swap and at every step since; any other rule on a former attacker is a re-added rule.

### C1: benign surge

| Parameter | A2 | A1 | A3 | static |
|---|---|---|---|---|
| Other benign hosts with a rule, per step, after the event | 1.086 [0.546, 3.282] | 1.533 [0.467, 3.844] | 0.000 [0.000, 0.000] | 1.171 [1.082, 1.241] |
| Per-host share of post steps with a rule, other benign hosts | 0.155 [0.078, 0.469] | 0.219 [0.067, 0.549] | 0.000 [0.000, 0.000] | 0.167 [0.155, 0.177] |
| Per-host share of pre steps with a rule, future surge hosts | 0.090 [0.014, 0.336] | 0.177 [0.050, 0.497] | 0.000 [0.000, 0.000] | 0.225 [0.201, 0.244] |
| Per-host share of post steps with a rule, surge hosts | 0.499 [0.449, 0.727] | 0.518 [0.375, 0.765] | 0.000 [0.000, 0.000] | 0.454 [0.438, 0.466] |
| Rise in surge-host blocking minus rise in other-benign blocking (post - pre) | 0.339 [0.267, 0.375] | 0.266 [0.192, 0.321] | 0.000 [0.000, 0.000] | 0.277 [0.243, 0.323] |

### C2: attacker swap

| Parameter | A2 | A1 | A3 | static |
|---|---|---|---|---|
| Other benign hosts with a rule, per step, after the event | 0.695 [0.421, 2.297] | 1.065 [0.347, 3.252] | 0.202 [0.147, 0.245] | 0.963 [0.905, 1.046] |
| Per-host share of post steps with a rule, other benign hosts | 0.116 [0.070, 0.383] | 0.178 [0.058, 0.542] | 0.034 [0.025, 0.041] | 0.161 [0.151, 0.174] |
| Post steps with a rule held over from before the swap | 4.067 [1.833, 26.833] | 9.067 [1.667, 29.000] | 0.533 [0.167, 1.467] | 2.867 [2.333, 3.467] |
| Post steps with only re-added rules on former attackers | 10.133 [5.233, 11.933] | 8.633 [6.067, 11.401] | 5.733 [4.800, 6.567] | 14.267 [12.666, 15.867] |
| Per-host rate of re-added rules, former attackers | 0.114 [0.059, 0.142] | 0.102 [0.059, 0.138] | 0.068 [0.058, 0.079] | 0.158 [0.139, 0.171] |

## Exploratory paired tests (A2 vs each arm, exact Wilcoxon, Holm within each family of 3)

| Condition | Parameter | Comparison | n (non-zero) | median diff | p | p (Holm) | r |
|---|---|---|---|---|---|---|---|
| C1_surge | other_blocked_post | A2 vs A1 | 10 (10) | -0.090 | 0.6250 | 1.0000 | -0.20 |
| C1_surge | other_blocked_post | A2 vs A3 | 10 (10) | 0.896 | 0.0020 | 0.0059 | 1.00 |
| C1_surge | other_blocked_post | A2 vs static | 10 (10) | -0.180 | 0.9219 | 1.0000 | 0.05 |
| C1_surge | other_host_block_post | A2 vs A1 | 10 (10) | -0.013 | 0.6250 | 1.0000 | -0.20 |
| C1_surge | other_host_block_post | A2 vs A3 | 10 (10) | 0.128 | 0.0020 | 0.0059 | 1.00 |
| C1_surge | other_host_block_post | A2 vs static | 10 (10) | -0.026 | 0.9219 | 1.0000 | 0.05 |
| C1_surge | surge_host_block_pre | A2 vs A1 | 10 (10) | -0.047 | 0.0898 | 0.1797 | -0.62 |
| C1_surge | surge_host_block_pre | A2 vs A3 | 10 (9) | 0.073 | 0.0039 | 0.0117 | 1.00 |
| C1_surge | surge_host_block_pre | A2 vs static | 10 (10) | -0.122 | 0.4316 | 0.4316 | -0.31 |
| C1_surge | surge_host_block_post | A2 vs A1 | 10 (10) | 0.027 | 0.7148 | 0.7148 | 0.15 |
| C1_surge | surge_host_block_post | A2 vs A3 | 10 (10) | 0.517 | 0.0020 | 0.0059 | 1.00 |
| C1_surge | surge_host_block_post | A2 vs static | 10 (10) | 0.050 | 0.0840 | 0.1680 | 0.64 |
| C1_surge | surge_rise_minus_other | A2 vs A1 | 10 (10) | 0.067 | 0.0371 | 0.0742 | 0.75 |
| C1_surge | surge_rise_minus_other | A2 vs A3 | 10 (10) | 0.342 | 0.0020 | 0.0059 | 1.00 |
| C1_surge | surge_rise_minus_other | A2 vs static | 10 (10) | 0.039 | 0.1602 | 0.1602 | 0.53 |
| C2_swap | other_blocked_post | A2 vs A1 | 10 (10) | -0.028 | 0.5566 | 1.0000 | -0.24 |
| C2_swap | other_blocked_post | A2 vs A3 | 10 (10) | 0.366 | 0.0020 | 0.0059 | 1.00 |
| C2_swap | other_blocked_post | A2 vs static | 10 (10) | -0.428 | 0.6094 | 1.0000 | -0.20 |
| C2_swap | other_host_block_post | A2 vs A1 | 10 (10) | -0.005 | 0.5566 | 1.0000 | -0.24 |
| C2_swap | other_host_block_post | A2 vs A3 | 10 (10) | 0.061 | 0.0020 | 0.0059 | 1.00 |
| C2_swap | other_host_block_post | A2 vs static | 10 (10) | -0.071 | 0.6094 | 1.0000 | -0.20 |
| C2_swap | held_steps | A2 vs A1 | 10 (9) | 0.300 | 0.6523 | 0.8945 | 0.20 |
| C2_swap | held_steps | A2 vs A3 | 10 (9) | 2.700 | 0.0117 | 0.0352 | 0.93 |
| C2_swap | held_steps | A2 vs static | 10 (10) | 0.000 | 0.4473 | 0.8945 | 0.29 |
| C2_swap | reflag_only_steps | A2 vs A1 | 10 (10) | -1.000 | 0.8457 | 0.8457 | -0.09 |
| C2_swap | reflag_only_steps | A2 vs A3 | 10 (10) | 5.200 | 0.1309 | 0.2617 | 0.56 |
| C2_swap | reflag_only_steps | A2 vs static | 10 (10) | -3.600 | 0.0020 | 0.0059 | -1.00 |
| C2_swap | former_reflag_host_rate | A2 vs A1 | 10 (10) | -0.017 | 0.6250 | 0.6250 | -0.20 |
| C2_swap | former_reflag_host_rate | A2 vs A3 | 10 (10) | 0.044 | 0.1816 | 0.3633 | 0.49 |
| C2_swap | former_reflag_host_rate | A2 vs static | 10 (10) | -0.035 | 0.0371 | 0.1113 | -0.75 |

## Exploratory within-arm comparisons (b - a, exact Wilcoxon, Holm over arms)

| Condition | Measure | Arm | a (IQM) | b (IQM) | median b - a | seeds up / n | n (non-zero) | p | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|
| C1_surge | Per-host blocking of the surge hosts (post - pre) | A2 | 0.090 | 0.499 | 0.403 | 10/10 | 10 | 0.0020 | 0.0059 |
| C1_surge | Per-host blocking of the surge hosts (post - pre) | A1 | 0.177 | 0.518 | 0.297 | 10/10 | 10 | 0.0020 | 0.0059 |
| C1_surge | Per-host blocking of the surge hosts (post - pre) | A3 | 0.000 | 0.000 | 0.000 | 0/10 | 0 | n/a | n/a |
| C1_surge | Per-host blocking of the surge hosts (post - pre) | static | 0.225 | 0.454 | 0.233 | 10/10 | 10 | 0.0020 | 0.0059 |
| C1_surge | Per-host blocking of the other benign hosts (post - pre) | A2 | 0.095 | 0.155 | 0.060 | 10/10 | 10 | 0.0020 | 0.0059 |
| C1_surge | Per-host blocking of the other benign hosts (post - pre) | A1 | 0.193 | 0.219 | 0.048 | 7/10 | 10 | 0.0684 | 0.0684 |
| C1_surge | Per-host blocking of the other benign hosts (post - pre) | A3 | 0.000 | 0.000 | 0.000 | 0/10 | 0 | n/a | n/a |
| C1_surge | Per-host blocking of the other benign hosts (post - pre) | static | 0.219 | 0.167 | -0.055 | 1/10 | 10 | 0.0039 | 0.0078 |
| C1_surge | Other benign hosts with a rule, per step (post - pre) | A2 | 0.665 | 1.086 | 0.420 | 10/10 | 10 | 0.0020 | 0.0059 |
| C1_surge | Other benign hosts with a rule, per step (post - pre) | A1 | 1.354 | 1.533 | 0.334 | 7/10 | 10 | 0.0684 | 0.0684 |
| C1_surge | Other benign hosts with a rule, per step (post - pre) | A3 | 0.000 | 0.000 | 0.000 | 0/10 | 0 | n/a | n/a |
| C1_surge | Other benign hosts with a rule, per step (post - pre) | static | 1.532 | 1.171 | -0.382 | 1/10 | 10 | 0.0039 | 0.0078 |
| C2_swap | Per-host rate after the swap: re-added rules on former attackers - rules on other benign hosts | A2 | 0.116 | 0.114 | 0.043 | 6/10 | 10 | 0.8262 | 1.0000 |
| C2_swap | Per-host rate after the swap: re-added rules on former attackers - rules on other benign hosts | A1 | 0.178 | 0.102 | 0.014 | 6/10 | 10 | 0.9219 | 1.0000 |
| C2_swap | Per-host rate after the swap: re-added rules on former attackers - rules on other benign hosts | A3 | 0.034 | 0.068 | 0.033 | 10/10 | 10 | 0.0020 | 0.0078 |
| C2_swap | Per-host rate after the swap: re-added rules on former attackers - rules on other benign hosts | static | 0.161 | 0.158 | -0.015 | 4/10 | 10 | 0.6426 | 1.0000 |

## Recovery (Fisher's exact test on never-recovered episodes, Holm)

| Condition | Comparison | A2 never / n | other never / n | p | p (Holm) |
|---|---|---|---|---|---|
| C1_surge | A2 vs A1 | 0/50 | 0/50 | 1.0000 | 1.0000 |
| C1_surge | A2 vs A3 | 0/50 | 0/50 | 1.0000 | 1.0000 |
| C1_surge | A2 vs static | 0/50 | 0/50 | 1.0000 | 1.0000 |
| C2_swap | A2 vs A1 | 0/50 | 0/50 | 1.0000 | 1.0000 |
| C2_swap | A2 vs A3 | 0/50 | 0/50 | 1.0000 | 1.0000 |
| C2_swap | A2 vs static | 0/50 | 1/50 | 1.0000 | 1.0000 |
