#!/usr/bin/env python3
"""Offline tests for scripts/analyze_iipem.py on constructed episodes.

Every expected value below is worked out by hand from the constructed rows,
not taken from the script. Run: ~/venv/bin/python scripts/test_analyze_iipem.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_iipem as A  # noqa: E402

PASSED, FAILED = [], []


def check(name, cond, detail=""):
    (PASSED if cond else FAILED).append(name)
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  {detail}"))


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


# ------------------------------------------------------------ builders
def episode(cond, agent, seed, ep=0, n=100, ev=50, S=None, mitigated=None, att0="h2,h3,h4",
            att1=None, former="", surge=""):
    """One episode of step rows. mitigated: dict step -> 'ip,ip'."""
    S = np.ones(n) if S is None else np.asarray(S, float)
    att1 = att0 if att1 is None else att1
    rows = []
    for t in range(n):
        post = t >= ev
        m = (mitigated or {}).get(t, "")
        mm = A.hosts(m)
        att = A.hosts(att1 if post else att0)
        coll = (len(mm - att) / len(mm)) if mm else 0.0
        rows.append(dict(cond=cond, agent=agent, seed=seed, ep=ep, step=t, action=0, S=S[t],
                         rule_count=len(mm), collateral=coll, probe_n=15,
                         infer_ms=0.5, apply_ms=2.0, ctl_cpu_pct=3.0, ctl_rss_mb=900.0,
                         event_fired=post, attackers=att1 if post else att0,
                         former_attackers=former if post else "", surge_hosts=surge if post else "",
                         flagged="", mitigated=m, t_wall=1000 + 0.5 * t))
    return rows


def deliv(cond, agent, seed, pre, post, ep=0):
    """pre/post: dict ip -> bytes."""
    rows = []
    for ph, d in (("pre", pre), ("post", post)):
        for src, b in d.items():
            rows.append(dict(cond=cond, agent=agent, seed=seed, ep=ep, phase=ph, src=src,
                             pkts=max(1, b // 1000), bytes=b))
    return rows


# ------------------------------------------------------------ statistics
def test_stats():
    n, w, p, r = A.wilcoxon_exact(np.arange(1, 11) * 0.1)
    check("wilcoxon: 10 positive differences give p = 2/1024", close(p, 2 / 1024), p)
    check("wilcoxon: rank-biserial is 1 when all differences are positive", close(r, 1.0), r)
    d = [0.645584, 1.121618, 0.630437, -1.003157, 1.205356, 0.746375, -0.236953, 0.881118, 0.664572]
    p = A.wilcoxon_exact(d)[2]
    check("wilcoxon: matches SciPy exact (0.09765625) on untied data", close(p, 0.09765625), p)
    check("wilcoxon: symmetric in sign", close(A.wilcoxon_exact(-np.array(d))[2], p))
    check("wilcoxon: all-zero differences give p = 1 and n = 0",
          A.wilcoxon_exact([0, 0, 0])[0] == 0 and A.wilcoxon_exact([0, 0, 0])[2] == 1.0)
    # Ties: d = -1,1,2,3,4 -> ranks 1.5,1.5,3,4,5; W+ = 13.5 of 15 (centre 7.5).
    # Sign patterns with W+ <= 1.5 are {}, {1.5}, {1.5'}; their complements give
    # W+ >= 13.5. So p = 6/32 (sequential ranks 1..5 would give 4/32).
    check("wilcoxon: exact with tied ranks (6/32)", close(A.wilcoxon_exact([-1, 1, 2, 3, 4])[2], 6 / 32))
    h = A.holm([0.01, 0.04, 0.03])
    check("holm: [0.01, 0.04, 0.03] -> [0.03, 0.06, 0.06]", np.allclose(h, [0.03, 0.06, 0.06]), h)
    check("holm: NaN stays NaN and is not counted", np.allclose(A.holm([0.01, np.nan, 0.02])[[0, 2]],
                                                               [0.02, 0.02]))
    check("fisher: matches SciPy (0.034965) for [[8,2],[1,5]]",
          close(A.fisher_exact(8, 2, 1, 5), 0.034965034965034975, 1e-12))
    check("iqm: 1..10 -> 5.5 (trim 2 each side)", close(A.iqm(np.arange(1, 11)), 5.5))
    lo, hi = A.bootstrap_ci(np.arange(1, 11), reps=2000)
    check("bootstrap: interval brackets the IQM", lo <= 5.5 <= hi, (lo, hi))
    noisy = [-0.004000000000000115, -0.0040000000000000036, 0.0040000000000000036,
             -0.0040000000000000036, 1e-17, -0.02, -0.044, 0.0, 0.02, 0.016]
    clean = [-0.004, -0.004, 0.004, -0.004, 0.0, -0.02, -0.044, 0.0, 0.02, 0.016]
    check("wilcoxon: float noise does not break ties or hide zeros",
          close(A.wilcoxon_exact(noisy)[2], A.wilcoxon_exact(clean)[2])
          and A.wilcoxon_exact(noisy)[0] == 8, (A.wilcoxon_exact(noisy), A.wilcoxon_exact(clean)))


# ------------------------------------------------------------ episode metrics
def test_c2_metrics():
    # Swap at 50: old h2,h3,h4 -> new h8,h12,h14. Rules on former attackers 10.0.0.2 at
    # steps 50..54 and 60 (6 steps), first clear at step 55 (latency 5). One rule on
    # benign 10.0.0.5 at step 70 together with new attacker 10.0.0.8.
    m = {t: "10.0.0.2" for t in range(50, 55)}
    m[60] = "10.0.0.2,10.0.0.8"
    m[70] = "10.0.0.5,10.0.0.8"
    m[10] = "10.0.0.2,10.0.0.9"         # pre: one attacker, one benign -> collateral 0.5
    S = np.ones(100)
    S[50] = 0.0                          # transient: excluded from worst_post
    S[51] = 0.1                          # transient: excluded from worst_post
    S[52] = 0.25                         # inside the post window
    rows = episode("C2_swap", "A2", 1000, S=S, mitigated=m, att0="h2,h3,h4",
                   att1="h8,h12,h14", former="h2,h3,h4")
    pre = {"10.0.0.1": 5000, "10.0.0.2": 8000, "10.0.0.3": 8000, "10.0.0.4": 8000,
           "10.0.0.5": 1000, "10.0.0.9": 3000, "10.0.0.8": 2000, "10.0.0.12": 2000, "10.0.0.14": 2000}
    post = {"10.0.0.1": 5000, "10.0.0.2": 2000, "10.0.0.3": 2000, "10.0.0.4": 2000,
            "10.0.0.5": 500, "10.0.0.9": 3000, "10.0.0.8": 6000, "10.0.0.12": 6000, "10.0.0.14": 6000}
    st = pd.DataFrame(rows)
    dl = pd.DataFrame(deliv("C2_swap", "A2", 1000, pre, post))
    em, roles = A.episode_metrics(st, dl)
    e = em.iloc[0]
    check("C2: stale steps = 6", e.stale_steps == 6, e.stale_steps)
    check("C2: stale clear latency = 5", e.stale_clear_latency == 5, e.stale_clear_latency)
    # post collateral: steps 50-54 -> 1.0 each (5), step 60 -> 0.5, step 70 -> 0.5; /50
    check("C2: collateral_post = 6/50", close(e.collateral_post, 6.0 / 50), e.collateral_post)
    check("C2: collateral on former attackers = 5.5/50", close(e.collateral_role_post, 5.5 / 50))
    check("C2: collateral on other benign = 0.5/50", close(e.collateral_other_post, 0.5 / 50))
    check("C2: collateral_pre = 0.5/50", close(e.collateral_pre, 0.5 / 50))
    check("C2: role split agrees with the logged collateral", np.allclose(roles.coll, roles.collateral))
    check("C2: worst_post skips the two transient steps (0.25)", close(e.worst_post, 0.25), e.worst_post)
    check("C2: service_post = (47 + 0.1 + 0.25)/50",
          close(e.service_post, (47 + 0.1 + 0.25) / 50), e.service_post)
    # other benign = .5 and .9 (probe .1 excluded): pre 4000, post 3500 per 50 steps
    check("C2: benign delivery (other) = 3500/4000", close(e.benign_delivery_other, 3500 / 4000))
    # former attackers mean post 2000 vs other-benign mean pre 2000
    check("C2: former-attacker delivery = 2000/2000", close(e.former_attacker_delivery, 1.0))
    check("C2: attack delivery ratio = 18000/24000", close(e.attack_delivery_ratio, 0.75))
    check("C2: attack MB pre/post", close(e.attack_mb_pre, 0.024) and close(e.attack_mb_post, 0.018))
    check("C2: worst_isasf (ISASF minS from step 2) includes the event dip (0.0)",
          close(e.worst_isasf, 0.0), e.worst_isasf)
    # ISASF recovery starts at event + 1 = step 51 (ISASF's switch step is one row after
    # the dip). Baseline S[31:51] = 19 ones and S[50] = 0 -> 0.95, target 0.9025.
    # 5-step rolling mean of S[51:] = .1, .175, .45, .5875, .67, .85, 1.0 ... -> t = 6
    check("recovery: ISASF latency 6 (clock starts one step after the event)",
          bool(e.recovered) and e.adapt_latency == 6, (e.recovered, e.adapt_latency))


def isasf_recovery_latency(S, regime):
    """Verbatim logic of ISASF scripts/analyze.py recovery_latency (tag isasf-2026)."""
    first = regime[0]
    idx = np.nonzero(regime != first)[0]
    sw = int(idx[0]) if len(idx) else None
    if sw is None or sw < 20:
        return None
    base = float(np.mean(S[sw - 20:sw]))
    if base <= 1e-6:
        return None
    post = S[sw:]
    roll = pd.Series(post).rolling(5, min_periods=1).mean().to_numpy()
    ok = roll >= 0.95 * base
    for t in range(len(ok) - 5 + 1):
        if ok[t:t + 5].all():
            return float(t), False
    return float(len(post)), True


def test_recovery_matches_isasf():
    rng = np.random.default_rng(7)
    bad = []
    for i in range(300):
        ev = int(rng.integers(20, 90))
        S = np.clip(rng.normal(0.9, 0.15, 100), 0, 1)
        S[ev:ev + int(rng.integers(0, 40))] *= rng.uniform(0, 1)
        if i % 10 == 0:
            S[ev:] = 0.2                                  # never recovers
        regime = np.array(["R1"] * ev + ["R2"] * (100 - ev))
        ref = isasf_recovery_latency(S, regime)
        rec, lat = A.recovery(S, ev)
        want = (True, ref[0]) if not ref[1] else (False, None)
        got = (rec, None if np.isnan(lat) else lat)
        if want != got:
            bad.append((i, want, got))
    check("recovery: equals ISASF recovery_latency on 300 random episodes (30 censored)",
          not bad, bad[:3])
    check("recovery: undefined before 20 baseline steps", A.recovery(np.ones(100), 10)[0] is None)
    base08 = np.r_[np.full(50, 0.8), np.full(50, 0.77)]
    check("recovery: target is relative to the pre-event level (0.77 >= 0.95 x 0.8)",
          A.recovery(base08, 50) == (True, 0.0))


def test_c1_metrics():
    m = {t: "10.0.0.8,10.0.0.12" for t in range(55, 75)}       # 20 post steps on both surge hosts
    m.update({t: "10.0.0.9" for t in range(20, 30)})          # 10 pre steps on a benign host
    rows = episode("C1_surge", "static", 1000, mitigated=m, surge="h8,h12")
    pre = {"10.0.0.8": 50000, "10.0.0.12": 50000, "10.0.0.9": 40000, "10.0.0.5": 60000,
           "10.0.0.6": 5000}
    offered = 4.0e6 / 8 * 0.5 * 50                                # bytes per surge host after the event
    post = {"10.0.0.8": int(offered * 0.5), "10.0.0.12": int(offered * 0.3), "10.0.0.9": 40000,
            "10.0.0.5": 30000, "10.0.0.6": 5000}
    em, _ = A.episode_metrics(pd.DataFrame(rows),
                              pd.DataFrame(deliv("C1_surge", "static", 1000, pre, post)))
    e = em.iloc[0]
    check("C1: surge block share = 20/50", close(e.surge_block_share, 0.4))
    check("C1: collateral on surge hosts = 20/50", close(e.collateral_role_post, 0.4))
    check("C1: benign rule share before the event = 10/50", close(e.benign_rule_share_pre, 0.2))
    check("C1: surge delivery = mean share of 4 Mbps offered (0.4)",
          close(e.surge_delivery, 0.4, 1e-6), e.surge_delivery)
    check("C1: benign delivery (other) = 70000/100000", close(e.benign_delivery_other, 0.7))
    check("C1: no stale-rule columns", np.isnan(e.get("stale_steps", np.nan)))


# ------------------------------------------------------------ step 22 (exploratory) metrics
def test_step22_c1():
    # Hosts in the delivery log: probes .1 .6 .11, silent listed attackers .2 .3 .4,
    # surge .8 .12, other benign .5 .7 .9 (3 hosts). Pre window = steps 2..49 (48 steps).
    m = {1: "10.0.0.8"}                                        # reset step: not counted
    m.update({t: "10.0.0.9" for t in range(20, 30)})           # other benign, 10 pre steps
    m.update({t: "10.0.0.8" for t in range(30, 40)})           # future surge host, 10 pre steps
    m.update({t: "10.0.0.8,10.0.0.12" for t in range(55, 75)})  # both surge hosts, 20 post steps
    m.update({t: "10.0.0.5,10.0.0.7" for t in range(80, 90)})   # two other benign, 10 post steps
    rows = episode("C1_surge", "A2", 1000, mitigated=m, surge="h8,h12")
    srcs = ["10.0.0.%d" % i for i in (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12)]
    d = deliv("C1_surge", "A2", 1000, dict.fromkeys(srcs, 5000), dict.fromkeys(srcs, 5000))
    e = A.episode_metrics(pd.DataFrame(rows), pd.DataFrame(d))[0].iloc[0]
    check("C1x: surge per-host blocking before the surge = 10/96",
          close(e.surge_host_block_pre, 10 / 96), e.surge_host_block_pre)
    check("C1x: surge per-host blocking after = 40/100", close(e.surge_host_block_post, 0.4))
    check("C1x: other per-host blocking before = 10/144 (3 other hosts, probes excluded)",
          close(e.other_host_block_pre, 10 / 144), e.other_host_block_pre)
    check("C1x: other per-host blocking after = 20/150", close(e.other_host_block_post, 20 / 150))
    check("C1x: other benign hosts with a rule per step: before 10/48, after 20/50",
          close(e.other_blocked_pre, 10 / 48) and close(e.other_blocked_post, 0.4),
          (e.other_blocked_pre, e.other_blocked_post))
    check("C1x: rise minus other rise = (0.4 - 10/96) - (20/150 - 10/144)",
          close(e.surge_rise_minus_other, (0.4 - 10 / 96) - (20 / 150 - 10 / 144)))
    check("C1x: S = 0 at the event is flagged as 0 when S = 1", e.S_zero_at_event == 0.0)
    S = np.ones(100)
    S[50] = 0.0                                    # the probe-restart dip alone
    e2 = A.episode_metrics(pd.DataFrame(episode("C1_surge", "A2", 1000, S=S, surge="h8,h12")),
                           None)[0].iloc[0]
    check("recovery: a dip at the event row alone gives latency 0, as in ISASF E1/E3",
          bool(e2.recovered) and e2.adapt_latency == 0, (e2.recovered, e2.adapt_latency))
    no_dl = A.episode_metrics(pd.DataFrame(rows), None)[0].iloc[0]
    check("C1x: without delivery counters the host universe is unknown (NaN)",
          np.isnan(no_dl.other_blocked_post) and np.isnan(no_dl.get("surge_host_block_pre", np.nan)))


def test_step22_c2():
    # Former .2 .3 .4 -> new .8 .12 .14; stayed benign .5 .7 .9 .10 .13 .15 (6 hosts).
    m = {}
    for t in range(0, 100):
        s = set()
        if 40 <= t <= 52:
            s.add("10.0.0.2")                 # held 50-52
        if 45 <= t <= 56 or 60 <= t <= 62:
            s.add("10.0.0.3")                 # held 50-56, re-added 60-62 (3 host-steps)
        if t in (50, 51, 57, 58):
            s.add("10.0.0.4")                 # no rule at 49: re-added (4 host-steps)
        if t >= 53:
            s.add("10.0.0.8")                 # a new attacker
        if 80 <= t <= 84:
            s.add("10.0.0.5")                 # stayed benign, 5 post host-steps
        if 10 <= t <= 19:
            s.add("10.0.0.9")                 # stayed benign, 10 pre host-steps
        if s:
            m[t] = ",".join(sorted(s))
    rows = episode("C2_swap", "A2", 1000, mitigated=m, att0="h2,h3,h4", att1="h8,h12,h14",
                   former="h2,h3,h4")
    srcs = ["10.0.0.%d" % i for i in range(1, 16)]
    d = deliv("C2_swap", "A2", 1000, dict.fromkeys(srcs, 5000), dict.fromkeys(srcs, 5000))
    e = A.episode_metrics(pd.DataFrame(rows), pd.DataFrame(d))[0].iloc[0]
    check("C2x: held-over steps = 7 (steps 50-56)", e.held_steps == 7, e.held_steps)
    check("C2x: re-added-only steps = 5 (57, 58, 60, 61, 62)", e.reflag_only_steps == 5,
          e.reflag_only_steps)
    check("C2x: planned stale steps unchanged = 12, clear latency = 9",
          e.stale_steps == 12 and e.stale_clear_latency == 9, (e.stale_steps, e.stale_clear_latency))
    check("C2x: re-added host rate = 7/150 (a rule first seen at the event is not held over)",
          close(e.former_reflag_host_rate, 7 / 150), e.former_reflag_host_rate)
    check("C2x: other benign per-host = 5/300 after, 10/288 before",
          close(e.other_host_block_post, 5 / 300) and close(e.other_host_block_pre, 10 / 288),
          (e.other_host_block_post, e.other_host_block_pre))
    check("C2x: no surge columns in C2", np.isnan(e.get("surge_host_block_pre", np.nan)))


def test_within_arm():
    rows = []
    for k, seed in enumerate(range(1000, 1010)):
        for arm, pre, post in (("A2", 0.1, 0.1 + 0.01 * (k + 1)), ("A3", 0.0, 0.0),
                               ("static", 0.3, 0.3 - 0.01 * (k + 1))):
            rows.append(dict(cond="C1_surge", agent=arm, seed=seed, surge_host_block_pre=pre,
                             surge_host_block_post=post))
    w = A.within_arm_tests(pd.DataFrame(rows)).set_index("agent")
    check("within: A2 up on all 10 seeds -> 10/10, p = 2/1024",
          w.loc["A2"].seeds_up == 10 and close(w.loc["A2"].p_exact, 2 / 1024))
    check("within: static down on all 10 seeds -> 0/10 up, same p",
          w.loc["static"].seeds_up == 0 and close(w.loc["static"].p_exact, 2 / 1024))
    check("within: all-zero arm gives no p-value and Holm counts 2 arms (2 x 2/1024)",
          np.isnan(w.loc["A3"].p_exact) and close(w.loc["A2"].p_holm, 2 * 2 / 1024))
    check("fmt: -0.0001 prints as 0.000", A.fmt(-0.0001) == "0.000")


# ------------------------------------------------------------ validation
def expect_exit(name, fn):
    try:
        fn()
    except SystemExit:
        check(name, True)
        return
    check(name, False, "no SystemExit")


def test_validation():
    a = episode("C2_swap", "A2", 1000, att1="h8,h12,h14", former="h2,h3,h4")
    b = episode("C2_swap", "A1", 1000, att1="h8,h12,h15", former="h2,h3,h4")
    expect_exit("validate: different new attackers across arms are refused",
                lambda: A.validate(pd.DataFrame(a + b), None))
    c = episode("C2_swap", "A1", 1000, ev=51, att1="h8,h12,h14", former="h2,h3,h4")
    expect_exit("validate: different event steps across arms are refused",
                lambda: A.validate(pd.DataFrame(a + c), None))
    d = episode("C1_surge", "A2", 1000, mitigated={60: "10.0.0.1"}, surge="h8,h12")
    expect_exit("validate: a mitigated probe host is refused",
                lambda: A.validate(pd.DataFrame(d), None))
    e = pd.DataFrame(episode("C1_surge", "A2", 1000, surge="h8,h12"))
    e = e[e.step != 30]
    expect_exit("validate: a missing step is refused", lambda: A.validate(e, None))
    # Dead traffic generators: background hosts deliver nothing before the event.
    bg = ("10.0.0.5", "10.0.0.7", "10.0.0.8", "10.0.0.9")
    live = {h: 50000 for h in bg}
    f = pd.DataFrame(episode("C1_surge", "A2", 1000, surge="h8,h9"))
    expect_exit("validate: an episode with silent background hosts is refused",
                lambda: A.validate(f, pd.DataFrame(deliv("C1_surge", "A2", 1000,
                                                         dict.fromkeys(bg, 60), live))))
    three_dead = dict(live, **dict.fromkeys(bg[1:], 0))
    expect_exit("validate: three of four background hosts silent is refused",
                lambda: A.validate(f, pd.DataFrame(deliv("C1_surge", "A2", 1000, three_dead, live))))
    blocked = pd.DataFrame(episode("C1_surge", "A2", 1000, surge="h8,h9",
                                   mitigated={t: "10.0.0.7,10.0.0.8" for t in range(10, 50)}))
    A.validate(blocked, pd.DataFrame(deliv("C1_surge", "A2", 1000, three_dead, live)))
    check("validate: hosts silent because an arm blocked them pre-event are not dead traffic", True)
    surge_blocked = dict(live, **{"10.0.0.8": 0, "10.0.0.9": 0})
    A.validate(f, pd.DataFrame(deliv("C1_surge", "A2", 1000, live, surge_blocked)))
    check("validate: surge hosts blocked after the event are not dead traffic", True)
    ok = episode("C2_swap", "A1", 1000, att1="h8,h12,h14", former="h2,h3,h4")
    notes = A.validate(pd.DataFrame(a + ok), None)
    check("validate: paired arms pass and the seed-count note appears",
          any("seed" in n for n in notes), notes)


# ------------------------------------------------------------ end to end
def test_end_to_end():
    rows, drows = [], []
    rng = np.random.default_rng(0)
    for seed in range(1000, 1010):
        for ep in range(5):
            for agent, k in (("A2", 2), ("A1", 6), ("A3", 6), ("static", 10)):
                m = {t: "10.0.0.2" for t in range(50, 50 + k + ep % 2)}   # k, k+1, k, k+1, k
                S = np.ones(100)
                S[50] = 0.0
                rows += episode("C2_swap", agent, seed, ep, S=S, mitigated=m, att1="h8,h12,h14",
                                former="h2,h3,h4")
                pre = {h: int(rng.integers(1000, 2000)) for h in ("10.0.0.5", "10.0.0.9")}
                pre.update({"10.0.0.2": 9000, "10.0.0.8": 1500})
                post = dict(pre, **{"10.0.0.2": 1500, "10.0.0.8": 9000})
                drows += deliv("C2_swap", agent, seed, pre, post, ep)
    with tempfile.TemporaryDirectory() as td:
        sp, dp, out = Path(td, "s.parquet"), Path(td, "d.parquet"), Path(td, "out")
        pd.DataFrame(rows).to_parquet(sp)
        pd.DataFrame(drows).to_parquet(dp)
        A.main(["--steps", str(sp), "--delivered", str(dp), "--out", str(out), "--boot", "500"])
        t = pd.read_csv(out / "paired_tests.csv")
        s = t[t.parameter == "stale_steps"].set_index("comparison")
        check("e2e: A2 fewer stale steps on all 10 seeds -> exact p = 2/1024",
              np.allclose(s.p_exact, 2 / 1024), s.p_exact.tolist())
        check("e2e: Holm makes the smallest of three equal p-values 3x",
              np.allclose(s.p_holm, 3 * 2 / 1024), s.p_holm.tolist())
        check("e2e: significant after Holm", s["significant_0.05"].all())
        q = pd.read_csv(out / "summary_iqm.csv")
        v = q[(q.parameter == "stale_steps")].set_index("agent").iqm
        check("e2e: IQM of seed means (5 episodes each) = 2.4, 6.4, 6.4, 10.4",
              np.allclose([v.A2, v.A1, v.A3, v.static], [2.4, 6.4, 6.4, 10.4]), v.to_dict())
        sm = pd.read_csv(out / "seed_metrics.csv")
        check("e2e: 5 episodes per seed", (sm.n_episodes == 5).all())
        files = {"episode_metrics.csv", "seed_metrics.csv", "summary_iqm.csv", "paired_tests.csv",
                 "recovery_fisher.csv", "within_arm_tests.csv", "summary.md"}
        check("e2e: all outputs written", files <= {p.name for p in out.iterdir()})
        check("e2e: planned and exploratory tests are labelled separately",
              set(t.family) == {"planned", "exploratory"}
              and (t[t.parameter == "held_steps"].family == "exploratory").all()
              and (t[t.parameter == "stale_steps"].family == "planned").all())
        md = (out / "summary.md").read_text()
        check("e2e: summary shows non-zero n and the exploratory heading",
              "n (non-zero)" in md and "not pre-registered" in md
              and "the dip alone gives a latency of 0" in md)


if __name__ == "__main__":
    test_stats()
    test_c2_metrics()
    test_c1_metrics()
    test_recovery_matches_isasf()
    test_step22_c1()
    test_step22_c2()
    test_within_arm()
    test_validation()
    test_end_to_end()
    print(f"\n{len(PASSED)} of {len(PASSED) + len(FAILED)} checks passed")
    print("ALL PASSED" if not FAILED else "FAILED: " + ", ".join(FAILED))
    sys.exit(1 if FAILED else 0)
