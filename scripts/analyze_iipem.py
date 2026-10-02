#!/usr/bin/env python3
"""IIPEM analysis (WP7): performance parameters, paired statistics and figures.

Reads the per-step log written by scripts/evaluate_iipem.py and the victim's
per-source delivery counters, and writes every number the paper uses:

  <out>/episode_metrics.csv   one row per (cond, agent, seed, ep)
  <out>/seed_metrics.csv      mean of each seed's episodes (the statistical unit)
  <out>/summary_iqm.csv       IQM and bootstrap 95% interval per arm
  <out>/paired_tests.csv      exact paired Wilcoxon, A2 vs each other arm, Holm
  <out>/recovery_fisher.csv   never-recovered episodes, Fisher's exact test, Holm
  <out>/within_arm_tests.csv  C1 change from before to after the event, per arm (exploratory)
  <out>/summary.md            the same results as readable tables, with checks
  <out>/fig_*.pdf|png         figures (skipped if matplotlib is missing)

Plan revision 5, Sections 7 and 8. Depends on numpy and pandas only; the
statistics are implemented here so results do not depend on the SciPy version.

Usage:
  python scripts/analyze_iipem.py --steps runs/iipem.parquet \
      --delivered runs/iipem_delivered.parquet --out results/iipem
"""
import argparse
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = ["A2", "A1", "A3", "static"]          # A2 first: it is the reference arm
REF = "A2"
CONDS = ["C1_surge", "C2_swap"]
# Probe clients: configs/base.yaml roles.probes = [h1, h6, h11] (confirmed 2 Oct
# 2026); also checked against the data in validate().
PROBE_HOSTS = {"10.0.0.1", "10.0.0.6", "10.0.0.11"}
# Worst window, as ISASF scripts/analyze.py (tag isasf-2026): minS = lowest S from
# step WARMUP on. IIPEM also skips TRANSIENT steps after the event (worst_post),
# because the event restarts the probe clients in every arm (S ~ 0 at the event).
TRANSIENT = 2
# Recovery, as ISASF adaptation latency (plan 4.12): the ROLL-step rolling mean of
# S (min_periods 1) from the event on returns to RECOVER_FRAC x the mean S of the
# BASELINE_WIN steps before the event and holds for HOLD steps; latency = first
# such step - event; censored if it never holds.
# The recovery clock starts RECOVERY_OFFSET steps after the event row. ISASF's
# switch_step is the first row with the new regime label, which in all 195 ISASF
# E1/E3 episodes is step 51, one row after the probe-restart dip at step 50
# (status step 24). The IIPEM event row is the dip row itself (step 50), so the
# same clock starts at event + 1.
RECOVERY_OFFSET = 1
RECOVER_FRAC = 0.95
BASELINE_WIN = 20
ROLL = 5
HOLD = 5
MIN_SEEDS_FOR_TESTS = 8
LIVE_BYTES = 1000       # a host that delivered less than this in a phase sent nothing
SURGE_RATE_MBPS = 4.0  # C1 setting fixed by the WP2 calibration (plan rev. 5)

REQUIRED_STEP_COLS = [
    "cond", "agent", "seed", "ep", "step", "action", "S", "rule_count", "collateral",
    "probe_n", "infer_ms", "apply_ms", "ctl_cpu_pct", "ctl_rss_mb", "event_fired",
    "attackers", "former_attackers", "surge_hosts", "flagged", "mitigated",
]
REQUIRED_DELIV_COLS = ["cond", "agent", "seed", "ep", "phase", "src", "pkts", "bytes"]
KEY = ["cond", "agent", "seed", "ep"]

# Parameters tested with the paired Wilcoxon (each a family of 3 comparisons).
# (column, condition or None for both, label, higher_is_better)
TESTED = [
    ("service_post", None, "Mean service after the event", True),
    ("worst_post", None, "Worst-window service after the event", True),
    ("worst_isasf", None, "Worst-window service, ISASF minS (whole episode)", True),
    ("adapt_latency", None, "Recovery latency, recovered episodes (steps)", False),
    ("collateral_post", None, "Collateral after the event", False),
    ("collateral_other_post", None, "Collateral on other benign hosts after the event", False),
    ("benign_delivery_other", None, "Delivery from other benign hosts (post/pre)", True),
    ("surge_block_share", "C1_surge", "Share of post steps with a rule on a surge host", None),
    ("surge_delivery", "C1_surge", "Surge-host delivery (share of 4.0 Mbps offered)", None),
    ("stale_steps", "C2_swap", "Post steps with a rule on a former attacker", False),
    ("stale_clear_latency", "C2_swap", "Steps until no rule on a former attacker", False),
    ("former_attacker_delivery", "C2_swap", "Former-attacker delivery (vs benign level)", True),
    ("attack_delivery_ratio", "C2_swap", "Attack delivery (post/pre)", False),
    ("infer_ms_median", None, "Inference time, median (ms)", False),
    ("apply_ms_median", None, "Rule installation time, median (ms)", False),
    ("ctl_cpu_mean", None, "Controller CPU, mean (%)", False),
]

# Added on 2 Oct 2026 after the rerun results were seen (status step 22, fixes 3 to 5).
# Not pre-registered: reported in a separate, clearly labelled table, tested the same way.
# "Other benign" = hosts in the episode's delivery log that are not probes, not attackers
# before or after the event, not former attackers and not surge hosts.
EXPLORATORY = [
    ("other_blocked_post", None, "Other benign hosts with a rule, per step, after the event", False),
    ("other_host_block_post", None, "Per-host share of post steps with a rule, other benign hosts",
     False),
    ("surge_host_block_pre", "C1_surge", "Per-host share of pre steps with a rule, future surge hosts",
     None),
    ("surge_host_block_post", "C1_surge", "Per-host share of post steps with a rule, surge hosts", None),
    ("surge_rise_minus_other", "C1_surge",
     "Rise in surge-host blocking minus rise in other-benign blocking (post - pre)", None),
    ("held_steps", "C2_swap", "Post steps with a rule held over from before the swap", False),
    ("reflag_only_steps", "C2_swap", "Post steps with only re-added rules on former attackers", False),
    ("former_reflag_host_rate", "C2_swap", "Per-host rate of re-added rules, former attackers", False),
]
# Within-arm comparisons, exploratory: (condition, a, b, label); the test is on b - a.
WITHIN_ARM = [
    ("C1_surge", "surge_host_block_pre", "surge_host_block_post",
     "Per-host blocking of the surge hosts (post - pre)"),
    ("C1_surge", "other_host_block_pre", "other_host_block_post",
     "Per-host blocking of the other benign hosts (post - pre)"),
    ("C1_surge", "other_blocked_pre", "other_blocked_post",
     "Other benign hosts with a rule, per step (post - pre)"),
    ("C2_swap", "other_host_block_post", "former_reflag_host_rate",
     "Per-host rate after the swap: re-added rules on former attackers - rules on other benign hosts"),
]


FAMILIES = [("planned",) + t for t in TESTED] + [("exploratory",) + t for t in EXPLORATORY]


# ---------------------------------------------------------------- helpers
def hosts(s):
    """'h3,h4' or '10.0.0.3,10.0.0.4' -> {'10.0.0.3', '10.0.0.4'}."""
    out = set()
    for x in str(s or "").split(","):
        x = x.strip()
        if not x:
            continue
        out.add("10.0.0." + x[1:] if x.startswith("h") else x)
    return out


def iqm(x):
    """Interquartile mean, as scipy.stats.trim_mean(x, 0.25) (rliable)."""
    x = np.sort(np.asarray(x, float))
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return float("nan")
    cut = int(0.25 * n)
    return float(x[cut:n - cut].mean())


def bootstrap_ci(x, reps=10000, seed=0, alpha=0.05):
    """Percentile bootstrap interval for the IQM, resampling seeds."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(reps, len(x)))
    stats = np.array([iqm(x[i]) for i in idx])
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def avg_ranks(a):
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def wilcoxon_exact(diff):
    """Exact two-sided paired Wilcoxon signed-rank test.

    Differences are rounded to 9 decimals, so float noise neither breaks a
    tie nor hides a zero. Zero differences are dropped (Wilcoxon's method); tied |d| get average
    ranks and the null distribution is enumerated over all 2^n sign patterns
    of those ranks, so the p-value is exact even with ties.
    Returns (n_nonzero, W_plus, p, rank_biserial).
    """
    d = np.asarray(diff, float)
    d = d[~np.isnan(d)]
    # Seed means of step fractions carry float noise (0.004 vs 0.0040000000001):
    # round so equal differences tie and tiny residues count as zero.
    d = np.round(d, 9)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 0, 0.0, 1.0, 0.0
    r = avg_ranks(np.abs(d))
    w_plus = float(r[d > 0].sum())
    total = float(r.sum())
    centre = total / 2
    obs = abs(w_plus - centre)
    if n > 20:  # not expected (10 seeds); normal approximation
        sd = math.sqrt((r ** 2).sum() / 4)
        z = obs / sd if sd else 0.0
        p = math.erfc(z / math.sqrt(2))
    else:
        signs = np.array(list(itertools.product([0, 1], repeat=n)))
        dist = signs @ r
        p = float(np.mean(np.abs(dist - centre) >= obs - 1e-9))
    rbc = (2 * w_plus - total) / total
    return n, w_plus, min(p, 1.0), float(rbc)


def holm(pvals):
    p = np.asarray(pvals, float)
    out = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    idx = np.where(ok)[0][np.argsort(p[ok], kind="mergesort")]
    m = len(idx)
    running = 0.0
    for k, i in enumerate(idx):
        running = max(running, min(1.0, (m - k) * p[i]))
        out[i] = running
    return out


def fisher_exact(a, b, c, d):
    """Two-sided Fisher's exact test for [[a, b], [c, d]] (as SciPy)."""
    n1, n2, k = a + b, c + d, a + c
    lo, hi = max(0, k - n2), min(k, n1)

    def pmf(x):
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n1 + n2, k)

    p_obs = pmf(a)
    return min(1.0, sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= p_obs * (1 + 1e-7)))


# ---------------------------------------------------------------- loading
def load(steps_path, deliv_path):
    st = pd.read_parquet(steps_path)
    miss = [c for c in REQUIRED_STEP_COLS if c not in st.columns]
    if miss:
        sys.exit(f"step log is missing columns: {miss}")
    st = st.sort_values(KEY + ["step"]).reset_index(drop=True)
    dl = None
    if deliv_path:
        dl = pd.read_parquet(deliv_path)
        miss = [c for c in REQUIRED_DELIV_COLS if c not in dl.columns]
        if miss:
            sys.exit(f"delivery log is missing columns: {miss}")
    return st, dl


def dead_traffic(st, dl, min_bytes=LIVE_BYTES):
    """Episodes where the testbed stopped sending benign traffic.

    Uses the pre-event phase only, before any surge or swap, so that blocking
    by an arm cannot look like a dead generator. Background hosts are all
    hosts except probes and attackers (C1 surge hosts are ordinary benign
    hosts before the event). A host is live if it delivered more than
    min_bytes, or if an arm mitigated it pre-event. At least half of the
    background hosts must be live. Returns failing keys ordered by run time."""
    pre_st = st[~st["event_fired"].astype(bool)]
    blocked = pre_st.groupby(KEY)["mitigated"].agg(lambda m: set().union(*m.map(hosts)))
    roles = st.groupby(KEY).agg(a0=("attackers", "first"), t=("t_wall", "min"))
    pre = dl[dl.phase == "pre"].groupby(KEY)
    bad = []
    for k, d in pre:
        if k not in roles.index:
            continue
        r = roles.loc[k]
        live = set(d.src[d.bytes > min_bytes]) | blocked.get(k, set())
        bg = set(d.src) - PROBE_HOSTS - hosts(r.a0)
        if len(bg & live) * 2 < len(bg):
            bad.append((r.t, k))
    return [k for _, k in sorted(bad)]


def validate(st, dl):
    """Checks that must hold before any number is trusted. Returns notes."""
    notes, errors = [], []
    for k, g in st.groupby(KEY):
        steps = g["step"].to_numpy()
        if not (steps[0] == 0 and np.all(np.diff(steps) == 1)):
            errors.append(f"{k}: steps are not 0..n-1 without gaps")
        ev = g.loc[g["event_fired"].astype(bool), "step"]
        if ev.empty:
            errors.append(f"{k}: the event never fired")
    # Pairing: every arm on the same (cond, seed, ep) has the same event step and roles.
    first = st.groupby(KEY).agg(
        event=("step", lambda s: int(s[st.loc[s.index, "event_fired"].astype(bool)].min())
               if st.loc[s.index, "event_fired"].any() else -1),
        att0=("attackers", "first"), att1=("attackers", "last"),
        former=("former_attackers", "last"), surge=("surge_hosts", "last"),
        n=("step", "size"),
    ).reset_index()
    for c in ["att0", "att1", "former", "surge"]:
        first[c] = first[c].map(lambda s: ",".join(sorted(hosts(s))))
    for k, g in first.groupby(["cond", "seed", "ep"]):
        for c in ["event", "att0", "att1", "former", "surge", "n"]:
            if g[c].nunique() > 1:
                errors.append(f"pairing broken at {k}: '{c}' differs across arms "
                              f"({dict(zip(g.agent, g[c]))})")
    mitig = set().union(*st["mitigated"].map(hosts)) if len(st) else set()
    if mitig & PROBE_HOSTS:
        errors.append(f"probe hosts were mitigated: {sorted(mitig & PROBE_HOSTS)}; "
                      "PROBE_HOSTS is wrong or probe protection failed")
    if dl is not None:
        sz = dl[dl.pkts > 0].groupby("src").apply(
            lambda g: g.bytes.sum() / g.pkts.sum(), include_groups=False)
        big = [h for h in PROBE_HOSTS if h in sz.index and sz[h] > 200]
        if big:
            errors.append(f"PROBE_HOSTS {big} send large packets; check the probe list")
        small = [h for h in sz.index if sz[h] < 200 and h not in PROBE_HOSTS]
        if small:
            notes.append(f"hosts with probe-sized packets not in PROBE_HOSTS: {small}")
        dead = dead_traffic(st, dl)
        if dead:
            errors.append(f"{len(dead)} episode(s) have no working benign background or surge "
                          f"traffic (testbed traffic generators stopped); first by run time: "
                          f"{dead[0]}. Re-run those episodes; their numbers would be meaningless")
        have = set(map(tuple, dl[KEY].drop_duplicates().to_numpy()))
        lost = [k for k in map(tuple, st[KEY].drop_duplicates().to_numpy()) if k not in have]
        if lost:
            notes.append(f"{len(lost)} episodes have no delivery counters")
    arms = sorted(st.agent.unique())
    if set(arms) - set(ARMS):
        notes.append(f"unexpected arms ignored: {sorted(set(arms) - set(ARMS))}")
    ns = st.groupby(["cond", "agent"]).seed.nunique()
    if ns.min() < MIN_SEEDS_FOR_TESTS:
        notes.append(f"only {ns.min()} seed(s) per arm: paired tests need at least "
                     f"{MIN_SEEDS_FOR_TESTS} to reach the Holm threshold, so p-values "
                     "are reported but cannot be significant")
    if errors:
        for e in errors:
            print("ERROR:", e, file=sys.stderr)
        sys.exit(f"{len(errors)} validation error(s); no results written")
    return notes


# ---------------------------------------------------------------- metrics
def step_roles(st):
    """Per-step rule counts by the role of the mitigated host at that step."""
    rows = []
    for r in st.itertuples(index=False):
        m = hosts(r.mitigated)
        att, former, surge = hosts(r.attackers), hosts(r.former_attackers), hosts(r.surge_hosts)
        n_att = len(m & att)
        n_former = len(m & former - att)
        n_surge = len(m & surge - att)
        n_other = len(m) - n_att - n_former - n_surge
        rows.append((len(m), n_att, n_former, n_surge, n_other))
    a = np.array(rows, float).reshape(-1, 5)
    out = st.copy()
    out["n_rules"], out["n_att"], out["n_former"], out["n_surge"], out["n_other"] = a.T
    tot = out["n_rules"].replace(0, np.nan)
    # Collateral as the environment defines it: share of active rules on hosts that
    # are not current attackers; 0 when no rule is active.
    out["coll"] = ((out.n_rules - out.n_att) / tot).fillna(0.0)
    out["coll_role"] = ((out.n_former + out.n_surge) / tot).fillna(0.0)
    out["coll_other"] = (out.n_other / tot).fillna(0.0)
    return out


def recovery(S, event):
    """ISASF adaptation latency (scripts/analyze.py recovery_latency), with the
    IIPEM event step in place of the regime switch. Returns (recovered, latency):
    (True, t) if it recovers t steps after the event, (False, nan) if censored,
    (None, nan) if undefined (event before BASELINE_WIN, or zero baseline)."""
    S = np.asarray(S, float)
    if event < BASELINE_WIN:
        return None, np.nan
    base = float(S[event - BASELINE_WIN:event].mean())
    if base <= 1e-6:
        return None, np.nan
    roll = pd.Series(S[event:]).rolling(ROLL, min_periods=1).mean().to_numpy()
    ok = roll >= RECOVER_FRAC * base
    for t in range(len(ok) - HOLD + 1):
        if ok[t:t + HOLD].all():
            return True, float(t)
    return False, np.nan


def delivery_metrics(d, pre_steps, post_steps, att0, att1, former, surge, dt):
    """Per-step delivered bytes by role, from the victim's per-source counters."""
    nan = float("nan")
    out = dict(benign_delivery_other=nan, surge_delivery=nan, former_attacker_delivery=nan,
               attack_mb_pre=nan, attack_mb_post=nan, attack_delivery_ratio=nan)
    if d is None or d.empty:
        return out
    b = d.pivot_table(index="src", columns="phase", values="bytes", aggfunc="sum").reindex(
        columns=["pre", "post"]).fillna(0.0)
    pre = b["pre"] / pre_steps
    post = b["post"] / post_steps
    srcs = set(b.index) - PROBE_HOSTS
    other = sorted(srcs - att0 - att1 - former - surge)

    def ratio(num, den):
        return float(num / den) if den > 0 else nan

    out["benign_delivery_other"] = ratio(post[other].sum(), pre[other].sum())
    if surge:
        # Share of what the surge hosts offer after the event (SURGE_RATE_MBPS each).
        # Not post/pre: before the event they send ordinary benign traffic, which
        # can be tiny and makes a post/pre ratio unstable.
        s = sorted(surge & srcs)
        offered = SURGE_RATE_MBPS * 1e6 / 8 * dt * post_steps
        out["surge_delivery"] = ratio(b.loc[s, "post"].mean(), offered)
    if former:
        f = sorted(former & srcs)
        # Former attackers have no benign "before" level, so compare each with the
        # mean per-host level of the hosts that stayed benign before the event.
        out["former_attacker_delivery"] = ratio(post[f].mean(), pre[other].mean())
        a0, a1 = sorted(att0 & srcs), sorted(att1 & srcs)
        out["attack_mb_pre"] = float(b.loc[a0, "pre"].sum() / 1e6)
        out["attack_mb_post"] = float(b.loc[a1, "post"].sum() / 1e6)
        out["attack_delivery_ratio"] = ratio(post[a1].sum(), pre[a0].sum())
    return out


def block_metrics(g, ev, universe, att0, att1, former, surge):
    """Rule counts per host role, before and after the event (status step 22).

    g: the episode's step rows (sorted by step). universe: the episode's source
    hosts from the delivery log, or None (then everything here is NaN).
    Pre = steps TRANSIENT..ev-1 (the reset steps are skipped); post = steps ev..end.
    C1: per-host blocking of the surge hosts, also before the surge (the surge
    hosts are known from the post-event rows), against the other benign hosts.
    C2: a rule on a former attacker is "held over" if that host had a rule at
    step ev-1 and at every step since; any other rule on a former attacker is a
    re-added rule. Held-over rules can only form a prefix of the post phase."""
    nan = float("nan")
    out = dict(other_blocked_pre=nan, other_blocked_post=nan, other_host_block_pre=nan,
               other_host_block_post=nan)
    if universe is None:
        return out
    other = universe - PROBE_HOSTS - att0 - att1 - former - surge
    M = [hosts(x) for x in g.mitigated]
    steps = g.step.to_numpy()
    pre = [i for i, t in enumerate(steps) if TRANSIENT <= t < ev]
    post = [i for i, t in enumerate(steps) if t >= ev]

    def per_host(idx, group):
        if not idx or not group:
            return nan
        return sum(len(M[i] & group) for i in idx) / (len(idx) * len(group))

    def count(idx, group):
        return sum(len(M[i] & group) for i in idx) / len(idx) if idx else nan

    out.update(other_blocked_pre=count(pre, other), other_blocked_post=count(post, other),
               other_host_block_pre=per_host(pre, other), other_host_block_post=per_host(post, other))
    if surge:
        sp, so = per_host(pre, surge), per_host(post, surge)
        out.update(surge_host_block_pre=sp, surge_host_block_post=so,
                   surge_rise_minus_other=(so - sp) - (out["other_host_block_post"]
                                                       - out["other_host_block_pre"]))
    if former:
        i_last = [i for i, t in enumerate(steps) if t == ev - 1]
        alive = (M[i_last[0]] & former) if i_last else set()
        held_steps, reflag_only, reflag_hs = 0, 0, 0
        for i in post:
            alive = alive & M[i]
            re = (M[i] & former) - alive
            reflag_hs += len(re)
            if alive:
                held_steps += 1
            elif re:
                reflag_only += 1
        out.update(held_steps=held_steps, reflag_only_steps=reflag_only,
                   former_reflag_host_rate=reflag_hs / (len(post) * len(former)) if post else nan)
    return out


def episode_metrics(st, dl):
    st = step_roles(st)
    rows = []
    for k, g in st.groupby(KEY, sort=False):
        g = g.reset_index(drop=True)
        n = len(g)
        ev = int(g.loc[g.event_fired.astype(bool), "step"].min())
        pre, post = g[g.step < ev], g[g.step >= ev]
        S = g["S"].to_numpy()
        att0, att1 = hosts(g.attackers.iloc[0]), hosts(g.attackers.iloc[-1])
        former, surge = hosts(g.former_attackers.iloc[-1]), hosts(g.surge_hosts.iloc[-1])
        rec, lat = recovery(S, ev + RECOVERY_OFFSET)
        m = dict(zip(KEY, k))
        m.update(
            event_step=ev, n_steps=n,
            service_mean=float(S.mean()),
            service_pre=float(pre.S.mean()), service_post=float(post.S.mean()),
            worst_pre=float(g.S[(g.step >= TRANSIENT) & (g.step < ev)].min()),
            worst_post=float(g.S[g.step >= ev + TRANSIENT].min()),
            worst_isasf=float(g.S[g.step >= TRANSIENT].min()),
            collateral_pre=float(pre.coll.mean()), collateral_post=float(post.coll.mean()),
            collateral_role_post=float(post.coll_role.mean()),
            collateral_other_pre=float(pre.coll_other.mean()),
            collateral_other_post=float(post.coll_other.mean()),
            benign_rule_share_pre=float((pre.n_rules - pre.n_att > 0).mean()),
            benign_rule_share_post=float((post.n_rules - post.n_att > 0).mean()),
            env_collateral_mean=float(g.collateral.mean()),
            recovered=rec, adapt_latency=lat,
            infer_ms_median=float(g.infer_ms.iloc[1:].median()),
            infer_ms_p95=float(g.infer_ms.iloc[1:].quantile(0.95)),
            apply_ms_median=float(g.apply_ms.median()),
            apply_ms_p95=float(g.apply_ms.quantile(0.95)),
            ctl_cpu_mean=float(g.ctl_cpu_pct.mean()),
            ctl_rss_max=float(g.ctl_rss_mb.max()),
        )
        if surge:
            m["surge_block_share"] = float((post.n_surge > 0).mean())
        if former:
            stale = post.n_former > 0
            m["stale_steps"] = int(stale.sum())
            clear = np.where(~stale.to_numpy())[0]
            m["stale_clear_latency"] = int(clear[0]) if len(clear) else len(post)
        d = None
        if dl is not None:
            sel = np.ones(len(dl), bool)
            for c, v in zip(KEY, k):
                sel &= (dl[c] == v).to_numpy()
            d = dl[sel]
        dt = float(np.median(np.diff(g.t_wall))) if "t_wall" in g and n > 1 else 0.5
        m["step_s"] = dt
        m.update(delivery_metrics(d, len(pre), len(post), att0, att1, former, surge, dt))
        universe = set(d.src) if d is not None and len(d) else None
        m.update(block_metrics(g, ev, universe, att0, att1, former, surge))
        m["S_zero_at_event"] = float(g.S[g.step == ev].iloc[0] == 0.0)
        rows.append(m)
    em = pd.DataFrame(rows)
    em["agent"] = pd.Categorical(em.agent, [a for a in ARMS if a in set(em.agent)])
    return em.sort_values(KEY).reset_index(drop=True), st


def seed_metrics(em):
    num = [c for c in em.columns if c not in KEY and pd.api.types.is_numeric_dtype(em[c])]
    sm = em.groupby(["cond", "agent", "seed"], observed=True)[num].mean().reset_index()
    sm["n_episodes"] = em.groupby(["cond", "agent", "seed"], observed=True).size().to_numpy()
    return sm


def summarise(sm, reps):
    rows = []
    for cond in CONDS:
        for fam_name, col, only, label, _ in FAMILIES:
            if only and only != cond or col not in sm:
                continue
            for arm in ARMS:
                x = sm[(sm.cond == cond) & (sm.agent == arm)][col].to_numpy(float)
                x = x[~np.isnan(x)]
                if len(x) == 0:
                    continue
                lo, hi = bootstrap_ci(x, reps)
                rows.append(dict(cond=cond, family=fam_name, parameter=col, label=label,
                                 agent=arm, n_seeds=len(x),
                                 iqm=iqm(x), ci_low=lo, ci_high=hi, mean=float(x.mean()),
                                 median=float(np.median(x))))
    return pd.DataFrame(rows)


def paired_tests(sm):
    rows = []
    for cond in CONDS:
        for fam_name, col, only, label, better in FAMILIES:
            if only and only != cond or col not in sm:
                continue
            ref = sm[(sm.cond == cond) & (sm.agent == REF)].set_index("seed")[col]
            fam = []
            for arm in ARMS[1:]:
                oth = sm[(sm.cond == cond) & (sm.agent == arm)].set_index("seed")[col]
                j = pd.concat([ref, oth], axis=1, keys=["ref", "oth"]).dropna()
                diff = (j.ref - j.oth).to_numpy()
                n_nz, w, p, rbc = wilcoxon_exact(diff) if len(diff) else (0, np.nan, np.nan, np.nan)
                fam.append(dict(cond=cond, family=fam_name, parameter=col, label=label,
                                comparison=f"{REF} vs {arm}",
                                n_pairs=len(diff), n_nonzero=n_nz,
                                median_diff=float(np.median(diff)) if len(diff) else np.nan,
                                W_plus=w, p_exact=p if n_nz else np.nan,
                                rank_biserial=rbc if n_nz else np.nan,
                                higher_is_better=better))
            padj = holm([f["p_exact"] for f in fam])
            for f, pa in zip(fam, padj):
                f["p_holm"] = pa
                f["significant_0.05"] = bool(pa < 0.05) if not np.isnan(pa) else False
            rows += fam
    return pd.DataFrame(rows)


def within_arm_tests(sm):
    """Exploratory: paired comparison of two measures within each arm (WITHIN_ARM),
    exact Wilcoxon on the seed-level difference b - a, Holm over the arms that
    have a defined p-value."""
    rows = []
    for cond, pre_c, post_c, label in WITHIN_ARM:
        sub = sm[sm.cond == cond]
        if pre_c not in sub or post_c not in sub:
            continue
        fam = []
        for arm in ARMS:
            g = sub[sub.agent == arm][[pre_c, post_c]].dropna()
            if g.empty:
                continue
            diff = (g[post_c] - g[pre_c]).to_numpy()
            n_nz, w, p, rbc = wilcoxon_exact(diff)
            fam.append(dict(cond=cond, measure=label, agent=arm, n_pairs=len(diff),
                            n_nonzero=n_nz, pre_iqm=iqm(g[pre_c]), post_iqm=iqm(g[post_c]),
                            median_change=float(np.median(diff)),
                            seeds_up=int((np.round(diff, 9) > 0).sum()),
                            p_exact=p if n_nz else np.nan, rank_biserial=rbc if n_nz else np.nan))
        for f, pa in zip(fam, holm([f["p_exact"] for f in fam])):
            f["p_holm"] = pa
        rows += fam
    return pd.DataFrame(rows)


def recovery_tests(em):
    rows = []
    for cond in CONDS:
        e = em[(em.cond == cond) & em.recovered.notna()].copy()
        e["recovered"] = e.recovered.astype(bool)
        r = e[e.agent == REF]
        fam = []
        for arm in ARMS[1:]:
            o = e[e.agent == arm]
            if r.empty or o.empty:
                continue
            a, b = int((~r.recovered).sum()), int(r.recovered.sum())
            c, d = int((~o.recovered).sum()), int(o.recovered.sum())
            fam.append(dict(cond=cond, comparison=f"{REF} vs {arm}", ref_never=a, ref_n=a + b,
                            other_never=c, other_n=c + d, p_fisher=fisher_exact(a, b, c, d),
                            ref_latency_median=float(r.adapt_latency.median()),
                            other_latency_median=float(o.adapt_latency.median())))
        for f, pa in zip(fam, holm([f["p_fisher"] for f in fam])):
            f["p_holm"] = pa
        rows += fam
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures
COLORS = {"A2": "#2a78d6", "A1": "#eb6834", "A3": "#1baf7a", "static": "#eda100"}
STYLES = {"A2": ("-", "o"), "A1": ("--", "s"), "A3": ("-.", "^"), "static": (":", "D")}
COND_LABEL = {"C1_surge": "C1: benign surge", "C2_swap": "C2: attacker swap"}


def figures(st_roles, em, sm, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ["matplotlib not installed: figures skipped"]
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e",
                         "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
                         "legend.frameon": False, "savefig.dpi": 200})
    st = st_roles.copy()
    # Episode-level roles, so that rules on the future surge hosts are visible before
    # the surge too (the step log names the surge hosts only from the event on).
    last = st.groupby(KEY)[["surge_hosts", "former_attackers", "attackers"]].transform("last")
    surge_ep = last.surge_hosts.map(hosts)
    m_sets = st.mitigated.map(hosts)
    att_now = st.attackers.map(hosts)
    former_now = st.former_attackers.map(hosts)
    st["other_ct"] = [len(m - a - f - sg) for m, a, f, sg in
                      zip(m_sets, att_now, former_now, surge_ep)]
    st["role_rule"] = [float(bool(m & sg) or bool(m & (f - a))) for m, a, f, sg in
                       zip(m_sets, att_now, former_now, surge_ep)]
    panels = [("S", "Probe success"), ("other_ct", "Other benign hosts with a rule"),
              ("role_rule", "Episodes with a rule on a surge host\n(C1, all steps) / former attacker (C2)")]
    fig, axes = plt.subplots(len(CONDS), len(panels), figsize=(7.2, 2.7), sharex=True)
    for i, cond in enumerate(CONDS):
        for j, (col, title) in enumerate(panels):
            ax = axes[i, j]
            for arm in ARMS:
                g = st[(st.cond == cond) & (st.agent == arm)]
                if g.empty:
                    continue
                y = g.groupby("step")[col].mean()
                ls, mk = STYLES[arm]
                ax.plot(y.index, y.values, ls, color=COLORS[arm], lw=1.4, label=arm,
                        marker=mk, markevery=10, ms=3.5)
            ev = em[em.cond == cond].event_step
            if len(ev):
                ax.axvline(ev.iloc[0], color="#52514e", lw=0.8)
            if col != "other_ct":
                ax.set_ylim(-0.05, 1.05)
            if i == 0:
                ax.set_title(title, fontsize=8)
            if j == 0:   # two lines, so the row labels do not run into each other
                ax.set_ylabel(COND_LABEL[cond].replace(": ", ":\n"))
            if i == len(CONDS) - 1:
                ax.set_xlabel("Step (0.5 s)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    if h:   # one legend above the panels, so it covers no data
        fig.legend(h, lab, loc="upper center", ncol=len(lab), fontsize=7,
                   bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_timeseries.{ext}")
    plt.close(fig)

    # Planned parameters only (plan Section 7); the step-22 measures are exploratory and
    # are reported in their own table, not in this figure.
    def worst_title(cond):
        e = em[em.cond == cond]
        if e.empty:
            return "Worst window after the event"
        ev, n = int(e.event_step.iloc[0]), int(e.n_steps.iloc[0])
        return f"Worst window,\nsteps {ev + TRANSIENT}-{n - 1}"
    params = {"C1_surge": [("service_post", "Service after event"),
                           ("surge_delivery", "Surge-host delivery\n(share of 4.0 Mbps offered)"),
                           ("benign_delivery_other", "Other benign delivery\n(post/pre)"),
                           ("surge_block_share", "Share of post steps with\na rule on a surge host")],
              "C2_swap": [("service_post", "Service after event"),
                          ("worst_post", worst_title("C2_swap")),
                          ("benign_delivery_other", "Other benign delivery\n(post/pre)"),
                          ("stale_steps", "Post steps with a rule on\na former attacker (of 50)")]}
    # Both service panels on one honest scale (a 0.98-0.988 axis exaggerates 0.004).
    sv = sm["service_post"].dropna() if "service_post" in sm else pd.Series(dtype=float)
    s_lo = float(np.floor((sv.min() - 0.01) * 50) / 50) if len(sv) else 0.0
    fig, axes = plt.subplots(len(CONDS), 4, figsize=(7.2, 2.9))
    for i, cond in enumerate(CONDS):
        for j, (col, title) in enumerate(params[cond]):
            ax = axes[i, j]
            if col not in sm:
                ax.set_visible(False)
                continue
            for x, arm in enumerate(ARMS):
                v = sm[(sm.cond == cond) & (sm.agent == arm)][col].dropna().to_numpy()
                if len(v) == 0:
                    continue
                jit = np.linspace(-0.12, 0.12, len(v)) if len(v) > 1 else np.zeros(1)
                ax.scatter(x + jit, v, s=10, color=COLORS[arm], alpha=0.55, linewidths=0)
                c, (lo, hi) = iqm(v), bootstrap_ci(v)   # default reps: same interval as summary_iqm.csv
                if not np.isnan(lo):
                    ax.vlines(x + 0.3, lo, hi, color="#0b0b0b", lw=1)
                ax.plot([x + 0.3], [c], marker=STYLES[arm][1], color=COLORS[arm], ms=5,
                        markeredgecolor="#0b0b0b", markeredgewidth=0.6)
            ax.set_xticks(range(len(ARMS)), ARMS)
            if col == "service_post":
                ax.set_ylim(s_lo, 1.0)
            elif col == "worst_post":
                ax.set_ylim(-0.05, 1.05)
            ax.set_title(title, fontsize=8)
            if j == 0:
                ax.set_ylabel(COND_LABEL[cond])
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_seed_parameters.{ext}")
    plt.close(fig)
    return []


# ---------------------------------------------------------------- report
def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    v = round(float(x), nd)
    return f"{v + 0.0:.{nd}f}"          # + 0.0 turns -0.0 into 0.0


def _param_table(q, family):
    q = q[q.family == family]
    L = ["| Parameter | " + " | ".join(ARMS) + " |", "|---|" + "---|" * len(ARMS)]
    for col in dict.fromkeys(q.parameter):
        r = q[q.parameter == col].set_index("agent")
        cells = []
        for a in ARMS:
            if a in r.index:
                x = r.loc[a]
                ci = "" if np.isnan(x.ci_low) else f" [{fmt(x.ci_low)}, {fmt(x.ci_high)}]"
                cells.append(fmt(x.iqm) + ci)
            else:
                cells.append("n/a")
        L.append(f"| {r.label.iloc[0]} | " + " | ".join(cells) + " |")
    return L + [""]


def _test_table(tests, family):
    L = ["| Condition | Parameter | Comparison | n (non-zero) | median diff | p | p (Holm) | r |",
         "|---|---|---|---|---|---|---|---|"]
    for t in tests[tests.family == family].itertuples():
        L.append(f"| {t.cond} | {t.parameter} | {t.comparison} | {t.n_pairs} ({t.n_nonzero}) | "
                 f"{fmt(t.median_diff)} | {fmt(t.p_exact, 4)} | {fmt(t.p_holm, 4)} | "
                 f"{fmt(t.rank_biserial, 2)} |")
    return L + [""]


def write_summary(path, notes, em, iq, tests, rec, within, fig_notes, args):
    L = ["# IIPEM analysis summary", "",
         f"Inputs: `{args.steps}`" + (f", `{args.delivered}`" if args.delivered else ""),
         f"Episodes: {len(em)}; seeds per arm: "
         + ", ".join(f"{c} {a}: {n}" for (c, a), n in
                     em.groupby(['cond', 'agent'], observed=True).seed.nunique().items()), "",
         "## Checks", "", "- Steps contiguous, event fired, roles and event step identical across arms "
         "for every (cond, seed, episode): passed.",
         "- Probe hosts never mitigated: passed."]
    L += [f"- Note: {n}" for n in notes + fig_notes]
    # Measurement artefacts, computed here rather than typed.
    z = em.groupby(["cond", "agent"], observed=True).S_zero_at_event.mean()
    dip_only = recovery(np.r_[np.ones(BASELINE_WIN), 0.0, np.ones(HOLD + ROLL)],
                        BASELINE_WIN + RECOVERY_OFFSET)[1]
    L += ["", "## Reading the results", "",
          f"- The event restarts the probe clients, so S = 0 at the event step in "
          f"{z.min():.0%} to {z.max():.0%} of episodes per arm and condition. Consequences: "
          f"`worst_isasf` (ISASF minS over the whole episode) is close to 0 for every arm, as it "
          f"is for every ISASF E1/E3 episode; `worst_post` skips the event step and the next "
          f"({TRANSIENT} steps) for that reason. The recovery clock starts {RECOVERY_OFFSET} step "
          f"after the event, where ISASF's starts (its switch step is one row after the dip), so "
          f"the dip alone gives a latency of {dip_only:g}.",
          "- A3 in C1 is a forced no-op: its label is \"none\" in R0 and `agents/oracle_ddqn.py` "
          "(lines 48-49) returns a no-op without calling the network. Its zero blocking in C1 holds "
          "by construction, and its C1 inference and rule-installation times measure a branch.",
          "- Collateral is a share of the active rules. Report the count of blocked benign hosts "
          "(exploratory table) beside it: the share and the count can move by different amounts.",
          "- `n (non-zero)`: seeds whose difference is not zero. The exact test uses only those.",
          "", "Worst window and recovery follow ISASF `scripts/analyze.py`: minS from step "
          f"{TRANSIENT}; recovery when the {ROLL}-step rolling mean of S returns to {RECOVER_FRAC} x the "
          f"mean of the {BASELINE_WIN} steps before the start and holds {HOLD} steps; the start is "
          f"event + {RECOVERY_OFFSET}, ISASF's switch step (censored if never). `worst_post` additionally skips the "
          f"{TRANSIENT} steps after the event, where every arm loses its probes to the restart.", ""]
    L += ["## Planned parameters (plan Section 7; IQM over seeds, bootstrap 95% interval)", ""]
    for cond in CONDS:
        q = iq[iq.cond == cond]
        if not q.empty:
            L += [f"### {COND_LABEL[cond]}", ""] + _param_table(q, "planned")
    L += ["## Planned paired tests (exact Wilcoxon, Holm within each family of 3)", ""]
    L += _test_table(tests, "planned")
    L += ["## Exploratory parameters (added 2 Oct 2026 after the results were seen; "
          "not pre-registered)", "",
          "Pre = steps " + f"{TRANSIENT}" + " to the step before the event; post = the event step "
          "to the end. Other benign hosts exclude probes, attackers (before and after), former "
          "attackers and surge hosts. A held-over rule (C2) is on a former attacker that had a "
          "rule at the step before the swap and at every step since; any other rule on a former "
          "attacker is a re-added rule.", ""]
    for cond in CONDS:
        q = iq[iq.cond == cond]
        if not q.empty and (q.family == "exploratory").any():
            L += [f"### {COND_LABEL[cond]}", ""] + _param_table(q, "exploratory")
    L += ["## Exploratory paired tests (A2 vs each arm, exact Wilcoxon, Holm within each family "
          "of 3)", ""] + _test_table(tests, "exploratory")
    if within is not None and len(within):
        L += ["## Exploratory within-arm comparisons (b - a, exact Wilcoxon, Holm over arms)", "",
              "| Condition | Measure | Arm | a (IQM) | b (IQM) | median b - a | seeds up / n | "
              "n (non-zero) | p | p (Holm) |", "|---|---|---|---|---|---|---|---|---|---|"]
        for t in within.itertuples():
            L.append(f"| {t.cond} | {t.measure} | {t.agent} | {fmt(t.pre_iqm)} | {fmt(t.post_iqm)} | "
                     f"{fmt(t.median_change)} | {t.seeds_up}/{t.n_pairs} | {t.n_nonzero} | "
                     f"{fmt(t.p_exact, 4)} | {fmt(t.p_holm, 4)} |")
        L.append("")
    L += ["## Recovery (Fisher's exact test on never-recovered episodes, Holm)", "",
          "| Condition | Comparison | A2 never / n | other never / n | p | p (Holm) |",
          "|---|---|---|---|---|---|"]
    for t in rec.itertuples():
        L.append(f"| {t.cond} | {t.comparison} | {t.ref_never}/{t.ref_n} | {t.other_never}/{t.other_n} "
                 f"| {fmt(t.p_fisher, 4)} | {fmt(t.p_holm, 4)} |")
    path.write_text("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--steps", required=True, help="per-step parquet from evaluate_iipem.py")
    ap.add_argument("--delivered", help="per-source delivery parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=10000, help="bootstrap replicates")
    ap.add_argument("--no-figs", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    st, dl = load(args.steps, args.delivered)
    st = st[st.agent.isin(ARMS) & st.cond.isin(CONDS)]
    notes = validate(st, dl)
    em, st_roles = episode_metrics(st, dl)
    if not np.allclose(st_roles.coll, st_roles.collateral, atol=1e-9):
        sys.exit("collateral recomputed from roles does not match the logged collateral")
    notes.append("collateral recomputed from mitigated hosts and roles matches the logged "
                 "collateral at every step")
    sm = seed_metrics(em)
    iq = summarise(sm, args.boot)
    tests = paired_tests(sm)
    rec = recovery_tests(em)
    within = within_arm_tests(sm)
    fig_notes = [] if args.no_figs else figures(st_roles, em, sm, out)

    em.to_csv(out / "episode_metrics.csv", index=False)
    sm.to_csv(out / "seed_metrics.csv", index=False)
    iq.to_csv(out / "summary_iqm.csv", index=False)
    tests.to_csv(out / "paired_tests.csv", index=False)
    rec.to_csv(out / "recovery_fisher.csv", index=False)
    within.to_csv(out / "within_arm_tests.csv", index=False)
    write_summary(out / "summary.md", notes, em, iq, tests, rec, within, fig_notes, args)
    print(f"{len(em)} episodes analysed; results in {out}/")
    for n in notes + fig_notes:
        print("note:", n)


if __name__ == "__main__":
    main()
