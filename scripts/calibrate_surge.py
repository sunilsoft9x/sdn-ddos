"""WP2: calibrate the IIPEM benign surge (C1) before the main runs.

The surge only tests anything if the suspect rule flags the surge hosts. The
rule flags a source whose packet rate exceeds three times the median rate of
all active sources, probes included, and the surge hosts share the 10 Mbps
victim link. So this sweeps the number of surge hosts and their iperf3 rate:

    n_hosts 1, 2, 3  x  rate 2, 3, 4, 6 Mbps  x  3 seeds  = 36 episodes

R0 (no attack), no mitigation (action 0 throughout), surge at step 10, 40
steps per episode. Post-surge steps are 12 to 40: step 11 is the restart
window, as in the smoke run.

PRE-REGISTERED CHOICE (fixed before the sweep, reported in the paper):
  a combination passes if every surge host, in every seed, is flagged in at
  least 90% of post-surge steps. Take the largest n_hosts (3, then 2, then 1)
  with a passing rate, and the lowest passing rate for that n_hosts. If none
  passes, C1 is a negative calibration result and the paper rests on C2.

Run from the repo root, as root, with ODL up and no other Mininet running:

    sudo mn -c
    sudo ~/venv/bin/python scripts/calibrate_surge.py

About 15 to 20 minutes. Per-step rows go to runs/calibrate_surge.parquet.
Re-running the summary on saved rows: scripts/calibrate_surge.py --summarise-only
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

N_HOSTS = (1, 2, 3)
RATES_MBPS = (2.0, 3.0, 4.0, 6.0)
SEEDS = (2000001, 2000002, 2000003)
AT = 10
STEPS = 40
FIRST_POST = AT + 2
PASS_SHARE = 0.90
OUT = pathlib.Path("runs/calibrate_surge.parquet")


def run_episode(env, n_hosts, rate, seed):
    options = {"schedule": [["R0", 100]],
               "surge": {"n_hosts": n_hosts, "rate_mbps": rate, "at": AT}}
    env.reset(seed=seed, options=options)
    rows = []
    for t in range(1, STEPS + 1):
        _, _, _, _, info = env.step(0)
        det = env.detector
        terms = det.snapshot_terms()                 # median, threshold, suspects
        current = det._history[-1][1] if det._history else {}
        long_means = det._long_window_means()
        flagged = set(terms["suspects"])
        surge_ips = [env._src_ip(h) for h in env._surge_hosts]
        other_flagged = flagged - set(surge_ips)
        base = {"n_hosts": n_hosts, "rate_mbps": rate, "seed": seed, "step": t,
                "event_fired": info["event_fired"], "median_pps": terms["median_long_rate"],
                "threshold_pps": terms["threshold"], "n_active": terms["n_active"],
                "other_flagged": len(other_flagged), "S": info["reward_terms"]["S"],
                "probe_n": info["probe_n"]}
        for host, ip in zip(env._surge_hosts, surge_ips):
            rows.append({**base, "host": host,
                         "current_pps": current.get(ip, 0.0),
                         "long_pps": long_means.get(ip, 0.0),
                         "flagged": ip in flagged})
    return rows


def summarise(df):
    """Per-combination results and the pre-registered choice. Pure function."""
    post = df[df.step >= FIRST_POST]
    per_host = (post.groupby(["n_hosts", "rate_mbps", "seed", "host"])
                .agg(flag_share=("flagged", "mean"), current_pps=("current_pps", "mean"),
                     threshold_pps=("threshold_pps", "mean"))
                .reset_index())
    per_host["ratio"] = per_host.current_pps / per_host.threshold_pps.where(per_host.threshold_pps > 0)
    steps = post.drop_duplicates(["n_hosts", "rate_mbps", "seed", "step"])
    other = steps.groupby(["n_hosts", "rate_mbps"]).other_flagged.mean()
    service = steps.groupby(["n_hosts", "rate_mbps"]).S.mean()
    combo = (per_host.groupby(["n_hosts", "rate_mbps"])
             .agg(min_flag_share=("flag_share", "min"), mean_flag_share=("flag_share", "mean"),
                  mean_ratio=("ratio", "mean"), n_host_seeds=("flag_share", "size"))
             .reset_index())
    combo["other_flagged_per_step"] = [other[(n, r)] for n, r in zip(combo.n_hosts, combo.rate_mbps)]
    combo["mean_S"] = [service[(n, r)] for n, r in zip(combo.n_hosts, combo.rate_mbps)]
    combo["passes"] = combo.min_flag_share >= PASS_SHARE
    choice = None
    for n in sorted(combo.n_hosts.unique(), reverse=True):
        ok = combo[(combo.n_hosts == n) & combo.passes]
        if len(ok):
            choice = (int(n), float(ok.rate_mbps.min()))
            break
    return combo, choice


def report(combo, choice):
    print(f"\n{'n':>2} {'Mbps':>5} {'min flag':>9} {'mean flag':>10} {'pps/thr':>8} "
          f"{'other flagged/step':>19} {'mean S':>7}  pass")
    for _, r in combo.sort_values(["n_hosts", "rate_mbps"]).iterrows():
        print(f"{r.n_hosts:>2} {r.rate_mbps:>5.1f} {r.min_flag_share:>9.3f} {r.mean_flag_share:>10.3f} "
              f"{r.mean_ratio:>8.2f} {r.other_flagged_per_step:>19.2f} {r.mean_S:>7.3f}  "
              f"{'yes' if r.passes else 'no'}")
    print("\nmin flag: lowest share of post-surge steps in which a surge host was flagged,")
    print("over all surge hosts and seeds; pps/thr: surge-host packet rate over the threshold.")
    if choice:
        print(f"\nPRE-REGISTERED CHOICE: n_hosts = {choice[0]}, rate = {choice[1]:.1f} Mbps")
    else:
        print("\nPRE-REGISTERED CHOICE: none passes; C1 is a negative calibration result")


def main() -> int:
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarise-only", action="store_true")
    args = ap.parse_args()
    if not args.summarise_only:
        from env.testbed_env import TestbedEnv
        OUT.parent.mkdir(parents=True, exist_ok=True)
        env = TestbedEnv(run_id="calibrate_surge")
        rows = []
        try:
            for n in N_HOSTS:
                for rate in RATES_MBPS:
                    for seed in SEEDS:
                        ep = run_episode(env, n, rate, seed)
                        rows += ep
                        post = [r for r in ep if r["step"] >= FIRST_POST]
                        share = sum(r["flagged"] for r in post) / len(post)
                        print(f"n={n} {rate:.1f} Mbps seed {seed}: surge hosts flagged in "
                              f"{100 * share:.1f}% of host-steps", flush=True)
                        pd.DataFrame(rows).to_parquet(OUT, index=False)   # keep partial results
        finally:
            env.close()
    combo, choice = summarise(pd.read_parquet(OUT))
    report(combo, choice)
    print(f"\nrows in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
