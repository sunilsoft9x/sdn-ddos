"""Fidelity check: simulator vs held-out testbed rows (plan 7.1, gate G5).

The service model was fitted on the non-hold-out calibration rows. This script
runs the simulator at the held-out intensity (0.7) and compares it against the
testbed rows it never saw:

  - absolute error in mean probe success, per regime;
  - relative error in each of the 18 feature means;
  - a cross-domain check: a regime classifier trained on simulator features and
    tested on testbed features (if the sim is faithful, it still separates).

Results go to analysis/fidelity.json, a small table in Section V. G5 passes when
probe-success error stays small (< ~0.1) and the cross-domain classifier holds
up (> ~0.85).

    ~/venv/bin/python sim/fidelity.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sim.model import ServiceModel  # noqa: E402
from sim.sim_env import SimEnv  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
FEATURES = [
    "f_pkt_rate", "f_byte_rate", "f_mean_pkt_size", "f_tcp_share", "f_udp_share",
    "f_icmp_share", "f_src_entropy", "f_active_sources", "f_top_source_share",
    "f_victim_link_util", "f_victim_tx_dropped_rate", "f_edge_table_occupancy",
    "f_probe_success_feat", "f_probe_latency_median", "f_probe_goodput",
    "f_d_pkt_rate", "f_d_udp_share", "f_mitigated_fraction",
]
FEATURE_ORDER = [c[2:] for c in FEATURES]      # names as sim_env emits them


def sim_rows(env: SimEnv, regime: str, scale: float, n_windows: int = 40,
             warmup: int = 4) -> pd.DataFrame:
    """Run the simulator with no mitigation and collect per-window features."""
    env.reset(seed=999, options={"schedule": [[regime, 10**9]],
                                 "scale": scale, "n_attackers": 3})
    rows = []
    for i in range(warmup + n_windows):
        obs, r, term, trunc, info = env.step(0)
        if i < warmup:
            continue
        row = {f"f_{name}": float(v) for name, v in zip(FEATURE_ORDER, obs)}
        row["regime"] = regime
        row["probe_success"] = info["reward_terms"]["S"]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    cfg = yaml.safe_load(open(REPO / "configs" / "base.yaml"))
    model = ServiceModel.load(REPO / "sim" / "service_model.json")
    env = SimEnv(cfg=cfg, service_model=model)

    testbed = pd.read_parquet(REPO / "data" / "calib.parquet")
    held = testbed[(testbed["holdout"]) & (testbed["mit"] == "none")].copy()
    scale = float(held["scale"].iloc[0]) if len(held) else 0.7

    report = {"holdout_scale": scale, "per_regime": {}, "feature_rel_error": {}}
    sim_frames = []
    for regime in ["R1", "R2"]:
        tb = held[held["regime"] == regime]
        if len(tb) == 0:
            continue
        sm = sim_rows(env, regime, scale)
        sim_frames.append(sm)

        succ_err = abs(tb["probe_success"].mean() - sm["probe_success"].mean())
        report["per_regime"][regime] = {
            "testbed_success": round(float(tb["probe_success"].mean()), 3),
            "sim_success": round(float(sm["probe_success"].mean()), 3),
            "abs_error": round(float(succ_err), 3),
        }
        rel = {}
        for col in FEATURES:
            tb_m = tb[col].mean()
            sm_m = sm[col].mean()
            rel[col] = round(float(abs(tb_m - sm_m) / (abs(tb_m) + 1e-6)), 3)
        report["feature_rel_error"][regime] = rel

    # Cross-domain: train on sim R1/R2, test on testbed held-out R1/R2.
    if sim_frames and len(held):
        sim_all = pd.concat(sim_frames, ignore_index=True)
        tb_all = held[held["regime"].isin(["R1", "R2"])]
        Xs, ys = sim_all[FEATURES].to_numpy(), sim_all["regime"].to_numpy()
        Xt, yt = tb_all[FEATURES].to_numpy(), tb_all["regime"].to_numpy()
        scaler = StandardScaler().fit(Xs)
        clf = LogisticRegression(max_iter=2000).fit(scaler.transform(Xs), ys)
        acc = float((clf.predict(scaler.transform(Xt)) == yt).mean())
        report["cross_domain_accuracy"] = round(acc, 3)

    out = REPO / "analysis" / "fidelity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print("fidelity vs held-out testbed rows (intensity "
          f"{scale}):")
    for regime, r in report["per_regime"].items():
        print(f"  {regime}: probe success testbed {r['testbed_success']} vs "
              f"sim {r['sim_success']}  (abs err {r['abs_error']})")
    if "cross_domain_accuracy" in report:
        print(f"  cross-domain classifier accuracy: "
              f"{report['cross_domain_accuracy']}")
    worst = max((v for reg in report["feature_rel_error"].values()
                 for v in reg.values()), default=0.0)
    print(f"  worst feature relative error: {worst}")
    print(f"\nsaved -> {out}")

    ok = all(r["abs_error"] < 0.12 for r in report["per_regime"].values()) \
        and report.get("cross_domain_accuracy", 1.0) > 0.85
    print("PASS G5" if ok else "REVIEW: fidelity below target — inspect the table")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
