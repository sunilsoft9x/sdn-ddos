"""Regime separability from label-free features (plan 6.8, gate G4).

Trains a logistic-regression classifier to recover the regime (R0/R1/R2) from
the 18 observation features alone, using only the no-mitigation calibration
rows. If the features carry the regime, held-out accuracy clears 90% and the
label-free contribution has empirical footing: a policy reading these features
does not need an external detector label, because the regime is already in the
observation.

Reports test accuracy, the confusion matrix, and which features carry the
signal (standardised coefficients) — the last is a Section V figure, and it
should show protocol share doing most of the work, which is why volumetric
attacks stay separable.

    ~/venv/bin/python scripts/separability.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

REPO = pathlib.Path(__file__).resolve().parents[1]
FEATURES = [
    "f_pkt_rate", "f_byte_rate", "f_mean_pkt_size", "f_tcp_share", "f_udp_share",
    "f_icmp_share", "f_src_entropy", "f_active_sources", "f_top_source_share",
    "f_victim_link_util", "f_victim_tx_dropped_rate", "f_edge_table_occupancy",
    "f_probe_success_feat", "f_probe_latency_median", "f_probe_goodput",
    "f_d_pkt_rate", "f_d_udp_share", "f_mitigated_fraction",
]


def main() -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    df = pd.read_parquet(REPO / "data" / "calib.parquet")
    # No-mitigation rows of the training regimes only (R4 is held out).
    sub = df[(df["mit"] == "none") & (df["regime"].isin(["R0", "R1", "R2"]))].copy()
    print(f"rows: {len(sub)}  by regime: {sub['regime'].value_counts().to_dict()}")

    X = sub[FEATURES].to_numpy()
    y = sub["regime"].to_numpy()

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0)
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000)  # multinomial by default (sklearn >=1.7)
    clf.fit(scaler.transform(Xtr), ytr)
    pred = clf.predict(scaler.transform(Xte))
    acc = accuracy_score(yte, pred)

    print(f"\nHELD-OUT ACCURACY: {acc:.3f}  (G4 needs > 0.90)")
    labels = sorted(set(y))
    cm = confusion_matrix(yte, pred, labels=labels)
    print("\nconfusion matrix (rows true, cols pred):")
    print("        " + "  ".join(f"{l:>5}" for l in labels))
    for i, l in enumerate(labels):
        print(f"  {l:>4}  " + "  ".join(f"{v:5d}" for v in cm[i]))

    # Which features carry the signal: mean |coef| across the one-vs-rest rows.
    importance = np.abs(clf.coef_).mean(axis=0)
    order = np.argsort(importance)[::-1]
    print("\ntop features by |coefficient|:")
    for idx in order[:6]:
        print(f"  {FEATURES[idx]:28s} {importance[idx]:.2f}")

    print("\n" + ("PASS: features separate the regimes" if acc > 0.90
                  else "BELOW G4 THRESHOLD — inspect the confusion matrix"))
    return 0 if acc > 0.90 else 1


if __name__ == "__main__":
    raise SystemExit(main())
