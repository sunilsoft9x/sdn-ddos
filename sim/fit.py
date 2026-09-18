"""Fit the service model to calibration data (plan 7.1).

Reads data/calib.parquet, reconstructs (utilisation, passed packet rate,
tcp share) from the logged features, and fits:
  - probe success as a binomial logistic in those inputs (per-window success
    counts, so a window with more probes weighs more);
  - scaled probe latency as a monotone rise with utilisation;
  - multiplicative counter noise, per feature, from the R0 windows.

The hold-out rows (intensity 0.7) are excluded from fitting and kept for the
fidelity check in sim/fidelity is run separately.

    ~/venv/bin/python sim/fit.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sim.model import ServiceModel  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
MAX_PKT_RATE = 20000.0          # matches env/features._Scales.max_pkt_rate

FEATURE_COLS = ["f_victim_link_util", "f_pkt_rate", "f_tcp_share",
                "f_udp_share", "f_byte_rate", "f_mean_pkt_size"]


def _inputs(df: pd.DataFrame):
    util = df["f_victim_link_util"].to_numpy()
    pps = df["f_pkt_rate"].to_numpy() * MAX_PKT_RATE
    tcp = df["f_tcp_share"].to_numpy()
    # design matrix: [1, util, util^2, log1p(pps), tcp_share]
    X = np.column_stack([np.ones_like(util), util, util * util,
                         np.log1p(pps), tcp])
    return X


def fit_success(df: pd.DataFrame) -> list[float]:
    from scipy.optimize import minimize

    X = _inputs(df)
    # calibrate.py did not log probe_n; every window had 3 probe hosts x ~5
    # attempts = 15, so fall back to that constant when the column is absent.
    if "probe_n" in df.columns:
        n = np.maximum(df["probe_n"].to_numpy(), 1)
    else:
        n = np.full(len(df), 15)
    k = np.clip(np.round(df["probe_success"].to_numpy() * n), 0, n)

    def nll(w):
        z = X @ w
        z = np.clip(z, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -np.sum(k * np.log(p) + (n - k) * np.log(1 - p))

    w0 = np.array([4.0, -6.0, 0.0, 0.0, 0.0])
    res = minimize(nll, w0, method="L-BFGS-B")
    return res.x.tolist()


def fit_latency(df: pd.DataFrame) -> list[float]:
    # scaled latency (0..1) ~ base + slope * util^2, least squares
    util = df["f_victim_link_util"].to_numpy()
    y = df["f_probe_latency_median"].to_numpy()
    A = np.column_stack([np.ones_like(util), util * util])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return [float(coef[0]), float(coef[1])]


def fit_util_calibration(df: pd.DataFrame) -> dict[str, list[float]]:
    """Per-regime linear fit of measured utilisation vs intensity scale.

    Uses no-mitigation rows so the utilisation reflects the raw attack. The
    simulator sets its offered load to reproduce util = a + b*scale, so its
    service response tracks the testbed's by construction. R4 is excluded
    (held out, evaluation only).
    """
    cal: dict[str, list[float]] = {}
    none = df[df["mit"] == "none"]
    for regime in ["R0", "R1", "R2"]:
        sub = none[none["regime"] == regime]
        if len(sub) < 2:
            continue
        s = sub["scale"].to_numpy()
        u = sub["f_victim_link_util"].to_numpy()
        A = np.column_stack([np.ones_like(s), s])
        coef, *_ = np.linalg.lstsq(A, u, rcond=None)
        cal[regime] = [float(coef[0]), float(coef[1])]
    return cal


def fit_noise(df: pd.DataFrame) -> dict[str, float]:
    r0 = df[(df["regime"] == "R0")]
    noise = {}
    for col in FEATURE_COLS:
        vals = r0[col].to_numpy()
        m = float(np.mean(vals))
        s = float(np.std(vals))
        noise[col] = (s / m) if m > 1e-6 else 0.0
    return noise


def main() -> int:
    df = pd.read_parquet(REPO / "data" / "calib.parquet")
    fit_df = df[~df["holdout"]].copy()
    print(f"fitting on {len(fit_df)} non-holdout rows "
          f"({df['holdout'].sum()} held out)")

    model = ServiceModel(
        success_coef=fit_success(fit_df),
        latency=fit_latency(fit_df),
        noise_std=fit_noise(fit_df),
        util_cal=fit_util_calibration(fit_df),
        fitted=True,
    )
    print("util calibration (util = a + b*scale):")
    for regime, (a, b) in model.util_cal.items():
        print(f"  {regime}: a={a:.3f} b={b:.3f}")
    out = REPO / "sim" / "service_model.json"
    model.save(out)

    # Quick in-sample sanity: predicted vs actual mean success by regime.
    print("\nin-sample mean probe success (actual -> predicted):")
    for regime in ["R0", "R1", "R2", "R4"]:
        sub = fit_df[fit_df["regime"] == regime]
        if len(sub) == 0:
            continue
        util = sub["f_victim_link_util"].to_numpy()
        pps = sub["f_pkt_rate"].to_numpy() * MAX_PKT_RATE
        tcp = sub["f_tcp_share"].to_numpy()
        pred = np.array([model.success(u, p, t) for u, p, t in zip(util, pps, tcp)])
        print(f"  {regime}: {sub['probe_success'].mean():.2f} -> {pred.mean():.2f}")

    print(f"\nsuccess_coef: {[round(c, 3) for c in model.success_coef]}")
    print(f"latency: {[round(c, 3) for c in model.latency]}")
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
