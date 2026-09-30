"""Analysis of the testbed evaluation (plan 8): tables + figures from the shards.

Reads runs/eval_shard*.parquet (combined), aggregates per episode, then per
cell, and writes the paper's tables and figures to runs/analysis/. Every number
here comes straight from the testbed logs -- no simulator, no re-running.

The design decisions we settled before the run are baked in:

  * E3 headline is LABEL-FREE vs the LABEL-STARVED ORACLE, on mean S.
    A1/A2 (never needed a label) vs A3miss (oracle with no valid label for the
    unseen regime, forced to no-op). This is the large, defensible effect.
  * A2-vs-A1 (the GRU's value) is judged on the SWITCH conditions, not on a
    static regime, using post-switch recovery latency and worst-window S.
  * Worst-window S is reported alongside mean S everywhere in E3.
  * Adaptation latency follows the plan's exact definition (Section V): the
    5-step rolling mean of probe success back within 5% of the pre-switch
    steady level (the 20 steps before the switch) and HELD for 5 steps,
    censored if it never holds, censored count reported.

    ~/venv/bin/python scripts/analyze.py
    ~/venv/bin/python scripts/analyze.py --glob 'runs/eval_shard*.parquet'
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "runs" / "analysis"
RECOVER_FRAC = 0.95          # "recovered" = back within 5% of pre-switch service (plan 4.12)
BASELINE_WIN = 20            # pre-switch steady window (steps 30-49 when the switch is at 50)
ROLL = 5                     # rolling-mean smoothing for the recovery test
HOLD = 5                     # recovery must hold for this many steps (plan 4.12)
WARMUP = 2                   # startup windows skipped for worst-window S (traffic
                             # is still ramping and A2's GRU history is empty --
                             # a step-0 dip is a transient, not a mitigation failure)


# --------------------------------------------------------------------------- #
# loading                                                                      #
# --------------------------------------------------------------------------- #

def load(glob: str) -> pd.DataFrame:
    files = sorted(REPO.glob(glob))
    if not files:
        sys.exit(f"no parquet matched {glob!r} under {REPO}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"loaded {len(df):,} rows from {len(files)} shard(s): "
          f"{[f.name for f in files]}")
    df["epi"] = (df["exp"] + "|" + df["agent"] + "|" + df["cond"] + "|"
                 + df["seed"].astype(str) + "|" + df["ep"].astype(str))
    return df


# --------------------------------------------------------------------------- #
# per-episode reductions                                                       #
# --------------------------------------------------------------------------- #

def switch_step(regime: np.ndarray) -> int | None:
    """First index where the regime differs from step 0, else None."""
    first = regime[0]
    idx = np.nonzero(regime != first)[0]
    return int(idx[0]) if len(idx) else None


def recovery_latency(S: np.ndarray, regime: np.ndarray):
    """Plan 4.12 adaptation latency. Steps from the switch until the 5-step
    rolling mean of probe success returns within 5% of the pre-switch steady
    level (mean of the BASELINE_WIN steps before the switch) AND holds there
    for HOLD steps. Returns (latency, censored) or None if no switch.
    Censored (never held for HOLD steps, or too few steps left to confirm) is
    reported at the number of post-switch steps."""
    sw = switch_step(regime)
    if sw is None or sw < BASELINE_WIN:
        return None
    base = float(np.mean(S[sw - BASELINE_WIN:sw]))
    if base <= 1e-6:
        return None
    target = RECOVER_FRAC * base
    post = S[sw:]
    roll = pd.Series(post).rolling(ROLL, min_periods=1).mean().to_numpy()
    ok = roll >= target
    for t in range(len(ok) - HOLD + 1):
        if ok[t:t + HOLD].all():
            return float(t), False
    return float(len(post)), True     # censored: never held for HOLD steps


def per_episode(df: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for epi, g in df.sort_values("step").groupby("epi"):
        S = g["S"].to_numpy()
        warm = S[WARMUP:] if len(S) > WARMUP else S     # drop startup transient
        rec = recovery_latency(S, g["regime"].to_numpy())
        recs.append({
            "exp": g["exp"].iloc[0], "agent": g["agent"].iloc[0],
            "cond": g["cond"].iloc[0], "seed": g["seed"].iloc[0],
            "ep": g["ep"].iloc[0],
            "meanS": float(S.mean()),
            "minS": float(warm.min()),                  # worst steady-state window
            "min_step": int(WARMUP + int(np.argmin(warm))),
            "meanReward": float(g["reward"].mean()),
            "meanRules": float(g["rule_count"].mean()),
            "collateral": float(g["collateral"].mean()),
            "recovery": rec[0] if rec is not None else None,
            "recovery_censored": (rec[1] if rec is not None else None),
        })
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------- #
# cell aggregation + tables                                                    #
# --------------------------------------------------------------------------- #

def cell_table(ep: pd.DataFrame, exp: str, metrics=("meanS",)) -> pd.DataFrame:
    sub = ep[ep["exp"] == exp]
    rows = []
    for (agent, cond), g in sub.groupby(["agent", "cond"]):
        row = {"agent": agent, "cond": cond, "n": len(g)}
        for m in metrics:
            vals = g[m].dropna()
            row[f"{m}_mean"] = round(float(vals.mean()), 3) if len(vals) else np.nan
            row[f"{m}_sd"] = round(float(vals.std(ddof=1)), 3) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["cond", "agent"]).reset_index(drop=True)


def recovery_table(ep: pd.DataFrame, exp: str) -> pd.DataFrame:
    """Adaptation latency per cell: recovered vs censored counts, and mean +- sd
    latency over the recovered episodes only (censored ones would bias it)."""
    sub = ep[(ep["exp"] == exp) & ep["recovery"].notna()]
    rows = []
    for (agent, cond), g in sub.groupby(["agent", "cond"]):
        cens = g["recovery_censored"].fillna(False).astype(bool)
        rec = g.loc[~cens, "recovery"]
        rows.append({
            "agent": agent, "cond": cond, "n": len(g),
            "recovered": int((~cens).sum()), "censored": int(cens.sum()),
            "latency_mean": round(float(rec.mean()), 2) if len(rec) else np.nan,
            "latency_sd": round(float(rec.std(ddof=1)), 2) if len(rec) > 1 else 0.0,
        })
    return pd.DataFrame(rows).sort_values(["cond", "agent"]).reset_index(drop=True)


def area_vs_clean(df: pd.DataFrame, exp: str, cond: str,
                  ref_agent: str = "A3") -> dict:
    """Area between each agent's post-switch mean-S curve and the clean-label
    oracle's (plan 4.12). Positive = the agent held service better than the
    clean oracle over the recovery. Only meaningful where ref_agent ran with a
    true label in this cell (E1)."""
    sub = df[(df["exp"] == exp) & (df["cond"] == cond)]
    if sub.empty or ref_agent not in sub["agent"].unique():
        return {}
    sw = switch_step(sub.sort_values("step")["regime"].to_numpy())
    if sw is None:
        return {}
    curve = (sub[sub["step"] >= sw].groupby(["agent", "step"])["S"].mean()
             .unstack("agent"))
    if ref_agent not in curve:
        return {}
    ref = curve[ref_agent]
    return {a: round(float((curve[a] - ref).sum()), 2)
            for a in curve.columns if a != ref_agent}


def _vals(ep, exp, cond, agents, metric) -> np.ndarray:
    sub = ep[(ep["exp"] == exp) & (ep["cond"] == cond) & (ep["agent"].isin(agents))]
    return sub[metric].dropna().to_numpy()


def contrast(label_a: str, a: np.ndarray, label_b: str, b: np.ndarray) -> str:
    from scipy import stats
    if len(a) < 2 or len(b) < 2:
        return f"  {label_a} vs {label_b}: too few episodes ({len(a)}, {len(b)})"
    _, p = stats.ttest_ind(a, b, equal_var=False)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) or 1e-9
    d = (a.mean() - b.mean()) / pooled
    return (f"  {label_a} ({a.mean():.3f}) vs {label_b} ({b.mean():.3f}): "
            f"Δ={a.mean()-b.mean():+.3f}, d={d:+.2f}, p={p:.3g}")


def welch(ep: pd.DataFrame, exp: str, cond, agents_a, agents_b,
          metric="meanS") -> str:
    a = _vals(ep, exp, cond, agents_a, metric)
    b = _vals(ep, exp, cond, agents_b, metric)
    return contrast(f"{'/'.join(agents_a)} [{exp} {cond}]", a,
                    f"{'/'.join(agents_b)} [{exp} {cond}]", b)


# --------------------------------------------------------------------------- #
# figures                                                                      #
# --------------------------------------------------------------------------- #

def fig_switch_timeline(df: pd.DataFrame, exp: str, cond: str, path: pathlib.Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = df[(df["exp"] == exp) & (df["cond"] == cond)]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for agent, g in sub.groupby("agent"):
        piv = g.pivot_table(index="step", values="S", aggfunc="mean")
        ax.plot(piv.index, piv["S"], label=agent, linewidth=1.6)
    sw = switch_step(sub.sort_values("step")["regime"].to_numpy())
    if sw is not None:
        ax.axvline(sw, color="0.4", linestyle="--", linewidth=1)
        ax.text(sw + 0.5, ax.get_ylim()[0] + 0.03, "regime switch", fontsize=8,
                color="0.4")
    ax.set_xlabel("step"); ax.set_ylabel("service S (mean over episodes)")
    ax.set_title(f"{exp}  {cond}"); ax.set_ylim(0, 1.02); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  figure -> {path.name}")


def fig_e3_bars(ep: pd.DataFrame, path: pathlib.Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = ep[ep["exp"] == "E3"]
    if sub.empty:
        return
    agents = ["static", "A3miss", "A3nearest", "A1", "A2"]
    conds = sorted(sub["cond"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, metric, title in zip(
            axes, ["meanS", "minS"], ["mean service", "worst-window service"]):
        x = np.arange(len(conds)); w = 0.15
        for i, ag in enumerate(agents):
            means, sds = [], []
            for c in conds:
                v = sub[(sub["agent"] == ag) & (sub["cond"] == c)][metric].dropna()
                means.append(v.mean() if len(v) else np.nan)
                sds.append(v.std(ddof=1) if len(v) > 1 else 0.0)
            ax.bar(x + (i - 2) * w, means, w, yerr=sds, capsize=2, label=ag)
        ax.set_xticks(x); ax.set_xticklabels(conds); ax.set_ylim(0, 1.05)
        ax.set_title(title); ax.set_ylabel(metric)
    axes[1].legend(fontsize=8, ncol=1, loc="lower right")
    fig.suptitle("E3 held-out regime: label-free (A1/A2) vs label-starved oracle (A3miss)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  figure -> {path.name}")


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--glob", default="runs/eval_shard*.parquet")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    df = load(args.glob)
    ep = per_episode(df)
    ep.to_csv(OUT / "per_episode.csv", index=False)

    lines: list[str] = ["# Evaluation analysis\n"]

    t0 = cell_table(ep, "E0", ("meanS",))
    lines += ["## E0  each regime alone (sanity)\n", t0.to_markdown(index=False), ""]

    t1 = cell_table(ep, "E1", ("meanS",))
    r1 = recovery_table(ep, "E1")
    lines += ["## E1  R1->R2 switch\n", t1.to_markdown(index=False), "",
              "Adaptation latency (recovered episodes only; censored counted separately):\n",
              r1.to_markdown(index=False), ""]
    area1 = area_vs_clean(df, "E1", "R1->R2", ref_agent="A3")
    if area1:
        lines += ["Area between post-switch service curve and clean-label A3 "
                  "(positive = better than the clean oracle):", "```"]
        lines += [f"  {a}: {v:+.2f}" for a, v in sorted(area1.items())] + ["```", ""]

    t2 = cell_table(ep, "E2", ("meanS",))
    lines += ["## E2  oracle under degraded labels\n", t2.to_markdown(index=False), ""]

    t3 = cell_table(ep, "E3", ("meanS", "minS"))
    r3 = recovery_table(ep, "E3")
    lines += ["## E3  held-out R4 (mean AND worst-window S)\n",
              t3.to_markdown(index=False), "",
              "Adaptation latency on the R1->R4 switch (recovered only; censored separate):\n",
              r3.to_markdown(index=False), ""]

    lines += ["## Key contrasts\n",
              "(meanS/minS: higher is better. latency: FEWER steps is better.)\n",
              "```"]
    lines.append("E3 headline -- label-free vs label-starved oracle (mean S):")
    for c in sorted(ep[ep["exp"] == "E3"]["cond"].unique()):
        lines.append(welch(ep, "E3", c, ["A1", "A2"], ["A3miss"], "meanS"))
    lines.append("")
    lines.append("A2 vs A1 -- does the GRU earn its place? (switch conditions)")
    for exp, c in [("E1", "R1->R2"), ("E3", "R1->R4")]:
        lines.append(welch(ep, exp, c, ["A2"], ["A1"], "recovery"))
        lines.append(welch(ep, exp, c, ["A2"], ["A1"], "minS"))
    lines.append("")
    lines.append("E2 -- oracle degrades with bad labels; label-free (A2) does not.")
    lines.append("     A3[true] is the clean-label oracle on the same R1->R2 schedule (from E1).")
    a3_true = _vals(ep, "E1", "R1->R2", ["A3"], "meanS")
    a2_free = _vals(ep, "E1", "R1->R2", ["A2"], "meanS")
    for mode in ("wrong", "missing", "delayed"):
        a3_bad = _vals(ep, "E2", mode, ["A3"], "meanS")
        lines.append(contrast("A3[true, E1]", a3_true, f"A3[{mode}, E2]", a3_bad))
        lines.append(contrast("A2[label-free, E1]", a2_free, f"A3[{mode}, E2]", a3_bad))
    lines += ["```", ""]

    (OUT / "report.md").write_text("\n".join(str(x) for x in lines))
    print(f"\ntables -> {OUT/'report.md'}")

    fig_switch_timeline(df, "E1", "R1->R2", OUT / "fig_e1_switch.png")
    fig_switch_timeline(df, "E3", "R1->R4", OUT / "fig_e3_switch.png")
    fig_e3_bars(ep, OUT / "fig_e3_bars.png")

    print("\ndone. Read runs/analysis/report.md first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
