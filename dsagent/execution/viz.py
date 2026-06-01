"""Real charts — the deliverable graphs, not tile names.

Produces the canonical causal-inference figures as PNGs:
  * forest plot     — estimates vs ground truth with 95% CIs across methods
  * event_study     — pre/post dynamic effects with a zero reference (the plot
                      a referee looks at first for a DiD/CS design)
  * overlap         — propensity histograms by treatment group (positivity)
  * calibration     — reliability curve for the predictive model

Uses the non-interactive Agg backend so it renders headless. Returns file paths.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import datagen as dg, estimators as est
from .ml import fit_predictive

_C = {"pt": "#2563eb", "ci": "#93c5fd", "ref": "#dc2626", "t": "#1d4ed8",
      "c": "#9ca3af", "ok": "#059669"}


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def forest_plot(rows, outdir, name="forest.png"):
    """rows: list of {label, point, ci_low, ci_high, truth}."""
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(rows) + 1.2))
    ys = range(len(rows))
    for y, r in zip(ys, rows):
        ax.plot([r["ci_low"], r["ci_high"]], [y, y], color=_C["ci"], lw=4, zorder=1)
        ax.plot(r["point"], y, "o", color=_C["pt"], zorder=3, ms=7)
        if r.get("truth") is not None:
            ax.plot(r["truth"], y, "|", color=_C["ref"], ms=16, mew=2, zorder=4)
    ax.set_yticks(list(ys)); ax.set_yticklabels([r["label"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("treatment effect (point, 95% CI; red = ground truth)")
    ax.set_title("Estimates vs. known truth across methods", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, outdir, name)


def event_study_plot(df, outdir, name="event_study.png"):
    """Dynamic pre/post effects vs never-treated, centered at adoption."""
    pre = est.pretrends_test(df)
    cells = pre.get("cells", [])
    # post-period effects via the same vs-never-treated differencing
    wide = df.pivot_table(index="unit", columns="period", values="y")
    cohorts = df.groupby("unit")["cohort"].first()
    never = cohorts.index[cohorts == 0]
    base = sorted(df["period"].unique())[0]
    pts = {}
    for g in sorted(c for c in cohorts.unique() if c != 0):
        tu = cohorts.index[cohorts == g]
        for t in sorted(df["period"].unique()):
            if t == base:
                continue
            dt = wide.loc[tu, t] - wide.loc[tu, base]
            dc = wide.loc[never, t] - wide.loc[never, base]
            eff = dt.mean() - dc.mean()
            se = np.sqrt(dt.var(ddof=1)/len(dt) + dc.var(ddof=1)/len(dc))
            pts.setdefault(t - g, []).append((eff, se))
    rel = sorted(pts)
    effs = [np.mean([e for e, _ in pts[r]]) for r in rel]
    ses = [np.mean([s for _, s in pts[r]]) for r in rel]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.axhline(0, color=_C["c"], lw=1)
    ax.axvline(-0.5, color=_C["ref"], ls="--", lw=1, label="adoption")
    ax.errorbar(rel, effs, yerr=[1.96*s for s in ses], fmt="o-", color=_C["pt"],
                ecolor=_C["ci"], capsize=3)
    ax.set_xlabel("event time (periods relative to adoption)")
    ax.set_ylabel("effect vs never-treated")
    ax.set_title("Event study: flat pre-trends, effect emerges post-adoption",
                 fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    return _save(fig, outdir, name)


def overlap_plot(df, outdir, name="overlap.png", controls=("x",)):
    from sklearn.linear_model import LogisticRegression
    cs = [c for c in controls if c in df.columns]
    ps = LogisticRegression(max_iter=200).fit(df[cs], df["t"]).predict_proba(df[cs])[:, 1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ps[df["t"] == 1], bins=30, alpha=0.6, color=_C["t"], label="treated", density=True)
    ax.hist(ps[df["t"] == 0], bins=30, alpha=0.6, color=_C["c"], label="control", density=True)
    ax.axvspan(0.0, 0.05, color=_C["ref"], alpha=0.08)
    ax.axvspan(0.95, 1.0, color=_C["ref"], alpha=0.08)
    ax.set_xlabel("propensity score"); ax.set_ylabel("density")
    ax.set_title("Overlap / positivity (shaded = poor support)", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    return _save(fig, outdir, name)


def calibration_plot(df, outdir, name="calibration.png", target="t", features=("x",)):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict
    from sklearn.calibration import calibration_curve
    X = df[list(features)].to_numpy(); y = df[target].to_numpy().astype(int)
    proba = cross_val_predict(GradientBoostingClassifier(random_state=0), X, y,
                              cv=5, method="predict_proba")[:, 1]
    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=10)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color=_C["c"], label="perfect")
    ax.plot(mean_pred, frac_pos, "o-", color=_C["pt"], label="model")
    ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency")
    ax.set_title("Calibration (reliability curve)", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    return _save(fig, outdir, name)


def render_standard_pack(outdir: str) -> dict:
    """Build the full figure pack from fresh, known-truth fits."""
    # forest across methods
    rows = []
    for label, (mk, truth, fit) in {
        "RCT (diff-in-means)": (lambda: dg.make_rct(2.0), 2.0,
                                lambda d: est.diff_in_means(d)),
        "Observational (DML)": (lambda: dg.make_observational(2.0), 2.0,
                                lambda d: est.double_ml(d, controls=("x",))),
        "Staggered (CS)": (lambda: dg.make_staggered(0.4)[0], 0.8357,
                           lambda d: est.callaway_santanna(d)),
        "IV (2SLS)": (lambda: dg.make_iv(1.5)[0], 1.5, lambda d: est.iv_2sls(d)),
    }.items():
        r = fit(mk())
        rows.append({"label": label, "point": r.point, "ci_low": r.ci_low,
                     "ci_high": r.ci_high, "truth": truth})
    obs = dg.make_observational(2.0)
    sdf, _ = dg.make_staggered(0.4)
    return {
        "forest": forest_plot(rows, outdir),
        "event_study": event_study_plot(sdf, outdir),
        "overlap": overlap_plot(obs, outdir),
        "calibration": calibration_plot(obs, outdir),
    }
