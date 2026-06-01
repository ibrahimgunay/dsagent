"""Estimators that actually fit — numpy / scipy / sklearn.

Each returns an `EstimateResult` with a point estimate, robust SE, 95% CI, and
p-value, plus diagnostics. These are the real numerical engines the modeling
agents call; the agent decides *which* estimator (LLM judgment), the estimator
produces the number (deterministic computation), and the critic gates it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


@dataclass
class EstimateResult:
    method: str
    point: float
    se: float
    ci_low: float
    ci_high: float
    pvalue: float
    n: int
    diagnostics: dict = field(default_factory=dict)

    def covers(self, truth: float) -> bool:
        return self.ci_low <= truth <= self.ci_high


def _ols(X: np.ndarray, y: np.ndarray):
    """OLS with HC1 heteroskedasticity-robust SE. Returns (beta, se)."""
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    # HC1 robust covariance
    meat = (X * resid[:, None]).T @ (X * resid[:, None])
    cov = XtX_inv @ meat @ XtX_inv * (n / (n - k))
    se = np.sqrt(np.diag(cov))
    return beta, se, n - k


def _result(method, beta_j, se_j, dof, n, diagnostics=None):
    t = beta_j / se_j if se_j > 0 else np.inf
    p = 2 * stats.t.sf(abs(t), dof)
    crit = stats.t.ppf(0.975, dof)
    return EstimateResult(method, float(beta_j), float(se_j),
                          float(beta_j - crit * se_j), float(beta_j + crit * se_j),
                          float(p), int(n), diagnostics or {})


def diff_in_means(df, treat="t", outcome="y") -> EstimateResult:
    """Naive treated-vs-control mean difference (biased under confounding)."""
    n = len(df)
    X = np.column_stack([np.ones(n), df[treat].to_numpy()])
    beta, se, dof = _ols(X, df[outcome].to_numpy())
    return _result("diff_in_means", beta[1], se[1], dof, n)


def regression_adjust(df, treat="t", outcome="y", controls=("x",)) -> EstimateResult:
    """OLS adjusting for observed confounders."""
    n = len(df)
    cols = [np.ones(n), df[treat].to_numpy()] + [df[c].to_numpy() for c in controls]
    X = np.column_stack(cols)
    beta, se, dof = _ols(X, df[outcome].to_numpy())
    return _result("regression_adjust", beta[1], se[1], dof, n,
                   {"controls": list(controls)})


def did_2x2(df, treat="treat", post="post", outcome="y") -> EstimateResult:
    """Difference-in-differences: coefficient on treat*post interaction = ATT."""
    n = len(df)
    tr, po = df[treat].to_numpy(), df[post].to_numpy()
    X = np.column_stack([np.ones(n), tr, po, tr * po])
    beta, se, dof = _ols(X, df[outcome].to_numpy())
    return _result("did_2x2", beta[3], se[3], dof, n,
                   {"interaction": "treat:post"})


def double_ml(df, treat="t", outcome="y", controls=("x",), folds=5,
              n_rep=3, seed=0) -> EstimateResult:
    """Partially-linear Double ML (Chernozhukov et al.), DML2 with REPEATED
    cross-fitting and median aggregation across repetitions (the recommended
    practice: reduces dependence on a single random fold split). Variance from
    the Neyman-orthogonal score.
    """
    Xc = df[list(controls)].to_numpy()
    y = df[outcome].to_numpy().astype(float)
    t = df[treat].to_numpy().astype(float)
    n = len(df)
    thetas, vars_ = [], []
    for rep in range(n_rep):
        y_res = np.zeros(n); t_res = np.zeros(n)
        kf = KFold(n_splits=folds, shuffle=True, random_state=seed + rep)
        for train, test in kf.split(Xc):
            my = GradientBoostingRegressor(random_state=seed).fit(Xc[train], y[train])
            mt = GradientBoostingRegressor(random_state=seed).fit(Xc[train], t[train])
            y_res[test] = y[test] - my.predict(Xc[test])
            t_res[test] = t[test] - mt.predict(Xc[test])
        theta = float((t_res @ y_res) / (t_res @ t_res))
        psi = (y_res - theta * t_res) * t_res
        var = (psi @ psi) / (t_res @ t_res) ** 2
        thetas.append(theta); vars_.append(var)
    # median aggregation (Chernozhukov et al. recommendation)
    theta = float(np.median(thetas))
    # median variance + between-rep dispersion of point estimates
    var = float(np.median(vars_) + np.median((np.array(thetas) - theta) ** 2))
    se = float(np.sqrt(var))
    return _result("double_ml", theta, se, n - 1, n,
                   {"folds": folds, "repetitions": n_rep})


def propensity_overlap(df, treat="t", controls=("x",), seed=0) -> dict:
    """Estimate propensity scores and check common support (positivity gate)."""
    Xc = df[list(controls)].to_numpy()
    t = df[treat].to_numpy()
    ps = LogisticRegression(max_iter=200).fit(Xc, t).predict_proba(Xc)[:, 1]
    out_of_support = float(np.mean((ps < 0.05) | (ps > 0.95)))
    return {"ps_min": float(ps.min()), "ps_max": float(ps.max()),
            "share_outside_[0.05,0.95]": round(out_of_support, 4),
            "overlap_ok": bool(ps.min() >= 0.02 and ps.max() <= 0.98)}


def callaway_santanna(df, unit="unit", period="period", cohort="cohort",
                      outcome="y") -> EstimateResult:
    """Callaway-Sant'Anna staggered DiD with not-yet/never-treated controls.

    For each cohort g and post-period t>=g, estimate the group-time ATT(g,t) as
    the change from the pre-period (g-1) for cohort g minus the same change for
    the never-treated. Aggregate to an overall ATT (group-size weighted). SE via
    a clustered (by unit) bootstrap.
    """
    wide = df.pivot_table(index=unit, columns=period, values=outcome)
    cohorts = df.groupby(unit)[cohort].first()
    never = cohorts.index[cohorts == 0]
    periods = sorted(df[period].unique())

    def _att_point(units_idx):
        coh = cohorts.loc[units_idx]
        cell_atts, weights = [], []
        for g in sorted(c for c in coh.unique() if c != 0):
            base_t = g - 1
            if base_t not in wide.columns:
                continue
            treated_units = coh.index[coh == g]
            for t in periods:
                if t < g or t not in wide.columns:
                    continue
                dtreat = (wide.loc[treated_units, t] - wide.loc[treated_units, base_t]).mean()
                dctrl = (wide.loc[never, t] - wide.loc[never, base_t]).mean()
                cell_atts.append(dtreat - dctrl)
                weights.append(len(treated_units))
        if not cell_atts:
            return np.nan
        w = np.array(weights, dtype=float)
        return float(np.average(cell_atts, weights=w))

    point = _att_point(cohorts.index)

    # clustered bootstrap over units
    rng = np.random.default_rng(0)
    treated_pool = cohorts.index[cohorts != 0]
    boot = []
    all_units = cohorts.index.to_numpy()
    for _ in range(200):
        samp = rng.choice(all_units, size=len(all_units), replace=True)
        # need both treated and never in sample
        try:
            val = _att_point(pd.Index(samp))
            if not np.isnan(val):
                boot.append(val)
        except Exception:
            continue
    se = float(np.std(boot)) if boot else float("nan")
    n = df[unit].nunique()
    return _result("callaway_santanna", point, se, n - 1, n,
                   {"controls": "never-treated", "aggregation": "group-size weighted",
                    "n_treated_units": int(len(treated_pool))})


def pretrends_test(df, unit="unit", period="period", cohort="cohort",
                   outcome="y") -> dict:
    """Joint pre-trends test (proper statistics, not a magnitude threshold).

    For each pre-treatment cohort/time cell we estimate the placebo DiD vs
    never-treated AND its standard error, form a per-cell t-test, and combine
    via a Bonferroni-adjusted joint p-value. Parallel trends is *supported* when
    we fail to reject (joint p > 0.05) — a real hypothesis test with a p-value.
    """
    wide = df.pivot_table(index=unit, columns=period, values=outcome)
    cohorts = df.groupby(unit)[cohort].first()
    never = cohorts.index[cohorts == 0]
    periods = sorted(df[period].unique())
    base_t = periods[0]
    cells = []
    for g in sorted(c for c in cohorts.unique() if c != 0):
        treated_units = cohorts.index[cohorts == g]
        for t in periods:
            if base_t < t < g:
                dt = (wide.loc[treated_units, t] - wide.loc[treated_units, base_t])
                dc = (wide.loc[never, t] - wide.loc[never, base_t])
                eff = dt.mean() - dc.mean()
                se = np.sqrt(dt.var(ddof=1) / len(dt) + dc.var(ddof=1) / len(dc))
                tstat = eff / se if se > 0 else 0.0
                p = 2 * stats.norm.sf(abs(tstat))
                cells.append({"cohort": int(g), "period": int(t),
                              "effect": float(eff), "se": float(se), "p": float(p)})
    if not cells:
        return {"pretrends_ok": True, "joint_p": 1.0, "max_pre_effect": 0.0,
                "n_pre_periods_tested": 0}
    min_p = min(c["p"] for c in cells)
    joint_p = min(1.0, min_p * len(cells))            # Bonferroni
    max_pre = max(abs(c["effect"]) for c in cells)
    return {"pretrends_ok": bool(joint_p > 0.05), "joint_p": round(float(joint_p), 4),
            "max_pre_effect": round(float(max_pre), 4),
            "n_pre_periods_tested": len(cells), "cells": cells}


def twfe_static(df, unit="unit", period="period", treated="treated",
                outcome="y") -> EstimateResult:
    """Naive single-dummy two-way fixed effects (biased under dynamic/staggered
    effects). Included to demonstrate the bias the critic gate prevents."""
    d = df.copy()
    u = pd.get_dummies(d[unit], prefix="u", drop_first=True).astype(float)
    p = pd.get_dummies(d[period], prefix="p", drop_first=True).astype(float)
    X = np.column_stack([np.ones(len(d)), d[treated].to_numpy().astype(float),
                         u.to_numpy(), p.to_numpy()])
    beta, se, dof = _ols(X, d[outcome].to_numpy())
    return _result("twfe_static", beta[1], se[1], dof, len(d),
                   {"warning": "biased under heterogeneous/dynamic timing"})


def cate_by_subgroup(df, group="tier", treat="t", outcome="y",
                     controls=("x",)) -> dict:
    """Heterogeneous treatment effects: adjusted estimate within each subgroup."""
    out = {}
    for g, sub in df.groupby(group):
        r = regression_adjust(sub, treat=treat, outcome=outcome, controls=controls)
        out[int(g)] = {"point": round(r.point, 3),
                       "ci": [round(r.ci_low, 3), round(r.ci_high, 3)]}
    return out


# ============================ v3: robustness & rigor ============================

def iv_2sls(df, outcome="y", treat="t", instrument="z") -> EstimateResult:
    """Two-stage least squares with a weak-instrument diagnostic.

    Stage 1: treat ~ instrument (record first-stage F). Stage 2: outcome ~ t_hat.
    The first-stage F flags weak instruments (Staiger-Stock rule of thumb F>=10);
    below that, 2SLS is unreliable and the skill gate blocks it.
    """
    n = len(df)
    z = df[instrument].to_numpy(); t = df[treat].to_numpy(); y = df[outcome].to_numpy()
    # stage 1
    Z = np.column_stack([np.ones(n), z])
    b1, se1, dof1 = _ols(Z, t)
    t_hat = Z @ b1
    first_stage_F = float((b1[1] / se1[1]) ** 2)        # single-instrument F = t^2
    # stage 2
    Th = np.column_stack([np.ones(n), t_hat])
    b2, _se2, _ = _ols(Th, y)
    # correct 2SLS SE using structural residuals (not t_hat residuals)
    resid = y - np.column_stack([np.ones(n), t]) @ b2
    XtX_inv = np.linalg.inv(Th.T @ Th)
    sigma2 = (resid @ resid) / (n - 2)
    se = float(np.sqrt((sigma2 * XtX_inv)[1, 1]))
    return _result("iv_2sls", b2[1], se, n - 2, n,
                   {"first_stage_F": round(first_stage_F, 2),
                    "weak_instrument": first_stage_F < 10})


def robustness_value(estimate: "EstimateResult") -> dict:
    """Cinelli-Hazlett robustness value: the share of residual variance an
    unobserved confounder would need to explain in BOTH treatment and outcome to
    bring the estimate to zero. RV near 1 = very robust; near 0 = fragile."""
    t = abs(estimate.point / estimate.se) if estimate.se > 0 else float("inf")
    fq = t / np.sqrt(estimate.n - 2) if estimate.n > 2 else 0.0
    rv = 0.5 * (np.sqrt(fq ** 4 + 4 * fq ** 2) - fq ** 2)
    return {"robustness_value": round(float(rv), 4),
            "interpretation": ("confounder must explain >="
                               f"{rv*100:.1f}% of residual variance in both "
                               "treatment and outcome to nullify the effect"),
            "fragile": bool(rv < 0.1)}


def refutation_battery(fit_point, df, treat="t", n_perm=200, seed=0) -> dict:
    """DoWhy-style refutations with a PROPER permutation null.

    * permutation placebo : shuffle treatment `n_perm` times to build the null
      distribution of the estimate; report a two-sided p-value. A real effect
      has p_perm < 0.05 (not reproducible by random assignment). The mean placebo
      effect should also sit near 0.
    * random common cause : adding an irrelevant covariate must not move it.
    * subset stability    : an 80% subsample must not move it.
    """
    rng = np.random.default_rng(seed)
    base = fit_point(df)

    perm_effects = []
    for _ in range(n_perm):
        pdf = df.copy()
        pdf[treat] = rng.permutation(pdf[treat].to_numpy())
        perm_effects.append(fit_point(pdf))
    perm = np.array(perm_effects)
    p_perm = float((np.abs(perm) >= abs(base)).mean())     # permutation p-value
    placebo_mean = float(perm.mean())

    rcc_df = df.copy(); rcc_df["_rcc"] = rng.normal(size=len(df))
    rcc = fit_point(rcc_df)
    subset = fit_point(df.sample(frac=0.8, random_state=seed))

    def stable(v):
        return abs(v - base) <= max(0.2 * abs(base), 0.1)

    res = {"estimate": round(base, 4),
           "permutation_p": round(p_perm, 4),
           "placebo_mean_effect": round(placebo_mean, 4),
           "placebo_ok": bool(p_perm < 0.05 and abs(placebo_mean) <= max(0.1 * abs(base), 0.1)),
           "random_common_cause": round(rcc, 4), "rcc_ok": bool(stable(rcc)),
           "subset_estimate": round(subset, 4), "subset_ok": bool(stable(subset))}
    res["survived_all"] = bool(res["placebo_ok"] and res["rcc_ok"] and res["subset_ok"])
    return res


def benjamini_hochberg(pvalues, alpha=0.05) -> dict:
    """BH false-discovery-rate control across a battery of tests."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    thresh = (np.arange(1, m + 1) / m) * alpha
    passed = p[order] <= thresh
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    cutoff = (k / m) * alpha if k else 0.0
    rejected = p <= cutoff
    return {"alpha": alpha, "n_tests": m, "n_rejected": int(rejected.sum()),
            "cutoff": round(float(cutoff), 5),
            "rejected": rejected.tolist()}


def e_value(point, ci_low, ci_high, sd_outcome) -> dict:
    """VanderWeele-Ding E-value: the minimum strength of association (on the
    risk-ratio scale) that an unmeasured confounder would need with BOTH
    treatment and outcome to fully explain away the observed effect.

    Continuous effects are mapped to an approximate risk ratio via the
    standardized mean difference (Chinn/VanderWeele: RR ~ exp(0.91*d)). We
    report the E-value for the point estimate and for the confidence limit
    nearest the null (the more conservative, decision-relevant number).
    """
    import numpy as np
    sd = sd_outcome if sd_outcome and sd_outcome > 1e-9 else 1.0

    def _rr(eff):
        rr = float(np.exp(0.91 * (eff / sd)))
        return rr if rr >= 1 else 1.0 / rr

    def _ev(rr):
        return float(rr + np.sqrt(rr * (rr - 1)))

    ev_point = _ev(_rr(point))
    crosses_null = ci_low is not None and ci_high is not None and ci_low <= 0 <= ci_high
    if crosses_null:
        ev_ci = 1.0
    else:
        near = min([ci_low, ci_high], key=abs)
        ev_ci = _ev(_rr(near))
    return {"e_value_point": round(ev_point, 3),
            "e_value_ci": round(ev_ci, 3),
            "interpretation": (f"a confounder would need risk-ratio associations "
                               f">= {ev_point:.2f} with both treatment and outcome to "
                               f"explain the point estimate; >= {ev_ci:.2f} to explain "
                               f"away the CI bound nearest the null"),
            "fragile": ev_ci < 1.25}
