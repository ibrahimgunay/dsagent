"""Synthetic data with KNOWN ground-truth effects.

Each generator plants a true ATE/ATT so the eval harness can score whether the
system recovers it. Crucially, the observational generator builds in confounding
so a naive comparison is *biased* — that's how we prove the eval catches a wrong
design, not just a wrong number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_rct(ate: float = 2.0, n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Randomized treatment: diff-in-means is unbiased."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)                      # covariate (prognostic, not confounding)
    t = rng.binomial(1, 0.5, size=n)            # randomized -> independent of x
    y = 1.0 + ate * t + 1.5 * x + rng.normal(scale=1.0, size=n)
    return pd.DataFrame({"y": y, "t": t, "x": x})


def make_observational(ate: float = 2.0, confounding: float = 1.8,
                       n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Treatment depends on a confounder that also drives the outcome.

    Naive diff-in-means is biased upward by ~confounding * (Δ mean x). Adjusting
    for x (regression / DML) recovers `ate`.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    p = _sigmoid(1.2 * x)                        # propensity depends on x
    t = rng.binomial(1, p)
    y = 1.0 + ate * t + confounding * x + rng.normal(scale=1.0, size=n)
    return pd.DataFrame({"y": y, "t": t, "x": x})


def make_did_panel(att: float = 1.5, n_units: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Two-period panel with parallel trends and a known ATT.

    Treated-group selection is correlated with the unit fixed effect (so a
    cross-sectional comparison is biased), but DiD differences it out.
    """
    rng = np.random.default_rng(seed)
    treat = rng.binomial(1, 0.5, size=n_units)
    unit_fe = rng.normal(size=n_units) + 0.8 * treat   # selection on level
    rows = []
    for i in range(n_units):
        for post in (0, 1):
            time_fe = 0.5 * post                        # common trend -> parallel
            eff = att * (treat[i] == 1 and post == 1)
            y = 2.0 + unit_fe[i] + time_fe + eff + rng.normal(scale=0.8)
            rows.append((i, treat[i], post, y))
    return pd.DataFrame(rows, columns=["unit", "treat", "post", "y"])


def make_heterogeneous(ate_low: float = 0.5, ate_high: float = 3.0,
                       n: int = 6000, seed: int = 0) -> pd.DataFrame:
    """Effect varies by subgroup (for CATE recovery). True ATE is the mix."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    tier = rng.integers(0, 2, size=n)               # 0 = low, 1 = high responder
    p = _sigmoid(0.8 * x)
    t = rng.binomial(1, p)
    eff = np.where(tier == 1, ate_high, ate_low)
    y = 1.0 + eff * t + 1.5 * x + rng.normal(scale=1.0, size=n)
    return pd.DataFrame({"y": y, "t": t, "x": x, "tier": tier}), {0: ate_low, 1: ate_high}


def make_null(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """A/A: no true effect. Used to check false-positive control."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t = rng.binomial(1, 0.5, size=n)
    y = 1.0 + 1.5 * x + rng.normal(scale=1.0, size=n)   # t has zero coefficient
    return pd.DataFrame({"y": y, "t": t, "x": x})


def make_staggered(base: float = 0.4, n_units: int = 900, n_periods: int = 6,
                   seed: int = 0):
    """Staggered adoption with DYNAMIC effects (effect grows with exposure).

    Cohorts adopt at different times; a never-treated group serves as clean
    controls. Effect_{i,t} = base * (t - g_i + 1) for t >= g_i. Because effect
    size varies across cohort/time cells, a single-dummy TWFE is biased
    (Goodman-Bacon), while group-time aggregation (Callaway-Sant'Anna) recovers
    the true average treated-cell effect.

    Returns (long_df, true_att) where true_att is the mean realized effect over
    all treated (unit, period) cells.
    """
    rng = np.random.default_rng(seed)
    cohorts = [2, 3, 4]                          # adoption periods (0-indexed)
    g = rng.choice(cohorts + [0], size=n_units)  # 0 == never-treated
    unit_fe = rng.normal(size=n_units)
    time_fe = np.linspace(0, 1.0, n_periods)
    rows, eff_cells = [], []
    for i in range(n_units):
        gi = g[i]
        for t in range(n_periods):
            treated_now = (gi != 0) and (t >= gi)
            eff = base * (t - gi + 1) if treated_now else 0.0
            if treated_now:
                eff_cells.append(eff)
            y = 3.0 + unit_fe[i] + time_fe[t] + eff + rng.normal(scale=0.5)
            rows.append((i, gi, t, int(treated_now), y))
    df = pd.DataFrame(rows, columns=["unit", "cohort", "period", "treated", "y"])
    true_att = float(np.mean(eff_cells))
    return df, true_att


def make_simpsons(effect: float = 1.0, n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Simpson's paradox trap: the within-group effect is POSITIVE, but the
    naive aggregate association is NEGATIVE because treatment is concentrated in
    the lower-baseline group. Adjusting for x (the group) recovers the truth.
    Returns df with columns y, t, x; true effect = `effect`.
    """
    rng = np.random.default_rng(seed)
    x = rng.binomial(1, 0.5, size=n)               # group / confounder
    p = np.where(x == 1, 0.2, 0.8)                 # high-baseline group rarely treated
    t = rng.binomial(1, p)
    y = 1.0 + effect * t + 3.0 * x + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"y": y, "t": t, "x": x})


def make_leaky(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Target-leak trap: feature `leak` is a near-copy of the label. A leakage
    scan must flag it; otherwise CV AUC is deceptively ~1.0. Returns df with
    target `t` and features x, leak."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t = rng.binomial(1, _sigmoid(0.8 * x))
    leak = t + rng.normal(scale=0.01, size=n)      # leaks the label
    return pd.DataFrame({"t": t, "x": x, "leak": leak})


def make_within_study(ate: float = 2.0, n: int = 5000, seed: int = 0):
    """Within-study design: a randomized experiment defines the TRUE ATE, and a
    separate OBSERVATIONAL slice (treatment depends on a confounder) is what the
    agent must analyze. Recovering the experiment's answer from the observational
    slice is the strongest credibility proof.

    Returns (observational_df, experiment_truth).
    """
    rng = np.random.default_rng(seed)
    # observational slice: confounded selection on x
    x = rng.normal(size=n)
    p = _sigmoid(1.0 * x)
    t = rng.binomial(1, p)
    y = 1.0 + ate * t + 1.5 * x + rng.normal(scale=1.0, size=n)
    obs = pd.DataFrame({"y": y, "t": t, "x": x})
    # the experiment would estimate `ate` unbiasedly; that's our ground truth
    return obs, float(ate)


def make_iv(ate: float = 1.5, strength: float = 1.2, n: int = 4000, seed: int = 0):
    """Endogenous treatment with a valid instrument z.

    An unobserved confounder u drives both treatment and outcome, so OLS y~t is
    biased. The instrument z affects t only through the first stage, so 2SLS
    recovers `ate`. Lower `strength` -> weak instrument (low first-stage F).
    Returns (df[y,t,z], true_ate).
    """
    rng = np.random.default_rng(seed)
    u = rng.normal(size=n)                       # unobserved confounder
    z = rng.normal(size=n)                       # instrument
    t = strength * z + u + rng.normal(scale=0.5, size=n)
    y = 1.0 + ate * t + 2.0 * u + rng.normal(scale=1.0, size=n)
    return pd.DataFrame({"y": y, "t": t, "z": z}), float(ate)
