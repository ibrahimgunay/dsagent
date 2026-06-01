"""Executor + design selector.

`Executor.fit` turns an estimator name + data into an `EstimateResult` — the
deterministic compute the modeling agents delegate to. `select_design` encodes
the rule an econometrician applies: pick the identification strategy from the
data's structure (randomized? panel? observed confounders?). The eval harness
scores whether this selection recovers truth.
"""
from __future__ import annotations

import pandas as pd
from . import estimators as est


class Executor:
    """Maps an estimator name -> fit. Production swaps internals for warehouse
    pushdown / a heavier stats backend without changing this surface."""

    def fit(self, estimator: str, data: pd.DataFrame, **cfg) -> est.EstimateResult:
        e = estimator.lower()
        if e in ("diff_in_means", "naive"):
            return est.diff_in_means(data, **cfg)
        if e in ("regression_adjust", "adjusted", "ancova"):
            return est.regression_adjust(data, **cfg)
        if e in ("did", "did_2x2", "diff_in_diff"):
            return est.did_2x2(data, **cfg)
        if e in ("callaway_santanna", "cs", "staggered_did"):
            return est.callaway_santanna(data, **cfg)
        if e in ("twfe", "twfe_static"):
            return est.twfe_static(data, **cfg)
        if e in ("double_ml", "dml"):
            return est.double_ml(data, **cfg)
        if e in ("iv_2sls", "iv", "2sls"):
            return est.iv_2sls(data, **cfg)
        raise ValueError(f"unknown estimator: {estimator}")

    def overlap(self, data, **cfg):
        return est.propensity_overlap(data, **cfg)

    def cate(self, data, **cfg):
        return est.cate_by_subgroup(data, **cfg)


def profile_data(data: pd.DataFrame) -> dict:
    """Lightweight structural profile used to choose a design."""
    cols = set(data.columns)
    staggered = False
    if "cohort" in cols:
        adopt = [c for c in data["cohort"].unique() if c != 0]
        staggered = len(set(adopt)) > 1            # >1 distinct adoption time
    return {
        "is_staggered": staggered,
        "is_panel": {"post", "treat"}.issubset(cols) or "unit" in cols,
        "has_controls": any(c in cols for c in ("x", "tier")),
        "has_instrument": "z" in cols,
        "randomized": False,   # set by the scenario / experiment metadata
    }


def select_design(profile: dict) -> dict:
    """Encode the estimand-first design choice.

    Returns the recommended estimator + a one-line identification rationale.
    """
    if profile.get("has_instrument"):
        return {"estimator": "iv_2sls",
                "identification": "Instrument satisfies relevance + exclusion; 2SLS.",
                "config": {}}
    if profile.get("is_staggered"):
        return {"estimator": "callaway_santanna",
                "identification": ("Staggered adoption with heterogeneous timing; "
                                   "group-time ATT vs never-treated (avoids TWFE bias)."),
                "config": {}}
    if profile.get("is_panel"):
        return {"estimator": "did_2x2",
                "identification": "Parallel-trends DiD (unit + time effects).",
                "config": {"treat": "treat", "post": "post", "outcome": "y"}}
    if profile.get("randomized"):
        return {"estimator": "diff_in_means",
                "identification": "Randomization -> unconfounded; simple contrast.",
                "config": {"treat": "t", "outcome": "y"}}
    if profile.get("has_controls"):
        return {"estimator": "double_ml",
                "identification": "Unconfoundedness given observed controls; DML for robustness.",
                "config": {"treat": "t", "outcome": "y", "controls": ("x",)}}
    return {"estimator": "diff_in_means",
            "identification": "No structure detected; naive contrast (weakest).",
            "config": {"treat": "t", "outcome": "y"}}
