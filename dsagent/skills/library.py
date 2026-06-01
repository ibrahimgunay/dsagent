"""The verified-skill library.

Each estimator becomes a Skill with a precondition (when retrieval surfaces it),
a blocking gate (run before fitting), and a fixture that proves it recovers
truth. `build_default_registry()` is what the agent retrieves over.
"""
from __future__ import annotations

import numpy as np
from .base import Skill, GateResult, SkillRegistry
from ..execution import estimators as est, datagen as dg


# ---- gates -------------------------------------------------------------------
def gate_overlap(df, controls=("x",)) -> GateResult:
    try:
        o = est.propensity_overlap(df, controls=[c for c in controls if c in df.columns])
        share = o["share_outside_[0.05,0.95]"]
        # Trim-and-proceed below 5% out-of-support; block on a real positivity
        # violation. (A gate that blocks a correct analysis is a bad gate.)
        passed = share < 0.05
        return GateResult(passed, "overlap_positivity",
                          f"share outside [.05,.95]={share}"
                          + ("" if passed else " — exceeds 5%, estimate withheld"))
    except Exception as e:
        return GateResult(False, "overlap_positivity", f"check failed: {e}")


def gate_pretrends(df) -> GateResult:
    if "cohort" not in df.columns:
        return GateResult(True, "parallel_trends", "no cohort structure; n/a")
    p = est.pretrends_test(df)
    return GateResult(p["pretrends_ok"], "parallel_trends",
                      f"max pre-effect={p['max_pre_effect']}")


def gate_balance(df, treat="t", controls=("x",)) -> GateResult:
    """RCT covariate-balance check: standardized mean diff should be small."""
    cs = [c for c in controls if c in df.columns]
    if not cs or treat not in df.columns:
        return GateResult(True, "randomization_balance", "no covariates to check")
    worst = 0.0
    for c in cs:
        a = df.loc[df[treat] == 1, c]; b = df.loc[df[treat] == 0, c]
        sd = np.sqrt((a.var() + b.var()) / 2) or 1.0
        worst = max(worst, abs(a.mean() - b.mean()) / sd)
    return GateResult(worst < 0.1, "randomization_balance",
                      f"max standardized mean diff={worst:.3f}")


def gate_instrument_strength(df, instrument="z") -> GateResult:
    if instrument not in df.columns:
        return GateResult(False, "instrument_strength", "no instrument column")
    iv = est.iv_2sls(df, instrument=instrument)
    F = iv.diagnostics["first_stage_F"]
    return GateResult(F >= 10, "instrument_strength",
                      f"first-stage F={F} (Staiger-Stock rule F>=10)")


# ---- registry ----------------------------------------------------------------
def build_default_registry() -> SkillRegistry:
    r = SkillRegistry()

    r.register(Skill(
        id="iv_2sls",
        description="Endogenous treatment with a valid instrument -> 2SLS.",
        preconditions=lambda p: p.get("has_instrument", False),
        estimator="iv_2sls",
        identification="Instrument affects outcome only through treatment (exclusion).",
        gate=gate_instrument_strength,
        fixture=lambda: dg.make_iv(ate=1.5, strength=1.2)))

    r.register(Skill(
        id="rct_contrast",
        description="Randomized experiment -> simple difference in means.",
        preconditions=lambda p: p.get("randomized", False),
        estimator="diff_in_means", identification="Randomization removes confounding.",
        gate=gate_balance, config={"treat": "t", "outcome": "y"},
        fixture=lambda: (dg.make_rct(ate=2.0), 2.0)))

    r.register(Skill(
        id="staggered_did_cs",
        description="Staggered adoption with heterogeneous timing -> Callaway-Sant'Anna.",
        preconditions=lambda p: p.get("is_staggered", False),
        estimator="callaway_santanna",
        identification="Group-time ATT vs never-treated; avoids TWFE negative weights.",
        gate=gate_pretrends,
        fixture=lambda: dg.make_staggered(base=0.4)))

    r.register(Skill(
        id="two_period_did",
        description="Two-period panel with parallel trends -> 2x2 DiD.",
        preconditions=lambda p: p.get("is_panel", False) and not p.get("is_staggered", False),
        estimator="did_2x2",
        identification="Differences out time-invariant confounding.",
        config={"treat": "treat", "post": "post", "outcome": "y"},
        fixture=lambda: (dg.make_did_panel(att=1.5), 1.5)))

    r.register(Skill(
        id="observational_dml",
        description="Observational with observed confounders -> Double ML.",
        preconditions=lambda p: (p.get("has_controls", False)
                                 and not p.get("randomized", False)
                                 and not p.get("is_panel", False)),
        estimator="double_ml",
        identification="Unconfoundedness given controls; Neyman-orthogonal DML.",
        gate=gate_overlap, config={"treat": "t", "outcome": "y", "controls": ("x",)},
        fixture=lambda: (dg.make_observational(ate=2.0), 2.0)))

    return r
