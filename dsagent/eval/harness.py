"""The eval harness — does the system recover known truth, and does it pick the
right design?

Scores four things the architecture doc demands of a trustworthy analyst:
  * recovery  : selected design's estimate is close to truth
  * coverage  : its 95% CI contains truth
  * discrimination : on confounded data the naive contrast is correctly flagged
                     as biased (the system would not be fooled)
  * fdr / null : on an A/A dataset it does NOT falsely declare an effect
Plus a process eval: the econometrician agent emits estimand + identifying
assumptions + sensitivity analysis (estimand-first compliance) and the critic
gates pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import pandas as pd

from ..execution.executor import Executor, select_design
from ..execution import estimators as est
from .scenarios import SCENARIOS, Scenario

REL_TOL = 0.15      # |est-truth| must be within 15% of truth (abs floor for null)
ABS_TOL = 0.15


@dataclass
class ScenarioScore:
    name: str
    selected_estimator: str
    point: float
    ci: tuple
    truth: float
    checks: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(self.checks.values())


def _close(point: float, truth: float) -> bool:
    if abs(truth) < 1e-9:
        return abs(point) <= ABS_TOL
    return abs(point - truth) / abs(truth) <= REL_TOL


def run_scenario(s: Scenario, executor: Executor) -> ScenarioScore:
    made = s.make()
    if isinstance(made, tuple):
        data, truth = made
    else:
        data, truth = made, s.truth
    design = select_design(s.profile)
    res = executor.fit(design["estimator"], data, **design.get("config", {}))

    checks = {}
    if s.is_null:
        checks["no_false_positive"] = res.pvalue >= 0.05
        checks["null_estimate_near_zero"] = abs(res.point) <= ABS_TOL
    else:
        checks["recovers_truth"] = _close(res.point, truth)
        checks["ci_covers_truth"] = res.covers(truth)

    # discrimination: the naive design must be wrong on this data
    if s.expects_naive_bias:
        naive = executor.fit(s.naive_estimator, data, **s.naive_config)
        checks["naive_correctly_biased"] = not naive.covers(truth)
        checks["selected_beats_naive"] = res.covers(truth) and not naive.covers(truth)

    return ScenarioScore(
        name=s.name, selected_estimator=design["estimator"],
        point=round(res.point, 4), ci=(round(res.ci_low, 3), round(res.ci_high, 3)),
        truth=round(truth, 4), checks=checks)


def process_eval(catalog, llm) -> dict:
    """Does the econometrician follow estimand-first, and do the critic gates pass?"""
    from ..agents.modeling import EconometricianAgent, CausalMLAgent
    from ..agents.critic import CriticAgent
    from ..runtime import Blackboard, ToolContext
    from ..execution import datagen as dg

    executor = Executor()
    data = dg.make_observational(ate=2.0)
    bb = Blackboard()

    # estimand-first econometrics, fitting a real estimate
    econ = EconometricianAgent(llm, executor)
    econ.run(ToolContext(blackboard=bb, task_id="econ", depends_on=[],
                         params={"data": data, "profile": {"has_controls": True}}))
    art = bb.value("econ")

    # causal-ML track (runs the overlap/positivity check the critic gates on)
    causal = CausalMLAgent(llm, executor)
    causal.run(ToolContext(blackboard=bb, task_id="causal", depends_on=[],
                           params={"data": data, "controls": ("x",)}))

    # critic sees the full modeling set, as in the real pipeline
    critic = CriticAgent(llm)
    critic.run(ToolContext(blackboard=bb, task_id="critic",
                           depends_on=["econ", "causal"]))
    verdict = bb.value("critic")

    est_obj = art.get("estimate", {})
    return {
        "states_estimand": "estimand" in art,
        "has_identifying_assumptions": bool(art.get("identifying_assumptions")),
        "has_sensitivity_analysis": bool(art.get("sensitivity_analysis")),
        "emitted_numeric_estimate": "point" in est_obj,
        "estimate_recovers_truth": _close(est_obj.get("point", 999), 2.0),
        "critic_gates_pass": not verdict.get("blocking_failures"),
    }


def within_study_eval(executor: Executor) -> dict:
    """Strongest credibility proof: recover an EXPERIMENT's known ATE from its
    OBSERVATIONAL slice. Activates when experiments are available."""
    from ..execution import datagen as dg
    obs, truth = dg.make_within_study(ate=2.0)
    dml = executor.fit("double_ml", obs, controls=("x",))
    naive = est.diff_in_means(obs)
    return {"experiment_truth": truth,
            "observational_dml": round(dml.point, 4),
            "recovered_experiment": dml.covers(truth),
            "naive_biased": not naive.covers(truth),
            "naive_point": round(naive.point, 4)}


def leakage_trap_eval() -> dict:
    """A target-leak feature must be flagged, not silently exploited."""
    from ..execution import datagen as dg
    from ..execution.ml import fit_predictive
    r = fit_predictive(dg.make_leaky(), target="t", features=["x", "leak"])
    return {"auc": round(r.auc, 4), "leakage_flagged": len(r.leakage_flags) > 0,
            "flags": r.leakage_flags}


def retrieval_eval() -> dict:
    """Precision@1 of skill selection across labeled data shapes — directly
    measures 'messy schema -> right method'."""
    from ..skills import build_default_registry
    reg = build_default_registry()
    cases = [
        ({"randomized": True}, "rct_contrast"),
        ({"is_staggered": True}, "staggered_did_cs"),
        ({"is_panel": True}, "two_period_did"),
        ({"has_controls": True}, "observational_dml"),
        ({"has_instrument": True}, "iv_2sls"),
    ]
    hits = []
    for profile, expected in cases:
        best = reg.best(profile)
        hits.append(best is not None and best.id == expected)
    return {"cases": len(cases), "precision_at_1": round(sum(hits) / len(cases), 3),
            "all_correct": all(hits)}


def run_all(catalog=None, llm=None, include_experiments: bool = False) -> dict:
    executor = Executor()
    scores = [run_scenario(s, executor) for s in SCENARIOS]
    out = {"scenarios": scores,
           "passed": sum(s.passed for s in scores), "total": len(scores),
           "leakage": leakage_trap_eval(),
           "retrieval": retrieval_eval()}
    if include_experiments:
        out["within_study"] = within_study_eval(executor)
    if catalog is not None and llm is not None:
        out["process"] = process_eval(catalog, llm)
    return out
