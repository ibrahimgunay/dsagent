"""Modeling sub-agents — LLM-driven judgment.

These are the "Harvard professor / Meta principal DS / frontier-lab" roles.
Each calls the LLM for a *structured* plan (not prose), which the critic then
gates. The econometrician is bound by the estimand-first contract: it must
return an estimand, a DAG, identifying assumptions, and a sensitivity analysis,
or the critic fails the run.
"""
from __future__ import annotations

from ..runtime.tools import Tool, ToolContext
from ..runtime.blackboard import Artifact
from ..llm.base import LLMClient


class _LLMAgent(Tool):
    intent = ""
    system = "You are an expert. Return structured JSON only."

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def _ask(self, ctx: ToolContext, prompt: str) -> dict:
        obj, usage = self.llm.complete_json(self.system, prompt, intent=self.intent)
        ctx.usage = usage
        return obj


def _resolve_data(ctx: ToolContext):
    """Find a dataframe for this task: from the DataStore via an upstream
    dataset reference (pipeline path), else from params (eval/direct path)."""
    store = ctx.services.get("datastore")
    if store is not None:
        for k in ctx.depends_on:
            a = ctx.blackboard.get(k)
            if a and a.kind == "dataset" and a.value.get("dataset_ref"):
                df = store.get(a.value["dataset_ref"])
                return df, a.value
    return ctx.params.get("data"), {}


class EconometricianAgent(_LLMAgent):
    name = "econometrician"
    kind = "econometrics"
    description = ("Estimand-first causal econometrics: states estimand + DAG + "
                   "identification, picks the modern estimator, attaches sensitivity analysis.")
    intent = "econometrics"
    system = ("You are a Harvard econometrics professor. Always work estimand-first: "
              "estimand -> DAG -> identification -> estimator -> robustness. Prefer "
              "modern estimators (Callaway-Sant'Anna over TWFE for staggered timing).")

    def __init__(self, llm, executor=None, skills=None):
        super().__init__(llm)
        self.executor = executor
        if skills is None:
            from ..skills import build_default_registry
            skills = build_default_registry()
        self.skills = skills

    def run(self, ctx: ToolContext) -> Artifact:
        q = ctx.params.get("question", "effect of feature adoption on retention")
        plan = self._ask(ctx, f"question: {q}\nDesign the causal study.")
        data, meta = _resolve_data(ctx)
        if data is not None and self.executor is not None:
            from ..execution.executor import profile_data
            profile = {**profile_data(data), **ctx.params.get("profile", {})}
            skill = self.skills.best(profile)
            if skill is None:
                plan["estimate"] = None
                plan["blocked_reason"] = "no skill matched the data shape"
                return self._emit(ctx, plan)

            plan["skill_used"] = skill.id
            plan["identification"] = skill.identification
            # GATE-AS-YOU-GO: the skill's assumption check must pass before fitting
            gate = skill.check_gate(data)
            # SELF-REPAIR: if overlap fails, attempt a principled fix (trim to
            # common support) and re-check, rather than just giving up.
            if not gate.passed and gate.name == "overlap_positivity":
                repaired = _trim_common_support(data, skill.config.get("controls", ("x",)))
                if repaired is not None and len(repaired) >= 0.5 * len(data):
                    regate = skill.check_gate(repaired)
                    if regate.passed:
                        data = repaired
                        gate = regate
                        plan["repaired_by"] = ("trimmed to common support "
                                               f"(kept {len(repaired)}/{len(_resolve_data(ctx)[0])} rows)")
            plan["precondition_gate"] = {"name": gate.name, "passed": gate.passed,
                                         "detail": gate.detail}
            if not gate.passed:
                plan["estimate"] = None
                plan["blocked_reason"] = f"gate '{gate.name}' failed before fitting: {gate.detail}"
                return self._emit(ctx, plan)

            res = self.executor.fit(skill.estimator, data, **skill.config)
            est = {"estimator": skill.estimator, "skill": skill.id,
                   "identification": skill.identification,
                   "point": round(res.point, 4), "se": round(res.se, 4),
                   "ci": [round(res.ci_low, 4), round(res.ci_high, 4)],
                   "pvalue": round(res.pvalue, 4), "n": res.n}
            if "true_effect" in meta:
                est["true_effect"] = meta["true_effect"]
                est["recovered"] = res.covers(meta["true_effect"])
            if skill.estimator in ("callaway_santanna", "did_2x2") \
                    and "cohort" in getattr(data, "columns", []):
                from ..execution.estimators import pretrends_test
                est["pretrends"] = pretrends_test(data)
            # v3: sensitivity analysis + automated refutation battery (always, for
            # cross-sectional designs where the refuters are well-defined)
            from ..execution import estimators as _e
            est["sensitivity"] = _e.robustness_value(res)
            outcome_col = skill.config.get("outcome", "y")
            if outcome_col in getattr(data, "columns", []):
                est["e_value"] = _e.e_value(res.point, res.ci_low, res.ci_high,
                                            float(data[outcome_col].std()))
            from ..execution.estimand import estimand_for, data_quality_report
            est["estimand"] = estimand_for(skill.id).to_dict()
            plan["data_quality"] = data_quality_report(
                data, target=skill.config.get("treat", "t"))
            if skill.estimator in ("double_ml", "regression_adjust", "diff_in_means") \
                    and "t" in getattr(data, "columns", []):
                ctrls = tuple(skill.config.get("controls", ("x",)))
                est["refutations"] = _e.refutation_battery(
                    lambda df: _e.regression_adjust(
                        df, controls=[c for c in ctrls if c in df.columns]).point
                    if any(c in df.columns for c in ctrls) else _e.diff_in_means(df).point,
                    data)
            plan["estimate"] = est
        return self._emit(ctx, plan)


class MLEngineerAgent(_LLMAgent):
    name = "ml_engineer"
    kind = "ml"
    description = "Predictive modeling with leakage-safe CV, calibration, and uncertainty."
    intent = "ml_plan"
    system = ("You are a Meta principal data scientist. Insist on point-in-time "
              "features, leakage-safe CV, and calibrated uncertainty.")

    def run(self, ctx: ToolContext) -> Artifact:
        plan = self._ask(ctx, "Design the predictive model and validation.")
        data, _meta = _resolve_data(ctx)
        if data is not None:
            from ..execution.ml import fit_predictive
            # demonstrate a real, leakage-safe fit (propensity-style model)
            target = ctx.params.get("target", "t" if "t" in data.columns else None)
            feats = ctx.params.get("features",
                                   [c for c in ("x", "tier") if c in data.columns])
            if target and feats:
                res = fit_predictive(data, target=target, features=feats,
                                     group=ctx.params.get("group"))
                plan["fit"] = res.as_dict()
                try:
                    from ..execution.ml import conformal_classify
                    plan["fit"]["conformal"] = conformal_classify(
                        data, target=target, features=feats)
                except Exception:
                    pass
        return self._emit(ctx, plan)


class CausalMLAgent(_LLMAgent):
    name = "causal_ml"
    kind = "causal_ml"
    description = "Heterogeneous treatment effects via DML / causal forests with overlap gates."
    intent = "causal_plan"
    system = ("You are a causal-ML research scientist. Use Double ML / causal forests; "
              "positivity/overlap is a blocking gate before any estimate.")

    def __init__(self, llm, executor=None):
        super().__init__(llm)
        self.executor = executor

    def run(self, ctx: ToolContext) -> Artifact:
        plan = self._ask(ctx, "Design the heterogeneous-effects analysis.")
        data, _meta = _resolve_data(ctx)
        if data is not None and self.executor is not None:
            controls = tuple(ctx.params.get("controls", ("x",)))
            overlap = self.executor.overlap(data, controls=controls)
            plan["overlap_check"] = overlap
            if overlap.get("overlap_ok", True):       # blocking gate respected
                res = self.executor.fit("double_ml", data, controls=controls)
                plan["estimate"] = {"estimator": "double_ml",
                                    "point": round(res.point, 4), "se": round(res.se, 4),
                                    "ci": [round(res.ci_low, 4), round(res.ci_high, 4)]}
                if "tier" in data.columns:
                    plan["cate"] = self.executor.cate(data, group="tier", controls=controls)
            else:
                plan["estimate"] = None
                plan["blocked_reason"] = "overlap/positivity failed; estimate withheld"
        return self._emit(ctx, plan)


class LabelerAgent(_LLMAgent):
    name = "labeler"
    kind = "labeling"
    description = "Weak-supervision + LLM labeling plan for free-text fields, with QC and provenance."
    intent = "labeling_plan"
    system = ("You design LLM labeling pipelines with weak supervision, gold-set QC, "
              "confidence routing, and provenance on every label.")
    reads = ["profiling"]

    def run(self, ctx: ToolContext) -> Artifact:
        free_text = []
        for k in ctx.depends_on:
            art = ctx.blackboard.get(k)
            if art and art.kind == "profiling":
                free_text = art.value.get("free_text_fields", [])
        prompt = f"free_text_fields: {free_text}\nDesign the labeling/extraction pipeline."
        return self._emit(ctx, self._ask(ctx, prompt))


def _trim_common_support(data, controls=("x",)):
    """Trim rows whose propensity is outside [0.05, 0.95] — restores positivity
    on the overlapping subpopulation (estimand becomes ATT on common support)."""
    try:
        from sklearn.linear_model import LogisticRegression
        import numpy as np
        cs = [c for c in controls if c in data.columns]
        if not cs or "t" not in data.columns:
            return None
        ps = LogisticRegression(max_iter=200).fit(data[cs], data["t"]).predict_proba(data[cs])[:, 1]
        keep = (ps >= 0.05) & (ps <= 0.95)
        return data[keep].copy()
    except Exception:
        return None
