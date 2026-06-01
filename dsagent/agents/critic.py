"""Critic — the seminar that tears the result apart.

Two layers:
  1. Deterministic BLOCKING gates that cannot be argued with. The headline one:
     'identification before estimation' — a causal artifact without an estimand,
     identifying assumptions, AND a sensitivity analysis fails the run. Plus the
     SQL fan-out gate and the causal overlap gate.
  2. LLM reconciliation across tracks (do the DiD and causal-forest agree?).

Gates run first; if any blocks, the run is marked failed regardless of what the
LLM says. This is what keeps the system honest.
"""
from __future__ import annotations

from ..runtime.tools import Tool, ToolContext
from ..runtime.blackboard import Artifact
from ..llm.base import LLMClient


class CriticAgent(Tool):
    name = "critic"
    kind = "validation"
    description = "Runs blocking validity gates and reconciles results across modeling tracks."
    reads = ["econometrics", "ml", "causal_ml", "sql", "labeling"]

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, ctx: ToolContext) -> Artifact:
        arts = {}
        for k in ctx.depends_on:
            a = ctx.blackboard.get(k)
            if a:
                arts[a.kind] = a.value

        gates = self._run_gates(ctx, arts)
        blocking = [g for g in gates if not g["passed"] and g["blocking"]]

        prompt = ("Reconcile these modeling results and list residual risks.\n"
                  f"econometrics: {arts.get('econometrics')}\n"
                  f"causal_ml: {arts.get('causal_ml')}")
        review, usage = self.llm.complete_json(
            "You reconcile causal estimates across methods.", prompt,
            intent="critic_review")
        ctx.usage = usage

        verdict = "FAILED_GATES" if blocking else review.get("verdict", "pass")
        return self._emit(ctx, {
            "verdict": verdict,
            "gates": gates,
            "blocking_failures": [g["name"] for g in blocking],
            "reconciliation": review.get("reconciliation"),
            "required_caveats": review.get("required_caveats", []),
            "remaining_risks": review.get("remaining_risks", []),
        })

    @staticmethod
    def _run_gates(ctx, arts) -> list[dict]:
        gates = []

        def gate(name, passed, blocking, detail):
            gates.append({"name": name, "passed": bool(passed),
                          "blocking": blocking, "detail": detail})

        econ = arts.get("econometrics", {})
        gate("identification_before_estimation",
             all(k in econ for k in ("estimand", "identifying_assumptions"))
             and bool(econ.get("sensitivity_analysis")),
             True,
             "Causal estimate must ship with estimand + assumptions + sensitivity analysis.")

        est = str(econ.get("estimator", "")).lower().replace("-", " ")
        is_naive_twfe = (("twfe" in est or "two way fixed" in est
                          or "twoway fixed" in est)
                         and not any(k in est for k in
                                     ("callaway", "sun", "abraham", "sant")))
        gate("no_naive_twfe", not is_naive_twfe, True,
             "Reject plain TWFE under staggered timing.")

        sql = arts.get("sql", {})
        unresolved_fanout = [w for w in sql.get("fanout_warnings", []) if "DOUBLE-COUNT" in w]
        gate("sql_fanout_resolved", not unresolved_fanout, True,
             f"{len(unresolved_fanout)} unresolved fan-out double-count risk(s).")

        causal = arts.get("causal_ml", {})
        gate("overlap_positivity_checked",
             "overlap" in str(causal).lower() or "positivity" in str(causal).lower(),
             True, "Causal-ML estimate requires an overlap/positivity check.")

        pre = econ.get("estimate", {}).get("pretrends")
        if pre is not None:
            gate("parallel_trends_supported", bool(pre.get("pretrends_ok")), True,
                 f"Pre-trends placebo max effect = {pre.get('max_pre_effect')}.")

        est_obj = econ.get("estimate") or {}
        if est_obj:
            gate("sensitivity_analysis_present", "sensitivity" in est_obj, True,
                 "Every causal estimate must ship a sensitivity analysis.")
            gate("e_value_reported", "e_value" in est_obj, True,
                 "Report the E-value (min confounder strength to nullify).")
            ref = est_obj.get("refutations")
            if ref is not None:
                gate("survived_refutations", bool(ref.get("survived_all")), True,
                     f"placebo={ref.get('placebo_effect')}, "
                     f"subset_ok={ref.get('subset_ok')}, rcc_ok={ref.get('rcc_ok')}.")
        dq = econ.get("data_quality")
        if dq is not None:
            gate("data_quality_checked", bool(dq.get("ok")), True,
                 f"data-quality issues: {dq.get('issues') or 'none'}")

        return gates


class DashboardBuilderAgent(Tool):
    name = "dashboard_builder"
    kind = "dashboard"
    description = "Produces a semantic-layer-backed dashboard spec (tiles -> governed metrics)."
    reads = ["validation", "semantic"]

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, ctx: ToolContext) -> Artifact:
        obj, usage = self.llm.complete_json(
            "You design analytics dashboards backed by governed metrics.",
            "Design the dashboard for the feature-impact analysis.",
            intent="dashboard_spec")
        ctx.usage = usage
        # If a figures directory is wired in, render REAL charts (not just a spec).
        figdir = ctx.services.get("figures_dir")
        if figdir:
            try:
                from ..execution.viz import render_standard_pack
                obj["figures"] = render_standard_pack(figdir)
            except Exception as e:
                obj["figures_error"] = str(e)
        return self._emit(ctx, obj)


class MemoWriterAgent(Tool):
    name = "memo_writer"
    kind = "memo"
    description = "Writes the stakeholder memo: headline, effect with uncertainty, caveats."
    reads = ["validation"]

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, ctx: ToolContext) -> Artifact:
        caveats = []
        for k in ctx.depends_on:
            a = ctx.blackboard.get(k)
            if a and a.kind == "validation":
                caveats = a.value.get("required_caveats", [])
        obj, usage = self.llm.complete_json(
            "You write executive analytics memos with explicit uncertainty.",
            f"Write the memo. Required caveats to include: {caveats}", intent="memo")
        ctx.usage = usage
        obj.setdefault("caveats", caveats)
        return self._emit(ctx, obj)
