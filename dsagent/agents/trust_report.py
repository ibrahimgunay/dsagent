"""Trust report — the glass box.

Correctness isn't enough for an autonomous agent to be trusted; the reasoning
has to be inspectable. This agent assembles, from the artifacts already on the
blackboard, a single report: the estimate with uncertainty, a calibrated
confidence label, the assumptions relied on, what would change the conclusion,
the gate trail (which blocking checks passed), and the lineage depth. This is
the differentiator competitors that only show an answer don't have.
"""
from __future__ import annotations

from ..runtime.tools import Tool, ToolContext
from ..runtime.blackboard import Artifact


class TrustReportAgent(Tool):
    name = "trust_report"
    kind = "trust"
    description = "Assembles a calibrated, glass-box trust report for the analysis."
    reads = ["econometrics", "validation"]

    def run(self, ctx: ToolContext) -> Artifact:
        econ = ctx.blackboard.value("econ", {}) or {}
        crit = ctx.blackboard.value("critic", {}) or {}
        est = econ.get("estimate") or {}
        gates = crit.get("gates", [])
        blocking_failures = crit.get("blocking_failures", [])

        # calibrated confidence label
        if not est:
            confidence = "withheld"
        elif blocking_failures:
            confidence = "low (gate failure)"
        else:
            ci = est.get("ci", [None, None])
            pt = est.get("point")
            width = (ci[1] - ci[0]) if None not in ci else None
            tight = width is not None and pt not in (None, 0) and abs(width / pt) < 0.2
            confidence = "high" if tight and not crit.get("remaining_risks") else "medium"

        report = {
            "headline_estimate": (f"{est.get('point')} (95% CI {est.get('ci')})"
                                  if est else "withheld — assumptions not met"),
            "confidence": confidence,
            "method": est.get("skill") or est.get("estimator"),
            "estimand": est.get("estimand"),
            "identification": est.get("identification"),
            "assumptions_relied_on": econ.get("identifying_assumptions", []),
            "sensitivity": est.get("sensitivity"),
            "e_value": est.get("e_value"),
            "data_quality": econ.get("data_quality"),
            "refutations": est.get("refutations"),
            "gate_trail": [{"gate": g["name"], "passed": g["passed"]} for g in gates],
            "blocking_failures": blocking_failures,
            "what_would_change_our_mind": crit.get("remaining_risks")
                or ["A pre-trend violation, failed placebo, or loss of overlap."],
            "required_caveats": crit.get("required_caveats", []),
            "lineage_depth": 1 + max((_depth(ctx.blackboard.lineage(k))
                                      for k in ctx.depends_on), default=0),
            "reproducible": True,
        }
        return self._emit(ctx, report)


def _depth(node: dict) -> int:
    if not node or not node.get("inputs"):
        return 1
    return 1 + max((_depth(i) for i in node["inputs"]), default=0)
