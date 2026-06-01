"""The adaptive planner.

Three capabilities a fixed DAG doesn't have:

  1. REFLECTION — propose an initial plan, critique it against staff-level rules
     (approval gates present? a critic before delivery? a sign-off? dangling
     deps? within budget?), revise, and loop until the critique is clean. The
     transcript shows the plan improving.

  2. MULTI-HYPOTHESIS BRANCHING — emit several candidate causal designs (DiD /
     staggered-CS / DML / IV) as mutually-exclusive branches, and commit to one
     at runtime once the data is profiled (`resolve_design`).

  3. EVENT-DRIVEN REPLANNING — during execution, react to events: a gate failure
     inserts a repair step; thin budget prunes optional branches; a discovery
     selects a branch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from .plangraph import PlanGraph, PlanNode


_DESIGN_BY_PROFILE = [
    ("has_instrument", "iv"),
    ("is_staggered", "cs"),
    ("is_panel", "did"),
    ("has_controls", "dml"),
]
_DESIGN_TOOLS = {"iv": "econometrician", "cs": "econometrician",
                 "did": "econometrician", "dml": "econometrician"}


@dataclass
class CritiqueIssue:
    code: str
    detail: str
    fix: str


class AdaptivePlanner:
    def __init__(self, tool_names: set[str] | None = None, budget_cost: float = 999):
        self.tools = tool_names or set()
        self.budget_cost = budget_cost
        self.reflection_log: list[str] = []

    # ---------------------------------------------------------------- propose
    def propose_initial(self, goal: str) -> PlanGraph:
        """A reasonable but deliberately imperfect first draft (no approvals, no
        critic, no sign-off) — the reflection loop will repair it."""
        g = PlanGraph()
        g.add(PlanNode("profile", "profiler", "P0"))
        g.add(PlanNode("semantic", "semantic_modeler", "P0", ["profile"]))
        g.add(PlanNode("joins", "join_analyzer", "P0", ["semantic"]))
        g.add(PlanNode("sql", "sql_author", "P1", ["joins"]))
        g.add(PlanNode("fetch", "data_executor", "P1", ["sql"]))
        # candidate causal designs (one survives at runtime)
        for val in ("iv", "cs", "did", "dml"):
            g.add(PlanNode(f"design_{val}", _DESIGN_TOOLS[val], "P3", ["fetch"],
                           branch_key="design", branch_value=val,
                           cost=2.0, value=5.0))
        g.add(PlanNode("ml", "ml_engineer", "P3", ["fetch"], optional=True,
                       cost=1.5, value=1.5))
        g.add(PlanNode("causal", "causal_ml", "P3", ["fetch"], cost=2.0, value=3.0))
        g.add(PlanNode("labeler", "labeler", "P3", ["fetch"], optional=True,
                       cost=1.0, value=1.0))
        # delivery (draft wires straight off modeling — missing critic/approval)
        g.add(PlanNode("dashboard", "dashboard_builder", "P5",
                       ["design_dml", "causal"]))
        g.add(PlanNode("memo", "memo_writer", "P5", ["design_dml", "causal"]))
        g.add(PlanNode("trust", "trust_report", "P5", ["design_dml"]))
        return g

    # --------------------------------------------------------------- critique
    def critique(self, g: PlanGraph) -> list[CritiqueIssue]:
        issues: list[CritiqueIssue] = []
        active = {n.id: n for n in g.active()}

        modeling = [n for n in active.values()
                    if n.tool in ("econometrician", "ml_engineer", "causal_ml", "labeler")]
        # 1. human approval before modeling
        if not any(n.requires_human_approval and n.phase in ("P2", "P1")
                   for n in active.values()):
            issues.append(CritiqueIssue(
                "no_plan_approval", "Modeling can start without human review.",
                "insert_plan_review"))
        # 2. a critic before delivery
        if not any(n.tool == "critic" for n in active.values()):
            issues.append(CritiqueIssue(
                "no_critic", "No validity-gate critic before deliverables ship.",
                "insert_critic"))
        # 3. a final sign-off
        if not any(n.requires_human_approval and n.phase == "P5"
                   for n in active.values()):
            issues.append(CritiqueIssue(
                "no_signoff", "Deliverables ship with no final human sign-off.",
                "append_signoff"))
        # 4. dangling dependencies
        for n in active.values():
            missing = [d for d in n.depends_on if d not in active]
            if missing:
                issues.append(CritiqueIssue(
                    "dangling_dep", f"{n.id} depends on missing {missing}.",
                    f"drop_dep::{n.id}"))
        # 5. budget
        if g.est_cost() > self.budget_cost:
            issues.append(CritiqueIssue(
                "over_budget", f"est cost {g.est_cost():.1f} > budget {self.budget_cost}.",
                "prune_optional"))
        # 6. unknown tools
        for n in active.values():
            if self.tools and n.tool not in self.tools:
                issues.append(CritiqueIssue(
                    "unknown_tool", f"{n.id} uses unregistered tool {n.tool!r}.",
                    f"drop_node::{n.id}"))
        return issues

    # ----------------------------------------------------------------- revise
    def revise(self, g: PlanGraph, issues: list[CritiqueIssue]) -> PlanGraph:
        for iss in issues:
            if iss.fix == "insert_plan_review":
                g.add(PlanNode("plan_review", "noop", "P2", ["fetch"],
                               requires_human_approval=True))
                for n in g.active():
                    if n.tool in ("econometrician", "ml_engineer", "causal_ml", "labeler") \
                            and "plan_review" not in n.depends_on:
                        n.depends_on.append("plan_review")
            elif iss.fix == "insert_critic":
                modeling = [n.id for n in g.active() if n.tool in
                            ("econometrician", "ml_engineer", "causal_ml", "labeler")]
                g.add(PlanNode("critic", "critic", "P4", modeling))
                for n in g.active():
                    if n.tool in ("dashboard_builder", "memo_writer", "trust_report") \
                            and "critic" not in n.depends_on:
                        n.depends_on.append("critic")
            elif iss.fix == "append_signoff":
                delivery = [n.id for n in g.active()
                            if n.tool in ("dashboard_builder", "memo_writer", "trust_report")]
                g.add(PlanNode("signoff", "noop", "P5", delivery,
                               requires_human_approval=True))
            elif iss.fix == "prune_optional":
                for n in sorted(g.active(), key=lambda x: x.value):
                    if n.optional:
                        g.prune(n.id)
                        if g.est_cost() <= self.budget_cost:
                            break
            elif iss.fix.startswith("drop_dep::"):
                nid = iss.fix.split("::")[1]
                active = {n.id for n in g.active()}
                g.nodes[nid].depends_on = [d for d in g.nodes[nid].depends_on if d in active]
            elif iss.fix.startswith("drop_node::"):
                g.prune(iss.fix.split("::")[1])
        return g

    def plan(self, goal: str, max_iters: int = 6) -> PlanGraph:
        """Reflection loop: propose -> (critique -> revise)* until clean."""
        g = self.propose_initial(goal)
        self.reflection_log = [f"draft: {len(g.active())} nodes, "
                               f"est_cost={g.est_cost():.1f}"]
        for i in range(max_iters):
            issues = self.critique(g)
            if not issues:
                self.reflection_log.append(f"iter {i+1}: clean — plan accepted")
                break
            self.reflection_log.append(
                f"iter {i+1}: {len(issues)} issue(s): " +
                ", ".join(s.code for s in issues))
            g = self.revise(g, issues)
        g.validate()
        return g

    # ---------------------------------------------------- runtime adaptation
    def resolve_design(self, g: PlanGraph, profile: dict) -> str:
        """Commit to a design branch given the discovered data profile."""
        choice = "dml"
        for key, val in _DESIGN_BY_PROFILE:
            if profile.get(key):
                choice = val
                break
        kept = g.select_branch("design", choice)
        return kept or choice

    def on_event(self, g: PlanGraph, event: str, **kw) -> str:
        """React to a runtime event by mutating the plan. Returns a log line."""
        if event == "data_profiled":
            kept = self.resolve_design(g, kw.get("profile", {}))
            return f"replan: committed to design '{kept}' from data profile"
        if event == "gate_failed":
            target = kw.get("node_id"); gate = kw.get("gate", "gate")
            rep = PlanNode(f"repair_{target}", "noop", "P3")
            g.insert_before(target, rep)
            return f"replan: inserted repair before '{target}' (gate '{gate}' failed)"
        if event == "low_budget":
            pruned = []
            for n in sorted(g.active(), key=lambda x: x.value):
                if n.optional:
                    g.prune(n.id); pruned.append(n.id)
            return f"replan: pruned optional nodes under budget: {pruned}"
        return "no-op"
