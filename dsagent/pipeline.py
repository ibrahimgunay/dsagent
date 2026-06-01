"""One-call entry point for the whole system.

    from dsagent.pipeline import run_analysis
    result = run_analysis(goal, catalog, llm=AnthropicClient(), data_source=ws)

Builds the org, plans the DAG, executes it with data flowing end-to-end, and
returns a compact result: verdict, the causal estimate, deliverables, budget,
and the memo's lineage. This is the surface a product/app calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .catalog import Catalog
from .llm.base import LLMClient
from .llm.stub import StubLLM
from .agents import build_org
from .runtime import Blackboard, Orchestrator, Budget
from .data import SyntheticDataSource, DataStore
from .execution.executor import Executor


@dataclass
class AnalysisResult:
    goal: str
    completed: list[str]
    failed: dict
    halted: str
    verdict: str
    estimate: dict | None
    deliverables: dict
    budget: dict
    blackboard: Blackboard = field(repr=False, default=None)

    def summary(self) -> str:
        e = self.estimate or {}
        pt = f"{e.get('point')} (95% CI {e.get('ci')})" if e else "n/a"
        rec = f", recovered_truth={e.get('recovered')}" if e and "recovered" in e else ""
        return (f"goal: {self.goal}\n"
                f"verdict: {self.verdict}\n"
                f"estimate: {pt}{rec}\n"
                f"deliverables: {list(self.deliverables)}\n"
                f"budget: {self.budget}\n"
                f"failed: {self.failed or 'none'}  halted: {self.halted or 'no'}")


def run_adaptive(goal: str, catalog: Catalog, *, llm: LLMClient | None = None,
                 data_source=None, budget: Budget | None = None,
                 figures_dir: str | None = None) -> dict:
    """Adaptive pipeline: reflection-planned, design committed at runtime from the
    data profile, with event-driven replanning. Returns a summary dict including
    the reflection transcript and the replans that fired."""
    from .planning import AdaptivePlanner, AdaptiveOrchestrator
    llm = llm or StubLLM()
    data_source = data_source or SyntheticDataSource("observational", 2.0)
    registry, _ = build_org(catalog, llm, Executor())

    planner = AdaptivePlanner(set(registry.names()))
    plan = planner.plan(goal)

    bb = Blackboard()
    services = {"data_source": data_source, "datastore": DataStore()}
    if figures_dir:
        services["figures_dir"] = figures_dir
    orch = AdaptiveOrchestrator(registry, bb, planner,
                                budget=budget or Budget(max_usd=2.0), services=services)
    run = orch.run(plan)

    econ_id = next((n.id for n in plan.active()
                    if n.tool == "econometrician" and n.status == "done"), None)
    econ = (bb.value(econ_id, {}) or {}) if econ_id else {}
    return {
        "goal": goal,
        "reflection_log": planner.reflection_log,
        "replans": run.replans,
        "committed_design": econ.get("skill_used") or (econ.get("estimate") or {}).get("skill"),
        "estimate": econ.get("estimate"),
        "completed": run.completed,
        "failed": run.failed,
        "budget": run.budget,
        "blackboard": bb,
    }


def run_analysis(goal: str, catalog: Catalog, *, llm: LLMClient | None = None,
                 data_source=None, budget: Budget | None = None,
                 approval_fn=None, context: dict | None = None,
                 figures_dir: str | None = None) -> AnalysisResult:
    llm = llm or StubLLM()
    data_source = data_source or SyntheticDataSource(scenario="observational",
                                                     true_effect=2.0)
    executor = Executor()
    registry, planner = build_org(catalog, llm, executor)

    dag = planner.plan(goal, context=context or {"databases": sorted(catalog.databases())})

    bb = Blackboard()
    services = {"data_source": data_source, "datastore": DataStore()}
    if figures_dir:
        services["figures_dir"] = figures_dir
    kwargs = {"budget": budget or Budget(max_usd=2.0), "services": services}
    if approval_fn is not None:
        kwargs["approval_fn"] = approval_fn
    orch = Orchestrator(registry, bb, **kwargs)
    res = orch.run(dag)

    crit = bb.value("critic", {}) or {}
    econ = bb.value("econ", {}) or {}
    return AnalysisResult(
        goal=goal, completed=res.completed, failed=res.failed,
        halted=res.halted_reason, verdict=crit.get("verdict", "n/a"),
        estimate=econ.get("estimate"),
        deliverables={"dashboard": bb.value("dashboard"), "memo": bb.value("memo"),
                      "trust_report": bb.value("trust")},
        budget=res.budget, blackboard=bb)
