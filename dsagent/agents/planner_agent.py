"""Planner agent — the supervisor.

Given the CEO-level goal and the registry of available tools, the planner LLM
emits a task graph: which tool runs, what it depends on, where human approval is
required. The planner then VALIDATES that graph against the registry (no
hallucinated tools, no cycles, dependencies exist) before handing it to the
orchestrator. Planning is the one place we let the model design control flow;
everything else it does is bounded by a typed tool contract.
"""
from __future__ import annotations

from ..llm.base import LLMClient
from ..runtime.tools import ToolRegistry
from ..planner import AnalysisDAG, Task


class PlannerAgent:
    def __init__(self, llm: LLMClient, registry: ToolRegistry):
        self.llm = llm
        self.registry = registry

    def plan(self, goal: str, context: dict | None = None) -> AnalysisDAG:
        tools = self.registry.catalog_for_planner()
        prompt = (f"goal: {goal}\n"
                  f"available_tools: {tools}\n"
                  f"context: {context or {}}\n"
                  "Produce a task DAG. Each task: id, tool (from available_tools), "
                  "phase, depends_on (list of task ids), requires_human_approval (bool). "
                  "Run independent tasks in parallel by giving them the same dependency. "
                  "Gate the modeling phase behind a human-approval plan review, and gate "
                  "delivery behind a final sign-off.")
        obj, _usage = self.llm.complete_json(
            "You are the lead data-science program planner.", prompt, intent="plan")
        return self._to_dag(obj)

    def _to_dag(self, obj: dict) -> AnalysisDAG:
        valid_tools = set(self.registry.names())
        dag = AnalysisDAG()
        ids = {t["id"] for t in obj.get("tasks", [])}
        for t in obj.get("tasks", []):
            tool = t.get("tool", "")
            if tool not in valid_tools:
                raise ValueError(f"Planner chose unknown tool {tool!r}; "
                                 f"valid: {sorted(valid_tools)}")
            deps = [d for d in t.get("depends_on", []) if d in ids]
            dag.add(Task(
                id=t["id"], phase=t.get("phase", "P?"),
                kind=self.registry.get(tool).kind, tool=tool,
                depends_on=deps,
                requires_human_approval=bool(t.get("requires_human_approval", False)),
                params=t.get("params", {}),
            ))
        dag.validate()   # raises on cycles
        return dag
