"""Orchestration spine.

Compiles an analysis goal into a dependency DAG and resolves which tasks run in
parallel vs sequentially. Uses networkx topological *generations*: each
generation is a set of tasks with no remaining dependencies, i.e. a parallel
batch. This is the scaffolding the supervisor LLM populates; here we wire the
canonical phases from the design doc (ingest -> profile -> semantic ->
[EDA | DQ | discovery] -> plan -> [econ | ml | causal] -> reconcile -> deliver).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any
import networkx as nx


@dataclass
class Task:
    id: str
    phase: str
    kind: str                       # data_prep / eda / modeling / validation / delivery
    tool: str = ""                  # registry tool name the orchestrator dispatches to
    depends_on: list[str] = field(default_factory=list)
    fn: Callable[..., Any] | None = None
    requires_human_approval: bool = False
    params: dict = field(default_factory=dict)


class AnalysisDAG:
    def __init__(self) -> None:
        self.g = nx.DiGraph()

    def add(self, task: Task) -> "AnalysisDAG":
        self.g.add_node(task.id, task=task)
        for dep in task.depends_on:
            self.g.add_edge(dep, task.id)
        return self

    def validate(self) -> None:
        if not nx.is_directed_acyclic_graph(self.g):
            cycle = nx.find_cycle(self.g)
            raise ValueError(f"Cyclic dependency in analysis plan: {cycle}")

    def execution_batches(self) -> list[list[Task]]:
        """Ordered list of parallel batches. Items within a batch run concurrently."""
        self.validate()
        return [[self.g.nodes[n]["task"] for n in sorted(gen)]
                for gen in nx.topological_generations(self.g)]

    def describe(self) -> str:
        lines = []
        for i, batch in enumerate(self.execution_batches(), 1):
            kind = "sequential" if len(batch) == 1 else f"parallel x{len(batch)}"
            lines.append(f"  Batch {i} ({kind}):")
            for t in batch:
                hp = "  [HUMAN APPROVAL]" if t.requires_human_approval else ""
                lines.append(f"     - [{t.phase}] {t.id}{hp}")
        return "\n".join(lines)


def default_plan() -> AnalysisDAG:
    """The canonical pilot plan from the design doc, as an executable DAG."""
    dag = AnalysisDAG()
    dag.add(Task("ingest_schema", "P0-ground-truth", "data_prep"))
    dag.add(Task("profile_columns", "P0-ground-truth", "data_prep",
                 depends_on=["ingest_schema"]))
    dag.add(Task("build_semantic_layer", "P0-ground-truth", "data_prep",
                 depends_on=["profile_columns"]))
    dag.add(Task("build_join_graph", "P0-ground-truth", "data_prep",
                 depends_on=["build_semantic_layer"]))

    # Phase 1 fan-out (parallel)
    for tid in ("eda_distributions", "data_quality", "question_discovery"):
        dag.add(Task(tid, "P1-discovery", "eda",
                     depends_on=["build_join_graph"]))

    # Phase 2 plan synthesis (sequential, human checkpoint)
    dag.add(Task("synthesize_plan", "P2-plan", "data_prep",
                 depends_on=["eda_distributions", "data_quality", "question_discovery"],
                 requires_human_approval=True))

    # Phase 3 modeling tracks (parallel)
    for tid in ("econometrics_track", "ml_track", "causal_ml_track"):
        dag.add(Task(tid, "P3-modeling", "modeling",
                     depends_on=["synthesize_plan"]))

    # Phase 4 reconciliation (sequential)
    dag.add(Task("reconcile_and_validate", "P4-reconcile", "validation",
                 depends_on=["econometrics_track", "ml_track", "causal_ml_track"]))

    # Phase 5 delivery (parallel), then human sign-off
    for tid in ("build_dashboards", "write_memo", "model_cards"):
        dag.add(Task(tid, "P5-delivery", "delivery",
                     depends_on=["reconcile_and_validate"]))
    dag.add(Task("final_sign_off", "P5-delivery", "delivery",
                 depends_on=["build_dashboards", "write_memo", "model_cards"],
                 requires_human_approval=True))
    return dag
