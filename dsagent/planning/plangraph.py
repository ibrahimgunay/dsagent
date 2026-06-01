"""A mutable plan graph — the substrate for adaptive planning.

Unlike a static DAG, this graph is rewritten during execution: the planner can
insert a repair step before a node, replace a node's tool, prune low-value or
losing branches, and select among conditional design branches once the data is
profiled. `ready_batches()` always reflects the *current* graph over nodes that
are still pending, so the orchestrator can re-batch after every mutation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import networkx as nx


@dataclass
class PlanNode:
    id: str
    tool: str
    phase: str = "P?"
    depends_on: list[str] = field(default_factory=list)
    requires_human_approval: bool = False
    params: dict = field(default_factory=dict)
    # adaptivity metadata
    cost: float = 1.0                 # relative compute/$ weight
    value: float = 1.0               # expected analytical value
    optional: bool = False           # may be pruned under budget pressure
    branch_key: str | None = None     # e.g. "design" — mutually-exclusive group
    branch_value: str | None = None   # e.g. "dml" / "cs" / "iv"
    status: str = "pending"           # pending/running/done/failed/skipped/pruned


class PlanGraph:
    def __init__(self):
        self.nodes: dict[str, PlanNode] = {}

    # ---- construction ----
    def add(self, node: PlanNode) -> "PlanGraph":
        self.nodes[node.id] = node
        return self

    def get(self, nid: str) -> PlanNode | None:
        return self.nodes.get(nid)

    def active(self) -> list[PlanNode]:
        return [n for n in self.nodes.values() if n.status not in ("pruned", "skipped")]

    # ---- mutation primitives (what makes it adaptive) ----
    def replace_tool(self, nid: str, tool: str, **params):
        n = self.nodes[nid]
        n.tool = tool
        n.params.update(params)

    def insert_before(self, target_id: str, node: PlanNode):
        """Make `node` run before target; target now depends on node, and node
        inherits target's prior (active) dependencies."""
        target = self.nodes[target_id]
        node.depends_on = list(target.depends_on)
        self.add(node)
        target.depends_on = [d for d in target.depends_on] + [node.id]
        # ensure no duplicate: target depends on node, not on node's parents directly
        target.depends_on = list(dict.fromkeys(target.depends_on))

    def prune(self, nid: str):
        """Remove a node; its dependents inherit its dependencies."""
        if nid not in self.nodes:
            return
        n = self.nodes[nid]
        n.status = "pruned"
        for other in self.nodes.values():
            if nid in other.depends_on:
                other.depends_on = [d for d in other.depends_on if d != nid] + \
                                   [d for d in n.depends_on if d not in other.depends_on]

    def select_branch(self, branch_key: str, keep_value: str) -> str | None:
        """Keep the chosen branch node in a group; prune the rest, rewiring any
        dependents of the losing branches onto the kept node so links survive."""
        group = [n for n in self.nodes.values()
                 if n.branch_key == branch_key and n.status != "pruned"]
        kept = next((n.id for n in group if n.branch_value == keep_value), None)
        for n in group:
            if n.id == kept:
                continue
            # rewire dependents of this losing branch onto the kept node
            for other in self.nodes.values():
                if n.id in other.depends_on:
                    other.depends_on = [d for d in other.depends_on if d != n.id]
                    if kept and kept not in other.depends_on:
                        other.depends_on.append(kept)
            n.status = "pruned"
        return kept

    def mark(self, nid: str, status: str):
        if nid in self.nodes:
            self.nodes[nid].status = status

    # ---- scheduling ----
    def _graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        active = {n.id for n in self.active()}
        for n in self.active():
            g.add_node(n.id)
            for d in n.depends_on:
                if d in active:
                    g.add_edge(d, n.id)
        return g

    def validate(self):
        g = self._graph()
        if not nx.is_directed_acyclic_graph(g):
            raise ValueError(f"plan has a cycle: {nx.find_cycle(g)}")

    def ready_batches(self) -> list[list[PlanNode]]:
        """Topological generations over not-yet-done active nodes."""
        self.validate()
        g = self._graph()
        done = {n.id for n in self.nodes.values() if n.status in ("done", "running")}
        pending = g.subgraph([n for n in g.nodes if n not in done])
        return [[self.nodes[i] for i in sorted(gen)]
                for gen in nx.topological_generations(pending)]

    def est_cost(self) -> float:
        return sum(n.cost for n in self.active())

    def describe(self) -> str:
        out = []
        for n in self.active():
            tag = f" [branch {n.branch_key}={n.branch_value}]" if n.branch_key else ""
            opt = " (optional)" if n.optional else ""
            hp = " [APPROVAL]" if n.requires_human_approval else ""
            out.append(f"  {n.id:<16} <- {n.tool:<18} deps={n.depends_on}{tag}{opt}{hp}")
        return "\n".join(out)
