"""Foundation sub-agents — deterministic on purpose.

Schema profiling, semantic modeling, and join analysis must be reproducible and
auditable, so these wrap the rule-based engine rather than an LLM. They write
the ground-truth artifacts every judgment agent downstream depends on.
"""
from __future__ import annotations

from ..runtime.tools import Tool, ToolContext
from ..runtime.blackboard import Artifact
from ..catalog import Catalog
from ..graph import JoinGraph
from ..ontology import Ontology
from ..profiling import profile_table
from .. import sql as sqlmod


class ProfilerAgent(Tool):
    name = "profiler"
    kind = "profiling"
    description = "Profiles every table: semantic typing, PII detection, nested-leaf inventory."

    def __init__(self, catalog: Catalog):
        self.catalog = catalog

    def run(self, ctx: ToolContext) -> Artifact:
        profiles, pii, free_text = [], [], []
        for t in self.catalog.tables.values():
            p = profile_table(t)
            profiles.append(p)
            pii += [f"{t.fqn}.{c}" for c in p["pii_fields"]]
            free_text += [f"{t.fqn}.{c}" for c in p["free_text_fields"]]
        return self._emit(ctx, {"tables_profiled": len(profiles),
                                "pii_fields": pii, "free_text_fields": free_text,
                                "profiles": profiles})


class SemanticModelerAgent(Tool):
    name = "semantic_modeler"
    kind = "semantic"
    description = "Builds logical entities + a governed metric registry from physical tables."
    reads = ["profiling"]

    def __init__(self, catalog: Catalog):
        self.catalog = catalog

    def run(self, ctx: ToolContext) -> Artifact:
        onto = Ontology(self.catalog).build()
        return self._emit(ctx, {
            "entities": {k: v.physical_table for k, v in onto.entities.items()},
            "metrics": {k: m.expression for k, m in onto.metrics.items()},
        })


class JoinAnalyzerAgent(Tool):
    name = "join_analyzer"
    kind = "joins"
    description = ("Builds the cross-database join graph; reports recommended paths, "
                   "fan-out (double-count) risks, and ambiguous join keys.")
    reads = ["semantic"]

    def __init__(self, catalog: Catalog):
        self.catalog = catalog

    def run(self, ctx: ToolContext) -> Artifact:
        jg = JoinGraph(self.catalog)
        fanout = [{"left": e.left_table, "right": e.right_table,
                   "cardinality": e.cardinality} for e in jg.fanout_edges()]
        ambiguous = [{"a": a, "b": b, "keys": keys}
                     for a, b, keys in jg.ambiguous_joins()]
        return self._emit(ctx, {"nodes": jg.g.number_of_nodes(),
                                "edges": jg.g.number_of_edges(),
                                "fanout_joins": fanout,
                                "ambiguous_joins": ambiguous})
