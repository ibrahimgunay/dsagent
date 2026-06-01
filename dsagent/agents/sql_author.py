"""SQL author — LLM drafts, deterministic engine verifies.

The model proposes SQL for the analysis question; the rule-based engine then
analyzes it (complexity, anti-patterns) and runs the fan-out gate against the
join graph. This is the pattern throughout: LLM for generation, deterministic
code for verification, so a hallucinated double-counting query is caught.
"""
from __future__ import annotations

from ..runtime.tools import Tool, ToolContext
from ..runtime.blackboard import Artifact
from ..catalog import Catalog
from ..graph import JoinGraph
from ..llm.base import LLMClient
from .. import sql as sqlmod


class SqlAuthorAgent(Tool):
    name = "sql_author"
    kind = "sql"
    description = "Drafts analysis SQL against the semantic layer and self-verifies it."
    reads = ["semantic", "joins"]

    def __init__(self, catalog: Catalog, llm: LLMClient):
        self.catalog = catalog
        self.llm = llm
        self.jg = JoinGraph(catalog)

    def run(self, ctx: ToolContext) -> Artifact:
        semantic = ctx.blackboard.value(_first(ctx, "semantic"), {})
        prompt = ("Write warehouse SQL for the analysis question.\n"
                  f"question: {ctx.params.get('question', 'measure feature impact on retention')}\n"
                  f"entities: {list(semantic.get('entities', {}))}\n"
                  "Pre-aggregate to the join grain to avoid fan-out double counting.")
        obj, usage = self.llm.complete_json(
            "You are a principal analytics engineer.", prompt, intent="author_sql")
        ctx.usage = usage

        sql_text = obj.get("sql", "")
        qa = sqlmod.analyze(sql_text, self.catalog)
        fanout_warnings = self.jg.query_fanout_warnings(qa)

        return self._emit(ctx, {
            "sql": sql_text,
            "grain": obj.get("grain"),
            "complexity": qa.complexity,
            "anti_patterns": qa.anti_patterns,
            "fanout_warnings": fanout_warnings,
            "referenced_tables": sorted(set(qa.referenced_tables)),
        })


def _first(ctx: ToolContext, kind: str) -> str:
    for k in ctx.depends_on:
        art = ctx.blackboard.get(k)
        if art and art.kind == kind:
            return k
    return ctx.depends_on[0] if ctx.depends_on else ""
