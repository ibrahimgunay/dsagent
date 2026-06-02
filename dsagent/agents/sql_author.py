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
        from ..sql.nl2sql import NL2SQLAgent
        self.nl2sql = NL2SQLAgent(catalog, llm)

    def run(self, ctx: ToolContext) -> Artifact:
        question = ctx.params.get("question", "measure feature impact on retention")
        # agentic: link schema -> LLM draft -> validate -> self-repair -> select
        result = self.nl2sql.author(question)
        ctx.usage = ctx.usage  # usage accrued inside nl2sql drafts via llm
        sql_text = result.sql
        qa = sqlmod.analyze(sql_text, self.catalog)
        fanout_warnings = self.jg.query_fanout_warnings(qa)

        return self._emit(ctx, {
            "sql": sql_text,
            "complexity": qa.complexity,
            "anti_patterns": qa.anti_patterns,
            "fanout_warnings": fanout_warnings,
            "referenced_tables": sorted(set(qa.referenced_tables)),
            # agentic NL2SQL trail
            "linked_tables": result.linked_tables,
            "join_path": result.join_path,
            "validation": result.validation,
            "repairs": result.repairs,
            "candidates_tried": result.candidates_tried,
            "confidence": result.confidence,
        })


def _first(ctx: ToolContext, kind: str) -> str:
    for k in ctx.depends_on:
        art = ctx.blackboard.get(k)
        if art and art.kind == kind:
            return k
    return ctx.depends_on[0] if ctx.depends_on else ""
