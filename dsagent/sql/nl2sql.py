"""Agentic NL2SQL — the LLM reasons, deterministic guardrails guarantee safety.

Pipeline for turning a natural-language question + an unseen schema into trusted
SQL with minimal instruction:

  1. LINK     retrieve the relevant slice of the schema (schema_linker)
  2. PLAN     find the join path among linked tables (join graph)
  3. DRAFT    the LLM writes SQL given ONLY the linked slice + join path
  4. VALIDATE column-existence + fan-out + join-validity gates (validate.py)
  5. REPAIR   deterministically fix near-misses; re-prompt the LLM on hard fails
  6. SELECT   generate several candidates, keep the one that passes the most gates

The LLM is the reasoning engine ("agentic AI is what it depends on"); the gates
make it impossible to SHIP a hallucinated-column or double-counting query. The
loop is provider-agnostic (any LLMClient) and testable offline by passing
`drafts=` instead of calling the model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..graph import JoinGraph
from .schema_linker import link_schema, LinkedSchema
from .validate import validate_sql, repair_sql, ValidationReport


@dataclass
class NL2SQLResult:
    question: str
    sql: str
    valid: bool
    linked_tables: list[str]
    join_path: list[str]
    repairs: list[str] = field(default_factory=list)
    validation: dict = field(default_factory=dict)
    candidates_tried: int = 1
    confidence: str = "medium"


def _score(r: ValidationReport) -> int:
    if r.ok:
        return 100 - len(r.referenced_tables)          # prefer the simplest valid query
    return (4 * r.parses - 2 * len(r.unknown_columns)
            - 3 * (not r.fanout_ok) - 2 * (not r.join_valid))


class NL2SQLAgent:
    def __init__(self, catalog, llm=None):
        self.catalog = catalog
        self.llm = llm
        self.jg = JoinGraph(catalog)

    def _join_path(self, linked: LinkedSchema) -> list[str]:
        fqns = linked.fqns()
        path = []
        for i in range(len(fqns) - 1):
            edges = self.jg.join_path(fqns[i], fqns[i + 1])
            if edges:
                path += [f"{e.left_table}.{e.left_key} = {e.right_table}.{e.right_key}"
                         for e in edges]
        return sorted(set(path))

    def _draft(self, question: str, linked: LinkedSchema, joins: list[str],
               fix: str = "") -> str:
        prompt = (f"question: {question}\n"
                  f"available schema (only these tables/columns exist):\n{linked.as_prompt()}\n"
                  f"valid join keys: {joins}\n"
                  "Use ONLY columns shown. Pre-aggregate the many-side before "
                  "aggregating to avoid fan-out double counting.")
        if fix:
            prompt += f"\nThe previous attempt failed these checks, fix them: {fix}"
        obj, usage = self.llm.complete_json(
            "You are a principal analytics engineer.", prompt, intent="author_sql")
        return obj.get("sql", "")

    def author(self, question: str, *, drafts: list[str] | None = None,
               n_candidates: int = 3, max_repairs: int = 2) -> NL2SQLResult:
        linked = link_schema(question, self.catalog)
        joins = self._join_path(linked)

        # candidate drafts: provided (offline/testing) or from the LLM (prod)
        if drafts is None:
            if self.llm is None:
                raise ValueError("provide drafts= or an llm=")
            drafts = []
            for _ in range(n_candidates):
                drafts.append(self._draft(question, linked, joins))

        best = None
        for sql in drafts:
            report = validate_sql(sql, self.catalog, self.jg)
            repairs = []
            tries = 0
            while not report.ok and tries < max_repairs:
                sql, notes = repair_sql(sql, report, self.catalog)
                repairs += notes
                new = validate_sql(sql, self.catalog, self.jg)
                # if a deterministic fix didn't help and we have an LLM, re-prompt
                if not new.ok and self.llm is not None:
                    sql = self._draft(question, linked, joins, fix=str(new.summary()))
                    new = validate_sql(sql, self.catalog, self.jg)
                report = new
                tries += 1
            cand = (sql, report, repairs)
            if best is None or _score(report) > _score(best[1]):
                best = cand

        sql, report, repairs = best
        conf = "high" if report.ok and not repairs else \
               "medium" if report.ok else "low (failed gates)"
        return NL2SQLResult(
            question=question, sql=sql, valid=report.ok,
            linked_tables=linked.fqns(), join_path=joins, repairs=repairs,
            validation=report.summary(), candidates_tried=len(drafts), confidence=conf)
