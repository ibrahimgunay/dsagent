"""Validate-and-repair — the deterministic guardrails around an agentic drafter.

The LLM proposes SQL; this layer guarantees what ships is safe on an UNSEEN
schema, no matter what the model wrote:

  * column-existence gate  — every qualified `alias.column` must resolve to a
    real column in the catalog. Kills hallucinated columns (Spider 2.0's #1
    error). Self-repair snaps a near-miss to the closest real column.
  * fan-out gate           — no aggregation across a one-to-many join (reuses
    dsagent.sql.gates).
  * join-validity gate     — every joined table pair has a path in the join graph.

Returns a structured report so the agent loop can repair or reject, and so the
trust report can show exactly which checks passed.
"""
from __future__ import annotations

import re
import difflib
from dataclasses import dataclass, field

from ..sql import analyze, parse_query
from ..graph import JoinGraph
from .gates import fanout_gate


@dataclass
class ValidationReport:
    parses: bool = False
    unknown_columns: list[str] = field(default_factory=list)
    fanout_ok: bool = True
    fanout_detail: str = ""
    join_valid: bool = True
    join_detail: str = ""
    referenced_tables: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (self.parses and not self.unknown_columns
                and self.fanout_ok and self.join_valid)

    def summary(self) -> dict:
        return {"ok": self.ok, "parses": self.parses,
                "unknown_columns": self.unknown_columns,
                "fanout_ok": self.fanout_ok, "join_valid": self.join_valid,
                "tables": self.referenced_tables}


def _alias_map(pq, catalog):
    """alias / bare name -> resolved Table."""
    m = {}
    for tref in pq.tables:
        tbl = catalog.resolve(tref.name)
        if tbl:
            if tref.alias:
                m[tref.alias.lower()] = tbl
            m[tref.name.split(".")[-1].lower()] = tbl
    return m


def _column_universe(tbl):
    cols = set()
    for c in tbl.leaf_columns():
        cols.add(c.name.lower())
        cols.add(c.full_path.lower())
        cols.add(c.full_path.split(".")[-1].lower())   # leaf segment
    return cols


def validate_sql(sql: str, catalog, join_graph: JoinGraph | None = None) -> ValidationReport:
    jg = join_graph or JoinGraph(catalog)
    r = ValidationReport()
    try:
        pq = parse_query(sql)
        r.parses = True
    except Exception:
        return r

    qa = analyze(sql, catalog)
    r.referenced_tables = sorted(set(qa.referenced_tables))
    aliases = _alias_map(pq, catalog)

    # column-existence: check every qualified alias.column reference
    for m in re.finditer(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b", sql):
        a, col = m.group(1).lower(), m.group(2).lower()
        tbl = aliases.get(a)
        if tbl is None:
            continue                         # alias is a db/schema qualifier, skip
        if col not in _column_universe(tbl):
            ref = f"{m.group(1)}.{m.group(2)}"
            if ref not in r.unknown_columns:
                r.unknown_columns.append(ref)

    # fan-out
    g = fanout_gate(sql, catalog, jg)
    r.fanout_ok, r.fanout_detail = g.passed, g.detail

    # join validity: each adjacent referenced table pair should have a path
    fqns = r.referenced_tables
    if len(fqns) >= 2:
        bad = []
        for i in range(len(fqns) - 1):
            if jg.join_path(fqns[i], fqns[i + 1]) is None:
                bad.append((fqns[i], fqns[i + 1]))
        if bad and len(bad) == len(fqns) - 1:    # no pair connects at all
            r.join_valid, r.join_detail = False, f"no join path among {fqns}"
    return r


def repair_sql(sql: str, report: ValidationReport, catalog) -> tuple[str, list[str]]:
    """Deterministically fix what we safely can. Returns (sql, notes)."""
    notes = []
    fixed = sql
    pq = parse_query(sql)
    aliases = _alias_map(pq, catalog)
    for ref in report.unknown_columns:
        a, col = ref.split(".")
        tbl = aliases.get(a.lower())
        if not tbl:
            continue
        candidates = [c.name for c in tbl.leaf_columns()]
        near = difflib.get_close_matches(col, candidates, n=1, cutoff=0.7)
        if near:
            fixed = re.sub(rf"\b{re.escape(a)}\.{re.escape(col)}\b",
                           f"{a}.{near[0]}", fixed)
            notes.append(f"snapped {ref} -> {a}.{near[0]} (nearest real column)")
    if not report.fanout_ok:
        notes.append("fan-out hazard: pre-aggregate the many-side to the join grain "
                     "before aggregating (estimate withheld until resolved)")
    return fixed, notes
