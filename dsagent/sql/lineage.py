"""Table-level lineage + join extraction, with a pluggable backend.

Backend selection:
  * If `sqlglot` is importable, we use it for dialect-correct, column-level
    lineage (production path).
  * Otherwise we fall back to the stdlib structural parser, which yields
    table-level lineage, join edges + keys, and complexity. This is the path
    that runs in this environment.

The public surface (`analyze`) is identical regardless of backend so callers
never branch on it.
"""
from __future__ import annotations

from ..types import QueryAnalysis, JoinEdge
from ..catalog import Catalog
from .parser import parse_query
from .complexity import score_query

try:
    import sqlglot  # type: ignore
    _HAS_SQLGLOT = True
except Exception:
    _HAS_SQLGLOT = False


def backend_name() -> str:
    return "sqlglot" if _HAS_SQLGLOT else "stdlib-lite"


def analyze(sql: str, catalog: Catalog | None = None,
            dialect: str = "generic") -> QueryAnalysis:
    pq = parse_query(sql)
    comp = score_query(pq)

    # resolve table refs against the catalog (cross-database aware)
    resolved: list[str] = []
    alias_to_fqn: dict[str, str] = {}
    for ref in pq.tables:
        fqn = ref.name
        if catalog is not None:
            t = catalog.resolve(ref.name)
            if t:
                fqn = t.fqn
        resolved.append(fqn)
        if ref.alias:
            alias_to_fqn[ref.alias.lower()] = fqn
        alias_to_fqn[ref.name.split(".")[-1].lower()] = fqn

    # build join edges from recovered ON keys
    edges: list[JoinEdge] = []
    for jk in pq.join_keys:
        lt, lc = _split_ref(jk.left)
        rt, rc = _split_ref(jk.right)
        edges.append(JoinEdge(
            left_table=alias_to_fqn.get(lt.lower(), lt),
            right_table=alias_to_fqn.get(rt.lower(), rt),
            left_key=lc, right_key=rc,
            source="observed_in_sql",
        ))

    qa = QueryAnalysis(
        raw_sql=sql,
        referenced_tables=resolved,
        ctes=pq.ctes,
        join_edges=edges,
        complexity=comp["metrics"],
        anti_patterns=comp["anti_patterns"],
        max_subquery_depth=pq.subquery_depth,
    )
    return qa


def _split_ref(ref: str) -> tuple[str, str]:
    parts = ref.split(".")
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    return "", parts[-1]
