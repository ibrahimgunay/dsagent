"""Spaghetti / complexity analysis.

Turns the structural parse into a complexity score and a list of concrete
anti-patterns. This is the "spaghetti detector": it flags the things that make
SQL unmaintainable or silently wrong (cartesian risk, deep nesting, fan-out
joins, SELECT * in transforms, correlated subqueries, etc.).
"""
from __future__ import annotations

from .parser import ParsedQuery


# weight, message
_RULES = {
    "implicit_join":      (4, "Implicit (comma) join — cartesian-product risk; use explicit JOIN ... ON."),
    "cross_join":         (3, "Explicit CROSS JOIN — confirm the cartesian product is intended."),
    "missing_on":         (3, "JOIN without an ON/USING key — likely accidental cross join."),
    "deep_nesting":       (3, "Subquery nesting depth >= 4 — refactor into CTEs."),
    "many_joins":         (2, "High join count (>= 6) — consider a conformed/denormalized model."),
    "select_star":        (2, "SELECT * in a transform — breaks on schema drift; enumerate columns."),
    "correlated_subq":    (2, "Correlated subquery in WHERE — possible performance cliff; consider a join."),
    "many_distinct":      (1, "Multiple DISTINCTs — often masks a fan-out join (de-dup band-aid)."),
    "no_cte":             (1, "Deeply nested without CTEs — readability/lineage suffers."),
}


def score_query(pq: ParsedQuery) -> dict:
    flags: list[str] = []
    score = 0

    if pq.has_implicit_join:
        flags.append("implicit_join")
    if pq.has_cross_join:
        flags.append("cross_join")
    # a join with no recovered key is a missing-ON signal
    if pq.join_count > 0 and len(pq.join_keys) < pq.join_count and not pq.has_cross_join:
        flags.append("missing_on")
    if pq.subquery_depth >= 4:
        flags.append("deep_nesting")
    if pq.join_count >= 6:
        flags.append("many_joins")
    if pq.select_star:
        flags.append("select_star")
    if pq.correlated_subquery:
        flags.append("correlated_subq")
    if pq.distinct_count >= 2:
        flags.append("many_distinct")
    if pq.subquery_depth >= 3 and not pq.ctes:
        flags.append("no_cte")

    messages = []
    for f in flags:
        w, msg = _RULES[f]
        score += w
        messages.append(msg)

    band = "low" if score <= 2 else "moderate" if score <= 6 else "high" if score <= 11 else "critical"

    metrics = {
        "complexity_score": score,
        "complexity_band": band,
        "join_count": pq.join_count,
        "subquery_depth": pq.subquery_depth,
        "cte_count": len(pq.ctes),
        "window_funcs": pq.window_funcs,
        "set_ops": pq.set_ops,
        "distinct_count": pq.distinct_count,
    }
    return {"metrics": metrics, "anti_patterns": messages, "flags": flags}
