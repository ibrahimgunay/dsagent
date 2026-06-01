"""Structural SQL parsing for the stdlib backend.

Extracts the structure we need for lineage + complexity from arbitrarily nested
SQL: CTE names, referenced tables with aliases, JOIN edges with their ON keys,
subquery nesting depth, and a list of anti-patterns. It is bracket-aware so
deeply nested subqueries and CTE chains ("spaghetti") are handled.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class TableRef:
    name: str
    alias: str = ""


@dataclass
class JoinKeyRef:
    left: str   # alias.col or col
    right: str


@dataclass
class ParsedQuery:
    ctes: list[str] = field(default_factory=list)
    tables: list[TableRef] = field(default_factory=list)
    join_keys: list[JoinKeyRef] = field(default_factory=list)
    has_implicit_join: bool = False
    has_cross_join: bool = False
    join_count: int = 0
    subquery_depth: int = 0
    select_star: bool = False
    correlated_subquery: bool = False
    distinct_count: int = 0
    window_funcs: int = 0
    set_ops: int = 0


_IDENT = r'[A-Za-z_][\w$]*|"[^"]*"|`[^`]*`'
_QUALIFIED = rf'(?:{_IDENT})(?:\.(?:{_IDENT}))*'


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    return sql


def _match_paren(s: str, open_idx: int) -> int:
    depth = 0
    for i in range(open_idx, len(s)):
        if s[i] == "(":
            depth += 1
        elif s[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return len(s) - 1


def parse_query(sql: str) -> ParsedQuery:
    sql = _strip_comments(sql)
    pq = ParsedQuery()
    pq.subquery_depth = _max_subquery_depth(sql)
    pq.select_star = bool(re.search(r"select\s+(?:distinct\s+)?\*", sql, re.IGNORECASE))
    pq.distinct_count = len(re.findall(r"\bdistinct\b", sql, re.IGNORECASE))
    pq.window_funcs = len(re.findall(r"\bover\s*\(", sql, re.IGNORECASE))
    pq.set_ops = len(re.findall(r"\b(union|intersect|except)\b", sql, re.IGNORECASE))

    pq.ctes = _extract_ctes(sql)

    # JOIN clauses
    for jm in re.finditer(r"\b(inner|left|right|full|cross)?\s*(outer)?\s*join\b",
                          sql, re.IGNORECASE):
        pq.join_count += 1
        if (jm.group(1) or "").lower() == "cross":
            pq.has_cross_join = True

    # table refs from FROM/JOIN
    pq.tables = _extract_table_refs(sql)
    cte_names = {c.lower() for c in pq.ctes}
    pq.tables = [t for t in pq.tables if t.name.lower() not in cte_names]

    # join keys from ON clauses
    for on in re.finditer(rf"\bon\b\s+(.*?)(?=\b(?:join|where|group|order|"
                          rf"having|qualify|limit|union|inner|left|right|full|cross)\b|\)|$)",
                          sql, re.IGNORECASE | re.DOTALL):
        cond = on.group(1)
        for eq in re.finditer(rf"({_QUALIFIED})\s*=\s*({_QUALIFIED})", cond):
            pq.join_keys.append(JoinKeyRef(eq.group(1), eq.group(2)))

    # implicit join: comma-separated tables in FROM (cartesian risk)
    pq.has_implicit_join = _has_implicit_join(sql)

    # correlated subquery heuristic: an EXISTS / IN subquery referencing an
    # outer alias is hard to detect perfectly; flag nested SELECT inside WHERE
    pq.correlated_subquery = bool(
        re.search(r"where\b.*?\(\s*select\b", sql, re.IGNORECASE | re.DOTALL))

    return pq


def _max_subquery_depth(sql: str) -> int:
    depth = max_d = 0
    i = 0
    # count nesting only of parens that open a SELECT
    while i < len(sql):
        if sql[i] == "(":
            close = _match_paren(sql, i)
            inner = sql[i + 1:close]
            if re.match(r"\s*select\b", inner, re.IGNORECASE):
                d = 1 + _max_subquery_depth(inner)
                max_d = max(max_d, d)
            i = close + 1
        else:
            i += 1
    return max_d


def _extract_ctes(sql: str) -> list[str]:
    m = re.search(r"\bwith\b", sql, re.IGNORECASE)
    if not m:
        return []
    names = []
    # find "name as (" patterns at the top of the WITH clause
    for cm in re.finditer(rf"({_IDENT})\s+as\s*\(", sql[m.end():], re.IGNORECASE):
        names.append(cm.group(1).strip('"`'))
    # dedupe preserve order; cap to avoid matching inline "x as (" in subqueries
    seen, out = set(), []
    for n in names:
        if n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    return out


def _extract_table_refs(sql: str) -> list[TableRef]:
    refs: list[TableRef] = []
    # match FROM ... and JOIN ... up to next clause keyword, ignoring subqueries
    pattern = re.compile(
        rf"\b(?:from|join)\b\s+({_QUALIFIED})(?:\s+(?:as\s+)?({_IDENT}))?",
        re.IGNORECASE)
    for m in pattern.finditer(sql):
        name = m.group(1).strip('"`')
        alias = (m.group(2) or "").strip('"`')
        if alias.lower() in ("on", "using", "where", "group", "inner", "left",
                              "right", "full", "cross", "join", "order"):
            alias = ""
        refs.append(TableRef(name=name, alias=alias))
    return refs


def _has_implicit_join(sql: str) -> bool:
    # FROM a, b  (old-style cartesian). Must scan EVERY from-clause, because the
    # offending one is often the outer query while an inner subquery's FROM comes
    # first textually.
    for fm in re.finditer(r"\bfrom\b(.*?)(?=\bwhere\b|\bgroup\b|\border\b|"
                          r"\bjoin\b|\bhaving\b|\bqualify\b|\blimit\b|\)|$)",
                          sql, re.IGNORECASE | re.DOTALL):
        seg = fm.group(1)
        depth = 0
        for ch in seg:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == "," and depth == 0:
                return True
    return False
