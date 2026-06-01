"""A small, robust SQL tokenizer (stdlib only).

This is the fallback backend used when sqlglot is unavailable. It is good
enough to extract structure (CTEs, FROM/JOIN/ON, subqueries, column refs)
from genuinely messy SQL, which is what the lineage and complexity analyzers
need. For full, dialect-perfect column-level lineage in production, install
sqlglot and the LineageExtractor will use it automatically.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

KEYWORDS = {
    "with", "select", "from", "where", "join", "inner", "left", "right", "full",
    "outer", "cross", "on", "using", "group", "by", "order", "having", "limit",
    "union", "all", "distinct", "as", "and", "or", "not", "in", "exists",
    "case", "when", "then", "else", "end", "over", "partition", "qualify",
    "lateral", "unnest", "flatten",
}


@dataclass
class Token:
    kind: str   # ws, comment, string, ident, number, punct, keyword, op
    value: str


_TOKEN_RE = re.compile(
    r"""
      (?P<comment> --[^\n]* | /\*.*?\*/ )
    | (?P<ws>      \s+ )
    | (?P<string>  '(?:[^']|'')*' )
    | (?P<qident>  "(?:[^"]|"")*" | `[^`]*` | \[[^\]]*\] )
    | (?P<number>  \d+\.?\d*(?:[eE][+-]?\d+)? )
    | (?P<ident>   [A-Za-z_][A-Za-z0-9_$]* )
    | (?P<op>      <=|>=|<>|!=|\|\||::|->>|-> )
    | (?P<punct>   [(),.*=<>+\-/%;] )
    | (?P<other>   . )
    """,
    re.VERBOSE | re.DOTALL,
)


def tokenize(sql: str) -> list[Token]:
    toks: list[Token] = []
    for m in _TOKEN_RE.finditer(sql):
        kind = m.lastgroup
        val = m.group()
        if kind in ("ws", "comment"):
            continue
        if kind == "ident" and val.lower() in KEYWORDS:
            kind = "keyword"
        elif kind == "qident":
            kind = "ident"
        toks.append(Token(kind, val))
    return toks
