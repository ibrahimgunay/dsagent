"""Schema linking — retrieve the relevant slice of ANY schema for a question.

Spider 2.0's dominant failure (27.6% of errors) is wrong schema linking, because
real databases have hundreds-to-thousands of columns and you cannot put them all
in the prompt. The fix is retrieval: score every table and column for relevance
to the question and surface only the top-k. This is what lets the agent work on
an unseen database with minimal instruction — it discovers the relevant tables
itself instead of being told them.

The scorer here is dependency-free (token overlap + substring + light synonyms)
so it runs anywhere; it is a drop-in seam for embeddings later (swap `_score`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_SYNONYMS = {
    "revenue": {"mrr", "amount", "monthly", "invoice", "billing", "payment"},
    "retention": {"active", "retained", "sessions", "engagement", "return"},
    "user": {"users", "account", "customer", "member", "owner"},
    "signup": {"signup", "created", "joined", "registration"},
    "churn": {"canceled", "cancelled", "ended", "inactive", "lapsed"},
    "adoption": {"adopted", "feature", "activated", "first"},
    "funnel": {"stage", "step", "converted", "activation"},
    "cohort": {"cohort", "month", "week", "period"},
}


def _tok(s: str) -> set[str]:
    return {w for w in re.split(r"[^a-z0-9]+", s.lower()) if len(w) > 1}


def _expand(tokens: set[str]) -> set[str]:
    out = set(tokens)
    for t in list(tokens):
        for key, syns in _SYNONYMS.items():
            if t == key or t in syns:
                out |= syns | {key}
    return out


@dataclass
class LinkedTable:
    fqn: str
    score: float
    columns: list[str] = field(default_factory=list)   # relevant leaf paths


@dataclass
class LinkedSchema:
    question: str
    tables: list[LinkedTable]

    def fqns(self) -> list[str]:
        return [t.fqn for t in self.tables]

    def as_prompt(self) -> str:
        """A compact, model-ready schema snippet — only the relevant slice."""
        lines = []
        for t in self.tables:
            cols = ", ".join(t.columns[:20])
            lines.append(f"{t.fqn} ({cols})")
        return "\n".join(lines)


def link_schema(question: str, catalog, k_tables: int = 6,
                k_cols: int = 12) -> LinkedSchema:
    q = _expand(_tok(question))
    ranked = []
    for t in catalog.tables.values():
        name_toks = _tok(t.name) | _tok(t.schema)
        leaves = t.leaf_columns()
        # per-column relevance
        col_scores = []
        for c in leaves:
            ct = _tok(c.full_path)
            overlap = len(q & ct)
            substr = sum(1 for w in q if any(w in cc or cc in w for cc in ct))
            col_scores.append((overlap * 2 + substr, c.full_path))
        col_scores.sort(reverse=True)
        # table relevance = name overlap + best column signal + id-joinability
        name_score = len(q & name_toks) * 3
        col_signal = sum(s for s, _ in col_scores[:k_cols] if s > 0)
        has_ids = sum(1 for c in leaves if c.name.lower().endswith("_id"))
        score = name_score + col_signal + 0.1 * has_ids
        rel_cols = [p for s, p in col_scores if s > 0][:k_cols] or \
                   [c.full_path for c in leaves[:k_cols]]
        ranked.append(LinkedTable(t.fqn, float(score), rel_cols))
    ranked.sort(key=lambda x: -x.score)
    # keep tables with any signal; always return at least the top few so a
    # join path can still be planned
    keep = [t for t in ranked if t.score > 0][:k_tables] or ranked[:k_tables]
    return LinkedSchema(question, keep)
