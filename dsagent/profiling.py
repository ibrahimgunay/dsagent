"""Semantic typing + sensitivity classification.

In production this is LLM-assisted (the classifier prompt sees column name,
type, table context, and sample values and returns SemanticType + Sensitivity
+ a human definition). Here we ship a deterministic heuristic classifier with
the same interface, so the pipeline runs end-to-end offline. `classify_column`
is the single hook to swap for an LLM call.
"""
from __future__ import annotations

import re
from .types import Column, Table, SemanticType, Sensitivity

_PII_PATTERNS = [
    "email", "e_mail", "phone", "ssn", "social_security", "passport",
    "address", "street", "zip", "postal", "dob", "birth", "first_name",
    "last_name", "full_name", "credit_card", "card_number", "ip_address",
    "lat", "lon", "latitude", "longitude", "device_id", "fingerprint",
]
_ID_PATTERNS = ["_id", "id", "uuid", "guid", "key", "sk", "pk"]
_TIME_PATTERNS = ["_at", "_ts", "timestamp", "time", "date", "_dt"]
_MONEY_PATTERNS = ["amount", "revenue", "price", "cost", "spend", "ltv",
                   "gmv", "arpu", "mrr", "arr", "balance", "fee"]
_MEASURE_PATTERNS = ["count", "qty", "quantity", "num_", "_num", "total",
                     "sum", "score", "duration", "sessions", "clicks", "views"]


def classify_column(col: Column, table: Table) -> tuple[SemanticType, Sensitivity, str]:
    """Return (semantic_type, sensitivity, human_definition).

    SWAP POINT: replace the body with an LLM call for production-grade typing.
    """
    name = col.name.lower()
    nt = col.normalized_type

    if any(p in name for p in _PII_PATTERNS):
        return SemanticType.PII, Sensitivity.PII, f"Likely PII field: {col.name}"

    if nt in ("timestamp", "date", "time") or any(name.endswith(p) or p in name
                                                  for p in _TIME_PATTERNS):
        return SemanticType.EVENT_TIME, Sensitivity.INTERNAL, f"Temporal field: {col.name}"

    if name in table.primary_key or any(name.endswith(p) for p in _ID_PATTERNS):
        return SemanticType.IDENTIFIER, Sensitivity.INTERNAL, f"Identifier / join key: {col.name}"

    if any(p in name for p in _MONEY_PATTERNS):
        return SemanticType.MONETARY, Sensitivity.CONFIDENTIAL, f"Monetary measure: {col.name}"

    if nt in ("int", "float") and any(p in name for p in _MEASURE_PATTERNS):
        return SemanticType.MEASURE, Sensitivity.INTERNAL, f"Additive measure: {col.name}"

    if nt == "bool" or name.startswith(("is_", "has_", "did_")):
        return SemanticType.BOOLEAN_FLAG, Sensitivity.INTERNAL, f"Boolean flag: {col.name}"

    if nt == "string":
        if col.normalized_type == "string" and any(
                k in name for k in ("description", "comment", "review", "body",
                                    "message", "text", "note", "feedback")):
            return SemanticType.FREE_TEXT, Sensitivity.INTERNAL, f"Free text (LLM-extractable): {col.name}"
        return SemanticType.DIMENSION, Sensitivity.INTERNAL, f"Categorical dimension: {col.name}"

    if nt in ("int", "float"):
        return SemanticType.MEASURE, Sensitivity.INTERNAL, f"Numeric measure: {col.name}"

    return SemanticType.UNKNOWN, Sensitivity.INTERNAL, col.name


def profile_table(table: Table) -> dict:
    """Annotate every leaf column in place and return a profile summary."""
    counts: dict[str, int] = {}
    pii: list[str] = []
    free_text: list[str] = []
    for col in table.leaf_columns():
        st, sens, _defn = classify_column(col, table)
        col.semantic_type = st
        col.sensitivity = sens
        counts[st.value] = counts.get(st.value, 0) + 1
        if sens in (Sensitivity.PII, Sensitivity.REGULATED):
            pii.append(col.full_path)
        if st == SemanticType.FREE_TEXT:
            free_text.append(col.full_path)
    return {
        "table": table.fqn,
        "leaf_column_count": len(table.leaf_columns()),
        "semantic_type_counts": counts,
        "pii_fields": pii,
        "free_text_fields": free_text,
        "nested": any(c.children for c in table.columns),
    }
