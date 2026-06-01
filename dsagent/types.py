"""Core typed artifacts for the DS agent.

Everything the agent touches is one of these typed objects. They carry a
content hash so the orchestrator can cache and track lineage (see planner.py).
Pure stdlib (dataclasses) so the package has zero hard dependencies beyond
networkx for the join graph.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


class Dialect(str, Enum):
    SNOWFLAKE = "snowflake"
    BIGQUERY = "bigquery"
    POSTGRES = "postgres"
    DATABRICKS = "databricks"
    REDSHIFT = "redshift"
    GENERIC = "generic"


class SemanticType(str, Enum):
    """LLM-assignable semantic role of a column. Drives PII policy + modeling."""
    IDENTIFIER = "identifier"          # user_id, account_id (join keys)
    EVENT_TIME = "event_time"          # timestamps that define grain/ordering
    MEASURE = "measure"                # additive numeric (revenue, count)
    DIMENSION = "dimension"            # categorical slice
    MONETARY = "monetary"
    FREE_TEXT = "free_text"            # feeds the LLM extraction subsystem
    GEO = "geo"
    BOOLEAN_FLAG = "boolean_flag"
    PII = "pii"
    UNKNOWN = "unknown"


class Sensitivity(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    PII = "pii"
    REGULATED = "regulated"            # HIPAA / PCI / etc.


def _hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass
class Column:
    name: str
    raw_type: str                      # as written in DDL, e.g. ARRAY<STRUCT<...>>
    normalized_type: str = "unknown"   # canonical: string/int/float/bool/timestamp/struct/array/variant
    nullable: bool = True
    path: str = ""                     # dotted path for nested leaves, e.g. event.payload.amount
    is_nested_leaf: bool = False
    semantic_type: SemanticType = SemanticType.UNKNOWN
    sensitivity: Sensitivity = Sensitivity.INTERNAL
    children: list["Column"] = field(default_factory=list)

    @property
    def full_path(self) -> str:
        return self.path or self.name


@dataclass
class Table:
    database: str
    schema: str
    name: str
    dialect: Dialect = Dialect.GENERIC
    columns: list[Column] = field(default_factory=list)
    primary_key: list[str] = field(default_factory=list)
    foreign_keys: list["ForeignKey"] = field(default_factory=list)
    row_estimate: Optional[int] = None

    @property
    def fqn(self) -> str:
        return f"{self.database}.{self.schema}.{self.name}"

    def leaf_columns(self) -> list[Column]:
        """All terminal columns including flattened nested struct/array leaves."""
        out: list[Column] = []

        def walk(col: Column):
            if col.children:
                for c in col.children:
                    walk(c)
            else:
                out.append(col)

        for c in self.columns:
            walk(c)
        return out


@dataclass
class ForeignKey:
    column: str
    ref_table: str        # fqn
    ref_column: str
    inferred: bool = False
    confidence: float = 1.0


@dataclass
class JoinEdge:
    left_table: str
    right_table: str
    left_key: str
    right_key: str
    cardinality: str = "unknown"   # one_to_one / one_to_many / many_to_many
    source: str = "declared"       # declared / inferred / observed_in_sql
    fanout_risk: bool = False


@dataclass
class QueryAnalysis:
    raw_sql: str
    referenced_tables: list[str] = field(default_factory=list)
    ctes: list[str] = field(default_factory=list)
    join_edges: list[JoinEdge] = field(default_factory=list)
    output_columns: list[str] = field(default_factory=list)
    complexity: dict[str, Any] = field(default_factory=dict)
    anti_patterns: list[str] = field(default_factory=list)
    max_subquery_depth: int = 0

    @property
    def fingerprint(self) -> str:
        return _hash(self.raw_sql)
