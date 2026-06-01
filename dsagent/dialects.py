"""Dialect-aware type handling.

The headline capability here is parsing *nested* column types that are typical
in Snowflake and BigQuery and that naive schema tools choke on:

    BigQuery:  ARRAY<STRUCT<id INT64, tags ARRAY<STRING>, meta STRUCT<k STRING>>>
    Snowflake: VARIANT / OBJECT / ARRAY  (semi-structured)

We parse these into a tree of Column objects so downstream profiling, lineage,
and the ontology can reason about leaf paths like `events.payload.amount`.
"""
from __future__ import annotations

import re
from .types import Column, Dialect

# Map of raw type token -> canonical normalized type.
_SCALAR_MAP = {
    # strings
    "string": "string", "varchar": "string", "char": "string", "text": "string",
    "nvarchar": "string", "clob": "string",
    # ints
    "int": "int", "integer": "int", "int64": "int", "bigint": "int",
    "smallint": "int", "tinyint": "int", "number": "float",
    # floats / decimals
    "float": "float", "float64": "float", "double": "float", "real": "float",
    "decimal": "float", "numeric": "float",
    # bool
    "bool": "bool", "boolean": "bool",
    # temporal
    "date": "date", "datetime": "timestamp", "timestamp": "timestamp",
    "timestamp_ntz": "timestamp", "timestamp_tz": "timestamp",
    "timestamp_ltz": "timestamp", "time": "time",
    # semi-structured (Snowflake)
    "variant": "variant", "object": "struct", "array": "array",
    # bigquery
    "struct": "struct", "record": "struct", "bytes": "bytes", "json": "variant",
    "geography": "geo", "geometry": "geo",
}


def normalize_scalar(raw: str) -> str:
    base = re.split(r"[(<]", raw.strip().lower())[0].strip()
    return _SCALAR_MAP.get(base, "unknown")


def _split_top_level(s: str, sep: str = ",") -> list[str]:
    """Split on `sep` but only at bracket depth 0. Handles <>, (), []."""
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch in "<([":
            depth += 1
        elif ch in ">)]":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def parse_type(name: str, raw_type: str, dialect: Dialect, path: str = "") -> Column:
    """Recursively parse a (possibly nested) type into a Column tree.

    Examples it handles:
      ARRAY<STRUCT<id INT64, tags ARRAY<STRING>>>
      STRUCT<a INT64, b STRUCT<c STRING>>
      OBJECT(amount NUMBER, currency STRING)   -- snowflake style
      VARIANT                                  -- opaque; flagged for runtime inference
    """
    raw_type = raw_type.strip()
    cur_path = f"{path}.{name}" if path else name
    base = normalize_scalar(raw_type)
    col = Column(name=name, raw_type=raw_type, normalized_type=base,
                 path=cur_path, is_nested_leaf=False)

    low = raw_type.lower()

    # ARRAY<inner>  or  ARRAY(inner)
    m = re.match(r"array\s*[<(](.*)[>)]\s*$", low, re.DOTALL)
    if m:
        inner_raw = raw_type[m.start(1):m.end(1)]
        col.normalized_type = "array"
        # element type becomes a single child named "element"
        child = parse_type("element", inner_raw, dialect, cur_path)
        col.children = [child]
        return col

    # STRUCT<...> / RECORD<...> / OBJECT(...)
    m = re.match(r"(struct|record|object)\s*[<(](.*)[>)]\s*$", low, re.DOTALL)
    if m:
        inner_raw = raw_type[m.start(2):m.end(2)]
        col.normalized_type = "struct"
        for fld in _split_top_level(inner_raw):
            # field is "name TYPE" (bigquery) or "name TYPE" (snowflake object)
            fm = re.match(r"([`\"\w]+)\s+(.*)$", fld.strip(), re.DOTALL)
            if fm:
                fname = fm.group(1).strip("`\"")
                ftype = fm.group(2).strip()
                col.children.append(parse_type(fname, ftype, dialect, cur_path))
        return col

    # VARIANT / opaque semi-structured: leaf, but flagged
    if base == "variant":
        col.is_nested_leaf = True
        col.normalized_type = "variant"
        return col

    # plain scalar leaf
    col.is_nested_leaf = True
    return col
