"""Multi-database catalog and DDL ingestion.

"Track all possible databases": the Catalog holds tables across any number of
databases/schemas/dialects in one registry, so cross-database joins and lineage
work uniformly. DDL ingestion parses CREATE TABLE statements (Snowflake /
BigQuery / Postgres flavored) including nested STRUCT/ARRAY/VARIANT columns.
"""
from __future__ import annotations

import re
from .types import Table, Column, ForeignKey, Dialect
from .dialects import parse_type, _split_top_level


class Catalog:
    def __init__(self) -> None:
        self.tables: dict[str, Table] = {}   # keyed by fqn

    def add(self, table: Table) -> None:
        self.tables[table.fqn] = table

    def get(self, fqn: str) -> Table | None:
        return self.tables.get(fqn)

    def databases(self) -> set[str]:
        return {t.database for t in self.tables.values()}

    def resolve(self, name: str) -> Table | None:
        """Resolve a possibly-unqualified table name to a catalog table."""
        if name in self.tables:
            return self.tables[name]
        # match on bare table name across databases
        matches = [t for t in self.tables.values() if t.name == name.split(".")[-1]]
        return matches[0] if len(matches) == 1 else (matches[0] if matches else None)

    # ------------------------------------------------------------------ DDL
    def ingest_ddl(self, ddl: str, dialect: Dialect = Dialect.GENERIC,
                   default_db: str = "DB", default_schema: str = "PUBLIC") -> list[Table]:
        """Parse one or more CREATE TABLE statements into the catalog."""
        out = []
        for stmt in self._split_statements(ddl):
            t = self._parse_create_table(stmt, dialect, default_db, default_schema)
            if t:
                self.add(t)
                out.append(t)
        return out

    @staticmethod
    def _split_statements(sql: str) -> list[str]:
        # strip comments
        sql = re.sub(r"--[^\n]*", "", sql)
        sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
        return [s.strip() for s in sql.split(";") if s.strip()]

    def _parse_create_table(self, stmt: str, dialect: Dialect,
                            default_db: str, default_schema: str) -> Table | None:
        m = re.search(r"create\s+(?:or\s+replace\s+)?(?:transient\s+|temporary\s+|temp\s+)?"
                      r"table\s+(?:if\s+not\s+exists\s+)?([`\"\w.]+)\s*\((.*)\)",
                      stmt, re.IGNORECASE | re.DOTALL)
        if not m:
            return None
        qualified = m.group(1).strip("`\"")
        body = m.group(2)

        parts = qualified.split(".")
        if len(parts) == 3:
            db, sch, name = parts
        elif len(parts) == 2:
            db, sch, name = default_db, parts[0], parts[1]
        else:
            db, sch, name = default_db, default_schema, parts[0]

        table = Table(database=db, schema=sch, name=name, dialect=dialect)

        for raw_def in _split_top_level(body):
            d = raw_def.strip()
            low = d.lower()
            # table-level constraints
            if low.startswith("primary key"):
                cols = re.findall(r"[`\"\w]+", d[len("primary key"):])
                table.primary_key = [c.strip("`\"") for c in cols if c.lower() not in ("key",)]
                continue
            if low.startswith("foreign key") or "references" in low:
                fk = self._parse_fk(d)
                if fk:
                    table.foreign_keys.append(fk)
                if low.startswith("foreign key"):
                    continue
            if low.startswith(("constraint", "unique", "check")):
                continue

            # column def: "name TYPE [modifiers]"
            cm = re.match(r"([`\"\w]+)\s+(.*)$", d, re.DOTALL)
            if not cm:
                continue
            col_name = cm.group(1).strip("`\"")
            rest = cm.group(2).strip()
            type_str, nullable = self._extract_type(rest)
            col = parse_type(col_name, type_str, dialect)
            col.nullable = nullable
            table.columns.append(col)
            if re.search(r"primary\s+key", rest, re.IGNORECASE):
                table.primary_key.append(col_name)

        return table

    @staticmethod
    def _extract_type(rest: str) -> tuple[str, bool]:
        """Pull the type expression off a column def, respecting nested <> ()."""
        nullable = "not null" not in rest.lower()
        # take everything up to the first top-level modifier keyword
        depth, end = 0, len(rest)
        tokens = rest
        # find first space-delimited modifier at depth 0 after the type
        i = 0
        # the type is the first token plus any bracketed group attached to it
        m = re.match(r"[\w`\"]+", tokens)
        if not m:
            return rest, nullable
        i = m.end()
        # consume attached bracket group if present
        while i < len(tokens) and tokens[i] in " ":
            # if next non-space is a bracket, it's part of the type
            j = i
            while j < len(tokens) and tokens[j] == " ":
                j += 1
            if j < len(tokens) and tokens[j] in "<([":
                i = j
            else:
                break
        if i < len(tokens) and tokens[i] in "<([":
            for k in range(i, len(tokens)):
                if tokens[k] in "<([":
                    depth += 1
                elif tokens[k] in ">)]":
                    depth -= 1
                    if depth == 0:
                        end = k + 1
                        break
            return tokens[:end].strip(), nullable
        return m.group(0), nullable

    @staticmethod
    def _parse_fk(d: str) -> ForeignKey | None:
        m = re.search(r"(?:foreign\s+key\s*\(\s*([`\"\w]+)\s*\)\s*)?references\s+"
                      r"([`\"\w.]+)\s*\(\s*([`\"\w]+)\s*\)", d, re.IGNORECASE)
        if not m:
            return None
        col = (m.group(1) or "").strip("`\"")
        return ForeignKey(column=col, ref_table=m.group(2).strip("`\""),
                          ref_column=m.group(3).strip("`\""))
