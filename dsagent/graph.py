"""The join graph: how every table connects to every other.

Builds an undirected multigraph of join relationships from three sources:
  1. declared foreign keys in the DDL,
  2. inferred keys (name-based + optional inclusion-dependency from samples),
  3. join keys observed in the sample SQL (often the real, de-facto joins).

Then it answers the questions that make complicated joins tractable:
  * What is the join path between table A and table B? (shortest = recommended)
  * Where are there multiple distinct keys joining the same two tables?
    ("which key do I join on?" - a classic source of wrong numbers)
  * Which joins fan out (one-to-many / many-to-many) and will double-count
    measures unless aggregated first - and which of those appear in a query.
"""
from __future__ import annotations

import networkx as nx
from .types import Table, JoinEdge, QueryAnalysis
from .catalog import Catalog


class JoinGraph:
    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog
        self.g = nx.MultiGraph()
        self._build()

    def _build(self) -> None:
        for t in self.catalog.tables.values():
            self.g.add_node(t.fqn, table=t)
        for t in self.catalog.tables.values():
            for fk in t.foreign_keys:
                ref = self.catalog.resolve(fk.ref_table)
                if ref:
                    self._add_edge(t.fqn, ref.fqn, fk.column, fk.ref_column,
                                   "declared", 1.0)
        self._infer_name_based_fks()

    def _add_edge(self, left: str, right: str, lkey: str, rkey: str,
                  source: str, confidence: float) -> None:
        ekey = "|".join(sorted([f"{left}.{lkey}", f"{right}.{rkey}"]))
        if self.g.has_edge(left, right):
            if ekey in self.g.get_edge_data(left, right):
                return
        cardinality, fanout = self._cardinality(left, lkey, right, rkey)
        self.g.add_edge(left, right, key=ekey,
                        edge=JoinEdge(left, right, lkey, rkey,
                                      cardinality=cardinality, source=source,
                                      fanout_risk=fanout),
                        confidence=confidence)

    def _infer_name_based_fks(self) -> None:
        pk_index: dict[str, list[tuple[str, str]]] = {}
        for t in self.catalog.tables.values():
            for col in t.leaf_columns():
                pk_index.setdefault(col.name.lower(), []).append((t.fqn, col.name))
        for t in self.catalog.tables.values():
            existing = {fk.column.lower() for fk in t.foreign_keys}
            for col in t.leaf_columns():
                cl = col.name.lower()
                if not cl.endswith("_id") or cl in existing:
                    continue
                prefix = cl[:-3]
                for fqn, target_col in pk_index.get(cl, []):
                    if fqn == t.fqn:
                        continue
                    target = self.catalog.get(fqn)
                    if target_col not in target.primary_key and \
                       target_col.lower() not in ("id", f"{prefix}_id"):
                        continue
                    conf = 0.9 if target.name.lower().rstrip("s") == prefix else 0.65
                    self._add_edge(t.fqn, fqn, col.name, target_col, "inferred", conf)

    def add_observed_edges(self, edges: list[JoinEdge]) -> None:
        for e in edges:
            lt = self.catalog.resolve(e.left_table)
            rt = self.catalog.resolve(e.right_table)
            if lt and rt and lt.fqn != rt.fqn:
                self._add_edge(lt.fqn, rt.fqn, e.left_key, e.right_key,
                               "observed_in_sql", 0.8)

    def _is_unique_key(self, fqn: str, col: str) -> bool:
        t = self.catalog.get(fqn)
        return bool(t and (col in t.primary_key or [col] == t.primary_key))

    def _cardinality(self, left: str, lkey: str, right: str, rkey: str):
        lu, ru = self._is_unique_key(left, lkey), self._is_unique_key(right, rkey)
        if lu and ru:
            return "one_to_one", False
        if ru and not lu:
            return "many_to_one", False
        if lu and not ru:
            return "one_to_many", True
        return "many_to_many", True

    def _best_edge(self, u: str, v: str) -> JoinEdge:
        data = self.g.get_edge_data(u, v)
        return max(data.values(), key=lambda d: d["confidence"])["edge"]

    def join_path(self, a: str, b: str):
        af, bf = self.catalog.resolve(a), self.catalog.resolve(b)
        if not af or not bf:
            return None
        try:
            nodes = nx.shortest_path(self.g, af.fqn, bf.fqn)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
        return [self._best_edge(u, v) for u, v in zip(nodes, nodes[1:])]

    def ambiguous_joins(self):
        """Table pairs joinable by >1 distinct key - 'which key?' hazard."""
        out = []
        pairs = {tuple(sorted((a, b))) for a, b in self.g.edges()}
        for u, v in pairs:
            data = self.g.get_edge_data(u, v) or {}
            keys = sorted({f"{d['edge'].left_key}={d['edge'].right_key}"
                           for d in data.values()})
            if len(keys) > 1:
                out.append((u, v, keys))
        return sorted(out, key=lambda x: -len(x[2]))

    def fanout_edges(self):
        seen, out = set(), []
        for u, v, d in self.g.edges(data=True):
            e = d["edge"]
            sig = tuple(sorted([e.left_table, e.right_table]))
            if e.fanout_risk and sig not in seen:
                seen.add(sig)
                out.append(e)
        return out

    def query_fanout_warnings(self, qa: QueryAnalysis):
        """Flag fan-out joins *used in a specific query* - the double-count bug."""
        tables = set(qa.referenced_tables)
        agg = any(k in qa.raw_sql.upper()
                  for k in ("SUM(", "AVG(", "COUNT(", "MAX(", "MIN("))
        warns = []
        for e in self.fanout_edges():
            if e.left_table in tables and e.right_table in tables:
                msg = f"Fan-out join {e.left_table} <-> {e.right_table} ({e.cardinality})"
                if agg:
                    msg += " with aggregation -> measures will DOUBLE-COUNT; pre-aggregate to the join grain."
                warns.append(msg)
        return warns
