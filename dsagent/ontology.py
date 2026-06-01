"""The semantic / ontology layer (the Palantir-style spine).

Maps physical tables to logical *entities* (User, Order, Event...) and builds a
governed *metric* registry so every downstream number references one canonical
definition. Entities are discovered from identifier columns; metrics are
proposed from monetary/measure columns and (in production) mined from the
sample SQL's aggregate expressions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from .types import Table, SemanticType
from .catalog import Catalog


@dataclass
class Entity:
    name: str
    grain: str                 # the identifier column defining one row
    physical_table: str        # fqn
    key_columns: list[str] = field(default_factory=list)


@dataclass
class Metric:
    name: str
    expression: str            # canonical SQL expression
    entity: str
    grain: str
    aggregation: str = "sum"
    sensitivity: str = "internal"


class Ontology:
    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog
        self.entities: dict[str, Entity] = {}
        self.metrics: dict[str, Metric] = {}

    def build(self) -> "Ontology":
        for t in self.catalog.tables.values():
            self._entity_from_table(t)
            self._metrics_from_table(t)
        return self

    def _entity_from_table(self, t: Table) -> None:
        ids = [c for c in t.leaf_columns()
               if c.semantic_type == SemanticType.IDENTIFIER]
        grain = t.primary_key[0] if t.primary_key else (ids[0].name if ids else "row")
        ent_name = t.name.rstrip("s").title().replace("_", "")
        self.entities[ent_name] = Entity(
            name=ent_name, grain=grain, physical_table=t.fqn,
            key_columns=[c.name for c in ids],
        )

    def _metrics_from_table(self, t: Table) -> None:
        ent_name = t.name.rstrip("s").title().replace("_", "")
        grain = self.entities[ent_name].grain if ent_name in self.entities else "row"
        for c in t.leaf_columns():
            if c.semantic_type in (SemanticType.MONETARY, SemanticType.MEASURE):
                mname = f"{t.name}_{c.name}".lower()
                self.metrics[mname] = Metric(
                    name=mname,
                    expression=f"SUM({t.fqn}.{c.full_path})",
                    entity=ent_name, grain=grain, aggregation="sum",
                    sensitivity=c.sensitivity.value,
                )
        # a row-count metric per entity is always useful
        self.metrics[f"{t.name}_count".lower()] = Metric(
            name=f"{t.name}_count".lower(),
            expression=f"COUNT(*)", entity=ent_name, grain=grain,
            aggregation="count",
        )

    def summary(self) -> dict:
        return {
            "entities": {k: v.physical_table for k, v in self.entities.items()},
            "metric_count": len(self.metrics),
            "metrics_sample": list(self.metrics.keys())[:12],
        }
