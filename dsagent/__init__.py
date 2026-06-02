"""dsagent — autonomous data-science agent core.

Handles many databases, deeply nested tables, spaghetti SQL, and complicated
joins. The SQL parsing backend uses sqlglot when installed (dialect-correct,
column-level) and a stdlib fallback otherwise.
"""
from .types import (Dialect, SemanticType, Sensitivity, Table, Column,
                    ForeignKey, JoinEdge, QueryAnalysis)
from .catalog import Catalog
from .graph import JoinGraph
from .ontology import Ontology
from .profiling import profile_table, classify_column
from .planner import default_plan, AnalysisDAG, Task
from . import sql
from . import llm
from . import runtime
from . import agents

__version__ = "3.5.0"
__all__ = [
    "Dialect", "SemanticType", "Sensitivity", "Table", "Column", "ForeignKey",
    "JoinEdge", "QueryAnalysis", "Catalog", "JoinGraph", "Ontology",
    "profile_table", "classify_column", "default_plan", "AnalysisDAG", "Task",
    "sql", "llm", "runtime", "agents",
]
