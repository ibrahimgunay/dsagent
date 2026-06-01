"""Data executor — runs the authored SQL and materializes a dataframe.

Sits between sql_author and the modeling agents. It reads the SQL artifact,
executes it through whatever DataSource is wired (synthetic offline, warehouse
in prod), stores the frame in the DataStore, and writes a small reference
artifact (id + shape + truth-if-known) to the blackboard. This is the step that
makes the end-to-end pipeline produce real numbers.
"""
from __future__ import annotations

from ..runtime.tools import Tool, ToolContext
from ..runtime.blackboard import Artifact


class DataExecutorAgent(Tool):
    name = "data_executor"
    kind = "dataset"
    description = "Executes the authored SQL against the data source and materializes a frame."
    reads = ["sql"]

    def run(self, ctx: ToolContext) -> Artifact:
        source = ctx.services.get("data_source")
        store = ctx.services.get("datastore")
        if source is None or store is None:
            return self._emit(ctx, {"status": "no_data_source", "dataset_ref": None})

        sql = ""
        for k in ctx.depends_on:
            a = ctx.blackboard.get(k)
            if a and a.kind == "sql":
                sql = a.value.get("sql", "")
        df = source.query(sql, **ctx.params.get("hints", {}))
        ref = store.put(df)

        payload = {"dataset_ref": ref, "rows": int(len(df)),
                   "columns": list(df.columns)}
        if hasattr(source, "truth"):
            payload["true_effect"] = source.truth()      # only known for synthetic
        return self._emit(ctx, payload)
