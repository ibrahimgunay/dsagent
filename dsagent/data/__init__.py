"""Data sources — the bridge from authored SQL to a dataframe the models fit.

`DataSource.query(sql)` returns a pandas DataFrame. Two implementations:

  * WarehouseDataSource — production. Wraps any DB-API connection (Snowflake,
    BigQuery, Postgres, DuckDB) or a connector callable, and executes the SQL
    the sql_author produced. Import-safe offline; only touches the DB on query.
  * SyntheticDataSource — offline. Returns planted-truth datasets so the full
    orchestrated pipeline flows real frames (and real estimates) without a
    warehouse, and exposes the ground truth for the eval.

Frames are kept in a DataStore and referenced by id on the blackboard, so the
blackboard stays small and json-serializable while large data moves by handle.
"""
from __future__ import annotations

import abc
import uuid
import pandas as pd

from .. import execution as _exec


class DataSource(abc.ABC):
    @abc.abstractmethod
    def query(self, sql: str, **hints) -> pd.DataFrame: ...

    def describe(self) -> str:
        return self.__class__.__name__


class WarehouseDataSource(DataSource):
    """Executes SQL against a real warehouse. `connection` is any PEP-249
    connection; or pass `connector` (a callable returning a fresh connection)."""

    def __init__(self, connection=None, connector=None):
        if connection is None and connector is None:
            raise ValueError("WarehouseDataSource needs a connection or connector.")
        self._conn = connection
        self._connector = connector

    def _conn_obj(self):
        return self._conn or self._connector()

    def query(self, sql: str, **hints) -> pd.DataFrame:
        conn = self._conn_obj()
        # pandas.read_sql works across DB-API drivers; row-count is logged by caller
        return pd.read_sql(sql, conn)


class SyntheticDataSource(DataSource):
    """Offline source with known ground truth, keyed by the analysis intent."""

    def __init__(self, scenario: str = "observational", true_effect: float = 2.0,
                 seed: int = 0):
        self.scenario = scenario
        self.true_effect = true_effect
        self.seed = seed
        self._truth = true_effect

    def query(self, sql: str, **hints) -> pd.DataFrame:
        s = (hints.get("scenario") or self.scenario).lower()
        if "staggered" in s:
            df, att = _exec.datagen.make_staggered(base=self.true_effect, seed=self.seed)
            self._truth = att
            return df
        if "panel" in s or "did" in s:
            self._truth = self.true_effect
            return _exec.datagen.make_did_panel(att=self.true_effect, seed=self.seed)
        if "iv" in s or "instrument" in s:
            df, ate = _exec.datagen.make_iv(ate=self.true_effect, seed=self.seed)
            self._truth = ate
            return df
        if "rct" in s or "random" in s:
            self._truth = self.true_effect
            return _exec.datagen.make_rct(ate=self.true_effect, seed=self.seed)
        if "null" in s:
            self._truth = 0.0
            return _exec.datagen.make_null(seed=self.seed)
        self._truth = self.true_effect
        return _exec.datagen.make_observational(ate=self.true_effect, seed=self.seed)

    def truth(self) -> float:
        return self._truth


class DataStore:
    """Holds dataframes by id; the blackboard carries only the id + a summary."""

    def __init__(self):
        self._frames: dict[str, pd.DataFrame] = {}

    def put(self, df: pd.DataFrame) -> str:
        key = f"df_{uuid.uuid4().hex[:8]}"
        self._frames[key] = df
        return key

    def get(self, key: str) -> pd.DataFrame:
        return self._frames[key]
