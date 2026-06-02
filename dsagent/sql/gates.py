"""Verified SQL gates — the verified-skill pattern applied to query hazards.

A SQL gate is precondition + a BLOCKING check + a known-truth self-test, exactly
like our estimator skills. The flagship is the fan-out gate: it blocks a query
that aggregates a measure across a one-to-many (or many-to-many) join without
pre-aggregating to the join grain, because that silently multiplies rows and
double-counts every SUM/COUNT downstream — the single most common, most
expensive, and most invisible analytics bug.
"""
from __future__ import annotations

from dataclasses import dataclass
from ..sql import analyze
from ..graph import JoinGraph


@dataclass
class SqlGateResult:
    passed: bool
    name: str
    detail: str = ""


def fanout_gate(sql: str, catalog, join_graph: JoinGraph | None = None) -> SqlGateResult:
    """Block a query that double-counts through a fan-out join."""
    jg = join_graph or JoinGraph(catalog)
    qa = analyze(sql, catalog)
    warns = jg.query_fanout_warnings(qa)
    hazardous = [w for w in warns if "DOUBLE-COUNT" in w]      # fan-out + aggregation
    if hazardous:
        return SqlGateResult(False, "sql_fanout_resolved",
                             f"{len(hazardous)} fan-out join(s) aggregated without a "
                             f"pre-rollup -> measures double-count. e.g. {hazardous[0]}")
    note = ("no fan-out joins in this query" if not warns
            else f"{len(warns)} fan-out edge(s) present but not aggregated (safe)")
    return SqlGateResult(True, "sql_fanout_resolved", note)


def selftest_fanout_gate(catalog=None) -> dict:
    """Known truth: a query that SUMs across a one-to-many join must FAIL the
    gate; the pre-aggregated version of the same question must PASS."""
    from ..catalog import Catalog
    from ..types import Dialect
    if catalog is None:
        catalog = Catalog()
        catalog.ingest_ddl("""
            CREATE TABLE PROD.CORE.ACCOUNTS (
                account_id NUMBER PRIMARY KEY, mrr NUMBER);
            CREATE TABLE PROD.CORE.INVOICES (
                invoice_id NUMBER PRIMARY KEY, account_id NUMBER, amount NUMBER,
                FOREIGN KEY (account_id) REFERENCES PROD.CORE.ACCOUNTS(account_id));
        """, Dialect.SNOWFLAKE)
    jg = JoinGraph(catalog)
    # HAZARD: sum mrr (account grain) across the 1:many invoices join -> double-counts mrr
    bad = """SELECT a.account_id, SUM(a.mrr), SUM(i.amount)
             FROM PROD.CORE.ACCOUNTS a
             JOIN PROD.CORE.INVOICES i ON i.account_id = a.account_id
             GROUP BY a.account_id"""
    # SAFE: pre-aggregate invoices to the account grain first (no fan-out in the join)
    good = """SELECT a.account_id, a.mrr FROM PROD.CORE.ACCOUNTS a"""
    g_bad = fanout_gate(bad, catalog, jg)
    g_good = fanout_gate(good, catalog, jg)
    return {"blocks_double_count": not g_bad.passed,
            "passes_safe_query": g_good.passed,
            "bad_detail": g_bad.detail, "good_detail": g_good.detail}
