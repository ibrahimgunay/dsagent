"""Minimal but real tests for the core capabilities. Run: python -m demo.test_core"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsagent import Catalog, JoinGraph, Ontology, profile_table
from dsagent.types import Dialect, SemanticType, Sensitivity
from dsagent import sql as sqlmod
from demo.fixtures import SNOWFLAKE_DDL, BIGQUERY_DDL, SPAGHETTI_QUERIES

PASS = 0
FAIL = 0

def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}")


def build():
    cat = Catalog()
    cat.ingest_ddl(SNOWFLAKE_DDL, Dialect.SNOWFLAKE)
    cat.ingest_ddl(BIGQUERY_DDL, Dialect.BIGQUERY)
    return cat


def main():
    cat = build()

    print("multi-database ingestion")
    check("two databases tracked", cat.databases() == {"PROD", "ANALYTICS"})
    check("seven tables", len(cat.tables) == 7)

    print("nested-type flattening")
    ev = cat.get("ANALYTICS.EVENTS.PRODUCT_EVENTS")
    leaves = {c.full_path for c in ev.leaf_columns()}
    check("deep struct path resolved", "page.utm.campaign" in leaves)
    check("array-of-struct leaf resolved", "properties.element.key" in leaves)
    users = cat.get("PROD.CORE.USERS")
    ul = {c.full_path for c in users.leaf_columns()}
    check("snowflake OBJECT flattened", "billing_address.zip" in ul)

    print("profiling / PII")
    for t in cat.tables.values():
        profile_table(t)
    pii = [c.full_path for c in users.leaf_columns()
           if c.sensitivity == Sensitivity.PII]
    check("email flagged PII", any(p == "email" for p in pii))
    check("nested address flagged PII", "billing_address.street" in pii)
    ip = next(c for c in ev.leaf_columns() if c.full_path == "device.ip_address")
    check("nested ip flagged PII", ip.sensitivity == Sensitivity.PII)

    print("join graph")
    jg = JoinGraph(cat)
    path = jg.join_path("USERS", "ACCOUNTS")
    check("users<->accounts path found", path is not None and len(path) == 1)
    fos = {tuple(sorted([e.left_table, e.right_table])) for e in jg.fanout_edges()}
    check("accounts->subscriptions fan-out detected",
          ("PROD.CORE.ACCOUNTS", "PROD.CORE.SUBSCRIPTIONS") in fos)

    print("spaghetti analysis")
    qa = sqlmod.analyze(SPAGHETTI_QUERIES["retention_blob"], cat)
    check("implicit join detected through subquery",
          any("Implicit" in a for a in qa.anti_patterns))
    check("subquery depth >= 3", qa.max_subquery_depth >= 3)
    clean = sqlmod.analyze(SPAGHETTI_QUERIES["clean_metric"], cat)
    check("clean query scores low",
          clean.complexity["complexity_band"] == "low")
    roll = sqlmod.analyze(SPAGHETTI_QUERIES["revenue_rollup"], cat)
    warns = jg.query_fanout_warnings(roll)
    check("double-count fan-out flagged in rollup",
          any("DOUBLE-COUNT" in w for w in warns))

    print("ontology")
    onto = Ontology(cat).build()
    check("user entity built", "Users" in onto.entities)
    check("metrics registered", len(onto.metrics) >= 10)

    print("verified sql fan-out gate")
    from dsagent.sql.gates import selftest_fanout_gate
    g = selftest_fanout_gate()
    check("gate blocks the double-count query", g["blocks_double_count"])
    check("gate passes the safe query", g["passes_safe_query"])

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
