"""
schema_benchmark.py — does dsagent's data layer actually handle hard schemas?

The causal benchmark (benchmark_suite.py) graded the ESTIMATORS. This grades the
DATA-ENGINEERING side: reading complex/nested/multi-database schemas, recovering
join structure, catching the fan-out double-count bug, and parsing spaghetti SQL.

It runs fully OFFLINE on built-in hard fixtures with known ground truth, so it
self-validates anywhere. With network (your machine) it also loads
REAL enterprise schemas from Spider 2.0 and BIRD and scores join-graph recovery
on those:

    python schema_benchmark.py                       # offline fixtures + scale + gate
    python schema_benchmark.py --spider tables.json   # add a real Spider/BIRD schema

Spider 2.0 is the hard target: enterprise databases of 700-3000 columns, nested
BigQuery/Snowflake types, where even frontier models score ~20% on full NL2SQL.
We don't attempt full NL2SQL here (that needs a live warehouse + LLM); we measure
the foundation: can we ingest, flatten, and structurally understand these schemas.
"""
import argparse
import json
import time
from dsagent.catalog import Catalog
from dsagent.types import Dialect
from dsagent.graph import JoinGraph
from dsagent.sql import analyze
from dsagent.sql.gates import fanout_gate, selftest_fanout_gate

# ============================================================ hard fixtures
# Multi-database, mixed dialect, deeply nested, with a planted one-to-many.
SNOWFLAKE_DDL = """
CREATE TABLE PROD.CORE.USERS (
    user_id NUMBER PRIMARY KEY, email VARCHAR, plan_tier VARCHAR,
    attributes VARIANT,
    billing_address OBJECT(street STRING, city STRING, zip STRING));
CREATE TABLE PROD.CORE.ACCOUNTS (
    account_id NUMBER PRIMARY KEY, owner_user_id NUMBER, mrr NUMBER,
    FOREIGN KEY (owner_user_id) REFERENCES PROD.CORE.USERS(user_id));
CREATE TABLE PROD.BILLING.INVOICES (
    invoice_id NUMBER PRIMARY KEY, account_id NUMBER, amount NUMBER,
    FOREIGN KEY (account_id) REFERENCES PROD.CORE.ACCOUNTS(account_id));
"""
BIGQUERY_DDL = """
CREATE TABLE ANALYTICS.EVENTS.PRODUCT_EVENTS (
    event_id STRING, user_id INT64,
    device STRUCT<os STRING, browser STRING>,
    page STRUCT<url STRING, utm STRUCT<source STRING, medium STRING, campaign STRING>>,
    properties ARRAY<STRUCT<key STRING, value STRING>>);
"""

# ground truth ---------------------------------------------------------------
EXPECTED_NESTED_LEAVES = {
    "device.os", "device.browser", "page.url",
    "page.utm.source", "page.utm.medium", "page.utm.campaign",
    "properties.element.key", "properties.element.value",
}
EXPECTED_FK_EDGES = {                              # unordered table pairs
    frozenset({"PROD.CORE.ACCOUNTS", "PROD.CORE.USERS"}),
    frozenset({"PROD.BILLING.INVOICES", "PROD.CORE.ACCOUNTS"}),
}
EXPECTED_FANOUT = frozenset({"PROD.BILLING.INVOICES", "PROD.CORE.ACCOUNTS"})

SPAGHETTI = {
    "deep_no_cte": ("""SELECT DISTINCT u.user_id FROM PROD.CORE.USERS u, PROD.CORE.ACCOUNTS a
                       WHERE u.user_id = a.owner_user_id
                       AND EXISTS (SELECT 1 FROM PROD.CORE.ACCOUNTS x WHERE x.owner_user_id=u.user_id)""",
                    {"implicit_join"}),
    "double_count": ("""SELECT a.account_id, SUM(a.mrr)
                        FROM PROD.CORE.ACCOUNTS a
                        JOIN PROD.BILLING.INVOICES i ON i.account_id=a.account_id
                        GROUP BY a.account_id""",
                     {"fanout"}),
    "cross_join": ("""SELECT * FROM PROD.CORE.USERS u
                      CROSS JOIN PROD.CORE.ACCOUNTS a""", {"cross_join", "select_star"}),
}


def _catalog():
    c = Catalog()
    c.ingest_ddl(SNOWFLAKE_DDL, Dialect.SNOWFLAKE)
    c.ingest_ddl(BIGQUERY_DDL, Dialect.BIGQUERY)
    return c


def _edges(jg):
    return {frozenset({e.left_table, e.right_table})
            for u, v, d in jg.g.edges(data=True) for e in [d["edge"]]}


# ============================================================ scorers
def score_nested(cat):
    leaves = {c.full_path for t in cat.tables.values() for c in t.leaf_columns()}
    found = EXPECTED_NESTED_LEAVES & leaves
    return len(found), len(EXPECTED_NESTED_LEAVES), sorted(EXPECTED_NESTED_LEAVES - leaves)


def score_joins(jg):
    edges = _edges(jg)
    found = EXPECTED_FK_EDGES & edges
    return len(found), len(EXPECTED_FK_EDGES), [set(e) for e in EXPECTED_FK_EDGES - edges]


def score_fanout(jg):
    fo = {frozenset({e.left_table, e.right_table}) for e in jg.fanout_edges()}
    return EXPECTED_FANOUT in fo


def score_spaghetti(cat, jg):
    rows = []
    for name, (sql, expect) in SPAGHETTI.items():
        qa = analyze(sql, cat)
        text = " ".join(qa.anti_patterns).lower()
        got = set()
        if "implicit (comma) join" in text:
            got.add("implicit_join")
        if "cross join" in text:
            got.add("cross_join")
        if "select *" in text:
            got.add("select_star")
        if "correlated subquery" in text:
            got.add("correlated")
        if jg.query_fanout_warnings(qa):
            got.add("fanout")
        rows.append((name, expect, expect & got, expect <= got))
    return rows


def score_scale(n_tables=40, cols_each=15):
    """Generate a wide chained schema (~n_tables*cols_each columns) and confirm
    ingest + flatten + join-graph build complete, with timing."""
    parts = []
    for i in range(n_tables):
        cols = [f"t{i}_id NUMBER PRIMARY KEY"]
        if i > 0:
            cols.append(f"t{i-1}_id NUMBER")                 # name-inferable FK chain
        cols += [f"col_{j} VARCHAR" for j in range(cols_each - 2)]
        parts.append(f"CREATE TABLE WIDE.S.T{i} (" + ", ".join(cols) + ");")
    c = Catalog()
    t0 = time.time()
    c.ingest_ddl("\n".join(parts), Dialect.GENERIC)
    jg = JoinGraph(c)
    leaves = sum(len(t.leaf_columns()) for t in c.tables.values())
    edges = len(_edges(jg))
    return {"tables": len(c.tables), "columns": leaves, "fk_edges_recovered": edges,
            "seconds": round(time.time() - t0, 3)}


# ============================================================ Spider/BIRD loader
def load_spider_schema(tables_json_path, which=0):
    """Build a Catalog from a Spider/BIRD `tables.json` entry (one DB) by
    synthesizing CREATE TABLE DDL, then return (catalog, declared_fk_pairs)."""
    data = json.load(open(tables_json_path))
    db = data[which] if isinstance(data, list) else data
    tnames = db["table_names_original"]
    cols = db["column_names_original"]              # [[table_idx, col_name], ...]
    ctypes = db.get("column_types", ["text"] * len(cols))
    by_table = {i: [] for i in range(len(tnames))}
    for (ti, cname), ctype in zip(cols, ctypes):
        if ti >= 0:
            by_table[ti].append((cname, ctype))
    ddl = []
    for ti, t in enumerate(tnames):
        defs = ", ".join(f"{c} {('NUMBER' if ct in ('number','integer') else 'VARCHAR')}"
                         for c, ct in by_table[ti])
        ddl.append(f"CREATE TABLE SPIDER.S.{t} ({defs});")
    cat = Catalog()
    cat.ingest_ddl("\n".join(ddl), Dialect.GENERIC)
    declared = set()
    for a, b in db.get("foreign_keys", []):
        ta = tnames[cols[a][0]]; tb = tnames[cols[b][0]]
        declared.add(frozenset({f"SPIDER.S.{ta}", f"SPIDER.S.{tb}"}))
    return cat, declared


def score_nl2sql(cat):
    """Agentic NL2SQL: schema-linking recall + hallucination gate + repair +
    candidate selection. (Offline: drafts are supplied; in prod the LLM writes
    them. Measures the guardrail + selection logic, not execution accuracy.)"""
    from dsagent.sql.schema_linker import link_schema
    from dsagent.sql.nl2sql import NL2SQLAgent
    rows = []
    link_cases = [("monthly revenue by account", "INVOICES"),
                  ("user retention after signup", "USERS"),
                  ("events by device and browser", "PRODUCT_EVENTS")]
    hits = sum(want in [t.fqn.split(".")[-1] for t in link_schema(q, cat).tables[:3]]
               for q, want in link_cases)
    rows.append(("schema-linking recall@3", f"{hits}/{len(link_cases)}", hits == len(link_cases)))

    ag = NL2SQLAgent(cat)
    drafts = [
        "SELECT a.account_id, a.revenuexx FROM PROD.CORE.ACCOUNTS a",
        ("SELECT a.account_id, SUM(a.mrr) FROM PROD.CORE.ACCOUNTS a "
         "JOIN PROD.BILLING.INVOICES i ON i.account_id=a.account_id GROUP BY a.account_id"),
        "SELECT a.account_id, a.mrr FROM PROD.CORE.ACCOUNTS a",
    ]
    sel = ag.author("revenue by account", drafts=drafts)
    rows.append(("selects the valid candidate", sel.confidence, sel.valid))
    rep = ag.author("account revenue", drafts=["SELECT a.account_id, a.mrrr FROM PROD.CORE.ACCOUNTS a"])
    rows.append(("repairs hallucinated column", "snapped" if rep.repairs else "no-fix",
                 bool(rep.repairs) and rep.valid))
    fo = ag.author("total revenue", drafts=[drafts[1]])
    rows.append(("flags fan-out double-count", "blocked" if not fo.validation["fanout_ok"] else "missed",
                 not fo.validation["fanout_ok"]))
    return rows


def score_spider(path):
    cat, declared = load_spider_schema(path)
    jg = JoinGraph(cat)
    edges = _edges(jg)
    recovered = len(declared & edges)
    cols = sum(len(t.leaf_columns()) for t in cat.tables.values())
    return {"tables": len(cat.tables), "columns": cols,
            "declared_fks": len(declared), "fks_recovered": recovered}


# ============================================================ runner
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spider", default=None, help="path to a Spider/BIRD tables.json")
    args = ap.parse_args()

    cat = _catalog()
    jg = JoinGraph(cat)
    print("=" * 66)
    print("SCHEMA-COMPLEXITY BENCHMARK (offline fixtures, known ground truth)")
    print("=" * 66)

    f, tot, missing = score_nested(cat)
    print(f"\n[1] Nested flattening (Snowflake VARIANT/OBJECT + BigQuery STRUCT/ARRAY)")
    print(f"    recovered {f}/{tot} deep leaf paths  {'PASS' if f == tot else 'MISS '+str(missing)}")

    f, tot, missing = score_joins(jg)
    print(f"\n[2] Join-graph / FK recovery across 3 databases")
    print(f"    recovered {f}/{tot} FK edges  {'PASS' if f == tot else 'MISS '+str(missing)}")

    print(f"\n[3] Fan-out (double-count) detection on the 1:many invoices join")
    print(f"    {'PASS — flagged' if score_fanout(jg) else 'MISS — not flagged'}")

    print(f"\n[4] Spaghetti-SQL anti-pattern detection")
    for name, expect, got, ok in score_spaghetti(cat, jg):
        print(f"    {name:<14} expect {sorted(expect)} -> {'PASS' if ok else 'MISS '+str(sorted(got))}")

    print(f"\n[5] Scale stress (wide chained schema)")
    sc = score_scale()
    print(f"    ingested {sc['tables']} tables / {sc['columns']} columns, "
          f"recovered {sc['fk_edges_recovered']} FK-chain edges in {sc['seconds']}s")

    print(f"\n[6] Verified fan-out GATE (blocks the double-count query)")
    g = selftest_fanout_gate()
    print(f"    blocks double-count: {g['blocks_double_count']} | passes safe query: {g['passes_safe_query']}"
          f"  {'PASS' if g['blocks_double_count'] and g['passes_safe_query'] else 'MISS'}")

    print(f"\n[7] Agentic NL2SQL (schema-link -> validate -> repair -> select)")
    for name, detail, ok in score_nl2sql(cat):
        print(f"    {name:<32} {str(detail):<10} {'PASS' if ok else 'MISS'}")

    if args.spider:
        print(f"\n[8] REAL enterprise schema (Spider/BIRD): {args.spider}")
        s = score_spider(args.spider)
        print(f"    {s['tables']} tables / {s['columns']} columns ingested; "
              f"recovered {s['fks_recovered']}/{s['declared_fks']} declared FK edges")
    else:
        print(f"\n[8] Real Spider/BIRD schema: pass --spider tables.json (needs the file)")
    print("\n" + "=" * 66)


if __name__ == "__main__":
    main()
