"""End-to-end demo: ingest -> profile -> ontology -> join graph -> SQL analysis.

Run:  python -m demo.run_demo     (from the package root)
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsagent import Catalog, JoinGraph, Ontology, profile_table, default_plan
from dsagent.types import Dialect
from dsagent import sql as sqlmod
from demo.fixtures import (SNOWFLAKE_DDL, BIGQUERY_DDL, SPAGHETTI_QUERIES)


def hr(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def main():
    cat = Catalog()
    hr("1. INGEST DDL ACROSS MULTIPLE DATABASES / DIALECTS")
    cat.ingest_ddl(SNOWFLAKE_DDL, Dialect.SNOWFLAKE)
    cat.ingest_ddl(BIGQUERY_DDL, Dialect.BIGQUERY)
    print(f"Databases tracked : {sorted(cat.databases())}")
    print(f"Tables ingested   : {len(cat.tables)}")
    for fqn in sorted(cat.tables):
        t = cat.tables[fqn]
        print(f"   - {fqn:<42} [{t.dialect.value}]  "
              f"{len(t.columns)} cols / {len(t.leaf_columns())} leaves")

    hr("2. NESTED-TYPE FLATTENING (Snowflake VARIANT/OBJECT, BigQuery STRUCT/ARRAY)")
    ev = cat.get("ANALYTICS.EVENTS.PRODUCT_EVENTS")
    print(f"{ev.fqn} flattened leaf paths:")
    for c in ev.leaf_columns():
        print(f"   - {c.full_path:<40} {c.normalized_type}")

    hr("3. PROFILING: SEMANTIC TYPING + PII DETECTION")
    for fqn in sorted(cat.tables):
        p = profile_table(cat.tables[fqn])
        pii = ", ".join(p["pii_fields"]) or "none"
        ft = ", ".join(p["free_text_fields"]) or "none"
        print(f"{fqn}")
        print(f"     types : {p['semantic_type_counts']}")
        print(f"     PII   : {pii}")
        if ft != "none":
            print(f"     text  : {ft}  (-> LLM extraction subsystem)")

    hr("4. JOIN GRAPH (declared + inferred + observed-in-SQL)")
    jg = JoinGraph(cat)
    # fold in join keys observed in the sample SQL
    for name, q in SPAGHETTI_QUERIES.items():
        qa = sqlmod.analyze(q, cat)
        jg.add_observed_edges(qa.join_edges)
    print(f"Nodes: {jg.g.number_of_nodes()}   Edges: {jg.g.number_of_edges()}")

    print("\nRecommended join path  USERS -> USER_FEATURES:")
    path = jg.join_path("USERS", "USER_FEATURES")
    if path:
        for e in path:
            flag = "  <-- FAN-OUT (aggregate first!)" if e.fanout_risk else ""
            print(f"   {e.left_table}.{e.left_key} = {e.right_table}.{e.right_key}"
                  f"  [{e.cardinality}, {e.source}]{flag}")

    print("\nFan-out joins (will double-count measures unless pre-aggregated):")
    for e in jg.fanout_edges()[:6]:
        print(f"   {e.left_table}  <->  {e.right_table}   ({e.cardinality})")

    print("\nAmbiguous joins (same table pair, >1 distinct key => 'which key?'):")
    amb = jg.ambiguous_joins()
    for a, b, keys in amb[:6]:
        short = lambda x: x.split(".")[-1]
        print(f"   {short(a)} <-> {short(b)} : {keys}")
    if not amb:
        print("   none")

    hr("5. ONTOLOGY / SEMANTIC LAYER")
    onto = Ontology(cat).build()
    s = onto.summary()
    print("Entities:")
    for ent, tbl in s["entities"].items():
        print(f"   - {ent:<14} <- {tbl}")
    print(f"\nGoverned metrics: {s['metric_count']}")
    print(f"   sample: {s['metrics_sample']}")

    hr(f"6. SPAGHETTI-SQL ANALYSIS  (parser backend: {sqlmod.backend_name()})")
    for name, q in SPAGHETTI_QUERIES.items():
        qa = sqlmod.analyze(q, cat)
        m = qa.complexity
        print(f"\n[{name}]  syntactic={m["complexity_band"].upper()}  "
              f"score={m['complexity_score']}  joins={m['join_count']}  "
              f"depth={m['subquery_depth']}  ctes={m['cte_count']}")
        print(f"   tables: {sorted(set(qa.referenced_tables))}")
        for ap in qa.anti_patterns:
            print(f"   ! {ap}")
        for w in jg.query_fanout_warnings(qa):
            print(f"   !! {w}")

    hr("7. ANALYSIS ORCHESTRATION DAG (parallel vs sequential)")
    dag = default_plan()
    print(dag.describe())

    hr("DONE")
    print("Every artifact above is typed + lineage-ready. Swap classify_column()")
    print("for an LLM call and install sqlglot for dialect-perfect column lineage.")


if __name__ == "__main__":
    main()
