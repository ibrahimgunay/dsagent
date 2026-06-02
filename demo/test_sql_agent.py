"""Tests for agentic NL2SQL: schema linking, validation gates, repair, selection.
Run: python -m demo.test_sql_agent"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsagent.catalog import Catalog
from dsagent.types import Dialect
from dsagent.sql.schema_linker import link_schema
from dsagent.sql.validate import validate_sql, repair_sql
from dsagent.sql.nl2sql import NL2SQLAgent

DDL = """
CREATE TABLE PROD.CORE.USERS (user_id NUMBER PRIMARY KEY, email VARCHAR, plan_tier VARCHAR);
CREATE TABLE PROD.CORE.ACCOUNTS (account_id NUMBER PRIMARY KEY, owner_user_id NUMBER, mrr NUMBER,
  FOREIGN KEY (owner_user_id) REFERENCES PROD.CORE.USERS(user_id));
CREATE TABLE PROD.BILLING.INVOICES (invoice_id NUMBER PRIMARY KEY, account_id NUMBER, amount NUMBER,
  FOREIGN KEY (account_id) REFERENCES PROD.CORE.ACCOUNTS(account_id));
"""

PASS = FAIL = 0
def check(n, c):
    global PASS, FAIL
    PASS, FAIL = PASS + bool(c), FAIL + (not c)
    print(f"  {'PASS' if c else 'FAIL'}  {n}")


def main():
    cat = Catalog(); cat.ingest_ddl(DDL, Dialect.SNOWFLAKE)

    print("schema linking (works on a schema with no instruction)")
    rev = [t.fqn.split(".")[-1] for t in link_schema("monthly revenue by account", cat).tables[:3]]
    check("revenue question links INVOICES/ACCOUNTS", "INVOICES" in rev or "ACCOUNTS" in rev)
    usr = [t.fqn.split(".")[-1] for t in link_schema("users by plan tier", cat).tables[:2]]
    check("user question links USERS", "USERS" in usr)

    print("validation gates")
    good = validate_sql("SELECT a.account_id, a.mrr FROM PROD.CORE.ACCOUNTS a", cat)
    check("clean query validates", good.ok)
    halluc = validate_sql("SELECT a.account_id, a.revenuexx FROM PROD.CORE.ACCOUNTS a", cat)
    check("hallucinated column flagged", "a.revenuexx" in halluc.unknown_columns)
    fan = validate_sql("SELECT a.account_id, SUM(a.mrr) FROM PROD.CORE.ACCOUNTS a "
                       "JOIN PROD.BILLING.INVOICES i ON i.account_id=a.account_id GROUP BY a.account_id", cat)
    check("fan-out double-count flagged", not fan.fanout_ok)

    print("deterministic self-repair")
    rep = validate_sql("SELECT a.mrrr FROM PROD.CORE.ACCOUNTS a", cat)
    fixed, notes = repair_sql("SELECT a.mrrr FROM PROD.CORE.ACCOUNTS a", rep, cat)
    check("typo column snapped to real one", "a.mrr" in fixed and "mrrr" not in fixed)

    print("agentic loop: candidate selection")
    ag = NL2SQLAgent(cat)
    res = ag.author("revenue by account", drafts=[
        "SELECT a.account_id, a.revenuexx FROM PROD.CORE.ACCOUNTS a",          # hallucinated
        "SELECT a.account_id, SUM(a.mrr) FROM PROD.CORE.ACCOUNTS a "
        "JOIN PROD.BILLING.INVOICES i ON i.account_id=a.account_id GROUP BY a.account_id",  # fan-out
        "SELECT a.account_id, a.mrr FROM PROD.CORE.ACCOUNTS a",                # clean
    ])
    check("selects a valid query from mixed candidates", res.valid)
    check("reports high confidence on the clean pick", res.confidence == "high")
    check("never ships unknown columns", res.validation["unknown_columns"] == [])

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
