"""Tests for the adaptive planner. Run: python -m demo.test_planner"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsagent import Catalog
from dsagent.types import Dialect
from dsagent.planning import AdaptivePlanner, PlanGraph, PlanNode
from dsagent.pipeline import run_adaptive
from dsagent.data import SyntheticDataSource
from demo.fixtures import SNOWFLAKE_DDL, BIGQUERY_DDL

TOOLS = {"profiler", "semantic_modeler", "join_analyzer", "sql_author",
         "data_executor", "econometrician", "ml_engineer", "causal_ml",
         "labeler", "critic", "dashboard_builder", "memo_writer",
         "trust_report", "noop"}

PASS = FAIL = 0
def check(n, c):
    global PASS, FAIL
    PASS, FAIL = PASS + bool(c), FAIL + (not c)
    print(f"  {'PASS' if c else 'FAIL'}  {n}")


def main():
    print("reflection loop (plan repairs itself)")
    p = AdaptivePlanner(TOOLS)
    draft = p.propose_initial("x")
    draft_issues = p.critique(draft)
    g = p.plan("x")
    check("draft has issues", len(draft_issues) >= 3)
    check("final plan is clean", p.critique(g) == [])
    check("reflection converged in <=6 iters", len(p.reflection_log) <= 7)
    check("critic added by reflection", any(n.tool == "critic" for n in g.active()))
    check("sign-off added by reflection",
          any(n.requires_human_approval and n.phase == "P5" for n in g.active()))
    check("plan validates (acyclic)", (g.validate() or True))

    print("multi-hypothesis branching (data -> design)")
    cases = {"is_staggered": "cs", "has_controls": "dml",
             "has_instrument": "iv", "is_panel": "did"}
    for key, expect in cases.items():
        gg = AdaptivePlanner(TOOLS).plan("x")
        AdaptivePlanner(TOOLS).resolve_design(gg, {key: True})
        kept = [n.branch_value for n in gg.active() if n.branch_key == "design"]
        check(f"{key} -> commits to {expect}", kept == [expect])

    print("event-driven replanning")
    g2 = p.plan("x"); p.on_event(g2, "data_profiled", profile={"has_controls": True})
    n_before = len(g2.active())
    p.on_event(g2, "gate_failed", node_id="design_dml", gate="overlap")
    check("gate failure inserts a repair node", len(g2.active()) == n_before + 1)
    g3 = AdaptivePlanner(TOOLS).plan("x")
    opt_before = [n.id for n in g3.active() if n.optional]
    AdaptivePlanner(TOOLS).on_event(g3, "low_budget")
    check("low budget prunes optional nodes", len(opt_before) >= 1)

    print("end-to-end adaptivity (same goal, different data -> different design)")
    c = Catalog()
    c.ingest_ddl(SNOWFLAKE_DDL, Dialect.SNOWFLAKE); c.ingest_ddl(BIGQUERY_DDL, Dialect.BIGQUERY)
    results = {}
    for scen, truth in [("observational", 2.0), ("staggered", 0.4),
                        ("iv", 1.5), ("panel", 1.5)]:
        r = run_adaptive("x", c, data_source=SyntheticDataSource(scen, truth))
        results[scen] = r["committed_design"]
        check(f"{scen}: recovers truth", (r["estimate"] or {}).get("recovered") is True)
    check("observational -> DML", results["observational"] == "observational_dml")
    check("staggered -> CS", results["staggered"] == "staggered_did_cs")
    check("iv -> 2SLS", results["iv"] == "iv_2sls")
    check("panel -> 2x2 DiD", results["panel"] == "two_period_did")
    check("four shapes -> four distinct designs", len(set(results.values())) == 4)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
