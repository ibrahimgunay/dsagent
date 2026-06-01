"""Full agentic run: a goal in, a reconciled analysis out.

  python -m demo.run_agent

Flow:  build catalog -> build org (planner + sub-agents) -> planner LLM emits a
DAG -> orchestrator executes it (parallel batches, human-approval checkpoints,
budget) -> sub-agents communicate via the blackboard -> critic gates + reconciles
-> delivery. Runs offline on StubLLM; export ANTHROPIC_API_KEY and swap the
client for a live run.
"""
from __future__ import annotations

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsagent import Catalog
from dsagent.types import Dialect
from dsagent.llm import StubLLM            # swap: AnthropicClient(model="claude-sonnet-4-...")
from dsagent.agents import build_org
from dsagent.runtime import Blackboard, Orchestrator, Budget
from demo.fixtures import SNOWFLAKE_DDL, BIGQUERY_DDL


def hr(t): print("\n" + "=" * 76 + f"\n{t}\n" + "=" * 76)


def main():
    GOAL = ("Determine whether adopting feature_X causally increased 7-day "
            "session retention for users, with heterogeneity by plan tier, and "
            "ship a dashboard + stakeholder memo.")

    cat = Catalog()
    cat.ingest_ddl(SNOWFLAKE_DDL, Dialect.SNOWFLAKE)
    cat.ingest_ddl(BIGQUERY_DDL, Dialect.BIGQUERY)

    llm = StubLLM()
    registry, planner = build_org(cat, llm)

    hr("CEO GOAL -> PLANNER")
    print(GOAL)
    print(f"\nAvailable sub-agents (tools): {registry.names()}")
    dag = planner.plan(GOAL, context={"databases": sorted(cat.databases())})
    print("\nPlanner-generated execution plan:")
    print(dag.describe())

    hr("ORCHESTRATOR EXECUTION (parallel batches, HITL, budget)")
    bb = Blackboard()
    orch = Orchestrator(registry, bb, budget=Budget(max_usd=2.0), max_workers=4)
    result = orch.run(dag)

    print("\nEvent log (agent communication via blackboard):")
    for line in bb.log:
        print("  " + line)

    hr("RESULTS")
    print(f"completed: {result.completed}")
    print(f"failed   : {result.failed or 'none'}")
    print(f"halted   : {result.halted_reason or 'no'}")
    print(f"budget   : {result.budget}")

    hr("CRITIC GATES (blocking validity checks)")
    crit = bb.value("critic", {})
    print(f"verdict: {crit.get('verdict')}")
    for g in crit.get("gates", []):
        mark = "PASS" if g["passed"] else ("BLOCK" if g["blocking"] else "warn")
        print(f"  [{mark}] {g['name']}")
    if crit.get("reconciliation"):
        print(f"reconciliation: {crit['reconciliation']}")

    hr("ECONOMETRICS ARTIFACT (estimand-first)")
    econ = bb.value("econ", {})
    for k in ("estimand", "estimator", "rejected_estimators",
              "identifying_assumptions", "sensitivity_analysis"):
        print(f"  {k}: {econ.get(k)}")

    hr("DELIVERABLES")
    print("dashboard tiles:",
          [t["name"] for t in bb.value("dashboard", {}).get("tiles", [])])
    memo = bb.value("memo", {})
    print("memo headline :", memo.get("headline"))
    print("memo caveats  :", memo.get("caveats"))

    hr("LINEAGE (memo provenance)")
    print(json.dumps(bb.lineage("memo"), indent=2)[:700])

    hr("DONE")
    print(f"LLM backend: {llm.model}  |  runtime LLM calls: {len(llm.transcript)}")
    print("Swap StubLLM -> AnthropicClient (set ANTHROPIC_API_KEY) for a live run.")


if __name__ == "__main__":
    main()
