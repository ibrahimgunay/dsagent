"""Tests for the agentic layer. Run: python -m demo.test_agent"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsagent import Catalog
from dsagent.types import Dialect
from dsagent.llm import StubLLM
from dsagent.agents import build_org
from dsagent.agents.critic import CriticAgent
from dsagent.runtime import Blackboard, Orchestrator, Budget, ToolContext, Artifact
from dsagent.planner import AnalysisDAG, Task
from demo.fixtures import SNOWFLAKE_DDL, BIGQUERY_DDL

PASS = FAIL = 0
def check(n, c):
    global PASS, FAIL
    PASS, FAIL = PASS + bool(c), FAIL + (not c)
    print(f"  {'PASS' if c else 'FAIL'}  {n}")


def org():
    cat = Catalog()
    cat.ingest_ddl(SNOWFLAKE_DDL, Dialect.SNOWFLAKE)
    cat.ingest_ddl(BIGQUERY_DDL, Dialect.BIGQUERY)
    return cat, StubLLM()


def main():
    cat, llm = org()
    reg, planner = build_org(cat, llm)

    print("planner")
    dag = planner.plan("measure feature_X causal effect on retention")
    batches = dag.execution_batches()
    check("planner builds acyclic DAG", isinstance(dag, AnalysisDAG))
    check("modeling batch runs in parallel (>=3 tasks)",
          any(len(b) >= 3 for b in batches))
    check("plan has human-approval checkpoints",
          any(t.requires_human_approval for b in batches for t in b))

    print("planner rejects bad plans")
    try:
        planner._to_dag({"tasks": [{"id": "x", "tool": "not_a_real_tool",
                                    "depends_on": []}]})
        check("unknown tool rejected", False)
    except ValueError:
        check("unknown tool rejected", True)

    print("orchestrator end-to-end")
    bb = Blackboard()
    res = Orchestrator(reg, bb, budget=Budget(max_usd=5.0)).run(dag)
    check("all tasks completed", not res.failed and not res.halted_reason)
    check("blackboard recorded artifacts", len(bb.keys()) >= 10)
    check("runtime LLM calls happened", len(llm.transcript) >= 5)
    check("budget tracked tokens", res.budget["tokens"] > 0)

    print("critic blocking gates")
    crit_art = bb.value("critic", {})
    check("happy-path critic passes gates", not crit_art.get("blocking_failures"))

    # negative case: feed a causal artifact missing sensitivity analysis -> must block
    bb2 = Blackboard()
    bb2.put(Artifact(key="econ_bad", kind="econometrics", producer="t",
                     value={"estimand": "ATT", "estimator": "TWO-WAY FIXED EFFECTS",
                            "identifying_assumptions": ["parallel trends"]}))  # no sensitivity, naive TWFE
    bb2.put(Artifact(key="sql_bad", kind="sql", producer="t",
                     value={"fanout_warnings": ["... DOUBLE-COUNT ..."]}))
    critic = CriticAgent(llm)
    ctx = ToolContext(blackboard=bb2, task_id="critic2",
                      depends_on=["econ_bad", "sql_bad"])
    critic.run(ctx)
    v = bb2.value("critic2")
    check("missing sensitivity analysis BLOCKS",
          "identification_before_estimation" in v["blocking_failures"])
    check("naive TWFE BLOCKS", "no_naive_twfe" in v["blocking_failures"])
    check("unresolved fan-out BLOCKS", "sql_fanout_resolved" in v["blocking_failures"])
    check("verdict is FAILED_GATES", v["verdict"] == "FAILED_GATES")

    print("provider factory (multi-LLM)")
    from dsagent.llm import make_client, OpenAIClient, GeminiClient, AnthropicClient
    check("factory: stub", type(make_client("stub")).__name__ == "StubLLM")
    check("factory: openai", isinstance(make_client("openai"), OpenAIClient))
    check("factory: gemini", isinstance(make_client("gemini"), GeminiClient))
    check("factory: anthropic", isinstance(make_client("anthropic"), AnthropicClient))
    try:
        make_client("bogus"); check("unknown provider rejected", False)
    except ValueError:
        check("unknown provider rejected", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
