"""Command-line entry point.

    python -m dsagent run     # full pipeline on the demo schema, real estimate
    python -m dsagent adaptive # adaptive planner: same goal, data-driven design choice
    python -m dsagent eval    # gold-standard eval scorecard
    python -m dsagent test    # run all test suites
    python -m dsagent demo    # the foundation (schema/joins/spaghetti) walkthrough
"""
from __future__ import annotations

import sys, subprocess, os


def _demo_catalog():
    from dsagent import Catalog
    from dsagent.types import Dialect
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(here))
    from demo.fixtures import SNOWFLAKE_DDL, BIGQUERY_DDL
    cat = Catalog()
    cat.ingest_ddl(SNOWFLAKE_DDL, Dialect.SNOWFLAKE)
    cat.ingest_ddl(BIGQUERY_DDL, Dialect.BIGQUERY)
    return cat


def cmd_run(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="dsagent run")
    ap.add_argument("--provider", default="stub",
                    choices=["stub", "openai", "gemini"],
                    help="LLM backend (default: stub, runs offline)")
    ap.add_argument("--model", default=None, help="override the default model")
    args = ap.parse_args(argv)

    from dsagent.pipeline import run_analysis
    from dsagent.llm import make_client
    goal = ("Determine whether adopting feature_X causally increased 7-day "
            "retention, with heterogeneity by plan tier; ship dashboard + memo.")
    llm = make_client(args.provider, args.model)
    print(f"[provider={args.provider} model={getattr(llm, 'model', '?')}]")
    res = run_analysis(goal, _demo_catalog(), llm=llm)
    print(res.summary())


def _run_module(mod: str) -> int:
    return subprocess.call([sys.executable, "-m", mod],
                           cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cmd_eval():
    sys.exit(_run_module("demo.run_eval"))


def cmd_demo():
    sys.exit(_run_module("demo.run_demo"))


def cmd_adaptive():
    from dsagent.pipeline import run_adaptive
    from dsagent.data import SyntheticDataSource
    cat = _demo_catalog()
    print("Same goal, different data shapes -> planner commits to different designs:\n")
    for scen, truth in [("observational", 2.0), ("staggered", 0.4),
                        ("iv", 1.5), ("panel", 1.5)]:
        r = run_adaptive("did feature_X raise retention?", cat,
                         data_source=SyntheticDataSource(scen, truth))
        e = r["estimate"] or {}
        if scen == "observational":
            print("Reflection loop on the initial plan:")
            for line in r["reflection_log"]:
                print("   " + line)
            print()
        print(f"  data={scen:14} -> design={r['committed_design']:18} "
              f"estimate={e.get('point')} recovered={e.get('recovered')}")
        for rp in r["replans"]:
            print(f"       replan: {rp}")


def cmd_test():
    rc = 0
    for t in ("demo.test_core", "demo.test_agent", "demo.test_exec_eval",
              "demo.test_skills", "demo.test_planner", "demo.test_sql_agent"):
        print(f"\n### {t}")
        rc |= _run_module(t)
    sys.exit(rc)


def main():
    cmds = {"run": cmd_run, "eval": cmd_eval, "test": cmd_test, "demo": cmd_demo,
            "adaptive": cmd_adaptive}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(__doc__)
        sys.exit(2)
    cmd = sys.argv[1]
    if cmd == "run":
        cmd_run(sys.argv[2:])
    else:
        cmds[cmd]()


if __name__ == "__main__":
    main()
