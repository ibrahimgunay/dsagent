"""Run the gold-standard eval.   python -m demo.run_eval"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsagent import Catalog
from dsagent.types import Dialect
from dsagent.llm import StubLLM
from dsagent.eval import run_all
from demo.fixtures import SNOWFLAKE_DDL, BIGQUERY_DDL


def hr(t): print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def main():
    cat = Catalog()
    cat.ingest_ddl(SNOWFLAKE_DDL, Dialect.SNOWFLAKE)
    cat.ingest_ddl(BIGQUERY_DDL, Dialect.BIGQUERY)

    report = run_all(catalog=cat, llm=StubLLM(), include_experiments=True)

    hr("GOLD-STANDARD EVAL: recovery of known truth (incl. adversarial traps)")
    print(f"{'scenario':<26}{'design':<16}{'estimate':>10}{'truth':>8}   checks")
    print("-" * 78)
    for s in report["scenarios"]:
        chk = "  ".join(f"{k}={'Y' if v else 'N'}" for k, v in s.checks.items())
        verdict = "PASS" if s.passed else "FAIL"
        print(f"{s.name:<26}{s.selected_estimator:<16}{s.point:>10}{s.truth:>8}   [{verdict}]")
        print(f"{'':<26}CI {s.ci}   {chk}")
    print(f"\nScenario score: {report['passed']}/{report['total']} passed")

    hr("WITHIN-STUDY (the credibility proof): recover an EXPERIMENT from observational data")
    w = report["within_study"]
    print(f"  experiment truth (ATE)      : {w['experiment_truth']}")
    print(f"  recovered from OBS via DML  : {w['observational_dml']}  -> covers truth: {w['recovered_experiment']}")
    print(f"  naive contrast on same data : {w['naive_point']}  -> biased: {w['naive_biased']}")

    hr("LEAKAGE TRAP: a target-leak feature must be flagged")
    lk = report["leakage"]
    print(f"  AUC={lk['auc']}  leakage_flagged={lk['leakage_flagged']}")
    for f in lk["flags"]:
        print(f"    ! {f}")

    hr("RETRIEVAL EVAL: did we pick the right skill for the data shape?")
    rt = report["retrieval"]
    print(f"  precision@1 = {rt['precision_at_1']} over {rt['cases']} data shapes")

    hr("PROCESS EVAL: estimand-first compliance + gate-as-you-go")
    for k, v in report["process"].items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")

    hr("INTERPRETATION")
    print("- observational_confounded: the eval confirms the NAIVE contrast is")
    print("  biased (would mislead) while the selected DML design recovers truth.")
    print("- aa_null: no false-positive effect declared (FDR control).")
    print("- process: the agent states an estimand, identifying assumptions, and a")
    print("  sensitivity analysis, FITS a real estimate, and clears the critic gates.")


if __name__ == "__main__":
    main()
