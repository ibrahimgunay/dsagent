"""Tests for the execution + eval layers. Run: python -m demo.test_exec_eval"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsagent.execution import datagen as dg, estimators as est
from dsagent.execution.executor import Executor, select_design
from dsagent.eval import run_all
from dsagent.llm import StubLLM
from dsagent.agents.modeling import EconometricianAgent
from dsagent.runtime import Blackboard, ToolContext

PASS = FAIL = 0
def check(n, c):
    global PASS, FAIL
    PASS, FAIL = PASS + bool(c), FAIL + (not c)
    print(f"  {'PASS' if c else 'FAIL'}  {n}")


def main():
    print("estimator recovery (known truth)")
    check("RCT recovers ATE=2", est.diff_in_means(dg.make_rct(2.0)).covers(2.0))
    check("DiD recovers ATT=1.5", est.did_2x2(dg.make_did_panel(1.5)).covers(1.5))
    d = dg.make_observational(2.0)
    check("adjusted recovers ATE on confounded", est.regression_adjust(d).covers(2.0))
    check("double_ml recovers ATE on confounded", est.double_ml(d).covers(2.0))

    print("discrimination")
    check("naive is BIASED on confounded data", not est.diff_in_means(d).covers(2.0))
    check("null shows no false positive",
          est.regression_adjust(dg.make_null()).pvalue >= 0.05)

    print("heterogeneity + overlap")
    het, truth = dg.make_heterogeneous()
    cate = est.cate_by_subgroup(het, group="tier")
    check("CATE recovers high responder ~3.0", abs(cate[1]["point"] - 3.0) < 0.4)
    check("CATE recovers low responder ~0.5", abs(cate[0]["point"] - 0.5) < 0.4)
    check("overlap check returns structured dict",
          "overlap_ok" in est.propensity_overlap(d))

    print("design selection")
    check("panel -> did", select_design({"is_panel": True})["estimator"] == "did_2x2")
    check("randomized -> diff_in_means",
          select_design({"randomized": True})["estimator"] == "diff_in_means")
    check("observational+controls -> dml",
          select_design({"has_controls": True})["estimator"] == "double_ml")

    print("agent actually FITS")
    bb = Blackboard()
    econ = EconometricianAgent(StubLLM(), Executor())
    econ.run(ToolContext(blackboard=bb, task_id="e", depends_on=[],
                         params={"data": d, "profile": {"has_controls": True}}))
    e = bb.value("e").get("estimate", {})
    check("agent emits numeric estimate", "point" in e)
    check("agent estimate covers truth",
          e.get("ci", [9, 9])[0] <= 2.0 <= e.get("ci", [-9, -9])[1])

    print("staggered DiD (Callaway-Santanna) + pre-trends")
    sdf, strue = dg.make_staggered(base=0.4)
    cs = est.callaway_santanna(sdf)
    twfe = est.twfe_static(sdf)
    check("CS recovers staggered ATT", cs.covers(strue))
    check("naive TWFE is biased on staggered data", not twfe.covers(strue))
    check("staggered -> selects callaway_santanna",
          select_design({"is_staggered": True})["estimator"] == "callaway_santanna")
    check("pre-trends pass on parallel data", est.pretrends_test(sdf)["pretrends_ok"])
    bad = sdf.copy()
    bad.loc[(bad["cohort"] == 3) & (bad["period"] == 1), "y"] += 2.0
    check("pre-trends FAIL on violation", not est.pretrends_test(bad)["pretrends_ok"])

    print("ML predictive fit (leakage-safe)")
    from dsagent.execution.ml import fit_predictive
    mlr = fit_predictive(dg.make_observational(2.0), target="t", features=["x"])
    check("ML fit returns AUC in (0.5,1)", 0.5 < mlr.auc < 1.0)
    check("ML reports a CV strategy", "Fold" in mlr.cv_strategy)
    check("ML leakage scan runs", isinstance(mlr.leakage_flags, list))

    print("full eval harness")
    rep = run_all(catalog=None, llm=None)
    check("all scenarios pass", rep["passed"] == rep["total"])
    check("staggered scenario included", rep["total"] >= 5)

    print("end-to-end pipeline (data flows -> real gated estimate)")
    from dsagent import Catalog
    from dsagent.types import Dialect
    from dsagent.pipeline import run_analysis
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from demo.fixtures import SNOWFLAKE_DDL, BIGQUERY_DDL
    cat = Catalog()
    cat.ingest_ddl(SNOWFLAKE_DDL, Dialect.SNOWFLAKE)
    cat.ingest_ddl(BIGQUERY_DDL, Dialect.BIGQUERY)
    r = run_analysis("did feature_X raise retention?", cat)
    check("pipeline completes without failure", not r.failed and not r.halted)
    check("pipeline emits a numeric estimate", r.estimate and "point" in r.estimate)
    check("pipeline estimate recovered planted truth", r.estimate.get("recovered") is True)
    check("pipeline produced deliverables", r.deliverables.get("memo") is not None)
    check("pipeline verdict not failed-gates", r.verdict != "FAILED_GATES")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
