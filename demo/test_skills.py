"""Tests for v2: verified skills, gate-as-you-go, adversarial evals, trust report.
Run: python -m demo.test_skills"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsagent.skills import build_default_registry
from dsagent.execution.executor import Executor
from dsagent.execution import datagen as dg, estimators as est
from dsagent.eval import run_all
from dsagent.eval.harness import within_study_eval, leakage_trap_eval
from dsagent.agents.modeling import EconometricianAgent
from dsagent.agents.trust_report import TrustReportAgent
from dsagent.llm import StubLLM
from dsagent.runtime import Blackboard, ToolContext, Artifact

PASS = FAIL = 0
def check(n, c):
    global PASS, FAIL
    PASS, FAIL = PASS + bool(c), FAIL + (not c)
    print(f"  {'PASS' if c else 'FAIL'}  {n}")


def main():
    print("verified skills (each proves itself on known truth)")
    reg = build_default_registry()
    v = reg.verify_all(Executor())
    check("all skills self-verify", v["passed"] == v["tested"] == v["skills"])
    check("retrieve staggered -> CS", reg.best({"is_staggered": True}).id == "staggered_did_cs")
    check("retrieve randomized -> RCT", reg.best({"randomized": True}).id == "rct_contrast")
    check("retrieve observational -> DML",
          reg.best({"has_controls": True}).id == "observational_dml")

    print("gate-as-you-go (withhold before fitting on failed assumption)")
    econ = EconometricianAgent(StubLLM(), Executor())
    bb = Blackboard()
    econ.run(ToolContext(blackboard=bb, task_id="ok", depends_on=[],
                         params={"data": dg.make_observational(2.0), "profile": {"has_controls": True}}))
    ok = bb.value("ok")
    check("fits when gate passes", (ok.get("estimate") or {}).get("point") is not None)
    check("records the precondition gate", ok.get("precondition_gate", {}).get("passed") is True)

    sdf, _ = dg.make_staggered(base=0.4)
    sdf.loc[(sdf["cohort"] == 3) & (sdf["period"] == 1), "y"] += 3.0   # break pre-trends
    bb2 = Blackboard()
    econ.run(ToolContext(blackboard=bb2, task_id="blk", depends_on=[],
                         params={"data": sdf, "profile": {"is_staggered": True}}))
    blk = bb2.value("blk")
    check("WITHHOLDS estimate when pre-trends fail", blk.get("estimate") is None)
    check("explains the block", "gate" in (blk.get("blocked_reason") or "").lower())

    print("adversarial traps")
    d = dg.make_simpsons(effect=1.0)
    check("Simpson's: naive has wrong sign", est.diff_in_means(d).point < 0)
    check("Simpson's: adjusted recovers +1", est.regression_adjust(d).covers(1.0))
    ws = within_study_eval(Executor())
    check("within-study recovers experiment truth", ws["recovered_experiment"])
    check("within-study: naive is biased", ws["naive_biased"])
    lk = leakage_trap_eval()
    check("leakage trap flagged", lk["leakage_flagged"])

    print("glass-box trust report")
    bb3 = Blackboard()
    bb3.put(Artifact(key="econ", kind="econometrics", producer="e",
                     value=ok))   # reuse the good run
    bb3.put(Artifact(key="critic", kind="validation", producer="c",
                     value={"gates": [{"name": "g1", "passed": True}],
                            "blocking_failures": [], "remaining_risks": [],
                            "required_caveats": ["ATT not ATE"]}))
    TrustReportAgent().run(ToolContext(blackboard=bb3, task_id="trust",
                                       depends_on=["econ", "critic"]))
    tr = bb3.value("trust")
    check("trust report has confidence", tr.get("confidence") in ("high", "medium", "low (gate failure)", "withheld"))
    check("trust report carries gate trail", len(tr.get("gate_trail", [])) >= 1)
    check("trust report states what-would-change-our-mind",
          len(tr.get("what_would_change_our_mind", [])) >= 1)

    print("expanded eval harness")
    rep = run_all(catalog=None, llm=None, include_experiments=True)
    check("eval all scenarios pass (incl. Simpson's + IV)", rep["passed"] == rep["total"])
    check("eval includes >=7 scenarios", rep["total"] >= 7)
    check("within-study reported", rep["within_study"]["recovered_experiment"])
    check("retrieval precision@1 == 1.0", rep["retrieval"]["precision_at_1"] == 1.0)

    print("v3 rigor: IV, sensitivity, refutations, conformal, FDR")
    div, ivtruth = dg.make_iv(ate=1.5, strength=1.2)
    iv = est.iv_2sls(div)
    check("IV recovers where OLS biased",
          iv.covers(ivtruth) and not est.diff_in_means(div).covers(ivtruth))
    check("strong instrument: F>=10", iv.diagnostics["first_stage_F"] >= 10)
    weak, _ = dg.make_iv(ate=1.5, strength=0.02)
    check("weak instrument flagged", est.iv_2sls(weak).diagnostics["weak_instrument"])
    check("IV skill gate blocks weak instrument",
          not reg.get("iv_2sls").check_gate(weak).passed)
    rv = est.robustness_value(est.double_ml(dg.make_observational(2.0), controls=("x",)))
    check("robustness value in [0,1]", 0 <= rv["robustness_value"] <= 1)
    ref = est.refutation_battery(
        lambda df: est.regression_adjust(df, controls=("x",)).point,
        dg.make_observational(2.0))
    check("valid estimate survives refutations", ref["survived_all"])
    from dsagent.execution.ml import conformal_classify
    cf = conformal_classify(dg.make_observational(2.0), target="t", features=["x"], alpha=0.1)
    check("conformal coverage >= target-0.05", cf["empirical_coverage"] >= 0.85)
    bh = est.benjamini_hochberg([0.001, 0.02, 0.2, 0.6, 0.9])
    check("BH rejects only small p-values", bh["n_rejected"] == 2)

    print("code-red: rigor upgrades")
    sdf2, _ = dg.make_staggered(base=0.4)
    pt = est.pretrends_test(sdf2)
    check("pre-trends is a joint TEST with p-value", "joint_p" in pt and pt["joint_p"] > 0.05)
    bad2 = sdf2.copy(); bad2.loc[(bad2["cohort"] == 3) & (bad2["period"] == 1), "y"] += 2.0
    check("pre-trends rejects a violation (p<=0.05)", est.pretrends_test(bad2)["joint_p"] <= 0.05)
    dml = est.double_ml(dg.make_observational(2.0), controls=("x",))
    check("DML uses 5-fold repeated cross-fitting", dml.diagnostics["folds"] == 5)
    real = est.refutation_battery(lambda d: est.regression_adjust(d, controls=("x",)).point,
                                  dg.make_observational(2.0), n_perm=120)
    null = est.refutation_battery(lambda d: est.regression_adjust(d, controls=("x",)).point,
                                  dg.make_null(), n_perm=120)
    check("permutation placebo: real effect significant", real["permutation_p"] < 0.05)
    check("permutation placebo: null effect NOT significant", null["permutation_p"] >= 0.05)

    print("code-red: self-repair loop")
    import numpy as np, pandas as pd
    rng = np.random.default_rng(0); x = rng.normal(size=4000)
    t = rng.binomial(1, 1/(1+np.exp(-2.5*x))); y = 1+2*t+1.5*x+rng.normal(size=4000)
    bb = Blackboard()
    EconometricianAgent(StubLLM(), Executor()).run(
        ToolContext(blackboard=bb, task_id="rep", depends_on=[],
                    params={"data": pd.DataFrame({"y": y, "t": t, "x": x}),
                            "profile": {"has_controls": True}}))
    rep = bb.value("rep")
    check("agent repairs overlap by trimming", rep.get("repaired_by") is not None)
    check("agent recovers an estimate after repair", (rep.get("estimate") or {}).get("point") is not None)

    print("code-red: real visualizations")
    import tempfile, os
    from dsagent.execution.viz import render_standard_pack
    figs = render_standard_pack(tempfile.mkdtemp())
    check("renders 4 figures", len(figs) == 4)
    check("figures are non-empty PNGs",
          all(os.path.getsize(p) > 5000 for p in figs.values()))

    print("scour adoptions: E-value, estimand, data-quality, SKILL.md standard")
    from dsagent.execution.estimand import estimand_for, data_quality_report
    from dsagent.skills.base import parse_skill_md
    ev = est.e_value(2.0, 1.9, 2.1, dg.make_observational(2.0)["y"].std())
    evn = est.e_value(0.0, -0.1, 0.1, 1.0)
    check("E-value: strong effect not fragile", ev["e_value_ci"] > 1.25 and not ev["fragile"])
    check("E-value: null effect is fragile (~1)", evn["e_value_ci"] == 1.0 and evn["fragile"])
    check("estimand names the summary measure",
          "Double ML" in estimand_for("observational_dml").summary_measure)
    check("estimand distinguishes IV (compliers)",
          estimand_for("iv_2sls").population == "compliers")
    dq_clean = data_quality_report(dg.make_observational(2.0), target="t")
    dq_leak = data_quality_report(dg.make_leaky(), target="t")
    check("data-quality: clean data ok", dq_clean["ok"])
    check("data-quality: leakage smell flagged", len(dq_leak["leakage_smell"]) >= 1)
    md = reg.get("observational_dml").to_skill_md()
    check("SKILL.md has frontmatter name", parse_skill_md(md).get("name") == "observational_dml")
    check("progressive-disclosure scan is metadata-only",
          set(reg.scan()[0].keys()) == {"name", "description"})

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
