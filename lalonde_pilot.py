"""
LaLonde / NSW pilot for dsagent — a real-world causal-inference stress test.

The NSW job-training experiment is the canonical "challenging" causal dataset:
a randomized experiment gives the TRUE effect (~$1,800 on 1978 earnings), and
non-experimental comparison groups (CPS/PSID) famously produce wildly biased
naive estimates. Recovering the experimental answer from observational controls
is the gold-standard "within-study" test.

USAGE
  python lalonde_pilot.py                      # experimental benchmark + rigor
  python lalonde_pilot.py --controls cps.csv   # ALSO run the within-study challenge
      (download cps_mixtape.csv or psid_controls.csv from the causaldata package
       / Rdatasets; same columns: treat,age,educ,black,hisp,marr,nodegree,re74,re75,re78)

This uses dsagent's own estimators + rigor layer (no agent-code changes).
"""
import argparse
import pandas as pd
from dsagent.execution import estimators as est
from dsagent.execution.estimand import data_quality_report, estimand_for

CTRLS = ["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"]


def _prep(df):
    return df.rename(columns={"re78": "y", "treat": "t"})[["y", "t"] + CTRLS].copy()


def experimental_benchmark(path):
    raw = pd.read_csv(path)
    df = _prep(raw)
    print(f"\nExperimental NSW data: {len(df)} rows "
          f"({int(df.t.sum())} treated, {int((1-df.t).sum())} control)")
    dim = est.diff_in_means(df)
    ra = est.regression_adjust(df, controls=tuple(CTRLS))
    dml = est.double_ml(df, controls=tuple(CTRLS))
    print("\nEXPERIMENTAL BENCHMARK (randomized = ground truth)")
    print(f"  diff-in-means      : ${dim.point:,.0f}  (95% CI [${dim.ci_low:,.0f}, ${dim.ci_high:,.0f}])")
    print(f"  regression-adjusted: ${ra.point:,.0f}")
    print(f"  double ML (5-fold) : ${dml.point:,.0f}")
    print("  literature (Dehejia-Wahba): ~$1,672-$1,794")
    ref = est.refutation_battery(
        lambda d: est.regression_adjust(d, controls=tuple(CTRLS)).point, df, n_perm=300)
    print("\nRIGOR LAYER (on real data)")
    print(f"  refutation: perm p={ref['permutation_p']}, placebo mean=${ref['placebo_mean_effect']:,.0f}, "
          f"survived_all={ref['survived_all']}")
    ev = est.e_value(dml.point, dml.ci_low, dml.ci_high, df["y"].std())
    print(f"  E-value: point={ev['e_value_point']}, ci={ev['e_value_ci']}")
    dq = data_quality_report(df, target="t")
    print(f"  data quality: {'OK' if dq['ok'] else dq['issues']}")
    return raw, df


def within_study(exp_raw, controls_path):
    """The famous challenge: experimental TREATED + NON-experimental controls.
    Naive estimate is badly biased; covariate adjustment / DML should recover
    the experimental ~$1,800."""
    ctrl_raw = pd.read_csv(controls_path)
    treated = exp_raw[exp_raw.treat == 1]
    obs = _prep(pd.concat([treated, ctrl_raw], ignore_index=True))
    print(f"\nWITHIN-STUDY CHALLENGE: {int(obs.t.sum())} experimental treated "
          f"+ {int((1-obs.t).sum())} non-experimental controls")
    naive = est.diff_in_means(obs)
    ra = est.regression_adjust(obs, controls=tuple(CTRLS))
    dml = est.double_ml(obs, controls=tuple(CTRLS))
    print(f"  naive diff-in-means : ${naive.point:,.0f}   <- famously biased")
    print(f"  regression-adjusted : ${ra.point:,.0f}")
    print(f"  double ML (5-fold)  : ${dml.point:,.0f}   <- should pull toward the experimental ~$1,800")
    ov = est.propensity_overlap(obs, controls=[c for c in CTRLS if c in obs.columns])
    print(f"  overlap: share outside [.05,.95] = {ov['share_outside_[0.05,0.95]']} "
          f"(trim-and-proceed if >5%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="lalonde_nsw_exp.csv")
    ap.add_argument("--controls", default=None,
                    help="non-experimental controls CSV (cps_mixtape / psid)")
    args = ap.parse_args()
    raw, _ = experimental_benchmark(args.exp)
    if args.controls:
        within_study(raw, args.controls)
    else:
        print("\n(Run with --controls cps_mixtape.csv to add the within-study challenge.)")
