"""
engagement_pilot.py — dsagent template for user-engagement causal analysis.

Three analyses on a user-week panel:
  1. Adoption -> retention causal effect  (design selected from data structure,
     precondition gate run BEFORE estimating, E-value + refutation battery).
  2. Cohort retention curve (weeks-since-adoption).
  3. Funnel stage-to-stage conversion rates.

USAGE
  python engagement_pilot.py                         # synthetic demo data
  python engagement_pilot.py --csv engagement.csv    # your real file

EXPECTED CSV COLUMNS (one row per user-week):
  user_id, week, adoption_week, adopted, active  + any controls in SCHEMA["controls"]
  adoption_week == 0 means never-treated.

SQL SKETCH (for reference — used by the production path below):

  -- fact_user_week: one row per (user_id, week)
  SELECT
      u.user_id,
      w.week_number                            AS week,
      COALESCE(c.adoption_week, 0)             AS adoption_week,
      CASE WHEN c.adoption_week IS NOT NULL THEN 1 ELSE 0 END AS adopted,
      CASE WHEN s.session_date IS NOT NULL THEN 1 ELSE 0 END  AS active,
      -- controls
      u.account_age_weeks,
      u.plan_tier,
      u.country_code
  FROM dim_users          u
  CROSS JOIN dim_weeks    w
  LEFT JOIN cohort_dates  c  ON c.user_id = u.user_id
  LEFT JOIN sessions_fact s  ON s.user_id = u.user_id
                             AND DATE_TRUNC('week', s.session_date) = w.week_start
  WHERE w.week_number BETWEEN :start_week AND :end_week
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from dsagent.execution import estimators as est
from dsagent.execution.executor import Executor, profile_data, select_design
from dsagent.execution.estimand import data_quality_report, estimand_for
from dsagent.skills.library import build_default_registry

# ---------------------------------------------------------------------------
# EDIT THESE to match your warehouse schema
# ---------------------------------------------------------------------------

SCHEMA = {
    "unit":     "user_id",       # user identifier
    "period":   "week",          # time dimension (integer week number)
    "cohort":   "adoption_week", # week of adoption; 0 = never-treated
    "treatment":"adopted",       # binary treatment indicator
    "outcome":  "active",        # binary or continuous engagement metric
    "controls": [                # pre-treatment confounders (all numeric or already encoded)
        "account_age_weeks",
        "plan_tier",
    ],
}

FUNNEL_STAGES = [
    "signed_up",
    "onboarded",
    "activated",     # first meaningful action
    "retained_w1",   # active in week 1
    "retained_w4",   # active in week 4
]

# ---------------------------------------------------------------------------
# Synthetic demo data (replace with pd.read_csv when you have real data)
# ---------------------------------------------------------------------------

def _make_demo_panel(n_users=200, n_weeks=10, seed=42) -> pd.DataFrame:
    """Staggered adoption panel with a known retention effect of ~0.12."""
    rng = np.random.default_rng(seed)
    users = np.arange(n_users)
    # 30% never-treated, rest adopt in weeks 3-8
    cohort = np.where(rng.random(n_users) < 0.3, 0,
                      rng.integers(3, 9, size=n_users))
    account_age = rng.integers(1, 52, size=n_users)
    plan_tier = rng.integers(0, 3, size=n_users)

    rows = []
    for w in range(1, n_weeks + 1):
        adopted = (cohort > 0) & (cohort <= w)
        base_p = 0.3 + 0.005 * account_age + 0.04 * plan_tier
        treat_lift = 0.12 * adopted.astype(float)
        noise = rng.normal(0, 0.05, size=n_users)
        active = (rng.random(n_users) < np.clip(base_p + treat_lift + noise, 0, 1)).astype(int)
        rows.append(pd.DataFrame({
            SCHEMA["unit"]:     users,
            SCHEMA["period"]:   w,
            SCHEMA["cohort"]:   cohort,
            SCHEMA["treatment"]: adopted.astype(int),
            SCHEMA["outcome"]:  active,
            "account_age_weeks": account_age,
            "plan_tier":         plan_tier,
        }))
    return pd.concat(rows, ignore_index=True)


def _make_demo_funnel(n_users=2000, seed=42) -> pd.DataFrame:
    """One row per user; columns are the FUNNEL_STAGES booleans."""
    rng = np.random.default_rng(seed)
    rates = [1.0, 0.72, 0.55, 0.41, 0.28]
    df = pd.DataFrame({SCHEMA["unit"]: np.arange(n_users)})
    prev = np.ones(n_users, dtype=bool)
    for stage, rate in zip(FUNNEL_STAGES, rates):
        curr = prev & (rng.random(n_users) < rate)
        df[stage] = curr.astype(int)
        prev = curr
    return df


# ---------------------------------------------------------------------------
# Analysis 1 — adoption -> retention causal effect
# ---------------------------------------------------------------------------

def analysis_adoption_retention(panel: pd.DataFrame) -> None:
    """Causal effect of feature adoption on retention (active next week).

    Steps:
      1. Profile the data structure (staggered? panel? controls?).
      2. Retrieve the right skill from the registry (gate-aware).
      3. Run the precondition gate BEFORE fitting — withholds if assumption fails.
      4. Fit and report with E-value, robustness value, and refutation battery.
    """
    print("\n" + "=" * 65)
    print("ANALYSIS 1 — Adoption → Retention (causal)")
    print("=" * 65)

    # --- remap to canonical estimator column names ---
    s = SCHEMA
    rename = {
        s["unit"]:      "unit",
        s["period"]:    "period",
        s["cohort"]:    "cohort",
        s["treatment"]: "t",
        s["outcome"]:   "y",
    }
    df = panel.rename(columns=rename)
    # keep controls that exist in the frame
    ctrl_cols = [c for c in s["controls"] if c in df.columns]

    n_users = df["unit"].nunique()
    n_obs = len(df)
    print(f"\nData: {n_obs:,} user-weeks, {n_users:,} users, "
          f"{df['period'].nunique()} weeks")

    # --- data quality gate ---
    dq = data_quality_report(df, target="t")
    print(f"Data quality: {'OK' if dq['ok'] else ', '.join(dq['issues'])}")
    if dq["leakage_smell"]:
        print(f"  WARNING: leakage suspects: {[x['column'] for x in dq['leakage_smell']]}")

    # --- structural profile -> design selection ---
    profile = profile_data(df)
    design = select_design(profile)
    print(f"\nData profile:  staggered={profile['is_staggered']}  "
          f"panel={profile['is_panel']}  has_controls={profile['has_controls']}")
    print(f"Selected design: {design['estimator']}")
    print(f"Identification:  {design['identification']}")

    # --- skill registry: retrieve the matching skill and run its gate ---
    registry = build_default_registry()
    skill = registry.best(profile)
    estimand = estimand_for(
        skill.id if skill else "observational_dml",
        outcome=s["outcome"],
        treatment="feature adoption",
    )
    print(f"\nEstimand: {estimand.one_line()}")

    if skill:
        gate = skill.check_gate(df)
        status = "PASS" if gate.passed else "BLOCK"
        print(f"Gate '{gate.name}': {status} — {gate.detail}")
        if not gate.passed:
            print("  Estimate withheld: assumption not supported. "
                  "Consider trimming overlap or revisiting the design.")
            return
    else:
        print("Gate: no skill matched; proceeding with selected design (no gate).")

    # --- fit ---
    ex = Executor()
    estimator = design["estimator"]
    cfg = {}
    if estimator in ("callaway_santanna",):
        cfg = {}   # uses canonical column names: unit, period, cohort, y
    elif estimator == "did_2x2":
        cfg = {"treat": "treat", "post": "post", "outcome": "y"}
    elif estimator == "double_ml" and ctrl_cols:
        cfg = {"treat": "t", "outcome": "y", "controls": tuple(ctrl_cols)}
    elif estimator == "diff_in_means":
        cfg = {"treat": "t", "outcome": "y"}

    result = ex.fit(estimator, df, **cfg)

    print(f"\nEstimate ({result.method}): {result.point:+.4f}  "
          f"(95% CI [{result.ci_low:+.4f}, {result.ci_high:+.4f}],  "
          f"p={result.pvalue:.4f},  n={result.n})")

    # --- rigor layer ---
    sd_y = float(df["y"].std())
    ev = est.e_value(result.point, result.ci_low, result.ci_high, sd_y)
    rv = est.robustness_value(result)
    print(f"\nRigor:")
    print(f"  E-value (point/CI): {ev['e_value_point']} / {ev['e_value_ci']} "
          f"{'(fragile)' if ev['fragile'] else '(robust)'}")
    print(f"  Robustness value:   {rv['robustness_value']} — {rv['interpretation']}")

    # Refutation battery uses a fast OLS proxy (regression_adjust) to keep the
    # demo snappy. The main estimate above uses the rigorous selected estimator.
    _ra_ctrl = tuple(ctrl_cols) if ctrl_cols else ("t",)
    def _fast_fit(d):
        return est.regression_adjust(d, treat="t", outcome="y",
                                     controls=_ra_ctrl).point

    ref = est.refutation_battery(_fast_fit, df, treat="t", n_perm=50)
    print(f"  Refutation battery: perm p={ref['permutation_p']}, "
          f"placebo mean={ref['placebo_mean_effect']:+.4f}, "
          f"survived_all={ref['survived_all']}")

    return result


# ---------------------------------------------------------------------------
# Analysis 2 — cohort retention curve (weeks-since-adoption)
# ---------------------------------------------------------------------------

def analysis_cohort_retention(panel: pd.DataFrame) -> None:
    """Average retention rate by weeks-since-adoption, for each adoption cohort."""
    print("\n" + "=" * 65)
    print("ANALYSIS 2 — Cohort Retention Curve")
    print("=" * 65)

    s = SCHEMA
    df = panel[panel[s["cohort"]] > 0].copy()   # only ever-treated users
    df["weeks_since"] = df[s["period"]] - df[s["cohort"]]
    df = df[df["weeks_since"] >= 0]

    curve = (df.groupby("weeks_since")[s["outcome"]]
               .agg(["mean", "count"])
               .rename(columns={"mean": "retention_rate", "count": "n_obs"})
               .reset_index())

    print(f"\n{'Week':>5}  {'Retention':>10}  {'N':>6}")
    print("-" * 27)
    for _, row in curve.iterrows():
        bar = "#" * int(row["retention_rate"] * 30)
        print(f"{int(row['weeks_since']):>5}  {row['retention_rate']:>9.1%}  "
              f"{int(row['n_obs']):>6}  {bar}")

    print(f"\nWeek-0 (day-of-adoption) retention: "
          f"{curve.loc[curve['weeks_since']==0,'retention_rate'].values[0]:.1%}"
          if len(curve) else "")
    return curve


# ---------------------------------------------------------------------------
# Analysis 3 — funnel stage-to-stage conversion
# ---------------------------------------------------------------------------

def analysis_funnel(funnel: pd.DataFrame) -> None:
    """Stage-to-stage conversion rates across the engagement funnel."""
    print("\n" + "=" * 65)
    print("ANALYSIS 3 — Funnel Stage-to-Stage Conversion")
    print("=" * 65)

    n = len(funnel)
    prev_n = n
    print(f"\n{'Stage':<18}  {'Users':>7}  {'Conv%':>7}  {'Drop':>7}")
    print("-" * 45)
    print(f"{'(total users)':<18}  {n:>7}")
    for stage in FUNNEL_STAGES:
        if stage not in funnel.columns:
            print(f"  {stage}: column not found in data, skipping")
            continue
        stage_n = int(funnel[stage].sum())
        conv = stage_n / prev_n if prev_n else 0.0
        drop = 1 - conv
        bar = "#" * int(conv * 20)
        print(f"{stage:<18}  {stage_n:>7}  {conv:>6.1%}  {drop:>6.1%}  {bar}")
        prev_n = stage_n

    # BH FDR correction across pairwise chi-square tests on consecutive stages
    pvalues = []
    for i in range(len(FUNNEL_STAGES) - 1):
        s1, s2 = FUNNEL_STAGES[i], FUNNEL_STAGES[i + 1]
        if s1 not in funnel.columns or s2 not in funnel.columns:
            continue
        a = int(funnel[s1].sum())
        b = int(funnel[s2].sum())
        # one-sample z-test: is stage-to-stage rate significantly below 100%?
        p = float(1 - (b / a)) if a else 0.0
        n_obs = a
        se = np.sqrt(p * (1 - p) / n_obs) if n_obs and 0 < p < 1 else 1.0
        z = p / se if se > 0 else 0.0
        from scipy import stats
        pvalues.append(float(2 * stats.norm.sf(abs(z))))

    if pvalues:
        fdr = est.benjamini_hochberg(pvalues)
        print(f"\nFDR (BH, α=0.05): {fdr['n_rejected']}/{fdr['n_tests']} "
              f"stage drops are statistically significant")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="dsagent engagement pilot — adoption/retention/funnel analysis")
    ap.add_argument("--csv", default=None,
                    help="Path to user-week CSV (default: synthetic demo data)")
    ap.add_argument("--funnel-csv", default=None,
                    help="Separate funnel CSV (one row per user, stage columns)")
    args = ap.parse_args()

    if args.csv:
        panel = pd.read_csv(args.csv)
        print(f"Loaded {len(panel):,} rows from {args.csv}")
    else:
        print("No --csv provided; using synthetic demo panel (n=500 users, 12 weeks).")
        panel = _make_demo_panel()

    if args.funnel_csv:
        funnel = pd.read_csv(args.funnel_csv)
    else:
        funnel = _make_demo_funnel()

    analysis_adoption_retention(panel)
    analysis_cohort_retention(panel)
    analysis_funnel(funnel)

    print("\n" + "=" * 65)
    print("Done. To run on real data:")
    print("  python engagement_pilot.py --csv your_data.csv")
    print("=" * 65)

    # -----------------------------------------------------------------------
    # PRODUCTION PATH (commented out — swap in when ready)
    # -----------------------------------------------------------------------
    # from dsagent import Catalog
    # from dsagent.pipeline import run_adaptive
    # from dsagent.llm import make_client
    # from dsagent.data import WarehouseDataSource
    # import snowflake.connector            # or google.cloud.bigquery, psycopg2, duckdb ...
    #
    # conn = snowflake.connector.connect(
    #     account=os.environ["SNOWFLAKE_ACCOUNT"],
    #     user=os.environ["SNOWFLAKE_USER"],
    #     password=os.environ["SNOWFLAKE_PASSWORD"],
    #     warehouse="COMPUTE_WH",
    #     database="PROD",
    #     schema="ANALYTICS",
    # )
    # cat = Catalog()
    # cat.ingest_ddl(open("schema.sql").read())        # point at your DDL
    #
    # llm = make_client("openai")                       # or "gemini" | "stub"; reads *_API_KEY
    # result = run_adaptive(
    #     goal="Did feature adoption causally increase weekly retention?",
    #     catalog=cat,
    #     llm=llm,
    #     data_source=WarehouseDataSource(connection=conn),
    # )
    # print(result["summary"])


if __name__ == "__main__":
    main()
