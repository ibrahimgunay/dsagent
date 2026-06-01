"""Estimand specification + data-quality profiling.

Two adoptions from the scour:

* `Estimand` formalizes the ICH E9(R1) framing (population, treatment, endpoint,
  intercurrent-event strategy, population-level summary) so "estimand-first" is a
  concrete object the trust report carries, not just a slogan.

* `data_quality_report` is the csv-summarizer / auto-EDA pattern: missingness,
  constant and high-cardinality columns, duplicate rows, class balance, and
  near-perfect predictors (leakage smell) — surfaced as a gateable object.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd


@dataclass
class Estimand:
    population: str = "all observed units"
    treatment: str = "feature adoption (vs none)"
    outcome: str = "retention"
    intercurrent_event_strategy: str = "treatment policy"
    summary_measure: str = "average treatment effect (difference in means)"

    def to_dict(self) -> dict:
        return asdict(self)

    def one_line(self) -> str:
        return (f"ATE-style: effect of [{self.treatment}] on [{self.outcome}] "
                f"in [{self.population}], intercurrent events handled by "
                f"[{self.intercurrent_event_strategy}], summarized as "
                f"[{self.summary_measure}].")


_SUMMARY_BY_SKILL = {
    "staggered_did_cs": "ATT, aggregated over group-time effects (Callaway-Sant'Anna)",
    "two_period_did": "ATT (2x2 difference-in-differences)",
    "observational_dml": "ATE under unconfoundedness (Double ML)",
    "iv_2sls": "Local ATE for compliers (2SLS)",
    "rct_contrast": "ATE (randomized difference in means)",
}


def estimand_for(skill_id: str, outcome: str = "retention",
                 treatment: str = "feature adoption (vs none)") -> Estimand:
    summary = _SUMMARY_BY_SKILL.get(skill_id, "average treatment effect")
    pop = "compliers" if skill_id == "iv_2sls" else (
        "treated units" if "did" in skill_id or skill_id == "staggered_did_cs"
        else "all observed units")
    return Estimand(population=pop, treatment=treatment, outcome=outcome,
                    summary_measure=summary)


def data_quality_report(df: pd.DataFrame, target: str | None = None) -> dict:
    n = len(df)
    miss = {c: round(float(df[c].isna().mean()), 4) for c in df.columns
            if df[c].isna().any()}
    constant = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    high_card = [c for c in df.columns
                 if df[c].dtype == object and n and df[c].nunique() / n > 0.9]
    dup_share = round(float(df.duplicated().mean()), 4) if n else 0.0
    balance = None
    if target and target in df.columns and df[target].nunique() <= 10:
        vc = df[target].value_counts(normalize=True)
        balance = {str(k): round(float(v), 4) for k, v in vc.items()}
    leak = []
    if target and target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
        for c in df.columns:
            if c != target and pd.api.types.is_numeric_dtype(df[c]):
                try:
                    r = abs(np.corrcoef(df[c].fillna(0), df[target])[0, 1])
                    if r > 0.98:
                        leak.append({"column": c, "abs_corr": round(float(r), 4)})
                except Exception:
                    pass
    issues = []
    if any(v > 0.2 for v in miss.values()):
        issues.append("columns with >20% missingness")
    if constant:
        issues.append("constant columns")
    if dup_share > 0.05:
        issues.append("duplicate rows >5%")
    if leak:
        issues.append("near-perfect predictor (possible leakage)")
    return {"n_rows": n, "n_cols": df.shape[1], "missingness": miss,
            "constant_columns": constant, "high_cardinality_columns": high_card,
            "duplicate_row_share": dup_share, "class_balance": balance,
            "leakage_smell": leak, "issues": issues, "ok": not issues}
