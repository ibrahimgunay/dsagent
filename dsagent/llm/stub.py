"""Deterministic offline LLM stub.

Returns sensible, schema-valid structured outputs keyed by `intent`, so the
full planner -> orchestrator -> sub-agent loop runs and is testable with no
network. Each branch mirrors the JSON contract the corresponding agent expects
from a real model. Swap `StubLLM` for `OpenAIClient` to go live; the agents
do not change.
"""
from __future__ import annotations

import json
from .base import LLMClient, LLMResponse, Usage


class StubLLM(LLMClient):
    model = "stub-deterministic"

    def __init__(self) -> None:
        self.transcript: list[tuple[str, str]] = []

    def complete(self, system, prompt, *, max_tokens=1500, intent="") -> LLMResponse:
        self.transcript.append((intent, prompt[:120]))
        payload = _ROUTES.get(intent, _default)(prompt)
        text = json.dumps(payload)
        # rough token accounting so the budget governor has something to track
        approx = (len(system) + len(prompt) + len(text)) // 4
        return LLMResponse(text=text, usage=Usage(input_tokens=approx // 2,
                                                  output_tokens=approx // 2,
                                                  calls=1, usd=0.0))


def _default(_prompt: str) -> dict:
    return {"result": "ok", "note": "stub default"}


def _plan(prompt: str) -> dict:
    # A causal-question plan: discovery -> approval -> parallel modeling -> reconcile -> deliver
    return {"tasks": [
        {"id": "profile", "tool": "profiler", "phase": "P0", "depends_on": []},
        {"id": "semantic", "tool": "semantic_modeler", "phase": "P0", "depends_on": ["profile"]},
        {"id": "joins", "tool": "join_analyzer", "phase": "P0", "depends_on": ["semantic"]},
        {"id": "author_sql", "tool": "sql_author", "phase": "P1", "depends_on": ["joins"]},
        {"id": "fetch_data", "tool": "data_executor", "phase": "P1", "depends_on": ["author_sql"]},
        {"id": "plan_review", "tool": "noop", "phase": "P2", "depends_on": ["fetch_data"],
         "requires_human_approval": True},
        {"id": "econ", "tool": "econometrician", "phase": "P3",
         "depends_on": ["plan_review", "fetch_data"]},
        {"id": "ml", "tool": "ml_engineer", "phase": "P3",
         "depends_on": ["plan_review", "fetch_data"]},
        {"id": "causal", "tool": "causal_ml", "phase": "P3",
         "depends_on": ["plan_review", "fetch_data"]},
        {"id": "label", "tool": "labeler", "phase": "P3", "depends_on": ["plan_review"]},
        {"id": "critic", "tool": "critic", "phase": "P4",
         "depends_on": ["econ", "ml", "causal", "label"]},
        {"id": "dashboard", "tool": "dashboard_builder", "phase": "P5", "depends_on": ["critic"]},
        {"id": "memo", "tool": "memo_writer", "phase": "P5", "depends_on": ["critic"]},
        {"id": "trust", "tool": "trust_report", "phase": "P5", "depends_on": ["critic", "econ"]},
        {"id": "signoff", "tool": "noop", "phase": "P5", "depends_on": ["dashboard", "memo", "trust"],
         "requires_human_approval": True},
    ]}


def _classify(prompt: str) -> dict:
    # The profiler sends one column; echo a typed decision. The heuristic in
    # profiling.py already does the work, so here we just confirm + add a defn.
    name = _field(prompt, "column")
    low = name.lower()
    if any(k in low for k in ("email", "name", "address", "ip_", "phone")):
        return {"semantic_type": "pii", "sensitivity": "pii",
                "definition": f"Personal data: {name}"}
    if low.endswith(("_at", "_ts")) or "time" in low or "date" in low:
        return {"semantic_type": "event_time", "sensitivity": "internal",
                "definition": f"Event timestamp: {name}"}
    if low.endswith("_id") or low == "id":
        return {"semantic_type": "identifier", "sensitivity": "internal",
                "definition": f"Join key: {name}"}
    return {"semantic_type": "dimension", "sensitivity": "internal",
            "definition": f"Attribute: {name}"}


def _author_sql(prompt: str) -> dict:
    return {"sql": (
        "WITH treated AS (SELECT user_id, MIN(event_time) AS first_seen "
        "FROM ANALYTICS.EVENTS.PRODUCT_EVENTS WHERE event_name = 'feature_x' "
        "GROUP BY user_id), "
        "panel AS (SELECT u.user_id, f.feature_date, f.sessions_7d, "
        "CASE WHEN t.user_id IS NOT NULL THEN 1 ELSE 0 END AS treated "
        "FROM PROD.CORE.USERS u "
        "JOIN ANALYTICS.MART.USER_FEATURES f ON f.user_id = u.user_id "
        "LEFT JOIN treated t ON t.user_id = u.user_id) "
        "SELECT treated, feature_date, AVG(sessions_7d) AS retention "
        "FROM panel GROUP BY treated, feature_date"),
        "grain": "user_id x feature_date",
        "notes": "Pre-aggregated to user grain before join to avoid event fan-out."}


def _econ(prompt: str) -> dict:
    return {
        "estimand": "ATT of feature_X adoption on 7-day session retention",
        "design": "Staggered adoption (treatment timing varies by user)",
        "dag": {"treatment": "feature_X_adoption", "outcome": "sessions_7d",
                "confounders": ["plan_tier", "tenure", "country"],
                "instruments": []},
        "estimator": "Callaway-Sant'Anna staggered DiD",
        "rejected_estimators": [
            {"name": "Two-way fixed effects", "reason":
             "Biased under heterogeneous treatment timing (negative weights)."}],
        "identifying_assumptions": [
            "Parallel trends conditional on covariates",
            "No anticipation prior to adoption",
            "Stable composition of cohorts"],
        "testable_checks": ["Event-study pre-trend plot", "Placebo timing test"],
        "sensitivity_analysis": ["Honest pre-trends (Rambachan-Roth)",
                                 "Oster delta for selection on unobservables"],
        "clustered_se": "user_id",
    }


def _ml(prompt: str) -> dict:
    return {
        "task": "Predict 30-day churn to contextualize the causal effect",
        "model_space": ["LogisticRegression (baseline)", "LightGBM"],
        "cv_strategy": "Grouped time-series split by user_id (no entity leakage)",
        "leakage_checks": ["Drop post-treatment features", "Point-in-time feature join"],
        "metrics": ["AUC", "PR-AUC", "calibration (reliability curve)"],
        "uncertainty": "Conformal prediction intervals",
    }


def _causal(prompt: str) -> dict:
    return {
        "method": "Double ML (DML) with LightGBM nuisance + cross-fitting",
        "heterogeneity": "Causal forest for CATE by plan_tier and tenure",
        "overlap_check": "Propensity trimming to common support [0.05, 0.95]",
        "policy": "DR-learner -> targeting rule, value via doubly-robust OPE",
        "blocking_gate": "positivity/overlap must pass before any estimate is emitted",
    }


def _label(prompt: str) -> dict:
    return {
        "target_fields": ["support_tickets.body"],
        "approach": "Weak supervision + LLM-as-annotator with confidence routing",
        "extracted_features": ["intent", "sentiment", "product_area", "severity"],
        "quality_control": "200-row gold set; route confidence<0.7 to human review",
        "provenance": "Every label carries source + quality estimate; never treated as ground truth in causal models",
    }


def _critic(prompt: str) -> dict:
    return {
        "verdict": "pass_with_caveats",
        "reconciliation": "Causal-forest CATE sign agrees with DiD ATT; magnitudes within 1 SE.",
        "remaining_risks": ["Pre-trends marginally significant in 1 of 6 cohorts",
                            "Overlap thin for enterprise tier (n small)"],
        "required_caveats": ["Effect is ATT, not ATE", "Enterprise tier under-powered"],
    }


def _dashboard(prompt: str) -> dict:
    return {"title": "Feature X Impact",
            "tiles": [
                {"name": "Event-study (pre-trends + ATT)", "metric": "sessions_7d", "type": "line"},
                {"name": "CATE by plan tier", "metric": "att_by_tier", "type": "bar"},
                {"name": "Adoption funnel", "metric": "users_count", "type": "funnel"}],
            "backed_by": "governed semantic metrics only"}


def _memo(prompt: str) -> dict:
    return {"headline": "Feature X increased 7-day retention for adopters",
            "effect": "+X.X sessions/week (95% CI [a, b]), ATT via staggered DiD",
            "caveats": ["ATT not ATE", "Enterprise tier under-powered",
                        "Observational design; unconfoundedness assumed"],
            "what_would_change_our_mind": "Pre-trend violation or failed placebo test"}


_ROUTES = {
    "plan": _plan,
    "classify_column": _classify,
    "author_sql": _author_sql,
    "econometrics": _econ,
    "ml_plan": _ml,
    "causal_plan": _causal,
    "labeling_plan": _label,
    "critic_review": _critic,
    "dashboard_spec": _dashboard,
    "memo": _memo,
}


def _field(prompt: str, key: str) -> str:
    # tiny helper to pull "key: value" out of a prompt the stub was given
    for line in prompt.splitlines():
        if line.lower().strip().startswith(key.lower()):
            return line.split(":", 1)[-1].strip()
    return ""
