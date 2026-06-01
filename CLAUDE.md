# CLAUDE.md — working context for this repo

This file is read automatically at the start of a Claude Code session. It is the
handoff from the chat session that built `dsagent`. Read it, then run the tests.

## What this is

`dsagent` is an autonomous data-science agent. Given a data schema and a goal,
it plans an analysis, writes and verifies SQL, fetches data, fits the right
causal/ML model, runs blocking validity gates, and ships a dashboard + memo —
orchestrating specialist sub-agents in parallel and sequence, and grading
itself against known-truth benchmarks.

It runs fully offline today and goes to production with two object swaps (below).

## Orient yourself first

```bash
pip install -e .          # core: networkx numpy scipy scikit-learn pandas
python -m dsagent test    # expect 129 passing (16 + 18 + 29 + 45 + 21)
python -m dsagent run     # end-to-end pipeline -> a real, gated estimate + trust report
python -m dsagent eval    # gold-standard + adversarial + within-study scorecard
python -m dsagent demo    # foundation walkthrough (nested schema, spaghetti SQL, joins)
```

If `python -m dsagent test` is green, the system is healthy. Always run it after
changes — the suite is the safety net that makes iteration safe (it already
caught one accidental deletion mid-build).

## Architecture (build order == dependency order)

```
Goal -> PlannerAgent(LLM) -> task DAG -> Orchestrator
                                          | parallel batches, HITL gates, budget
  FOUNDATION (deterministic)  DATA          MODELING (LLM picks, code fits)  VALIDATION/DELIVERY
  profiler                    data_executor econometrician                  critic (blocking gates)
  semantic_modeler            (runs SQL,    ml_engineer                     dashboard_builder
  join_analyzer                returns df)  causal_ml                       memo_writer
  sql_author (LLM+verify)                   labeler
                       \____ all communicate via the Blackboard (artifacts + lineage) ____/
```

Package map:
- `catalog.py`, `dialects.py`, `graph.py`, `ontology.py`, `profiling.py`, `sql/` — deterministic foundation: multi-DB schema ingestion, nested-type flattening, join graph + fan-out detection, spaghetti-SQL analysis, semantic layer.
- `llm/` — provider-agnostic client. `StubLLM` (offline fixtures, used in tests) and `AnthropicClient` (prod, stdlib urllib, reads `ANTHROPIC_API_KEY`).
- `runtime/` — `Blackboard` (comms + lineage), `ToolRegistry`, `Budget`, `Orchestrator` (topological batches, thread pool, human-approval, retries).
- `agents/` — `PlannerAgent` + sub-agents, each a typed `Tool`. `build_org(catalog, llm, executor)` is the composition root.
- `execution/` — real estimators (`estimators.py`: OLS w/ HC1 SE, DiD, Callaway–Sant'Anna, Double ML, pre-trends, overlap, CATE), `ml.py` (calibrated classifier, leakage-safe CV), `executor.py` (`Executor.fit` + `select_design`), `datagen.py` (known-truth generators).
- `data/` — `WarehouseDataSource` (prod, any DB-API conn) + `SyntheticDataSource` (offline) + `DataStore`.
- `eval/` — `harness.py` + `scenarios.py`: scores recovery, CI coverage, bias-discrimination, FDR/null, **adversarial traps** (Simpson's, leakage), **within-study** proof, plus a process eval.
- `skills/` — **Verified Skills (the moat).** Each `Skill` = preconditions + estimator + a BLOCKING gate + a known-truth self-test. `SkillRegistry.retrieve(profile)` is method-selection; `verify_all()` proves every skill. The econometrician runs a skill's gate BEFORE fitting (gate-as-you-go).
- `agents/trust_report.py` — glass-box report: calibrated confidence, assumptions, gate trail, what-would-change-our-mind, lineage depth.
- `pipeline.py` — `run_analysis(...)` one-call facade (returns estimate + deliverables incl. trust_report).
- `__main__.py` — the CLI.

## Key design decisions (do not regress these)

1. **Deterministic vs LLM split.** Schema parsing, join math, estimators, and
   the validity gates are deterministic and reproducible. The LLM is used only
   for judgment: planning, SQL authoring, method selection, reconciliation,
   memo/dashboard drafting. Keep numerical/gate logic out of the model.
2. **Estimand-first + blocking gates.** No causal number ships without an
   estimand, identifying assumptions, and a sensitivity analysis. The critic's
   gates are deterministic and OVERRIDE the LLM (`FAILED_GATES`). Current gates:
   identification-before-estimation, no-naive-TWFE, SQL fan-out resolved,
   overlap/positivity checked, parallel-trends supported. Negative cases are
   tested in `demo/test_agent.py` and `demo/test_exec_eval.py`.
3. **Agents communicate only via the Blackboard.** No direct agent-to-agent
   calls. They read upstream artifacts by task id and write one artifact back.
   Large dataframes move by handle through `DataStore`; the blackboard carries a
   reference + summary. This gives free lineage (`bb.lineage(key)`).
4. **Pluggable backends.** SQL parsing uses `sqlglot` if importable, else a
   stdlib parser. The LLM and data source are injected at the composition root.

## The two swaps to go live (no agent code changes)

```python
from dsagent.llm import make_client            # provider factory
from dsagent.data import WarehouseDataSource
from dsagent.pipeline import run_analysis

llm = make_client("anthropic")                 # or "openai" / "gemini" / "stub"
result = run_analysis(goal, catalog, llm=llm,
                      data_source=WarehouseDataSource(connection=conn))
```

Provider keys: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`
(or `GEMINI_API_KEY`). CLI: `python -m dsagent run --provider openai --model gpt-4o`.
All providers implement the same `LLMClient` interface (`llm/base.py`); the
agents never branch on provider. Live clients raise a clear error (not a crash)
when their key is absent, so offline `stub` stays the safe default.

## Things you can do here that the chat session could not

- `pip install sqlglot` to unlock dialect-perfect, column-level SQL lineage
  (the chat session stubbed it — `sql/lineage.py` auto-upgrades when present).
- Use the live `AnthropicClient` and a real warehouse (chat had no network).
- Real git: make the "continuous iteration" loop proper commits/branches.

## Backlog (next sprints, prioritized)

1. **IV / 2SLS** with weak-instrument diagnostics (first-stage F, Anderson–Rubin)
   in `estimators.py`; add a scenario to `eval/scenarios.py`.
2. **Live-LLM design-selection eval** — swap `StubLLM` -> `AnthropicClient` in
   `eval/harness.py` process_eval and measure whether the *real model* picks
   truth-recovering designs.
3. **Domain packs** — finance / medical / legal: compliance rules, method
   priors, and validity-threshold overrides as config + a retrieval corpus.
4. **Warehouse connector integration test** against DuckDB (in-process, no creds).
5. **Staggered SE** — replace the bootstrap in `callaway_santanna` with the
   analytic influence-function SE; add honest pre-trends (Rambachan–Roth).

## Conventions

- New estimator: add to `estimators.py` (return `EstimateResult`), register in
  `Executor.fit`, teach `select_design` when to choose it, add a known-truth
  scenario, and add a test asserting it recovers truth + that the wrong design
  is flagged. Then run `python -m dsagent test`.
- New sub-agent: subclass `Tool` (or `_LLMAgent`), emit exactly one artifact via
  `self._emit`, register in `build_org`, and add it to the planner's reachable
  tools.
- Every claim the system makes about an effect must trace to a gate. If you add
  a method, add the gate that keeps it honest.

## Companion docs
`README.md` (canonical), `AGENTIC_LAYER.md` (runtime), `EXECUTION_EVAL.md`
(estimators + eval), `CHANGELOG.md` (sprint history + self-caught regression).
The full system-architecture design doc lives in the chat that created this repo.
