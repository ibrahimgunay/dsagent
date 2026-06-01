# dsagent — Autonomous Data-Science Agent

Give it a data schema and a goal; it plans the analysis, writes and verifies the
SQL, fetches the data, fits the right causal/ML model, runs blocking validity
gates, and ships a dashboard + memo — orchestrating specialist agents in
parallel and sequence, and grading itself against known-truth benchmarks.

Built for commercial product data on Snowflake / BigQuery / Databricks;
adaptable to finance, medical, and legal via domain packs.

## Quickstart

```bash
pip install -e .               # core: networkx numpy scipy scikit-learn pandas
python -m dsagent test         # 129 tests — expect all green
python -m dsagent run          # end-to-end pipeline -> a real, gated estimate
python -m dsagent adaptive     # reflection-planned + runtime design selection
python -m dsagent eval         # gold-standard + adversarial scorecard
python -m dsagent demo         # foundation walkthrough (nested schema, spaghetti SQL)
```

One-call API:

```python
from dsagent import Catalog
from dsagent.pipeline import run_analysis
from dsagent.llm import make_client           # "anthropic" | "openai" | "gemini" | "stub"
from dsagent.data import WarehouseDataSource

cat = Catalog()
cat.ingest_ddl(open("schema.sql").read())

result = run_analysis(
    "Did feature adoption causally increase weekly retention?",
    cat,
    llm=make_client("anthropic"),             # set ANTHROPIC_API_KEY
    data_source=WarehouseDataSource(conn),    # any PEP-249 connection
)
print(result.summary())
```

Adaptive pipeline (reflection + event-driven replanning):

```python
from dsagent.pipeline import run_adaptive

result = run_adaptive(
    "Did feature adoption causally increase weekly retention?",
    cat, llm=make_client("anthropic"),
    data_source=WarehouseDataSource(conn),
)
```

See `engagement_pilot.py` for a ready-to-edit template covering adoption →
retention causal effect, cohort retention curves, and funnel conversion.

## Architecture

```
  Goal ─▶ PlannerAgent(LLM) ─▶ task DAG ─▶ Orchestrator
                                              │ parallel batches · HITL gates · budget
                                              ▼
  FOUNDATION (deterministic)   DATA            MODELING (LLM picks, code fits)   VALIDATION/DELIVERY
  profiler                     data_executor   econometrician ─┐                 critic (blocking gates)
  semantic_modeler             (runs SQL,       ml_engineer     │ fit via         dashboard_builder
  join_analyzer                 returns frame)  causal_ml       │ Executor        memo_writer
  sql_author (LLM+verify)                       labeler        ─┘
                        └──── all communicate via the Blackboard (artifacts + lineage) ────┘
```

| Layer | Package | What it does |
|---|---|---|
| Foundation | `catalog`, `dialects`, `graph`, `ontology`, `profiling`, `sql/` | Multi-DB schema ingestion, nested-type flattening, join graph + fan-out detection, spaghetti-SQL analysis, semantic layer |
| LLM | `llm/` | Provider-agnostic client; `AnthropicClient` / `OpenAIClient` / `GeminiClient` + `StubLLM` offline |
| Runtime | `runtime/` | Blackboard (comms + lineage), tool registry, budget governor, orchestrator (parallel/sequential, HITL) |
| Agents | `agents/` | Planner + 11 specialist sub-agents as typed tools |
| Execution | `execution/` | Real estimators (OLS, DiD, Callaway–Sant'Anna, Double ML, IV/2SLS, overlap, CATE) + design selector |
| Skills | `skills/` | Verified skill library: preconditions + estimator + blocking gate + known-truth self-test |
| Data | `data/` | Warehouse + synthetic data sources; frame store |
| Eval | `eval/` | Known-truth scorecard: recovery, CI coverage, bias discrimination, adversarial traps, within-study |
| Planning | `planning/` | Adaptive planner: reflection loop, multi-hypothesis branching, event-driven replanning |
| Entry | `pipeline.py`, `__main__.py` | One-call facade + CLI |

## Two swaps to go live (no agent code changes)

```python
llm         = make_client("anthropic")           # set ANTHROPIC_API_KEY
data_source = WarehouseDataSource(connection)    # any PEP-249 / DB-API conn
```

Everything else — planning, orchestration, estimators, gates, and the eval —
runs fully offline today.

## Real-world benchmark results

Each row below was run against a publicly available dataset and graded PASS if
dsagent's estimate landed within the published tolerance of the literature value.
**12 / 12 matched their published benchmark.**

| Dataset | Design | dsagent estimate | Published benchmark | Source |
|---|---|---|---|---|
| LaLonde NSW job training | RCT diff-in-means | ~$1,794 | ~$1,672–1,794 | Dehejia-Wahba 1999 |
| NHEFS smoking → weight gain | Observational (DML) | ~3.4 kg | ~3.4–3.5 kg | Hernan-Robins, *Causal Inference: What If* |
| Card proximity-to-college | IV / 2SLS | ~0.14 log-wage/yr | ~0.13 (IV vs OLS ~0.07) | Card 1995 |
| CigarettesSW demand | IV (tax instrument) | ~−1.08 | ~−1.0 to −1.1 | Stock-Watson |
| Beer tax → traffic fatalities | Panel FE (DiD) | ~−0.66 | ~−0.66 | Stock-Watson |
| Organ donation default | 2×2 DiD | ~−0.022 pp | ~−2 pp | Kessler-Roth 2014 |
| Castle doctrine → homicide | TWFE DiD | ~+0.08 log | ~+0.07–0.10 | Cheng-Hoekstra 2013 |
| Thornton HIV cash incentive | RCT diff-in-means | ~+0.45 pp | ~+45 pp uptake | Thornton 2008 |
| CA class size → test scores | Bivariate OLS | ~−2.28 pts | ~−2.28 | Stock-Watson |
| Journal price → subscriptions | Log-log OLS | ~−0.53 elasticity | ~−0.5 | Stock-Watson |
| Return to education (wage1) | OLS | ~+0.083 log-wage/yr | ~+8.3%/yr | Wooldridge |
| Cash transfers → political support | Local-linear RDD | ~+0.20 jump | ~+0.20–0.40 | Manacorda-Miguel-Vigorito 2011 |

Run the suite yourself (requires outbound network for `pd.read_csv`):

```bash
python benchmark_suite.py           # all 12 datasets
python benchmark_suite.py --only lalonde   # LaLonde only (bundled CSV, no network)
```

## Key design decisions

1. **Deterministic vs LLM split.** Schema parsing, estimators, and validity
   gates are deterministic. The LLM is used only for judgment: planning, SQL
   authoring, method selection, memo drafting.

2. **Estimand-first + blocking gates.** No causal number ships without an
   estimand, identifying assumptions, and a sensitivity analysis. The critic's
   gates are deterministic and OVERRIDE the LLM (`FAILED_GATES`). Current
   gates: identification-before-estimation, no-naive-TWFE, SQL fan-out resolved,
   overlap/positivity checked, parallel-trends supported.

3. **Agents communicate only via the Blackboard.** No direct agent-to-agent
   calls. Large dataframes move by handle through `DataStore`; the blackboard
   carries a reference + summary, giving free lineage (`bb.lineage(key)`).

4. **Verified Skills (the moat).** Each `Skill` = preconditions + estimator +
   a BLOCKING gate + a known-truth self-test. `SkillRegistry.retrieve(profile)`
   is method-selection; `verify_all()` proves every skill on synthetic fixtures.

## Tests

```
python -m dsagent test
```

129 tests across five modules — expect all green:

| Module | Tests | Covers |
|---|---|---|
| `test_core` | 16 | Schema ingestion, join graph, SQL analysis, ontology |
| `test_agent` | 18 | Planner DAG, orchestration, blocking gates, multi-LLM |
| `test_exec_eval` | 29 | Estimators, design selection, end-to-end pipeline |
| `test_skills` | 45 | Verified skills, adversarial traps, trust report, IV/refutations |
| `test_planner` | 21 | Reflection loop, multi-hypothesis branching, adaptive replanning |

## Design docs

- `AGENTIC_LAYER.md` — runtime: planner, orchestrator, blackboard, budget
- `EXECUTION_EVAL.md` — estimators, design selector, gold-standard eval
- `CHANGELOG.md` — sprint history and self-caught regressions
- `engagement_pilot.py` — ready-to-edit template for product engagement data
