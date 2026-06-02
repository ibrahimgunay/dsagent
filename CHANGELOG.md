# Changelog

## v3.5.0 — agentic NL2SQL for ANY unseen schema (RAG linking + validate/repair loop)

Turns a natural-language question + an unfamiliar database into trusted SQL with
minimal instruction. The LLM does the reasoning; deterministic gates make it
impossible to ship a hallucinated-column or double-counting query.

### RAG schema-linking (dsagent/sql/schema_linker.py)
- `link_schema(question, catalog)` retrieves only the relevant tables/columns
  from an arbitrarily large schema (the fix for Spider 2.0's #1 error: wrong
  schema linking on 700-3000-column DBs). Dependency-free scorer; embedding seam.
- Lets the agent work on an unseen database by discovering the relevant tables
  itself instead of being told them.

### Validate-and-repair guardrails (dsagent/sql/validate.py)
- column-existence gate: every `alias.column` must resolve to a real catalog
  column -> kills hallucinated columns; self-repair snaps a near-miss to the
  closest real column (e.g. a.mrrr -> a.mrr).
- fan-out gate (reused) + join-validity gate (path must exist in the join graph).

### Agentic NL2SQL loop (dsagent/sql/nl2sql.py)
- LINK -> PLAN joins -> DRAFT (any LLMClient) -> VALIDATE -> REPAIR -> SELECT the
  candidate that passes the most gates. Provider-agnostic; testable offline via
  `drafts=`. Now powers SqlAuthorAgent in the pipeline (backward-compatible keys).

### Benchmark + tests
- schema_benchmark.py [7]: schema-linking recall@3 = 3/3, selects the valid
  candidate over hallucinated + fan-out drafts, repairs hallucinated columns,
  flags fan-out. Spider/BIRD loader unchanged ([8], --spider).
- Tests: 131 -> 139 (new test_sql_agent: 9).

### Honest scope
- This is the architecture that GENERALIZES to unseen schemas and GUARANTEES
  safety (no hallucinated/double-counting query ships). It does NOT make full
  NL2SQL "flawless" — execution accuracy on Spider 2.0 (frontier ~20%) depends on
  the live LLM + warehouse and is measured in your environment. The guardrails
  hold regardless of model quality.

# Changelog

## v3.4.0 — schema-complexity benchmark + verified SQL fan-out gate

Validates the DATA-ENGINEERING side the way v3.x validated the estimators.

### Verified SQL fan-out gate (dsagent/sql/gates.py)
- The verified-skill pattern (precondition + BLOCKING check + known-truth self-test)
  applied to SQL: `fanout_gate` blocks a query that aggregates a measure across a
  one-to-many/many-to-one join without pre-aggregating to the grain (the silent
  double-count bug). Self-test: blocks the hazardous query, passes the safe one.

### Foundation fix (graph.py)
- `many_to_one` joins are now flagged as fan-out risks too (summing the one-side
  measure across the many-side double-counts). Previously only `one_to_many` was
  flagged — surfaced by the new benchmark.

### schema_benchmark.py (new harness)
- Grades the data layer on hard schemas with known ground truth: nested flattening
  (Snowflake VARIANT/OBJECT, BigQuery STRUCT/ARRAY incl. ARRAY<STRUCT>), join-graph
  / FK recovery across 3 databases, fan-out detection, spaghetti anti-patterns,
  scale stress (~600 columns in milliseconds), and the fan-out gate.
- Ships Spider 2.0 / BIRD loaders: `--spider tables.json` ingests a REAL enterprise
  schema (700-3000 columns, nested) and scores join-graph recovery. Runs on your
  machine where there's network; offline fixtures self-validate everywhere.
- Honest scope: does NOT attempt full NL2SQL (frontier models score ~20% on Spider
  2.0); measures foundation understanding. NL2SQL upgrade (RAG schema-linking +
  candidate generation) remains the next build.

### Totals
- Tests: 129 -> 131 (fan-out gate self-test in test_core).

# Changelog

## v3.3.0 — ecosystem adoptions (Agent Skills standard, clinical-grade rigor, data quality)

Folded in the best of the open-source scour (the open Agent Skills ecosystem,
GPTomics/bioSkills clinical statistics, the csv-summarizer auto-EDA pattern)
without diluting our verification moat.

### Open SKILL.md standard + progressive disclosure
- Verified skills now emit the standard SKILL.md format (YAML frontmatter +
  body) via `Skill.to_skill_md()` and `SkillRegistry.write_skill_files()`, so
  they are portable to agentic coding tools and the open Agent Skills ecosystem.
- `SkillRegistry.scan()` exposes cheap metadata-only entries (name+description)
  for progressive disclosure; `load()` fetches the full skill only when selected.
- Our differentiator is preserved: each emitted skill still carries a BLOCKING
  gate and a known-truth self-test — verified, not just instructions.

### Clinical-grade sensitivity + estimand (from bioSkills / ICH E9(R1))
- `estimators.e_value` (VanderWeele-Ding): minimum confounder strength (risk-ratio
  scale) to explain away the point estimate and the CI bound nearest the null.
  Reported alongside the Cinelli-Hazlett robustness value (strong effect E~3.4;
  null effect E=1.0). New critic gate `e_value_reported`.
- `estimand.Estimand`: ICH E9(R1) framing (population, treatment, endpoint,
  intercurrent-event strategy, summary measure). Makes "estimand-first" a
  concrete object the trust report carries; tailored per design (e.g. IV ->
  local ATE for compliers).

### Data-quality / auto-EDA (csv-summarizer pattern; closes the old "data" gap)
- `estimand.data_quality_report`: missingness, constant & high-cardinality
  columns, duplicate rows, class balance, and near-perfect-predictor leakage
  smell. New critic gate `data_quality_checked`.

### Gates + totals
- Critic now runs 8 blocking gates (added e_value_reported, data_quality_checked).
- Tests: 121 -> 129. SKILL.md files emitted under outputs/skills/.

### Deliberately NOT adopted
- Bio/chem-specific scientific skills (RNA-seq, docking, DICOM) — out of domain.
- Heavy framework migrations (LangGraph/AutoGen) — our adaptive PlanGraph already
  covers stateful graph orchestration; no rewrite warranted.

# Changelog

## v3.2.0 — state-of-the-art ADAPTIVE PLANNER

Replaced the fixed DAG with a planner that reflects, branches, and replans.
New package `planning/`.

### Reflection loop (self-correcting plans)
- The planner proposes an initial plan, CRITIQUES it against staff-level rules
  (human approval before modeling? a critic before delivery? a final sign-off?
  dangling deps? within budget? known tools?), REVISES, and loops until clean.
  The reflection transcript is exposed (draft -> issues -> accepted).

### Multi-hypothesis branching (data -> design at runtime)
- The plan carries mutually-exclusive candidate designs (DiD / staggered-CS /
  DML / IV). The committed design is chosen AT RUNTIME from the profiled data:
  same goal, four data shapes -> four designs, each recovering truth
  (observational->DML 2.01, staggered->CS 0.86, IV->2SLS 1.50, panel->DiD 1.49).

### Event-driven replanning
- A `PlanGraph` is mutable mid-run: a gate failure inserts a repair node, thin
  budget prunes optional branches, a discovery commits a branch. The
  `AdaptiveOrchestrator` re-derives the ready set after every mutation.

### Plumbing
- `pipeline.run_adaptive(...)` and `python -m dsagent adaptive`.
- Fixed `SyntheticDataSource` to route staggered/panel/iv scenarios (and carry
  per-scenario truth) — previously it silently fell back to observational.

### Totals
- Tests: 100 -> 121 (new `test_planner`: 21). Five suites.

### Next backlog
- LLM-proposed plans (planner uses the live model to draft, rules still gate).
- Cost/value learning from past runs; plan caching by goal+profile.
- Build evals + adaptive runs on the CEO's real data.

# Changelog

## v3.1.0 — CODE RED: statistical depth, real graphs, self-repair

Advisory-board review found three things "not working well" and fixed them.

### Statistics fixed (Ivy econ/stats + FAANG principal DS)
- Pre-trends is now a real JOINT TEST: per-cell placebo effect + SE, combined to
  a Bonferroni joint p-value (clean p=1.0; violation p=0.0). No more magnitude
  threshold.
- Double ML upgraded to 5-fold REPEATED cross-fitting with median aggregation
  (Chernozhukov et al. recommendation), tighter honest SE.
- Refutation placebo now uses a PERMUTATION NULL with a p-value (real effect
  p<0.05; A/A null p=0.87 -> correctly not significant), replacing arbitrary bands.

### Real visualizations (the dashboard the CEO flagged was fake)
- `execution/viz.py`: publication-quality PNGs — forest plot (estimates vs truth
  with CIs), event-study (flat pre-trends -> post ramp), propensity-overlap
  histogram, calibration curve. The dashboard agent renders them when a
  figures_dir is provided; `run_analysis(..., figures_dir=...)` emits the pack.

### Agentic orchestration: self-repair (AI-lab eng)
- On an overlap-gate failure the econometrician now TRIMS to common support and
  retries before withholding (kept 2998/4000 rows, recovered the estimate),
  recording `repaired_by`. The agent recovers, not just blocks.

### Totals
- Tests: 91 -> 100. New rigor + repair + viz tests in test_skills.

### Next backlog
- Anderson-Rubin weak-IV-robust CIs; honest pre-trends (Rambachan-Roth) bounds.
- Wire BH/FDR into the multi-question battery in the pipeline.
- Build the eval + figures from the CEO's real data.

# Changelog

## v3.0.0 — Robustness & Rigor (state-of-the-art causal battery)

Theme: competitors can write an analysis; they can't make it *survive* an
automated robustness battery. This sprint builds that battery.

### IV / 2SLS with weak-instrument diagnostics  (WS-1)
- `estimators.iv_2sls`: two-stage least squares + first-stage F. Recovers the
  effect (1.50) where OLS is confounded (2.22). Skill `iv_2sls` carries a
  Staiger-Stock gate (F>=10) that BLOCKS weak instruments (F=2.1 -> withheld).

### Sensitivity analysis, enforced  (WS-2)
- `estimators.robustness_value` (Cinelli-Hazlett): % of residual variance a
  confounder must explain in both treatment and outcome to nullify the effect.
- Critic gate `sensitivity_analysis_present`: no causal estimate ships without one.

### Automated refutation battery  (WS-3)
- `estimators.refutation_battery`: placebo treatment (effect must collapse),
  random common cause (estimate must be stable), subset stability. Critic gate
  `survived_refutations` blocks an estimate that fails any refuter.

### Calibrated ML + FDR  (WS-4)
- `ml.conformal_classify`: split-conformal prediction sets with a
  distribution-free coverage guarantee (target 0.90, empirical 0.915).
- `estimators.benjamini_hochberg`: FDR control across the question battery.

### Retrieval eval  (WS-5)
- `eval.retrieval_eval`: precision@1 of skill selection across data shapes = 1.0.

### Eval + gates
- Scenarios 6 -> 7 (added strong-IV recovery). Critic now runs 6 blocking gates;
  trust report ships the robustness value + refutation results.

### Totals
- Tests: 82 -> 91. Eval scenarios: 7 + within-study + leakage + retrieval.

### Next backlog
- Anderson-Rubin weak-IV-robust CIs; honest pre-trends (Rambachan-Roth).
- Sensitivity/refutations for panel (DiD/CS) designs, not just cross-section.
- Live-provider adversarial eval; build the eval from the CEO's real data.

# Changelog

## v2.0.0 — verified skills, gate-as-you-go, adversarial + within-study evals, glass-box trust

The meta-prompting sprint: the moat is *enforced, demonstrable correctness under
autonomy*, not raw knowledge.

### Verified Skills (the moat)  — `skills/`
- A Skill bundles preconditions + estimator + a BLOCKING gate + a known-truth
  self-test. `SkillRegistry.retrieve(profile)` ranks by data-shape match;
  `verify_all()` makes every skill prove itself. 4 reference skills (RCT, 2x2
  DiD, staggered CS, observational DML) all self-verify on planted truth.

### Gate-as-you-go autonomy
- The econometrician retrieves a skill and runs its gate BEFORE fitting,
  withholding the estimate if the assumption fails (overlap, pre-trends).
  Gates moved from end-of-pipeline to inline, so the autonomous agent cannot
  advance through a broken assumption. Overlap gate retuned (trim-and-proceed
  <5% out-of-support; block above) so it never blocks a correct analysis.

### Adversarial + within-study evals  — `eval/`
- Planted-trap scenarios: Simpson's reversal (naive gets the WRONG SIGN; DML
  recovers), target-leakage (flagged at AUC 1.0).
- Within-study proof: recover a randomized experiment's ATE (2.0) from its
  observational slice (DML 2.03; naive 3.30, biased). Activated via
  `run_all(include_experiments=True)`.
- Eval now 6/6 scenarios + within-study + leakage.

### Glass-box trust report  — `agents/trust_report.py`
- Every analysis ships calibrated confidence + assumptions relied on + gate
  trail + what-would-change-our-mind + lineage depth + caveats.

### Totals
- Tests: 63 -> 82 (new `test_skills`: 19). Eval scenarios: 5 -> 6 + within-study.

### Next backlog
- IV/2SLS skill (weak-instrument F, Anderson-Rubin) + a weak-IV trap scenario.
- Retrieval eval (precision@1 of skill selection across data shapes).
- Live-provider eval: run the adversarial suite through each LLM's design choice.
- Build the eval from the CEO's real data (observational-first; within-study if
  experiments become available).

# Changelog

## v1.2.0 — multi-provider LLM backends

- `llm/openai_client.py` (Chat Completions) and `llm/gemini_client.py`
  (generateContent), both stdlib-urllib, both with native JSON mode when the
  caller asks for JSON.
- `llm.make_client(provider, model)` factory: one string picks
  stub/openai/gemini. Agents never branch on provider.
- CLI: `python -m dsagent run --provider openai --model gpt-4o`.
- Live clients raise a clear error (not a crash) when their key is absent;
  offline `stub` remains the default.
- Tests: 58 -> 63 (provider-factory dispatch + unknown-provider rejection).
- Use `python -m dsagent eval` with each provider wired into the process eval to
  compare GPT vs Gemini vs others on THIS task, not generic benchmarks.

# Changelog

## v1.1.0 — "Continuous iteration" sprint

Run as the agent org (planner triage → sub-agent implements → critic gates with
tests). Each item shipped only after the critic verified it.

### BL-1 — Staggered-adoption Callaway–Sant'Anna  *(econometrician)*
- `estimators.callaway_santanna`: group-time ATT vs never-treated, group-size
  weighted, clustered bootstrap SE.
- `estimators.twfe_static`: naive single-dummy TWFE, kept to demonstrate bias.
- `datagen.make_staggered`: staggered panel with dynamic (exposure-growing)
  effects where TWFE is biased.
- `select_design` now detects staggered timing and routes to CS.
- **Critic verification:** on staggered data CS recovers true ATT (0.86 vs 0.84,
  CI covers); naive TWFE returns 0.59 — biased −0.25, CI misses.

### BL-2 — ML agent fits for real  *(ML engineer)*
- `execution/ml.fit_predictive`: calibrated gradient-boosted classifier with a
  leakage-safe CV strategy (GroupKFold by entity when an id is given), AUC /
  PR-AUC / Brier, and a single-feature target-leak scan.
- `MLEngineerAgent` now fits when data is present (degrades to plan-only without).
- **Critic verification:** fits in-pipeline — AUC 0.76, Brier 0.20, no leak flags.

### BL-3 — Parallel trends made testable + gated  *(econometrician + critic)*
- `estimators.pretrends_test`: event-study placebo on pre-treatment periods.
- Econometrician attaches it for DiD/CS designs.
- Critic gains a **blocking** `parallel_trends_supported` gate.
- **Critic verification:** 0.06 on clean data (pass); 1.94 on a planted
  violation (block).

### BL-4 — Eval scenario for staggered adoption  *(eval)*
- New `staggered_adoption` scenario; harness now handles dynamically-computed
  truth and a configurable naive baseline (TWFE here).
- **Result:** eval now 5/5, including "CS beats biased TWFE".

### Process note
A mid-sprint edit accidentally removed `cate_by_subgroup`; the test suite caught
it immediately and it was restored. The gates and tests are the safety net that
makes continuous iteration safe.

### Totals
- Tests: 49 → **58** (foundation 16, agents 13, execution+eval 29).
- Eval scenarios: 4 → **5**.

### Next backlog (not yet built)
- IV / 2SLS with weak-instrument diagnostics (first-stage F, Anderson–Rubin).
- Live LLM design-selection eval (swap StubLLM → OpenAIClient).
- Domain packs (finance / medical / legal threshold + compliance overrides).
- Warehouse connector integration test against DuckDB.
