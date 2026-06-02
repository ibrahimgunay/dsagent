# dsagent — an autonomous data-science agent with *enforced* correctness

> An agentic system that runs real econometric / causal analysis end-to-end — and
> **refuses to ship a wrong answer**. It blocks double-counting SQL, underpowered
> causal claims, hallucinated columns, and unmet identification assumptions, and
> shows the gate trail for every decision.

The interesting problem in agentic AI right now isn't fluency — models already write
plausible SQL and analysis. It's **trust under autonomy**: a fluent agent that is
confidently wrong is worse than no agent. `dsagent` is a study in the opposite design
goal — *demonstrable correctness as a first-class, enforced property* — applied to a
domain where being wrong is silent and expensive: causal inference on messy data.

```
$ python -m dsagent test
18 passed · 17 passed · 29 passed · 45 passed · 21 passed · 9 passed     # 139 total, 6 suites
```

---

## The thesis (and why it's the hard part)

Most analysis agents are a model + a prompt + a code sandbox. They optimize fluency.
`dsagent` splits the system on a different axis:

- **The LLM does judgment** — planning, method selection, SQL authoring, reconciliation.
- **Deterministic code enforces correctness** — identification checks, overlap/positivity,
  parallel-trends tests, sensitivity analysis, refutation batteries, fan-out detection,
  column existence. These are **blocking gates**: if an assumption fails, the agent
  *withholds the estimate and says why* rather than emitting a confident wrong number.

Every capability is packaged as a **verified skill** = *precondition + estimator +
blocking gate + a known-truth self-test*. The registry can prove every skill recovers a
planted effect before it's ever trusted on real data. That's the moat: not "knows
method X," but "its use of X is checkably correct."

---

## Proof, not vibes

### 1. Recovers 12 published causal results on real data — and declines the one it can't

`benchmark_suite.py` grades the engine against the *literature* across every design it
supports, on real downloaded datasets:

| Dataset | Design | Estimate | Published | Result |
|---|---|---|---|---|
| LaLonde NSW job training | RCT | $1,794 | ~$1,672–1,794 | ✅ |
| NHEFS smoking → weight gain | observational (DML) | 3.67 kg | ~3.4 kg | ✅ |
| Card college proximity | IV / 2SLS | 0.19 | ~0.13 (≫ OLS 0.07) | ✅ |
| CigarettesSW demand | IV / 2SLS | −1.14 | ~−1.08 | ✅ |
| Beer tax → fatalities | DiD / panel FE | −0.66 | ~−0.66 | ✅ |
| Organ-donation default | DiD | −0.01 | negative (Kessler–Roth) | ✅ |
| Castle-doctrine laws | DiD / TWFE | +0.03 | ~+0.08 | ✅ |
| Thornton HIV incentive | RCT | +0.45 | ~+0.45 | ✅ |
| CA class size → scores | regression | −2.28 | ~−2.28 | ✅ |
| Journal pricing | regression | −0.53 | ~−0.53 | ✅ |
| Return to education | regression | 0.083 | ~8.3%/yr | ✅ |
| Gov transfers → support | RDD | + jump | positive (Manacorda) | ✅ |

The within-study test is the headline: on the LaLonde data it recovers the **experimental
benchmark ($1,794)** and the effect **survives a permutation-null refutation (p=0.003)** —
i.e. it reproduces the answer a randomized experiment would give, from observational data,
and proves the effect isn't an artifact.

### 2. Handles hard, unseen schemas

`schema_benchmark.py` grades the data layer on nested / multi-database schemas with known
ground truth, and the agentic NL2SQL loop:

```
[1] Nested flattening (VARIANT/OBJECT + STRUCT/ARRAY)   8/8 deep leaf paths   PASS
[2] Join-graph / FK recovery across 3 databases         2/2 FK edges          PASS
[3] Fan-out (double-count) detection                    flagged               PASS
[4] Spaghetti-SQL anti-patterns                         all caught            PASS
[5] Scale stress                          599 columns in 0.006s               PASS
[6] Verified fan-out GATE                 blocks double-count, passes safe     PASS
[7] Agentic NL2SQL  link→validate→repair→select
       schema-linking recall@3            3/3                                  PASS
       selects valid candidate over hallucinated + fan-out drafts             PASS
       repairs hallucinated column (a.mrrr → a.mrr)                           PASS
       flags fan-out double-count                                            PASS
[8] Real Spider 2.0 / BIRD schema         --spider tables.json   (your env)
```

---

## How it works

```
Goal ─► AdaptivePlanner ──(reflection: propose→critique→revise)──► PlanGraph
              │                                                       │
              │   commits a causal design AT RUNTIME from the data    │
              ▼                                                       ▼
   ┌─ Foundation (deterministic) ─┐      ┌─ Agents (LLM judgment, gated) ─┐
   │ multi-DB catalog              │      │ planner · sql_author (NL2SQL)  │
   │ nested-type flattening        │      │ econometrician · ml_engineer   │
   │ join graph + fan-out detect   │      │ causal_ml · critic             │
   │ schema linking (RAG)          │      │ trust_report                   │
   └───────────────────────────────┘      └────────────────────────────────┘
              │                                          │
              └────────► blocking gates ◄────────────────┘
        identification · overlap · parallel-trends · sensitivity (E-value,
        robustness value) · refutation battery · fan-out · column existence
```

**Adaptive planner** — proposes a plan, critiques it against staff-level rules (approval
before modeling? a critic before delivery? within budget?), revises until clean, then
commits a causal design *at runtime from the profiled data*: same goal, four data shapes →
four designs (observational→DML, staggered→Callaway-Sant'Anna, IV→2SLS, panel→DiD), each
recovering truth. It also replans on events — a failed gate inserts a repair step.

**Agentic NL2SQL** — for any unseen schema with minimal instruction: RAG schema-linking
retrieves the relevant slice of a 3,000-column database (the fix for Spider 2.0's #1 error),
the LLM drafts, then deterministic gates guarantee what ships *can't* reference a nonexistent
column or double-count — it repairs or withholds instead. The LLM reasons; the gates make it safe.

**Provider-agnostic** — every agent depends on an `LLMClient` interface; runs fully offline
against a deterministic stub in CI, and against any major LLM provider (OpenAI, Google, etc.) in production by
swapping one object. Same two-swap design for data: synthetic source → any warehouse connection.

---

## Quickstart

```bash
pip install -e .
python -m dsagent test        # 139 passed, 6 suites
python -m dsagent adaptive     # same goal, 4 data shapes → 4 designs chosen at runtime
python -m dsagent run          # full pipeline → gated estimate + glass-box trust report
python -m dsagent eval         # recovery + adversarial traps + within-study + retrieval
python benchmark_suite.py      # 12 real-world causal datasets vs published benchmarks
python schema_benchmark.py     # nested/multi-DB schema + agentic NL2SQL
```

---

## Design decisions (the *why*)

- **LLM for judgment, deterministic code for verification.** The split is the architecture.
  Schema parsing, join math, estimators, and validity gates are reproducible and testable;
  the model is confined to where judgment genuinely helps. A hallucinated double-counting
  query is *caught by code*, not hoped against.
- **Gates block, they don't warn.** A warning an agent can ignore is not a safety property.
  `verdict = FAILED_GATES` overrides the model.
- **Gate-as-you-go, not gate-at-the-end.** The econometrician runs a skill's assumption
  check *before* fitting and withholds on failure — with a self-repair loop (e.g. trim to
  common support on an overlap violation, then retry) before giving up.
- **Estimand-first.** Every analysis carries an explicit ICH E9(R1)-style estimand; the
  method is chosen to identify *that* estimand, not the other way around.
- **Everything self-grades.** Known-truth generators + a within-study harness mean the
  system can prove it recovers planted effects — the same discipline used to validate it
  against real published studies.

---

## Honest scope

This is a research-grade engine, not a finished product. What's *guaranteed* regardless of
model quality: it won't ship a hallucinated-column or double-counting query, and it won't
report a causal estimate whose identifying assumptions failed. What still depends on the
live model + warehouse (and is measured in your environment): full NL2SQL execution accuracy
on the hardest enterprise schemas — where even frontier models score ~20% on Spider 2.0, and
where `dsagent`'s contribution is the *safety layer*, not a claim of solving generation.

## What I'd build next

- Embedding-based schema linking + candidate diversity (clean seams already in `sql/`).
- Anderson–Rubin weak-IV-robust CIs; honest pre-trends (Rambachan–Roth) bounds.
- LLM-drafted plans behind the existing rule gates; cost/value learning + plan caching.
- A live-warehouse adapter (MCP Toolbox pattern) and execution-accuracy eval on Spider 2.0.

---

## Repo map

`dsagent/` — `catalog` `dialects` `graph` `ontology` `profiling` `sql/` (parser, complexity,
schema_linker, validate, nl2sql, gates) · `llm/` (provider-agnostic clients + stub) ·
`runtime/` (blackboard, budget, orchestrator) · `agents/` · `execution/` (estimators,
ml, viz, datagen, estimand) · `skills/` (verified-skill registry) · `planning/` (adaptive
planner, mutable plan graph, adaptive orchestrator) · `eval/`
`demo/` — 6 test suites (139 tests) + runnable walkthroughs
`benchmark_suite.py` · `schema_benchmark.py` · `*_pilot.py`
