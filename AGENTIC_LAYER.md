# Agentic Layer — Delivery Brief (for the CEO)

**What you asked for:** stop acting like one engineer writing a script; act like
the whole org and ship the *agentic* system — a planner that orchestrates,
sub-agents as tools, and production Python where all the parts communicate.

**What was delivered:** a working multi-agent runtime on top of the foundation
engine. A planner LLM turns your goal into an execution plan; an orchestrator
runs it (parallel where safe, sequential where required, with human-approval
gates and a spend budget); ten specialist sub-agents do the work and
communicate through a shared, lineage-tracked blackboard; a critic enforces
blocking validity gates before anything ships. It runs end-to-end offline today
and goes live by swapping one object.

```bash
python -m demo.run_agent     # full org runs one goal end to end
python -m demo.test_agent    # 13 tests incl. negative gate cases
python -m demo.test_core     # 16 foundation tests
```

---

## How it works (the org chart, in code)

```
  CEO goal
     │
     ▼
 PlannerAgent ── LLM ──▶ task DAG ──▶  Orchestrator
 (agents/planner_agent)                (runtime/orchestrator)
                                          │  topological batches
                                          │  parallel within a batch (threads)
                                          │  human-approval checkpoints
                                          │  budget governor (USD/tokens/calls)
                                          ▼
                      ┌──────────── dispatches to tools ────────────┐
   FOUNDATION (deterministic)      MODELING (LLM judgment)     VALIDATION/DELIVERY
   profiler                        econometrician              critic  (blocking gates)
   semantic_modeler                ml_engineer                 dashboard_builder
   join_analyzer                   causal_ml                   memo_writer
   sql_author (LLM+verify)         labeler
                      └──────── all read/write the BLACKBOARD ───────┘
                                  (runtime/blackboard: artifacts + lineage + lock)
```

**Communication.** Agents never call each other directly. Each reads the
upstream artifacts it needs from the blackboard and writes its result back,
keyed by task id. That decoupling is what lets the orchestrator run a batch
concurrently, retry a single agent, and reconstruct full lineage for any output
(`bb.lineage("memo")` walks memo → critic → econ → … → source).

**Where intelligence lives.** Deliberately split:
- *Deterministic* (reproducible, auditable): schema parsing, profiling, join
  math, fan-out detection, the validity gates. These should never be
  probabilistic.
- *LLM judgment* (runtime model calls): the planner (designs control flow), SQL
  authoring, the econometrics / ML / causal / labeling plans, critic
  reconciliation, dashboard + memo drafting. The demo makes 9 real runtime LLM
  calls.

**The honest seam.** Everything runs on `StubLLM` (deterministic offline
fixtures) so it's testable in CI with no network. For production:

```python
from dsagent.llm import OpenAIClient
llm = OpenAIClient(model="")   # reads OPENAI_API_KEY
registry, planner = build_org(catalog, llm)          # nothing else changes
```

No agent code changes. The stub and the live client implement the same
`LLMClient` interface.

---

## What makes it production-shaped (not a toy)

- **Typed tool contracts.** Every sub-agent is a `Tool` with a name,
  description, declared inputs, and a single `run` that emits one artifact. The
  planner can only choose registered tools; unknown tools raise (tested).
- **Blocking validity gates.** The critic runs *deterministic* gates that
  override the LLM: identification-before-estimation (no causal estimate without
  estimand + assumptions + sensitivity analysis), no-naive-TWFE under staggered
  timing, SQL fan-out resolved, causal overlap checked. Negative cases are
  tested — feed a bad artifact and the run is marked `FAILED_GATES`.
- **Budget governor.** USD / token / tool-call ceilings, checked before each
  dispatch; a runaway plan halts instead of burning spend.
- **Human-in-the-loop.** Approval checkpoints (`requires_human_approval`) gate
  the modeling phase and final delivery; an `approval_fn` lets you wire a real
  UI/Slack approval in place of auto-approve.
- **Resilience.** Per-task retries; one failed agent doesn't crash the run.
- **Concurrency.** Independent tasks in a batch run on a thread pool; the
  blackboard is lock-guarded.

---

## What's real vs. still stubbed

**Real and running:** the full planner → orchestrator → sub-agent → critic →
delivery loop; parallel/sequential scheduling; blackboard comms + lineage;
budget; human-approval gating; the deterministic foundation engine; the
blocking validity gates; the production OpenAI client (works the moment a key
+ network are present).

**Stubbed (one swap each):** the LLM responses (StubLLM → OpenAIClient), and
the modeling agents currently emit *plans* (estimand, estimator, CV strategy)
rather than executing fits. The next build wires each modeling agent to a
sandboxed compute tool (statsmodels / EconML / LightGBM) so it returns a fitted
estimate the critic gates on — the contracts already accommodate it.

---

## Suggested next sprints

1. **Live LLM + golden-path eval.** Swap to OpenAIClient; run the agent eval
   harness on known-answer fixtures (does the planner pick the right design?).
2. **Execution tools for modeling agents.** Sandboxed runners that actually fit
   Callaway-Sant'Anna / DML / LightGBM and write numeric results + diagnostics.
3. **Warehouse connector.** Replace text-only SQL analysis with real pushdown
   execution + row-count reconciliation.
4. **Approval UI + audit export.** Wire `approval_fn` to a real reviewer; export
   the lineage graph as the audit artifact.
