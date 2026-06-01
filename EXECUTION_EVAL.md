# Execution + Eval — Delivery Brief

Two things shipped this sprint: the modeling agents now **actually fit models**
(not just plan them), and a **gold-standard eval harness** grades the system
against datasets with known answers.

```bash
python -m demo.run_eval          # the scorecard
python -m demo.test_exec_eval    # 15 tests: recovery, bias detection, fitting
```

---

## 1. Modeling agents now compute real numbers

`dsagent/execution/` is the numerical core (numpy / scipy / sklearn — no
statsmodels needed):

- **`estimators.py`** — real fits with HC1 robust SEs and 95% CIs:
  - `diff_in_means`, `regression_adjust` (ANCOVA),
  - `did_2x2` (difference-in-differences via interaction OLS),
  - `double_ml` (partially-linear Double ML with cross-fitting + GBM nuisance),
  - `propensity_overlap` (positivity gate), `cate_by_subgroup` (heterogeneity).
- **`executor.py`** — `Executor.fit(name, data)` dispatches a spec to a fit;
  `select_design(profile)` encodes the estimand-first design choice
  (panel→DiD, randomized→contrast, observed-confounders→DML).

The `EconometricianAgent` and `CausalMLAgent` now take an `Executor`. When a
dataset is wired in, they **fit and attach a numeric estimate** (point, SE, CI,
p); the `CausalMLAgent` runs the overlap check first and **withholds the
estimate if positivity fails** — the blocking gate in action. With no dataset
they degrade to plan-only, so the orchestrator demo is unchanged.

Verified recovery (offline, fixed seeds):

| data (known truth) | naive | selected design | recovers? |
|---|---|---|---|
| RCT, ATE=2.0 | 2.01 ✓ | diff-in-means | yes |
| Observational confounded, ATE=2.0 | **3.70 ✗ (biased)** | Double ML → 2.01 | yes |
| Panel DiD, ATT=1.5 | — | did_2x2 → 1.49 | yes |
| A/A null, effect=0 | — | DML → 0.00, p=0.90 | no false positive |

The confounded row is the point: the naive contrast is **wrong**, and the
system picks a design that fixes it.

---

## 2. The system grades itself (gold-standard eval)

`dsagent/eval/` implements the benchmark the architecture doc demands — "score
the agent on recovering known truth, calibration, and false-discovery rate."

`run_eval` scores four properties per scenario:
- **recovery** — selected design's estimate is within tolerance of truth,
- **coverage** — its 95% CI contains truth,
- **discrimination** — on confounded data the naive contrast is correctly
  flagged as biased (the system would not be fooled),
- **FDR / null** — on A/A data it does not declare a false effect.

Plus a **process eval**: the econometrician emits an estimand + identifying
assumptions + sensitivity analysis (estimand-first compliance), fits a real
estimate that recovers truth, and the critic gates pass.

Current result: **4/4 scenarios pass, 6/6 process checks pass.**

This is the bar a new analyst would have to clear, applied to the agent: if a
future change makes it pick a biased design or miss the null, the eval fails and
the regression is caught before it ships.

---

## What's still stubbed

- **LLM** — still `StubLLM` offline; swap to `AnthropicClient` for live design
  choices. The eval then measures whether the *live model* picks designs that
  recover truth (the most valuable use of the harness).
- **DiD** — `did_2x2` is the canonical two-period estimator. Staggered-adoption
  Callaway–Sant'Anna (which the econometrician already *names*) is the next
  estimator to add to `estimators.py`; the executor contract already fits it.
- **Data source** — datasets are synthetic generators with planted truth. The
  warehouse connector (run the authored SQL, return a frame) is the bridge to
  real data; the agents already accept a dataframe via params.
