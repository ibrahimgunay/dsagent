"""The Orchestrator — executes a plan.

Resolves the DAG into topological batches (tasks in a batch have no remaining
dependencies and run concurrently), dispatches each task to its registered
tool, injects the shared blackboard so tools read upstream artifacts, enforces
the budget, runs human-approval checkpoints, and retries transient failures.

This is the runtime the Planner agent targets: the planner decides *what* to do
(the DAG); the orchestrator makes it happen *safely and in the right order*.
"""
from __future__ import annotations

import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable

import networkx as nx

from ..planner import AnalysisDAG, Task
from .blackboard import Blackboard
from .tools import ToolRegistry, ToolContext
from .budget import Budget, BudgetExceeded


# approval_fn(task, blackboard) -> bool. Default auto-approves and logs.
ApprovalFn = Callable[[Task, Blackboard], bool]


def _auto_approve(task: Task, bb: Blackboard) -> bool:
    bb.log.append(f"APPROVAL auto-granted for checkpoint '{task.id}'")
    return True


@dataclass
class RunResult:
    completed: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    halted_reason: str = ""
    budget: dict = field(default_factory=dict)


class Orchestrator:
    def __init__(self, registry: ToolRegistry, blackboard: Blackboard,
                 budget: Budget | None = None, approval_fn: ApprovalFn = _auto_approve,
                 max_workers: int = 4, retries: int = 1, services: dict | None = None) -> None:
        self.registry = registry
        self.bb = blackboard
        self.budget = budget or Budget()
        self.approval_fn = approval_fn
        self.max_workers = max_workers
        self.retries = retries
        self.services = services or {}

    def run(self, dag: AnalysisDAG) -> RunResult:
        dag.validate()
        result = RunResult()
        try:
            for batch_no, batch in enumerate(dag.execution_batches(), 1):
                self.bb.log.append(f"--- BATCH {batch_no}: {[t.id for t in batch]} "
                                   f"({'parallel' if len(batch) > 1 else 'sequential'}) ---")
                self._run_batch(batch, result)
                if result.halted_reason:
                    return self._finish(result)
        except BudgetExceeded as e:
            result.halted_reason = str(e)
        return self._finish(result)

    def _run_batch(self, batch: list[Task], result: RunResult) -> None:
        # human-approval checkpoints run inline and gate the rest
        runnable = []
        for t in batch:
            if t.requires_human_approval:
                if not self.approval_fn(t, self.bb):
                    result.halted_reason = f"Human rejected checkpoint '{t.id}'"
                    return
                self._dispatch(t, result)  # record the checkpoint marker
            else:
                runnable.append(t)

        if not runnable:
            return
        if len(runnable) == 1:
            self._dispatch(runnable[0], result)
            return

        # genuine concurrency for independent tasks (LLM calls are I/O-bound)
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = {ex.submit(self._dispatch, t, result): t for t in runnable}
            for _ in as_completed(futs):
                pass

    def _dispatch(self, task: Task, result: RunResult) -> None:
        self.budget.check()
        tool = self.registry.get(task.tool or "noop")
        if tool is None:
            result.failed[task.id] = f"no tool registered: {task.tool!r}"
            return
        ctx = ToolContext(blackboard=self.bb, task_id=task.id,
                          depends_on=task.depends_on, params=task.params,
                          services=self.services)
        last_err = ""
        for attempt in range(self.retries + 1):
            try:
                t0 = time.time()
                tool.run(ctx)
                self.budget.record(ctx.usage)
                self.bb.log.append(
                    f"OK   {task.id} via {tool.name} "
                    f"({(time.time()-t0)*1000:.0f}ms, +${ctx.usage.usd:.4f})")
                result.completed.append(task.id)
                return
            except Exception as e:  # noqa: BLE001 — orchestrator must be resilient
                last_err = f"{type(e).__name__}: {e}"
                self.bb.log.append(f"ERR  {task.id} attempt {attempt+1}: {last_err}")
        result.failed[task.id] = last_err

    def _finish(self, result: RunResult) -> RunResult:
        result.budget = self.budget.summary()
        return result
