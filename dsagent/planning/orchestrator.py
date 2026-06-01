"""Adaptive orchestrator.

Executes a PlanGraph that changes while it runs. After each generation it
re-derives the ready set (so mutations take effect) and fires events to the
planner: once the data is fetched it triggers design-branch commitment; a
modeling block triggers a repair insertion; thin budget triggers pruning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..runtime.tools import ToolRegistry, ToolContext
from ..runtime.blackboard import Blackboard
from ..runtime.budget import Budget, BudgetExceeded
from .plangraph import PlanGraph
from .adaptive import AdaptivePlanner


def _auto_approve(node, bb):
    bb.log.append(f"APPROVAL auto-granted for '{node.id}'")
    return True


@dataclass
class AdaptiveRun:
    completed: list = field(default_factory=list)
    failed: dict = field(default_factory=dict)
    replans: list = field(default_factory=list)
    halted: str = ""
    budget: dict = field(default_factory=dict)


class AdaptiveOrchestrator:
    def __init__(self, registry: ToolRegistry, blackboard: Blackboard,
                 planner: AdaptivePlanner, budget: Budget | None = None,
                 services: dict | None = None, approval_fn=_auto_approve,
                 max_workers: int = 4):
        self.registry = registry
        self.bb = blackboard
        self.planner = planner
        self.budget = budget or Budget()
        self.services = services or {}
        self.approval_fn = approval_fn
        self.max_workers = max_workers
        self._profiled = False
        self._repaired: set[str] = set()

    def run(self, g: PlanGraph) -> AdaptiveRun:
        res = AdaptiveRun()
        guard = 0
        try:
            while True:
                guard += 1
                if guard > 100:
                    res.halted = "max generations exceeded"; break
                batches = g.ready_batches()
                if not batches:
                    break
                batch = batches[0]

                # budget-aware pruning before spending
                if self.budget.spent_usd > 0.6 * self.budget.max_usd:
                    line = self.planner.on_event(g, "low_budget")
                    if "pruned" in line and line not in res.replans:
                        res.replans.append(line)
                        continue   # re-derive after pruning

                self._run_generation(batch, res, g)
                if res.halted:
                    break
                self._post_generation_events(batch, res, g)
        except BudgetExceeded as e:
            res.halted = str(e)
        res.budget = self.budget.summary()
        return res

    def _run_generation(self, batch, res, g):
        runnable = []
        for node in batch:
            if node.requires_human_approval:
                if not self.approval_fn(node, self.bb):
                    res.halted = f"human rejected '{node.id}'"; return
                g.mark(node.id, "done"); res.completed.append(node.id)
                self.bb.log.append(f"OK   {node.id} (approval checkpoint)")
            else:
                runnable.append(node)
        if not runnable:
            return
        if len(runnable) == 1:
            self._dispatch(runnable[0], res, g)
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                list(as_completed([ex.submit(self._dispatch, n, res, g) for n in runnable]))

    def _dispatch(self, node, res, g):
        self.budget.check()
        tool = self.registry.get(node.tool or "noop")
        if tool is None:
            g.mark(node.id, "failed"); res.failed[node.id] = f"no tool {node.tool!r}"
            return
        g.mark(node.id, "running")
        ctx = ToolContext(blackboard=self.bb, task_id=node.id,
                          depends_on=[d for d in node.depends_on],
                          params=node.params, services=self.services)
        try:
            tool.run(ctx)
            self.budget.record(ctx.usage)
            g.mark(node.id, "done"); res.completed.append(node.id)
            self.bb.log.append(f"OK   {node.id} via {tool.name} (+${ctx.usage.usd:.4f})")
        except Exception as e:  # noqa: BLE001
            g.mark(node.id, "failed"); res.failed[node.id] = f"{type(e).__name__}: {e}"
            self.bb.log.append(f"ERR  {node.id}: {e}")

    def _post_generation_events(self, batch, res, g):
        for node in batch:
            # 1) data fetched -> commit to a design branch from the real profile
            if node.tool == "data_executor" and not self._profiled:
                profile = self._profile_from_fetch(node.id)
                line = self.planner.on_event(g, "data_profiled", profile=profile)
                res.replans.append(f"{line} (profile={ {k:v for k,v in profile.items() if v} })")
                self._profiled = True
            # 2) modeling node withheld its estimate -> insert a repair step
            if node.tool == "econometrician":
                art = self.bb.value(node.id, {}) or {}
                if art.get("estimate") is None and art.get("blocked_reason") \
                        and node.id not in self._repaired:
                    line = self.planner.on_event(g, "gate_failed", node_id=node.id,
                                                 gate=art.get("blocked_reason", "gate"))
                    res.replans.append(line); self._repaired.add(node.id)

    def _profile_from_fetch(self, fetch_id) -> dict:
        from ..execution.executor import profile_data
        art = self.bb.value(fetch_id, {}) or {}
        ref = art.get("dataset_ref")
        store = self.services.get("datastore")
        if ref and store is not None:
            try:
                return profile_data(store.get(ref))
            except Exception:
                return {}
        return {}
