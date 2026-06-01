"""Tools and the tool registry.

A sub-agent is exposed to the orchestrator as a Tool with a typed contract:
a name, a description (what the planner sees when choosing tools), the upstream
artifact keys it reads, and a `run` that writes one artifact to the blackboard.
The registry is the single place the planner and orchestrator discover
capabilities — add a tool here and it becomes plannable.
"""
from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field

from .blackboard import Blackboard, Artifact
from ..llm.base import Usage


@dataclass
class ToolContext:
    """Everything a tool needs at run time, injected by the orchestrator."""
    blackboard: Blackboard
    task_id: str
    depends_on: list[str]
    params: dict = field(default_factory=dict)
    services: dict = field(default_factory=dict)   # data_source, datastore, ...
    usage: Usage = field(default_factory=Usage)


class Tool(abc.ABC):
    name: str = "tool"
    description: str = ""
    kind: str = "generic"
    reads: list[str] = []          # documentation of expected upstream kinds

    @abc.abstractmethod
    def run(self, ctx: ToolContext) -> Artifact: ...

    def _emit(self, ctx: ToolContext, value) -> Artifact:
        return ctx.blackboard.put(Artifact(
            key=ctx.task_id, kind=self.kind, producer=self.name,
            value=value, inputs=list(ctx.depends_on)))


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def catalog_for_planner(self) -> list[dict]:
        """Capability list the planner LLM chooses from."""
        return [{"tool": t.name, "kind": t.kind, "description": t.description}
                for t in self._tools.values()]


class NoOpTool(Tool):
    """Used for checkpoint/human-approval nodes that carry no computation."""
    name = "noop"
    description = "Checkpoint node (e.g. human approval); produces a marker."
    kind = "checkpoint"

    def run(self, ctx: ToolContext) -> Artifact:
        return self._emit(ctx, {"checkpoint": ctx.task_id, "passed": True})
