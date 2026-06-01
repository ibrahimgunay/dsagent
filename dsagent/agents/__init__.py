"""Agent roster + the composition root.

`build_org` wires the whole organization: it registers every sub-agent as a
tool and returns the registry + a planner bound to the chosen LLM client. Swap
StubLLM for AnthropicClient here and the system goes live without touching any
agent code.
"""
from __future__ import annotations

from ..catalog import Catalog
from ..llm.base import LLMClient
from ..runtime.tools import ToolRegistry, NoOpTool
from .foundation import ProfilerAgent, SemanticModelerAgent, JoinAnalyzerAgent
from .sql_author import SqlAuthorAgent
from .data_agent import DataExecutorAgent
from .modeling import (EconometricianAgent, MLEngineerAgent, CausalMLAgent, LabelerAgent)
from .critic import CriticAgent, DashboardBuilderAgent, MemoWriterAgent
from .planner_agent import PlannerAgent


def build_org(catalog: Catalog, llm: LLMClient, executor=None, skills=None) -> tuple[ToolRegistry, PlannerAgent]:
    if executor is None:
        from ..execution.executor import Executor
        executor = Executor()
    if skills is None:
        from ..skills import build_default_registry
        skills = build_default_registry()
    reg = ToolRegistry()
    # foundation (deterministic)
    reg.register(ProfilerAgent(catalog))
    reg.register(SemanticModelerAgent(catalog))
    reg.register(JoinAnalyzerAgent(catalog))
    reg.register(SqlAuthorAgent(catalog, llm))
    reg.register(DataExecutorAgent())
    # modeling (LLM-driven judgment; econ + causal can also FIT via executor)
    reg.register(EconometricianAgent(llm, executor, skills))
    reg.register(MLEngineerAgent(llm))
    reg.register(CausalMLAgent(llm, executor))
    reg.register(LabelerAgent(llm))
    # validation + delivery
    reg.register(CriticAgent(llm))
    reg.register(DashboardBuilderAgent(llm))
    reg.register(MemoWriterAgent(llm))
    from .trust_report import TrustReportAgent
    reg.register(TrustReportAgent())
    # checkpoints
    reg.register(NoOpTool())

    planner = PlannerAgent(llm, reg)
    return reg, planner


__all__ = ["build_org", "PlannerAgent"]
