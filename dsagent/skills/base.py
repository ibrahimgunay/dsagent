"""Verified Skills — the moat.

A Skill is not just a prompt. It bundles four things so that "the agent knows
method X" becomes "the agent's use of X is provably correct":

  1. preconditions(profile) -> bool   : when retrieval should surface it
  2. estimator                        : the executor method it maps to
  3. gate(data) -> GateResult         : a BLOCKING check run *before* fitting
  4. self_test() -> bool              : recovers planted truth on a fixture

The registry can `verify_all()` (every skill proves itself) and `retrieve()`
(rank skills by precondition match). This is what turns a corpus of methods +
sample code into a *trust* asset competitors can't copy by fine-tuning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any
import pandas as pd

from ..execution import estimators as est
from ..execution.executor import Executor


@dataclass
class GateResult:
    passed: bool
    name: str
    detail: str = ""


@dataclass
class Skill:
    id: str
    description: str
    preconditions: Callable[[dict], bool]
    estimator: str
    identification: str
    gate: Callable[[pd.DataFrame], GateResult] | None = None
    config: dict = field(default_factory=dict)
    # self-test: a fixture generator returning (df, truth) and a tolerance
    fixture: Callable[[], tuple] | None = None
    tol: float = 0.15

    def match_score(self, profile: dict) -> float:
        try:
            return 1.0 if self.preconditions(profile) else 0.0
        except Exception:
            return 0.0

    def check_gate(self, data: pd.DataFrame) -> GateResult:
        if self.gate is None:
            return GateResult(True, f"{self.id}:no_gate")
        return self.gate(data)

    def self_test(self, executor: Executor) -> dict:
        if self.fixture is None:
            return {"skill": self.id, "tested": False}
        made = self.fixture()
        df, truth = made if isinstance(made, tuple) else (made, None)
        res = executor.fit(self.estimator, df, **self.config)
        rel = abs(res.point - truth) / (abs(truth) if abs(truth) > 1e-9 else 1.0)
        return {"skill": self.id, "tested": True, "point": round(res.point, 4),
                "truth": round(truth, 4), "covers": res.covers(truth),
                "within_tol": rel <= self.tol or abs(res.point - truth) <= self.tol}

    def to_skill_md(self) -> str:
        """Emit this verified skill in the open Agent Skills SKILL.md format
        (YAML frontmatter + markdown body) so it is portable to Claude Code,
        Cursor, etc. The blocking gate + self-test are what make ours *verified*,
        not just instructions."""
        gate = "none" if self.gate is None else self.gate.__name__
        return (f"---\nname: {self.id}\n"
                f"description: {self.description} Use when the data profile "
                f"matches; runs a blocking '{gate}' check before estimating.\n"
                f"---\n\n# {self.id}\n\n"
                f"## When to use\n{self.description}\n\n"
                f"## Identification\n{self.identification}\n\n"
                f"## Estimator\n`{self.estimator}` with config `{self.config}`.\n\n"
                f"## Blocking gate (verified)\nRuns `{gate}` BEFORE fitting; the "
                f"estimate is withheld if the assumption fails.\n\n"
                f"## Self-test (known truth)\nRecovers a planted effect on a "
                f"synthetic fixture within tolerance {self.tol}.\n")


def parse_skill_md(text: str) -> dict:
    """Minimal SKILL.md parser: returns the frontmatter name + description."""
    meta = {}
    if text.startswith("---"):
        fm = text.split("---", 2)[1]
        for line in fm.strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
    return meta


class SkillRegistry:
    def __init__(self):
        self._skills: dict[str, Skill] = {}

    def register(self, s: Skill):
        self._skills[s.id] = s

    def all(self):
        return list(self._skills.values())

    def get(self, sid: str) -> Skill | None:
        return self._skills.get(sid)

    def retrieve(self, profile: dict) -> list[Skill]:
        """Rank skills by precondition match (retrieval over the corpus)."""
        scored = [(s.match_score(profile), s) for s in self._skills.values()]
        return [s for sc, s in sorted(scored, key=lambda x: -x[0]) if sc > 0]

    def scan(self) -> list[dict]:
        """Progressive disclosure: cheap metadata-only view (name + description),
        as in the Agent Skills standard — the agent scans these first and only
        loads a full skill when relevant."""
        return [{"name": s.id, "description": s.description} for s in self._skills.values()]

    def load(self, sid: str) -> Skill | None:
        """Load the full skill once retrieval has selected it."""
        return self._skills.get(sid)

    def write_skill_files(self, outdir: str) -> list[str]:
        """Emit each verified skill as a standard SKILL.md folder."""
        import os
        paths = []
        for s in self._skills.values():
            d = os.path.join(outdir, s.id)
            os.makedirs(d, exist_ok=True)
            p = os.path.join(d, "SKILL.md")
            with open(p, "w") as f:
                f.write(s.to_skill_md())
            paths.append(p)
        return paths

    def best(self, profile: dict) -> Skill | None:
        r = self.retrieve(profile)
        return r[0] if r else None

    def verify_all(self, executor: Executor | None = None) -> dict:
        ex = executor or Executor()
        results = [s.self_test(ex) for s in self._skills.values()]
        tested = [r for r in results if r.get("tested")]
        passed = [r for r in tested if r.get("within_tol")]
        return {"skills": len(self._skills), "tested": len(tested),
                "passed": len(passed), "results": results}
