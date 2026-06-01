"""Known-answer scenarios for the eval harness.

Each scenario has a planted truth and the *design* that should recover it. The
observational scenario is the important one: it includes confounding so a naive
contrast is biased — the eval must mark the naive design as failing and the
adjusted/DML design as passing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
import pandas as pd
from ..execution import datagen as dg


@dataclass
class Scenario:
    name: str
    make: Callable[[], pd.DataFrame]   # returns df, or (df, truth)
    truth: float | None
    profile: dict                      # structural hints for select_design
    expects_naive_bias: bool = False   # should the naive contrast be wrong?
    is_null: bool = False
    naive_estimator: str = "diff_in_means"
    naive_config: dict = field(default_factory=lambda: {"treat": "t", "outcome": "y"})


SCENARIOS = [
    Scenario("rct_randomized",
             lambda: dg.make_rct(ate=2.0), truth=2.0,
             profile={"randomized": True, "has_controls": True}),
    Scenario("observational_confounded",
             lambda: dg.make_observational(ate=2.0), truth=2.0,
             profile={"has_controls": True}, expects_naive_bias=True),
    Scenario("did_panel",
             lambda: dg.make_did_panel(att=1.5), truth=1.5,
             profile={"is_panel": True},
             naive_config={"treat": "treat", "outcome": "y"}),
    Scenario("staggered_adoption",
             lambda: dg.make_staggered(base=0.4), truth=None,
             profile={"is_staggered": True}, expects_naive_bias=True,
             naive_estimator="twfe_static", naive_config={}),
    Scenario("iv_instrumented",
             lambda: dg.make_iv(ate=1.5, strength=1.2), truth=None,
             profile={"has_instrument": True}, expects_naive_bias=True,
             naive_estimator="diff_in_means", naive_config={"treat": "t", "outcome": "y"}),
    Scenario("simpsons_trap",
             lambda: dg.make_simpsons(effect=1.0), truth=1.0,
             profile={"has_controls": True}, expects_naive_bias=True,
             naive_estimator="diff_in_means"),
    Scenario("aa_null",
             lambda: dg.make_null(), truth=0.0,
             profile={"has_controls": True}, is_null=True),
]
