"""Brain selection: heuristic rules or Claude-API reasoning.

--brain heuristic  deterministic rules, no network, free
--brain llm        Claude reasons over the baseline profile (needs ANTHROPIC_API_KEY)
--brain auto       llm if a key is set, else heuristic
"""

from __future__ import annotations

import os
from typing import Protocol

from ..experiment import Trial
from ..profiler import ProfileResult


class Brain(Protocol):
    name: str

    def propose(self, baseline: ProfileResult, max_trials: int) -> list[Trial]:
        """Given a baseline profile, propose configs worth trialing."""
        ...

    def explain(self, baseline: ProfileResult, trials: list[Trial]) -> str:
        """Narrative summary of findings for the report."""
        ...


def make_brain(kind: str = "auto") -> Brain:
    from .heuristic import HeuristicBrain

    if kind == "heuristic":
        return HeuristicBrain()
    if kind == "llm":
        from .llm import LLMBrain

        return LLMBrain()
    if kind == "auto":
        if os.environ.get("ANTHROPIC_API_KEY"):
            from .llm import LLMBrain

            return LLMBrain()
        return HeuristicBrain()
    raise ValueError(f"unknown brain: {kind!r} (use heuristic|llm|auto)")
