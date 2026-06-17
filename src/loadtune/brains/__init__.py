"""Brain protocol for heuristic rules."""

from __future__ import annotations

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
