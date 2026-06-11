"""Experiment runner: executes trials in isolated subprocesses."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

from .knobs import Knobs


@dataclass
class Trial:
    knobs: Knobs
    reason: str  # why the brain proposed this config
    result: Optional[dict] = None  # ProfileResult dict, or {"error": ...}

    @property
    def ok(self) -> bool:
        return bool(self.result) and not self.result.get("error")

    @property
    def throughput(self) -> float:
        return self.result.get("throughput", 0.0) if self.ok else 0.0


def run_trial(
    workload_path: str,
    knobs: Knobs,
    steps: int,
    warmup: int,
    timeout_s: int = 900,
) -> dict:
    """Run one trial in a fresh Python process; return the result dict."""
    cmd = [
        sys.executable,
        "-m",
        "loadtune._trial",
        workload_path,
        knobs.to_json(),
        str(steps),
        str(warmup),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s
        )
    except subprocess.TimeoutExpired:
        return {"error": f"trial timed out after {timeout_s}s"}

    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("LOADTUNE_RESULT "):
            return json.loads(line[len("LOADTUNE_RESULT "):])
    return {
        "error": "trial produced no result",
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def run_trials(
    workload_path: str,
    trials: list[Trial],
    steps: int,
    warmup: int,
    on_progress=None,
) -> list[Trial]:
    for i, trial in enumerate(trials):
        if on_progress:
            on_progress(i, len(trials), trial)
        trial.result = run_trial(workload_path, trial.knobs, steps, warmup)
    return trials


def best_trial(trials: list[Trial], noise_tol: float = 0.02) -> Optional[Trial]:
    """Best = cheapest config within `noise_tol` of the top throughput.

    Throughput differences under ~2% are measurement noise; among the
    statistically tied winners, prefer fewer workers (less memory, fewer
    idle processes). This is the "num_workers=2 instead of 8" rule.
    """
    ok = [t for t in trials if t.ok]
    if not ok:
        return None
    top = max(t.throughput for t in ok)
    contenders = [t for t in ok if t.throughput >= top * (1 - noise_tol)]
    return min(
        contenders,
        key=lambda t: (t.knobs.num_workers, -t.throughput),
    )
