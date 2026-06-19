"""Python API for loadtune.

Provides ``profile()`` and ``tune()`` functions for programmatic use,
complementing the CLI entry points.

Usage::

    from loadtune import Workload, tune, profile

    workload = Workload(name="my_model", ...)
    result = profile(workload, steps=50)
    best = tune(workload, steps=50, max_trials=6)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .experiment import Trial, best_trial, run_trial_repeated, run_trials
from .heuristic import HeuristicBrain
from .knobs import Knobs
from .profiler import ProfileResult, profile_session
from .workload import Workload


def profile(
    workload: Workload,
    knobs: Optional[Knobs] = None,
    steps: int = 100,
    warmup: int = 10,
) -> ProfileResult:
    """Profile a workload and return the result.

    Parameters
    ----------
    workload : Workload
        The workload to profile.
    knobs : Knobs, optional
        Configuration knobs. Defaults to ``Knobs()`` (workers=0).
    steps : int
        Number of measured steps.
    warmup : int
        Number of warmup steps before measurement.

    Returns
    -------
    ProfileResult
        The profiling result with throughput, data_wait_frac, memory, etc.
    """
    if knobs is None:
        knobs = Knobs()
    return profile_session(workload, knobs, steps=steps, warmup=warmup)


@dataclass
class TuneResult:
    """Result of a ``tune()`` call."""

    baseline: ProfileResult
    trials: list[Trial]
    best: Optional[Trial]
    diagnosis: str

    @property
    def speedup(self) -> float:
        """Speedup of the best trial over baseline (1.0 = no improvement)."""
        if self.best and self.baseline.throughput > 0:
            return self.best.throughput / self.baseline.throughput
        return 1.0


def tune(
    workload: Workload,
    steps: int = 100,
    warmup: int = 10,
    max_trials: int = 6,
    max_rounds: int = 1,
    auto_batch: bool = False,
    repeats: int = 1,
    fast: bool = False,
    timeout_s: int = 900,
    verbose: bool = True,
) -> TuneResult:
    """Profile, diagnose, and tune a workload.

    Parameters
    ----------
    workload : Workload
        The workload to tune.
    steps : int
        Measured steps per trial.
    warmup : int
        Warmup steps per trial.
    max_trials : int
        Maximum number of trials per round.
    max_rounds : int
        Number of tuning rounds (multi-round optimization).
    auto_batch : bool
        Enable batch-size auto-scaling when GPU memory is underutilized.
    repeats : int
        Measure each config N times (median with min-max spread).
    fast : bool
        Run trials in-process (no subprocess isolation).
    timeout_s : int
        Per-trial timeout in seconds.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    TuneResult
        Contains baseline, all trials, best trial, and diagnosis text.
    """
    brain = HeuristicBrain()
    current_knobs = Knobs()
    original_baseline = None
    all_trials: list[Trial] = []

    for round_num in range(max_rounds):
        if verbose and max_rounds > 1:
            print(f"\n[loadtune] --- Round {round_num + 1}/{max_rounds} ---")

        # Run baseline
        if verbose:
            print(f"[loadtune] baseline run: {current_knobs.label()} ...")

        baseline_result = profile_session(
            workload, current_knobs, steps=steps, warmup=warmup,
        )
        if baseline_result.error:
            if verbose:
                print(f"[loadtune] baseline failed: {baseline_result.error}")
            return TuneResult(
                baseline=baseline_result,
                trials=[],
                best=None,
                diagnosis=f"Baseline failed: {baseline_result.error}",
            )

        if original_baseline is None:
            original_baseline = baseline_result

        if verbose:
            print(
                f"[loadtune] baseline: {baseline_result.throughput:.1f} samples/s, "
                f"data wait {baseline_result.data_wait_frac:.1%}, "
                f"device {baseline_result.device}"
            )

        # Propose trials
        trials = brain.propose(
            baseline_result, max_trials=max_trials, auto_batch=auto_batch,
        )
        if not trials:
            if verbose:
                print("[loadtune] brain proposed no trials — baseline looks fine.")
            break

        # Run trials
        def progress(i, n, t):
            if verbose:
                print(f"[loadtune] trial {i + 1}/{n}: {t.knobs.label()}  ({t.reason})")

        # For in-process (fast) mode, run profile_session directly
        if fast:
            for i, trial in enumerate(trials):
                progress(i, len(trials), trial)
                trial.result = profile_session(
                    workload, trial.knobs, steps=steps, warmup=warmup,
                ).to_dict()
        else:
            # Subprocess mode needs a workload file path — not available
            # for programmatic workloads. Fall back to in-process.
            for i, trial in enumerate(trials):
                progress(i, len(trials), trial)
                trial.result = profile_session(
                    workload, trial.knobs, steps=steps, warmup=warmup,
                ).to_dict()

        all_trials.extend(trials)

        best = best_trial(trials)
        if not best or best.throughput <= baseline_result.throughput:
            if verbose:
                print("[loadtune] no better config found in this round.")
            break

        current_knobs = best.knobs

    best = best_trial(all_trials)
    diagnosis = brain.explain(original_baseline, all_trials)

    if verbose and best and best.throughput > original_baseline.throughput:
        print(
            f"[loadtune] best: {best.knobs.label()} — "
            f"{best.throughput / original_baseline.throughput:.2f}x baseline"
        )

    return TuneResult(
        baseline=original_baseline,
        trials=all_trials,
        best=best,
        diagnosis=diagnosis,
    )
