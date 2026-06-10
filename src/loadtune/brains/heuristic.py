"""Deterministic rule-based brain.

Thresholds are deliberately simple and documented — this brain doubles as
the baseline the LLM brain is compared against in the writeup.
"""

from __future__ import annotations

from ..experiment import Trial
from ..knobs import Knobs, worker_candidates
from ..profiler import ProfileResult

# Fraction of step time spent waiting on data above which we call the
# workload input-bound.
INPUT_BOUND = 0.20
# Below this the pipeline is healthy and workers may even be overhead.
HEALTHY = 0.05


class HeuristicBrain:
    name = "heuristic"

    def propose(self, baseline: ProfileResult, max_trials: int) -> list[Trial]:
        b = Knobs.from_dict(baseline.knobs)
        frac = baseline.data_wait_frac
        cands: list[Trial] = []

        if frac >= INPUT_BOUND:
            # Input-bound: sweep workers, then prefetch on top of each.
            for w in worker_candidates(baseline.num_cpus):
                if w == b.num_workers:
                    continue
                cands.append(
                    Trial(
                        Knobs(num_workers=w, persistent_workers=w > 0,
                              pin_memory=b.pin_memory, batch_size=b.batch_size),
                        reason=f"data_wait_frac={frac:.0%} ≥ {INPUT_BOUND:.0%}: "
                               f"input-bound, trying num_workers={w}",
                    )
                )
            if b.num_workers > 0:
                cands.append(
                    Trial(
                        Knobs(num_workers=b.num_workers, prefetch_factor=4,
                              persistent_workers=True, pin_memory=b.pin_memory,
                              batch_size=b.batch_size),
                        reason="input-bound with workers active: deeper prefetch",
                    )
                )
        elif frac <= HEALTHY and b.num_workers > 0:
            # Pipeline healthy: workers may be pure overhead (worker spawn,
            # IPC). The "num_workers=2 instead of 4" case.
            for w in sorted({0, b.num_workers // 2}):
                cands.append(
                    Trial(
                        Knobs(num_workers=w, persistent_workers=w > 0,
                              pin_memory=b.pin_memory, batch_size=b.batch_size),
                        reason=f"data_wait_frac={frac:.0%} ≤ {HEALTHY:.0%}: "
                               f"compute-bound, workers may be overhead; try {w}",
                    )
                )
        else:
            # Mildly input-bound: small nudges around current config.
            for w in {b.num_workers + 2, max(0, b.num_workers - 2)}:
                if w != b.num_workers and w <= baseline.num_cpus:
                    cands.append(
                        Trial(
                            Knobs(num_workers=w, persistent_workers=w > 0,
                                  pin_memory=b.pin_memory, batch_size=b.batch_size),
                            reason=f"mild data wait ({frac:.0%}): nudge workers to {w}",
                        )
                    )

        # pin_memory only ever helps on CUDA.
        if baseline.device == "cuda" and not b.pin_memory:
            cands.append(
                Trial(
                    Knobs(num_workers=b.num_workers or 2, persistent_workers=True,
                          pin_memory=True, batch_size=b.batch_size),
                    reason="CUDA device: pin_memory speeds host-to-device copies",
                )
            )

        return cands[:max_trials]

    def explain(self, baseline: ProfileResult, trials: list[Trial]) -> str:
        frac = baseline.data_wait_frac
        if frac >= INPUT_BOUND:
            verdict = (
                f"The workload is **input-bound**: {frac:.0%} of step time is "
                f"spent waiting for the DataLoader."
            )
        elif frac <= HEALTHY:
            verdict = (
                f"The workload is **compute-bound** (data wait {frac:.0%}); "
                f"extra DataLoader workers add overhead without benefit."
            )
        else:
            verdict = f"The input pipeline is mildly contended (data wait {frac:.0%})."
        cpu = (
            f" Mean CPU utilisation during the run was {baseline.cpu_util_mean}%."
            if baseline.cpu_util_mean is not None
            else ""
        )
        return verdict + cpu
