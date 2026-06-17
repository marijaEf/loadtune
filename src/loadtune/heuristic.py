"""Deterministic rule-based brain.

Thresholds are deliberately simple and documented — this brain doubles as
the baseline the LLM brain is compared against in the writeup.

Signals used (all from one baseline profile):
  data_wait_frac    fraction of step time blocked on the DataLoader
  cpu_util_mean     system-wide CPU % during the run
  p90/p50 ratio     step-time jitter (straggler batches)
"""

from __future__ import annotations

from .experiment import Trial
from .knobs import Knobs, worker_candidates
from .profiler import ProfileResult

# Fraction of step time waiting on data above which the workload is input-bound.
INPUT_BOUND = 0.20
# Below this the pipeline is healthy and workers may even be overhead.
HEALTHY = 0.05
# System CPU % above which adding workers cannot help (cores are the limit).
CPU_SATURATED = 85.0
# p90/p50 step-time ratio above which the pipeline is jittery -> prefetch.
JITTERY = 1.5
# Worker counts at/above this get a paired intra-op thread cap trial.
MANY_WORKERS = 4


class HeuristicBrain:
    name = "heuristic"

    def propose(self, baseline: ProfileResult, max_trials: int) -> list[Trial]:
        b = Knobs.from_dict(baseline.knobs)
        frac = baseline.data_wait_frac
        cpu = baseline.cpu_util_mean
        jitter = (
            baseline.step_time_p90_ms / baseline.step_time_p50_ms
            if baseline.step_time_p50_ms
            else 1.0
        )
        cands: list[Trial] = []

        cpu_saturated = cpu is not None and cpu >= CPU_SATURATED

        if frac >= INPUT_BOUND and cpu_saturated:
            # Rule: input-bound but cores already maxed — more workers just
            # add contention. Try trimming main-process threads to give the
            # existing workers room; the real fix (cheaper transforms,
            # caching, GPU-side augmentation) is outside loadtune's knobs.
            if b.num_workers > 0:
                cands.append(
                    Trial(
                        Knobs(num_workers=b.num_workers, persistent_workers=True,
                              pin_memory=b.pin_memory, batch_size=b.batch_size,
                              num_threads=max(1, baseline.num_cpus - b.num_workers),
                              compile=getattr(b, "compile", False)),
                        reason=f"CPU saturated ({cpu:.0f}%): cap intra-op threads "
                               f"instead of adding workers",
                    )
                )
        elif frac >= INPUT_BOUND:
            # Rule: input-bound with CPU headroom — sweep workers; pair the
            # high worker counts with an intra-op thread cap so the main
            # process doesn't fight its own workers for cores.
            for w in worker_candidates(baseline.num_cpus):
                if w == b.num_workers:
                    continue
                cands.append(
                    Trial(
                        Knobs(num_workers=w, persistent_workers=w > 0,
                              pin_memory=b.pin_memory, batch_size=b.batch_size,
                              compile=getattr(b, "compile", False)),
                        reason=f"data_wait_frac={frac:.0%} ≥ {INPUT_BOUND:.0%}: "
                               f"input-bound, trying num_workers={w}",
                    )
                )
                if w >= MANY_WORKERS:
                    cands.append(
                        Trial(
                            Knobs(num_workers=w, persistent_workers=True,
                                  pin_memory=b.pin_memory, batch_size=b.batch_size,
                                  num_threads=max(1, baseline.num_cpus - w),
                                  compile=getattr(b, "compile", False)),
                            reason=f"workers={w} claim cores: cap intra-op "
                                   f"threads at {max(1, baseline.num_cpus - w)} "
                                   f"to avoid contention",
                        )
                    )
            if b.num_workers > 0:
                cands.append(
                    Trial(
                        Knobs(num_workers=b.num_workers, prefetch_factor=4,
                              persistent_workers=True, pin_memory=b.pin_memory,
                              batch_size=b.batch_size, compile=getattr(b, "compile", False)),
                        reason="input-bound with workers active: deeper prefetch",
                    )
                )
        elif frac <= HEALTHY and b.num_workers > 0:
            # Rule: pipeline healthy — workers may be pure overhead (spawn,
            # IPC). The "num_workers=2 instead of 8" case.
            for w in sorted({0, b.num_workers // 2}):
                cands.append(
                    Trial(
                        Knobs(num_workers=w, persistent_workers=w > 0,
                              pin_memory=b.pin_memory, batch_size=b.batch_size,
                              compile=getattr(b, "compile", False)),
                        reason=f"data_wait_frac={frac:.0%} ≤ {HEALTHY:.0%}: "
                               f"compute-bound, workers may be overhead; try {w}",
                    )
                )
            
            # Rule: compute-bound with CUDA/CPU -> try graph compilation
            if baseline.device in ("cuda", "cpu") and not getattr(b, "compile", False):
                cands.append(
                    Trial(
                        Knobs(**{**b.to_dict(), "compile": True}),
                        reason=f"compute-bound (data wait {frac:.0%} ≤ {HEALTHY:.0%}): "
                               f"try torch.compile for graph-level optimization",
                    )
                )
        else:
            # Rule: mildly input-bound — small nudges around current config.
            for w in {b.num_workers + 2, max(0, b.num_workers - 2)}:
                if w != b.num_workers and w <= baseline.num_cpus:
                    cands.append(
                        Trial(
                            Knobs(num_workers=w, persistent_workers=w > 0,
                                  pin_memory=b.pin_memory, batch_size=b.batch_size,
                                  compile=getattr(b, "compile", False)),
                            reason=f"mild data wait ({frac:.0%}): nudge workers to {w}",
                        )
                    )

        # Rule: jittery step times with active workers -> deeper prefetch
        # smooths straggler batches regardless of the mean wait.
        if jitter >= JITTERY and b.num_workers > 0 and b.prefetch_factor is None:
            cands.append(
                Trial(
                    Knobs(num_workers=b.num_workers, prefetch_factor=4,
                          persistent_workers=True, pin_memory=b.pin_memory,
                          batch_size=b.batch_size, compile=getattr(b, "compile", False)),
                    reason=f"step-time jitter p90/p50={jitter:.2f} ≥ {JITTERY}: "
                           f"deeper prefetch absorbs straggler batches",
                )
            )

        # Rule: pin_memory only ever helps on CUDA.
        if baseline.device == "cuda" and not b.pin_memory:
            cands.append(
                Trial(
                    Knobs(num_workers=b.num_workers or 2, persistent_workers=True,
                          pin_memory=True, batch_size=b.batch_size,
                          compile=getattr(b, "compile", False)),
                    reason="CUDA device: pin_memory speeds host-to-device copies",
                )
            )

        return cands[:max_trials]

    def explain(self, baseline: ProfileResult, trials: list[Trial]) -> str:
        frac = baseline.data_wait_frac
        cpu = baseline.cpu_util_mean
        if frac >= INPUT_BOUND and cpu is not None and cpu >= CPU_SATURATED:
            verdict = (
                f"The workload is **input-bound** ({frac:.0%} data wait) with "
                f"**CPU already saturated** ({cpu:.0f}%): more workers cannot "
                f"help. The preprocessing itself is the limit — consider "
                f"cheaper transforms, caching decoded samples, or moving "
                f"augmentation to the accelerator."
            )
        elif frac >= INPUT_BOUND:
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
        cpu_note = (
            f" Mean CPU utilisation during the run was {cpu}%."
            if cpu is not None and not (frac >= INPUT_BOUND and cpu >= CPU_SATURATED)
            else ""
        )
        return verdict + cpu_note
