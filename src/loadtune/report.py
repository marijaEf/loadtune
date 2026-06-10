"""Markdown report: baseline, trials table, verdict."""

from __future__ import annotations

from datetime import datetime

from .experiment import Trial, best_trial
from .knobs import Knobs
from .profiler import ProfileResult


def render_report(
    baseline: ProfileResult,
    trials: list[Trial],
    brain_name: str,
    narrative: str,
) -> str:
    best = best_trial(trials)
    lines = [
        f"# loadtune report — {baseline.workload}",
        "",
        f"*{datetime.now():%Y-%m-%d %H:%M} · device `{baseline.device}` · "
        f"{baseline.num_cpus} CPUs · brain `{brain_name}`*",
        "",
        "## Diagnosis",
        "",
        narrative,
        "",
        "## Baseline",
        "",
        f"- config: `{Knobs.from_dict(baseline.knobs).label()}`",
        f"- throughput: **{baseline.throughput:.1f} samples/s**",
        f"- data wait: {baseline.data_wait_frac:.1%} of step time "
        f"({baseline.data_wait_s:.2f}s of {baseline.total_s:.2f}s over "
        f"{baseline.steps} steps)",
        f"- step time p50/p90: {baseline.step_time_p50_ms:.1f} / "
        f"{baseline.step_time_p90_ms:.1f} ms",
        f"- dataloader startup: {baseline.startup_s:.2f}s",
        "",
        "## Trials",
        "",
        "| config | throughput (samples/s) | vs baseline | data wait | proposed because |",
        "|---|---|---|---|---|",
    ]
    for t in trials:
        if t.ok:
            r = t.result
            speedup = r["throughput"] / baseline.throughput if baseline.throughput else 0
            lines.append(
                f"| `{t.knobs.label()}` | {r['throughput']:.1f} | "
                f"{speedup:.2f}x | {r['data_wait_frac']:.1%} | {t.reason} |"
            )
        else:
            err = (t.result or {}).get("error", "unknown")
            first = str(err).strip().splitlines()[-1][:80]
            lines.append(f"| `{t.knobs.label()}` | failed | — | — | {first} |")

    lines += ["", "## Verdict", ""]
    if best and baseline.throughput and best.throughput > baseline.throughput * 1.02:
        speedup = best.throughput / baseline.throughput
        lines.append(
            f"**Recommended config: `{best.knobs.label()}` — "
            f"{speedup:.2f}x baseline throughput** "
            f"({baseline.throughput:.1f} → {best.throughput:.1f} samples/s)."
        )
    elif best:
        lines.append(
            "No trialed config beat the baseline by more than 2%. "
            "The baseline configuration appears close to optimal for this "
            "workload on this machine."
        )
    else:
        lines.append("All trials failed; see table above.")
    lines.append("")
    return "\n".join(lines)
