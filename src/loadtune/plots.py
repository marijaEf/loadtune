"""Charts for loadtune reports (matplotlib, saved as PNG).

GPU-utilization note: Apple Silicon has no public per-process GPU counter
API, so instead of an indirect utilization %, we plot what we measure
directly — time the training loop spends blocked on data vs computing.
A step that waits on data IS the accelerator sitting idle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .experiment import Trial
from .knobs import Knobs
from .profiler import ProfileResult

WAIT = "#d9534f"   # data wait — red
COMP = "#4878a8"   # compute — blue
BAND = "#dff0d8"   # noise-tolerance band


def _label(result: dict) -> str:
    return Knobs.from_dict(result["knobs"]).label()


def plot_breakdown(baseline: ProfileResult, trials: list[Trial], out: Path) -> Path:
    """Mean step time per config, stacked into data-wait vs compute."""
    rows = [("baseline\n" + Knobs.from_dict(baseline.knobs).label(),
             baseline.data_wait_s, baseline.compute_s, baseline.steps)]
    for t in trials:
        if t.ok:
            r = t.result
            rows.append((_label(r), r["data_wait_s"], r["compute_s"], r["steps"]))

    labels = [r[0] for r in rows]
    wait_ms = [1000 * r[1] / r[3] for r in rows]
    comp_ms = [1000 * r[2] / r[3] for r in rows]

    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(rows)), 4))
    x = range(len(rows))
    ax.bar(x, comp_ms, color=COMP, label="compute")
    ax.bar(x, wait_ms, bottom=comp_ms, color=WAIT, label="data wait (accelerator idle)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("mean step time (ms)")
    ax.set_title("Where each training step spends its time")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_throughput_vs_workers(
    baseline: ProfileResult,
    trials: list[Trial],
    out: Path,
    noise_tol: float = 0.02,
) -> Optional[Path]:
    """Throughput vs num_workers, with the noise band that drives the verdict.

    Configs with a capped num_threads are drawn as a separate series so
    worker×thread pairs don't collide at the same x position.
    """
    plain: dict[int, float] = {}
    capped: dict[int, float] = {}

    def add(knobs: dict, thr: float) -> None:
        target = capped if knobs.get("num_threads") is not None else plain
        target[knobs["num_workers"]] = thr

    add(baseline.knobs, baseline.throughput)
    for t in trials:
        if t.ok:
            add(t.result["knobs"], t.result["throughput"])
    if len(plain) + len(capped) < 3:
        return None
    top = max(list(plain.values()) + list(capped.values()))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhspan(top * (1 - noise_tol), top, color=BAND,
               label=f"within {noise_tol:.0%} of best (tie → fewest workers)")
    xs = sorted(plain)
    ax.plot(xs, [plain[x] for x in xs], marker="o", color=COMP,
            label="default threads")
    for x in xs:
        ax.annotate(f"{plain[x]:.0f}", (x, plain[x]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    if capped:
        cxs = sorted(capped)
        ax.plot(cxs, [capped[x] for x in cxs], marker="s", linestyle="--",
                color=WAIT, label="intra-op threads capped")
        for x in cxs:
            ax.annotate(f"{capped[x]:.0f}", (x, capped[x]),
                        textcoords="offset points", xytext=(0, -14),
                        ha="center", fontsize=8, color=WAIT)
    ax.set_xlabel("num_workers")
    ax.set_ylabel("throughput (samples/s)")
    ax.set_title("Throughput vs DataLoader workers")
    ax.set_xticks(sorted(set(plain) | set(capped)))
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_step_timeline(
    baseline: ProfileResult, best: Optional[Trial], out: Path
) -> Optional[Path]:
    """Per-step stacked area: baseline vs best config."""
    if not baseline.step_data_wait_ms:
        return None
    panels = [("baseline: " + Knobs.from_dict(baseline.knobs).label(),
               baseline.step_data_wait_ms, baseline.step_compute_ms)]
    if best and best.ok and best.result.get("step_data_wait_ms"):
        panels.append(("tuned: " + _label(best.result),
                       best.result["step_data_wait_ms"],
                       best.result["step_compute_ms"]))

    fig, axes = plt.subplots(
        len(panels), 1, figsize=(7, 2.6 * len(panels)), sharex=True, sharey=True
    )
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, waits, comps) in zip(axes, panels):
        steps = range(1, len(waits) + 1)
        ax.stackplot(steps, comps, waits, colors=[COMP, WAIT],
                     labels=["compute", "data wait"])
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("ms")
    axes[-1].set_xlabel("step")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def generate_plots(
    baseline: ProfileResult,
    trials: list[Trial],
    best: Optional[Trial],
    out_dir: Path,
    prefix: str,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_breakdown(baseline, trials, out_dir / f"{prefix}_breakdown.png"),
        plot_throughput_vs_workers(
            baseline, trials, out_dir / f"{prefix}_throughput.png"
        ),
        plot_step_timeline(baseline, best, out_dir / f"{prefix}_timeline.png"),
    ]
    return [p for p in paths if p is not None]
