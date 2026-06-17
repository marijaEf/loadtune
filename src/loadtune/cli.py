"""loadtune CLI.

  loadtune profile workloads/resnet50_cifar.py
  loadtune tune    workloads/resnet50_cifar.py --brain auto --out report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


from .experiment import Trial, best_trial, run_trial, run_trial_repeated, run_trials
from .knobs import Knobs
from .profiler import ProfileResult
from .report import render_report


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("workload", help="path to a workload .py file")
    p.add_argument("--steps", type=int, default=100, help="measured steps per trial")
    p.add_argument("--warmup", type=int, default=10, help="warmup steps per trial")
    p.add_argument("--workers", type=int, default=0, help="baseline num_workers")
    p.add_argument("--batch-size", type=int, default=None, help="override batch size")
    p.add_argument(
        "--timeout", type=int, default=900,
        help="per-trial timeout in seconds (first runs may download datasets)",
    )


def cmd_profile(args: argparse.Namespace) -> int:
    knobs = Knobs(num_workers=args.workers, batch_size=args.batch_size)
    print(f"[loadtune] profiling {args.workload} with {knobs.label()} ...")
    result = run_trial(args.workload, knobs, args.steps, args.warmup,
                       timeout_s=args.timeout)
    print(json.dumps(result, indent=2))
    return 0 if not result.get("error") else 1


def cmd_tune(args: argparse.Namespace) -> int:
    original_baseline_knobs = Knobs(num_workers=args.workers, batch_size=args.batch_size)
    current_knobs = original_baseline_knobs
    original_baseline = None
    all_trials = []
    
    max_rounds = getattr(args, "max_rounds", 1)
    for round_num in range(max_rounds):
        if max_rounds > 1:
            print(f"\n[loadtune] --- Round {round_num + 1}/{max_rounds} ---")
            
        print(f"[loadtune] baseline run: {current_knobs.label()} ...")
        baseline_dict = run_trial_repeated(args.workload, current_knobs, args.steps,
                                           args.warmup, timeout_s=args.timeout,
                                           repeats=args.repeats, fast=getattr(args, "fast", False))
        if baseline_dict.get("error"):
            print("[loadtune] baseline failed:\n" + str(baseline_dict["error"]))
            return 1
            
        baseline = ProfileResult(**baseline_dict)
        if original_baseline is None:
            original_baseline = baseline
            
        print(
            f"[loadtune] baseline: {baseline.throughput:.1f} samples/s, "
            f"data wait {baseline.data_wait_frac:.1%}, device {baseline.device}"
        )

        from .brains.heuristic import HeuristicBrain
        brain = HeuristicBrain()
        print(f"[loadtune] using brain: heuristic")
        print(f"[loadtune] brain: {brain.name}")
        trials = brain.propose(baseline, max_trials=args.max_trials)
        if not trials:
            print("[loadtune] brain proposed no trials — baseline looks fine.")
            break

        def progress(i: int, n: int, t: Trial) -> None:
            print(f"[loadtune] trial {i + 1}/{n}: {t.knobs.label()}  ({t.reason})")

        run_trials(args.workload, trials, args.steps, args.warmup,
                   on_progress=progress, timeout_s=args.timeout,
                   repeats=args.repeats, fast=getattr(args, "fast", False))
                   
        all_trials.extend(trials)
        
        best = best_trial(trials)
        if not best or best.throughput <= baseline.throughput:
            print("[loadtune] no better config found in this round.")
            break
            
        current_knobs = best.knobs

    out = Path(args.out)
    plot_files: list[str] = []
    if args.plots:
        try:
            from .plots import generate_plots

            paths = generate_plots(
                original_baseline, all_trials, best_trial(all_trials),
                out_dir=out.parent if str(out.parent) else Path("."),
                prefix=out.stem,
            )
            plot_files = [p.name for p in paths]
            print(f"[loadtune] charts: {', '.join(plot_files)}")
        except ImportError:
            print("[loadtune] matplotlib not installed; skipping charts "
                  "(pip install matplotlib)")

    narrative = brain.explain(original_baseline, all_trials)
    report = render_report(original_baseline, all_trials, brain.name, narrative, plot_files)
    out.write_text(report)
    print(f"[loadtune] report written to {out}")

    if args.html:
        from .report_html import render_html_report

        html_out = out.with_suffix(".html")
        html_out.write_text(
            render_html_report(original_baseline, all_trials, brain.name, narrative)
        )
        print(f"[loadtune] interactive report written to {html_out}")

    best = best_trial(all_trials)
    if best and best.throughput > original_baseline.throughput:
        print(
            f"[loadtune] best overall: {best.knobs.label()} — "
            f"{best.throughput / original_baseline.throughput:.2f}x baseline"
        )
        if getattr(args, "apply", False):
            from .apply import generate_apply_snippet
            out_file = generate_apply_snippet(best.knobs, Path(args.workload).parent)
            print(f"[loadtune] apply snippet written to {out_file}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="loadtune",
        description="Agentic profiler & tuner for ML training workloads.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_profile = sub.add_parser("profile", help="profile a workload once")
    _add_common(p_profile)
    p_profile.set_defaults(fn=cmd_profile)

    p_tune = sub.add_parser("tune", help="profile, diagnose, and trial better configs")
    _add_common(p_tune)

    p_tune.add_argument("--max-trials", type=int, default=6)
    p_tune.add_argument(
        "--repeats", type=int, default=1,
        help="measure each config N times; report median with min–max spread",
    )
    p_tune.add_argument(
        "--max-rounds", type=int, default=1,
        help="number of tuning rounds (multi-round optimization)",
    )
    p_tune.add_argument("--out", default="loadtune_report.md")
    p_tune.add_argument(
        "--no-plots", dest="plots", action="store_false", default=True,
        help="skip chart generation (on by default; needs matplotlib)",
    )
    p_tune.add_argument(
        "--html", action="store_true",
        help="also write a self-contained interactive HTML report",
    )
    p_tune.add_argument(
        "--apply", action="store_true",
        help="generate a python snippet to apply the recommended config",
    )
    p_tune.add_argument(
        "--fast", action="store_true",
        help="run trials in-process to avoid process startup overhead",
    )
    p_tune.set_defaults(fn=cmd_tune)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
