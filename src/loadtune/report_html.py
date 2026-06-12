"""Self-contained interactive HTML report (Plotly via CDN).

One file, shareable: stats, the trials table with the brain's reasoning,
and two interactive charts — throughput per config (with spread bars when
--repeats was used) and the per-step timeline, baseline vs tuned.
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from typing import Optional

from .experiment import Trial, best_trial
from .knobs import Knobs
from .profiler import ProfileResult

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"

CSS = """
body { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
       max-width: 980px; margin: 2rem auto; padding: 0 1rem; color: #1c2733; }
h1 { font-size: 1.5rem; } h2 { font-size: 1.15rem; margin-top: 2rem; }
.meta { color: #5f6b7a; font-size: 0.9rem; }
.stats { display: flex; gap: 2rem; flex-wrap: wrap; margin: 1rem 0; }
.stat .v { font-size: 1.4rem; font-weight: 600; }
.stat .k { color: #5f6b7a; font-size: 0.85rem; }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
th, td { text-align: left; padding: 0.5rem 0.75rem; border-bottom: 1px solid #e3e8ee; }
th { color: #5f6b7a; font-weight: 600; }
td.cfg { font-family: ui-monospace, Menlo, monospace; white-space: nowrap; }
tr.best { background: #f0f7f0; }
.verdict { background: #f0f7f0; border-left: 4px solid #5a9a5a;
           padding: 0.75rem 1rem; margin-top: 1.5rem; }
.chart { margin-top: 1rem; }
"""


def _fmt_thr(r: dict) -> str:
    s = f"{r['throughput']:.1f}"
    if r.get("repeats", 1) > 1 and r.get("throughput_min") is not None:
        s += f" ({r['throughput_min']:.1f}–{r['throughput_max']:.1f})"
    return s


def render_html_report(
    baseline: ProfileResult,
    trials: list[Trial],
    brain_name: str,
    narrative: str,
) -> str:
    best = best_trial(trials)
    base_label = Knobs.from_dict(baseline.knobs).label()

    rows = []
    for t in trials:
        if t.ok:
            r = t.result
            speedup = r["throughput"] / baseline.throughput if baseline.throughput else 0
            cls = ' class="best"' if t is best else ""
            rows.append(
                f"<tr{cls}><td class='cfg'>{html.escape(t.knobs.label())}</td>"
                f"<td>{_fmt_thr(r)}</td><td>{speedup:.2f}x</td>"
                f"<td>{r['data_wait_frac']:.1%}</td>"
                f"<td>{html.escape(t.reason)}</td></tr>"
            )
        else:
            err = str((t.result or {}).get("error", "unknown")).strip().splitlines()[-1][:90]
            rows.append(
                f"<tr><td class='cfg'>{html.escape(t.knobs.label())}</td>"
                f"<td>failed</td><td>—</td><td>—</td><td>{html.escape(err)}</td></tr>"
            )

    # Chart payloads
    bar = {"labels": [f"baseline: {base_label}"], "thr": [baseline.throughput],
           "lo": [baseline.throughput_min], "hi": [baseline.throughput_max]}
    for t in trials:
        if t.ok:
            bar["labels"].append(t.knobs.label())
            bar["thr"].append(t.result["throughput"])
            bar["lo"].append(t.result.get("throughput_min"))
            bar["hi"].append(t.result.get("throughput_max"))

    timeline = {"base_label": base_label,
                "base_wait": baseline.step_data_wait_ms,
                "base_comp": baseline.step_compute_ms,
                "best_label": None, "best_wait": [], "best_comp": []}
    if best and best.ok and best.result.get("step_data_wait_ms"):
        timeline["best_label"] = best.knobs.label()
        timeline["best_wait"] = best.result["step_data_wait_ms"]
        timeline["best_comp"] = best.result["step_compute_ms"]

    if best and baseline.throughput and best.throughput > baseline.throughput * 1.02:
        verdict = (
            f"Recommended config: <code>{html.escape(best.knobs.label())}</code> — "
            f"<b>{best.throughput / baseline.throughput:.2f}x</b> baseline throughput "
            f"({baseline.throughput:.1f} → {best.throughput:.1f} samples/s)."
        )
    elif best:
        verdict = ("No trialed config beat the baseline by more than 2%; "
                   "the baseline appears close to optimal on this machine.")
    else:
        verdict = "All trials failed."

    payload = json.dumps({"bar": bar, "timeline": timeline})

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>loadtune — {html.escape(baseline.workload)}</title>
<script src="{PLOTLY_CDN}"></script>
<style>{CSS}</style></head><body>
<h1>loadtune report — {html.escape(baseline.workload)}</h1>
<p class="meta">{datetime.now():%Y-%m-%d %H:%M} · device <code>{baseline.device}</code> ·
{baseline.num_cpus} CPUs · brain <code>{html.escape(brain_name)}</code> ·
{baseline.steps} steps/trial{f" · {baseline.repeats} repeats" if baseline.repeats > 1 else ""}</p>
<h2>Diagnosis</h2><p>{html.escape(narrative).replace("**", "")}</p>
<div class="stats">
<div class="stat"><div class="v">{baseline.throughput:.1f}</div><div class="k">baseline samples/s</div></div>
<div class="stat"><div class="v">{baseline.data_wait_frac:.1%}</div><div class="k">baseline data wait</div></div>
<div class="stat"><div class="v">{baseline.step_time_p50_ms:.1f} ms</div><div class="k">step p50</div></div>
<div class="stat"><div class="v">{1.0 / max(1e-6, 1.0 - baseline.data_wait_frac):.2f}x</div><div class="k">input-side ceiling</div></div>
</div>
<h2>Trials</h2>
<table><thead><tr><th>config</th><th>samples/s</th><th>vs baseline</th>
<th>data wait</th><th>proposed because</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<div class="verdict">{verdict}</div>
<h2>Throughput by config</h2><div id="bar" class="chart"></div>
<h2>Step timeline</h2>
<div id="tl_base" class="chart"></div>
<div id="tl_best" class="chart"></div>
<script>
const D = {payload};
const err = D.bar.hi.map((h, i) => h == null ? 0 : h - D.bar.thr[i]);
const errm = D.bar.lo.map((l, i) => l == null ? 0 : D.bar.thr[i] - l);
Plotly.newPlot("bar", [{{
  x: D.bar.labels, y: D.bar.thr, type: "bar", marker: {{color: "#4878a8"}},
  error_y: {{type: "data", array: err, arrayminus: errm, visible: true}}
}}], {{margin: {{t: 10}}, yaxis: {{title: "samples/s"}}}}, {{displayModeBar: false, responsive: true}});

const totals = (w, c) => w.map((v, i) => v + c[i]);
const ymax = 1.05 * Math.max(
  ...totals(D.timeline.base_wait, D.timeline.base_comp),
  ...(D.timeline.best_label
      ? totals(D.timeline.best_wait, D.timeline.best_comp) : [0])
);
function stacked(div, wait, comp, title) {{
  const steps = wait.map((_, i) => i + 1);
  Plotly.newPlot(div, [
    {{x: steps, y: comp, stackgroup: "s", name: "compute",
      line: {{color: "#4878a8"}}}},
    {{x: steps, y: wait, stackgroup: "s", name: "data wait",
      line: {{color: "#d9534f"}}}},
  ], {{margin: {{t: 32}}, height: 260, title: {{text: title, font: {{size: 13}}}},
      xaxis: {{title: "step"}}, yaxis: {{title: "ms", range: [0, ymax]}}}},
    {{displayModeBar: false, responsive: true}});
}}
stacked("tl_base", D.timeline.base_wait, D.timeline.base_comp,
        "baseline: " + D.timeline.base_label);
if (D.timeline.best_label) {{
  stacked("tl_best", D.timeline.best_wait, D.timeline.best_comp,
          "tuned: " + D.timeline.best_label);
}}
</script>
</body></html>
"""
