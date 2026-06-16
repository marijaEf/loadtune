"""Claude-API brain: reasons over the baseline profile to design experiments.

Falls back to the heuristic brain on any API failure so `--brain llm` never
blocks a tuning run. Set LOADTUNE_LLM_MODEL to override the model.
"""

from __future__ import annotations

import json
import os

from ..experiment import Trial
from ..knobs import KNOB_SPECS, Knobs, worker_candidates
from ..profiler import ProfileResult
from .heuristic import HeuristicBrain

DEFAULT_MODEL = os.environ.get("LOADTUNE_LLM_MODEL", "claude-sonnet-4-6")

SYSTEM = """\
You are a performance engineer tuning a PyTorch training workload.
You receive a baseline profile of a short training run. Propose DataLoader
configurations to trial. You may ONLY set these knobs:

{knob_docs}

Hardware notes:
- device "mps" (Apple Silicon): unified memory, pin_memory is a no-op,
  CPU-side augmentation competes with the GPU for memory bandwidth.
- device "cuda": pin_memory usually helps; workers feed H2D copies.
- num_workers must be between 0 and num_cpus.

Respond with ONLY a JSON object:
{{"diagnosis": "<2-3 sentence diagnosis>",
  "trials": [{{"knobs": {{...}}, "reason": "<why>"}}, ...]}}
"""


class LLMBrain:
    name = "llm"

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self._fallback = HeuristicBrain()
        self._diagnosis: str | None = None

    def propose(self, baseline: ProfileResult, max_trials: int) -> list[Trial]:
        try:
            return self._propose_llm(baseline, max_trials)
        except Exception as e:  # API error, bad JSON, missing key, ...
            print(f"[loadtune] LLM brain failed ({e!r}); falling back to heuristics")
            self.name = "llm (failed; heuristic fallback)"
            return self._fallback.propose(baseline, max_trials)

    def _propose_llm(self, baseline: ProfileResult, max_trials: int) -> list[Trial]:
        import anthropic

        knob_docs = "\n".join(
            f"- {s.name}: {s.description}" for s in KNOB_SPECS.values()
        )
        ceiling = 1.0 / max(1e-6, 1.0 - baseline.data_wait_frac)
        # Programmatic ceiling-aware budget clamping
        if ceiling < 1.05:
            max_trials = min(max_trials, 1)
        elif ceiling < 1.20:
            max_trials = min(max_trials, 2)
        elif ceiling < 1.50:
            max_trials = min(max_trials, 3)

        profile = baseline.to_dict()
        # Step series are large and add nothing to the diagnosis.
        profile.pop("step_data_wait_ms", None)
        profile.pop("step_compute_ms", None)
        user_msg = (
            f"Baseline profile:\n{json.dumps(profile, indent=2)}\n\n"
            f"Sensible num_workers values for this machine: "
            f"{worker_candidates(baseline.num_cpus)}\n"
            f"The data-wait fraction bounds the input-side speedup at "
            f"~{ceiling:.2f}x — no input-pipeline config can beat that.\n"
            f"Budget trials to the opportunity: if the ceiling is below "
            f"~1.2x, one or two trials suffice to claim it; propose more "
            f"only where the profile leaves real uncertainty about which "
            f"config wins. Each trial costs a full measured run.\n"
            f"Propose at most {max_trials} trials, most promising first. "
            f"Do not re-propose the baseline config."
        )
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=self.model,
            max_tokens=1500,
            system=SYSTEM.format(knob_docs=knob_docs),
            messages=[{"role": "user", "content": user_msg}],
        )
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        payload = json.loads(text)
        self._diagnosis = payload.get("diagnosis")

        trials = []
        for t in payload["trials"][:max_trials]:
            knobs = Knobs.from_dict(t["knobs"])
            # Guardrails: never trust the model with invalid configs.
            knobs.num_workers = max(0, min(knobs.num_workers, baseline.num_cpus))
            if knobs.num_workers == 0:
                knobs.prefetch_factor = None
                knobs.persistent_workers = False
            if knobs.num_threads is not None:
                knobs.num_threads = max(1, min(knobs.num_threads, baseline.num_cpus))
            trials.append(Trial(knobs, reason=t.get("reason", "LLM proposal")))
        return trials

    def explain(self, baseline: ProfileResult, trials: list[Trial]) -> str:
        if self._diagnosis:
            return self._diagnosis
        return self._fallback.explain(baseline, trials)
