"""Tunable knob definitions and trial configuration."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class KnobSpec:
    name: str
    description: str
    # Candidate values the search may try. None means "derive at runtime".
    candidates: Optional[list] = None


# Knobs loadtune owns. The harness builds the DataLoader itself, so these can
# be changed without touching the user's training code.
KNOB_SPECS: dict[str, KnobSpec] = {
    "num_workers": KnobSpec(
        "num_workers",
        "DataLoader worker processes. 0 = load in main process.",
        candidates=None,  # derived from cpu_count at runtime
    ),
    "prefetch_factor": KnobSpec(
        "prefetch_factor",
        "Batches prefetched per worker (only valid when num_workers > 0).",
        candidates=[2, 4, 8],
    ),
    "persistent_workers": KnobSpec(
        "persistent_workers",
        "Keep workers alive between epochs (only valid when num_workers > 0).",
        candidates=[True, False],
    ),
    "pin_memory": KnobSpec(
        "pin_memory",
        "Page-locked host memory for faster H2D copies. CUDA only; no-op on "
        "MPS/CPU (unified memory).",
        candidates=[True, False],
    ),
    "batch_size": KnobSpec(
        "batch_size",
        "Per-step batch size. Changes optimization dynamics — only tuned when "
        "explicitly allowed.",
        candidates=None,
    ),
    "num_threads": KnobSpec(
        "num_threads",
        "Intra-op CPU threads for the main process (torch.set_num_threads). "
        "Competes with DataLoader workers for cores; cap it when many "
        "workers are active. None = torch default (all cores).",
        candidates=None,  # derived from cpu_count and num_workers
    ),
    "compile": KnobSpec(
        "compile",
        "JIT compile the model using torch.compile (requires PyTorch 2.0+).",
        candidates=[True, False],
    ),
}


@dataclass
class Knobs:
    """One concrete configuration to trial."""

    num_workers: int = 0
    prefetch_factor: Optional[int] = None  # None -> torch default
    persistent_workers: bool = False
    pin_memory: bool = False
    batch_size: Optional[int] = None  # None -> workload default
    num_threads: Optional[int] = None  # None -> torch default (all cores)
    compile: bool = False

    def loader_kwargs(self) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if self.num_workers > 0:
            kw["persistent_workers"] = self.persistent_workers
            if self.prefetch_factor is not None:
                kw["prefetch_factor"] = self.prefetch_factor
        return kw

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Knobs":
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def label(self) -> str:
        parts = [f"workers={self.num_workers}"]
        if self.num_workers > 0:
            if self.prefetch_factor is not None:
                parts.append(f"prefetch={self.prefetch_factor}")
            if self.persistent_workers:
                parts.append("persistent")
        if self.pin_memory:
            parts.append("pin")
        if self.batch_size is not None:
            parts.append(f"bs={self.batch_size}")
        if self.num_threads is not None:
            parts.append(f"threads={self.num_threads}")
        if getattr(self, "compile", False):
            parts.append("compiled")
        return ", ".join(parts)


def worker_candidates(num_cpus: int) -> list[int]:
    """Reasonable num_workers values for this machine."""
    cands = {0, 1, 2, 4}
    if num_cpus >= 8:
        cands.add(num_cpus // 2)
        cands.add(num_cpus - 2)
    return sorted(c for c in cands if 0 <= c <= num_cpus)
