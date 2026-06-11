"""Workload interface.

A workload file is any Python file exposing `get_workload() -> Workload`.
loadtune owns DataLoader construction (so it can tune it); the workload
supplies the dataset, the model, and a single training step.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass
class Workload:
    name: str
    # Returns a torch.utils.data.Dataset (called once per trial, in a fresh process).
    make_dataset: Callable[[], Any]
    # Returns a fresh nn.Module (not yet moved to device).
    make_model: Callable[[], Any]
    # (model, optimizer, batch, device) -> loss tensor. Must call
    # loss.backward() and optimizer.step()/zero_grad() itself.
    train_step: Callable[..., Any]
    # (model) -> optimizer
    make_optimizer: Callable[[Any], Any]
    default_batch_size: int = 32
    # Optional collate_fn passed through to the DataLoader.
    collate_fn: Optional[Callable] = None
    description: str = ""


def load_workload(path: str) -> Workload:
    """Import `get_workload()` from a workload file path."""
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"workload file not found: {path}")
    # Import under the plain filename with its directory on sys.path.
    # Crucial on macOS/Windows: DataLoader workers are *spawned* and must
    # re-import this module by name to unpickle the dataset. sys.path is
    # propagated to spawned children; a synthetic module name is not.
    parent = str(p.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location(p.stem, p)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[p.stem] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "get_workload"):
        raise AttributeError(f"{path} must define get_workload() -> Workload")
    wl = mod.get_workload()
    if not isinstance(wl, Workload):
        raise TypeError(f"{path}: get_workload() must return loadtune.Workload")
    return wl
