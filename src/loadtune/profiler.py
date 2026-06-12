"""Instrumented training-loop profiler.

Core idea: a training step has two phases we can separate cheaply —
  data wait : time blocked in `next(dataloader_iter)`
  compute   : forward/backward/optimizer, measured with a device sync

A high data-wait fraction means the accelerator is starved by the input
pipeline; that is the signal the brains reason over.
"""

from __future__ import annotations

import dataclasses
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from .knobs import Knobs
from .workload import Workload

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


@dataclass
class ProfileResult:
    workload: str
    knobs: dict[str, Any]
    device: str
    num_cpus: int
    steps: int
    batch_size: int
    total_s: float
    data_wait_s: float
    compute_s: float
    startup_s: float  # dataloader iterator creation (worker spawn cost)
    throughput: float  # samples / s (measured steps only)
    data_wait_frac: float
    step_time_p50_ms: float
    step_time_p90_ms: float
    cpu_util_mean: Optional[float]  # system-wide %, all cores
    # Per measured step, in ms — used for plotting.
    step_data_wait_ms: list[float] = field(default_factory=list)
    step_compute_ms: list[float] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def profile_session(
    workload: Workload,
    knobs: Knobs,
    steps: int = 100,
    warmup: int = 10,
    seed: int = 0,
) -> ProfileResult:
    """Run `warmup + steps` training steps under `knobs` and measure."""
    torch.manual_seed(seed)
    if knobs.num_threads is not None:
        torch.set_num_threads(max(1, knobs.num_threads))
    device = pick_device()
    num_cpus = os.cpu_count() or 1
    batch_size = knobs.batch_size or workload.default_batch_size

    dataset = workload.make_dataset()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=workload.collate_fn,
        drop_last=True,
        **knobs.loader_kwargs(),
    )
    model = workload.make_model().to(device)
    model.train()
    optimizer = workload.make_optimizer(model)

    if psutil:
        psutil.cpu_percent(interval=None)  # prime the counter

    t0 = time.perf_counter()
    it = iter(loader)  # worker spawn happens here
    startup_s = time.perf_counter() - t0

    data_wait = 0.0
    compute = 0.0
    step_times: list[float] = []
    step_waits_ms: list[float] = []
    step_computes_ms: list[float] = []
    cpu_samples: list[float] = []
    done = 0
    total_needed = warmup + steps

    while done < total_needed:
        t_fetch = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            continue
        t_got = time.perf_counter()

        workload.train_step(model, optimizer, batch, device)
        device_sync(device)
        t_done = time.perf_counter()

        done += 1
        if done > warmup:
            data_wait += t_got - t_fetch
            compute += t_done - t_got
            step_times.append((t_done - t_fetch) * 1000)
            step_waits_ms.append(round((t_got - t_fetch) * 1000, 3))
            step_computes_ms.append(round((t_done - t_got) * 1000, 3))
            if psutil and done % 5 == 0:
                cpu_samples.append(psutil.cpu_percent(interval=None))

    total = data_wait + compute
    return ProfileResult(
        workload=workload.name,
        knobs=knobs.to_dict(),
        device=device.type,
        num_cpus=num_cpus,
        steps=steps,
        batch_size=batch_size,
        total_s=round(total, 4),
        data_wait_s=round(data_wait, 4),
        compute_s=round(compute, 4),
        startup_s=round(startup_s, 4),
        throughput=round(steps * batch_size / total, 2) if total > 0 else 0.0,
        data_wait_frac=round(data_wait / total, 4) if total > 0 else 0.0,
        step_time_p50_ms=round(statistics.median(step_times), 2) if step_times else 0.0,
        step_time_p90_ms=(
            round(statistics.quantiles(step_times, n=10)[8], 2)
            if len(step_times) >= 10
            else 0.0
        ),
        cpu_util_mean=(
            round(statistics.mean(cpu_samples), 1) if cpu_samples else None
        ),
        step_data_wait_ms=step_waits_ms,
        step_compute_ms=step_computes_ms,
    )
