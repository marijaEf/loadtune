"""Tests for Epic 3: Memory Profiling & Batch Size Auto-Scaling."""

import dataclasses
import math
from typing import Optional
from unittest.mock import patch

import pytest

from loadtune.knobs import Knobs
from loadtune.profiler import ProfileResult


# ---------------------------------------------------------------------------
# Helpers: build ProfileResult fixtures
# ---------------------------------------------------------------------------

def _make_profile(
    *,
    data_wait_frac: float = 0.02,
    gpu_mem_peak_mb: Optional[float] = None,
    gpu_mem_total_mb: Optional[float] = None,
    gpu_mem_utilization: Optional[float] = None,
    device: str = "cuda",
    batch_size: int = 64,
    throughput: float = 500.0,
    cpu_util_mean: Optional[float] = 40.0,
    num_cpus: int = 12,
    step_time_p50_ms: float = 100.0,
    step_time_p90_ms: float = 110.0,
    **overrides,
) -> ProfileResult:
    """Build a minimal ProfileResult for testing."""
    kwargs = dict(
        workload="test_workload",
        knobs=Knobs(batch_size=batch_size).to_dict(),
        device=device,
        num_cpus=num_cpus,
        steps=50,
        batch_size=batch_size,
        total_s=10.0,
        data_wait_s=round(10.0 * data_wait_frac, 4),
        compute_s=round(10.0 * (1 - data_wait_frac), 4),
        startup_s=0.01,
        throughput=throughput,
        data_wait_frac=data_wait_frac,
        step_time_p50_ms=step_time_p50_ms,
        step_time_p90_ms=step_time_p90_ms,
        cpu_util_mean=cpu_util_mean,
        gpu_mem_peak_mb=gpu_mem_peak_mb,
        gpu_mem_total_mb=gpu_mem_total_mb,
        gpu_mem_utilization=gpu_mem_utilization,
    )
    kwargs.update(overrides)
    return ProfileResult(**kwargs)


# ===================================================================
# 1. ProfileResult has memory fields
# ===================================================================

class TestProfileResultMemoryFields:
    """Verify the new gpu_mem_* fields exist and serialize correctly."""

    def test_fields_exist(self):
        pr = _make_profile()
        assert hasattr(pr, "gpu_mem_peak_mb")
        assert hasattr(pr, "gpu_mem_total_mb")
        assert hasattr(pr, "gpu_mem_utilization")

    def test_fields_default_none(self):
        pr = _make_profile()
        assert pr.gpu_mem_peak_mb is None
        assert pr.gpu_mem_total_mb is None
        assert pr.gpu_mem_utilization is None

    def test_fields_populated(self):
        pr = _make_profile(
            gpu_mem_peak_mb=4096.0,
            gpu_mem_total_mb=16384.0,
            gpu_mem_utilization=0.25,
        )
        assert pr.gpu_mem_peak_mb == 4096.0
        assert pr.gpu_mem_total_mb == 16384.0
        assert pr.gpu_mem_utilization == 0.25

    def test_to_dict_includes_memory(self):
        pr = _make_profile(
            gpu_mem_peak_mb=2048.0,
            gpu_mem_total_mb=8192.0,
            gpu_mem_utilization=0.25,
        )
        d = pr.to_dict()
        assert d["gpu_mem_peak_mb"] == 2048.0
        assert d["gpu_mem_total_mb"] == 8192.0
        assert d["gpu_mem_utilization"] == 0.25

    def test_to_dict_memory_none_when_absent(self):
        pr = _make_profile()
        d = pr.to_dict()
        assert d["gpu_mem_peak_mb"] is None
        assert d["gpu_mem_total_mb"] is None
        assert d["gpu_mem_utilization"] is None


# ===================================================================
# 2. Heuristic brain: batch_size doubling
# ===================================================================

class TestBatchSizeDoubling:
    """Verify the heuristic brain proposes batch_size changes correctly."""

    def test_proposes_doubling_when_low_utilization(self):
        from loadtune.heuristic import HeuristicBrain

        baseline = _make_profile(
            data_wait_frac=0.02,       # compute-bound
            gpu_mem_utilization=0.30,   # only 30% GPU memory used
            batch_size=64,
        )
        brain = HeuristicBrain()
        trials = brain.propose(baseline, max_trials=20, auto_batch=True)

        # Should have at least one trial with a larger batch_size
        bs_trials = [t for t in trials if t.knobs.batch_size is not None and t.knobs.batch_size > 64]
        assert len(bs_trials) >= 1, f"Expected batch_size trials, got: {[t.knobs.label() for t in trials]}"
        
        # First batch_size trial should double to 128
        assert bs_trials[0].knobs.batch_size == 128

    def test_proposes_up_to_4x_when_very_low_utilization(self):
        from loadtune.heuristic import HeuristicBrain

        baseline = _make_profile(
            data_wait_frac=0.02,
            gpu_mem_utilization=0.15,   # only 15% — room for 2x and 4x
            batch_size=32,
        )
        brain = HeuristicBrain()
        trials = brain.propose(baseline, max_trials=20, auto_batch=True)

        bs_trials = [t for t in trials if t.knobs.batch_size is not None and t.knobs.batch_size > 32]
        batch_sizes = [t.knobs.batch_size for t in bs_trials]
        assert 64 in batch_sizes, f"Expected bs=64, got {batch_sizes}"
        assert 128 in batch_sizes, f"Expected bs=128, got {batch_sizes}"

    def test_stops_at_high_utilization(self):
        from loadtune.heuristic import HeuristicBrain

        baseline = _make_profile(
            data_wait_frac=0.02,
            gpu_mem_utilization=0.85,   # already 85% — no room
            batch_size=256,
        )
        brain = HeuristicBrain()
        trials = brain.propose(baseline, max_trials=20, auto_batch=True)

        bs_trials = [t for t in trials if t.knobs.batch_size is not None and t.knobs.batch_size > 256]
        assert len(bs_trials) == 0, f"Should not propose larger bs at 85% util, got: {[t.knobs.batch_size for t in bs_trials]}"

    def test_no_doubling_when_input_bound(self):
        from loadtune.heuristic import HeuristicBrain

        baseline = _make_profile(
            data_wait_frac=0.45,        # input-bound
            gpu_mem_utilization=0.20,   # low memory, but input-bound
            batch_size=64,
        )
        brain = HeuristicBrain()
        trials = brain.propose(baseline, max_trials=20, auto_batch=True)

        bs_trials = [t for t in trials if t.knobs.batch_size is not None and t.knobs.batch_size > 64]
        assert len(bs_trials) == 0, "Should not propose larger batch_size when input-bound"

    def test_no_doubling_without_auto_batch_flag(self):
        from loadtune.heuristic import HeuristicBrain

        baseline = _make_profile(
            data_wait_frac=0.02,
            gpu_mem_utilization=0.20,
            batch_size=64,
        )
        brain = HeuristicBrain()
        # auto_batch=False (default)
        trials = brain.propose(baseline, max_trials=20, auto_batch=False)

        bs_trials = [t for t in trials if t.knobs.batch_size is not None and t.knobs.batch_size > 64]
        assert len(bs_trials) == 0, "Should not propose batch_size without --auto-batch"

    def test_no_doubling_on_cpu_mps(self):
        from loadtune.heuristic import HeuristicBrain

        for device in ["cpu", "mps"]:
            baseline = _make_profile(
                data_wait_frac=0.02,
                gpu_mem_utilization=None,  # no memory data on CPU/MPS
                device=device,
                batch_size=64,
            )
            brain = HeuristicBrain()
            trials = brain.propose(baseline, max_trials=20, auto_batch=True)

            bs_trials = [t for t in trials if t.knobs.batch_size is not None and t.knobs.batch_size > 64]
            assert len(bs_trials) == 0, f"Should not propose batch_size on {device}"


# ===================================================================
# 3. OOM graceful recovery
# ===================================================================

class TestOOMRecovery:
    """Verify that OOM errors produce a graceful error dict, not a crash."""

    def test_oom_returns_error_dict(self):
        """Simulate an OOM in profile_session by mocking the training loop."""
        from loadtune.profiler import ProfileResult

        # Build a ProfileResult as if OOM happened
        pr = ProfileResult(
            workload="test",
            knobs=Knobs(batch_size=512).to_dict(),
            device="cuda",
            num_cpus=12,
            steps=0,
            batch_size=512,
            total_s=0,
            data_wait_s=0,
            compute_s=0,
            startup_s=0,
            throughput=0,
            data_wait_frac=0,
            step_time_p50_ms=0,
            step_time_p90_ms=0,
            cpu_util_mean=None,
            error="OOM: batch_size=512 exceeded GPU memory",
        )
        assert pr.error is not None
        assert "OOM" in pr.error
        assert pr.steps == 0
        assert pr.throughput == 0

    def test_oom_in_trial_dict(self):
        """Verify that an OOM error in a trial result marks the trial as failed."""
        from loadtune.experiment import Trial

        trial = Trial(
            knobs=Knobs(batch_size=512),
            reason="testing OOM",
            result={"error": "OOM: batch_size=512 exceeded GPU memory"},
        )
        assert not trial.ok
        assert trial.throughput == 0.0


# ===================================================================
# 4. Explain method includes memory info
# ===================================================================

class TestExplainMemory:
    """Verify the explain() method includes GPU memory info when available."""

    def test_explain_includes_memory_utilization(self):
        from loadtune.heuristic import HeuristicBrain

        baseline = _make_profile(
            data_wait_frac=0.02,
            gpu_mem_utilization=0.45,
            batch_size=64,
        )
        brain = HeuristicBrain()
        explanation = brain.explain(baseline, [])
        assert "45%" in explanation, f"Expected memory info in explanation, got: {explanation}"

    def test_explain_no_memory_on_cpu(self):
        from loadtune.heuristic import HeuristicBrain

        baseline = _make_profile(
            data_wait_frac=0.02,
            gpu_mem_utilization=None,
            device="cpu",
        )
        brain = HeuristicBrain()
        explanation = brain.explain(baseline, [])
        assert "GPU memory" not in explanation


# ===================================================================
# 5. GPU-specific test (skipped without CUDA)
# ===================================================================

@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(),
    reason="CUDA not available",
)
class TestGPUMemoryMeasurement:
    """On a CUDA machine, verify memory fields are actually populated."""

    def test_memory_fields_populated_on_cuda(self):
        import torch
        from loadtune.profiler import profile_session
        from loadtune.workload import Workload

        # Minimal workload that allocates some GPU memory
        def make_dataset():
            return torch.utils.data.TensorDataset(
                torch.randn(200, 3, 32, 32),
                torch.randint(0, 10, (200,)),
            )

        def make_model():
            return torch.nn.Linear(3 * 32 * 32, 10)

        def train_step(model, optimizer, batch, device):
            x, y = batch
            x = x.to(device).view(x.size(0), -1)
            y = y.to(device)
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            return loss

        w = Workload(
            name="tiny_cuda_test",
            make_dataset=make_dataset,
            make_model=make_model,
            make_optimizer=lambda m: torch.optim.SGD(m.parameters(), lr=0.01),
            train_step=train_step,
            default_batch_size=16,
        )
        result = profile_session(w, Knobs(), steps=5, warmup=2)
        assert result.error is None, f"Profiling failed: {result.error}"
        assert result.gpu_mem_peak_mb is not None, "gpu_mem_peak_mb should be populated on CUDA"
        assert result.gpu_mem_total_mb is not None, "gpu_mem_total_mb should be populated on CUDA"
        assert result.gpu_mem_utilization is not None, "gpu_mem_utilization should be populated on CUDA"
        assert result.gpu_mem_peak_mb > 0, "Peak memory should be > 0"
        assert result.gpu_mem_total_mb > 0, "Total memory should be > 0"
        assert 0 < result.gpu_mem_utilization <= 1.0, f"Utilization should be in (0,1], got {result.gpu_mem_utilization}"
