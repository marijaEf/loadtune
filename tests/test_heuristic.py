import pytest
from loadtune.knobs import Knobs
from loadtune.profiler import ProfileResult
from loadtune.brains.heuristic import HeuristicBrain

def test_heuristic_propose_input_bound():
    brain = HeuristicBrain()
    baseline = ProfileResult(
        workload="test", knobs=Knobs(num_workers=0).to_dict(),
        device="mps", num_cpus=10, steps=10, batch_size=32, total_s=10.0,
        data_wait_s=5.0, compute_s=5.0, startup_s=0.1, throughput=32.0,
        data_wait_frac=0.5, step_time_p50_ms=100.0, step_time_p90_ms=110.0,
        cpu_util_mean=20.0
    )
    trials = brain.propose(baseline, max_trials=5)
    assert len(trials) > 0
    # Input bound, so it should propose more workers
    assert any(t.knobs.num_workers > 0 for t in trials)

def test_heuristic_propose_compute_bound():
    brain = HeuristicBrain()
    baseline = ProfileResult(
        workload="test", knobs=Knobs(num_workers=2).to_dict(),
        device="mps", num_cpus=10, steps=10, batch_size=32, total_s=10.0,
        data_wait_s=0.1, compute_s=9.9, startup_s=0.1, throughput=32.0,
        data_wait_frac=0.01, step_time_p50_ms=100.0, step_time_p90_ms=101.0,
        cpu_util_mean=80.0
    )
    trials = brain.propose(baseline, max_trials=5)
    # Since it's compute bound and we only tune input pipeline right now, it should propose 0 trials
    assert len(trials) == 0

def test_heuristic_explain():
    brain = HeuristicBrain()
    baseline = ProfileResult(
        workload="test", knobs=Knobs(num_workers=0).to_dict(),
        device="mps", num_cpus=10, steps=10, batch_size=32, total_s=10.0,
        data_wait_s=5.0, compute_s=5.0, startup_s=0.1, throughput=32.0,
        data_wait_frac=0.5, step_time_p50_ms=100.0, step_time_p90_ms=110.0,
        cpu_util_mean=20.0
    )
    trials = brain.propose(baseline, max_trials=5)
    explain = brain.explain(baseline, trials)
    assert "input pipeline" in explain.lower() or "data wait" in explain.lower()
