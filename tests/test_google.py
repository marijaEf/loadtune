import pytest
from loadtune.brains.google import GoogleBrain
from loadtune.profiler import ProfileResult
from loadtune.knobs import Knobs

class MockResponse:
    def __init__(self, text):
        self.text = text

class MockModels:
    def generate_content(self, model, contents, config=None):
        return MockResponse("""
        {
            "diagnosis": "Mocked diagnosis.",
            "trials": [
                {
                    "knobs": {"num_workers": 2, "prefetch_factor": 2},
                    "reason": "Test reason"
                }
            ]
        }
        """)

class MockClient:
    def __init__(self):
        self.models = MockModels()

def test_google_brain_propose(monkeypatch):
    import google.genai as genai
    monkeypatch.setattr(genai, "Client", MockClient)
    
    brain = GoogleBrain()
    baseline = ProfileResult(
        workload="test", knobs=Knobs(num_workers=0).to_dict(),
        device="mps", num_cpus=10, steps=10, batch_size=32, total_s=10.0,
        data_wait_s=5.0, compute_s=5.0, startup_s=0.1, throughput=32.0,
        data_wait_frac=0.5, step_time_p50_ms=100.0, step_time_p90_ms=110.0,
        cpu_util_mean=20.0
    )
    
    trials = brain.propose(baseline, max_trials=2)
    assert len(trials) == 1
    assert trials[0].knobs.num_workers == 2
    assert brain._diagnosis == "Mocked diagnosis."
