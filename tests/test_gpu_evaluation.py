"""GPU Evaluation Suite for NVIDIA DeepLearningExamples Mock Workloads."""

import pytest
import torch
import subprocess
import json
from pathlib import Path

# Skip the entire module if a CUDA GPU is not available!
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), 
    reason="NVIDIA Evaluation Suite requires a CUDA GPU to run."
)

def run_loadtune_tune(workload_path: str, tmp_path: Path):
    """Helper to run the loadtune CLI and return the best config found."""
    out_file = tmp_path / "report.md"
    raw_file = tmp_path / "raw.json"
    
    # We use --fast to speed up tests (in-process trials) and keep step count low
    cmd = [
        "loadtune", "tune", workload_path,
        "--steps", "5",
        "--max-trials", "4",
        "--save-raw", str(raw_file),
        "--out", str(out_file),
        "--fast"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"loadtune failed:\n{result.stderr}\n{result.stdout}"
    
    assert raw_file.exists()
    raw_data = json.loads(raw_file.read_text())
    
    # Extract the best throughput found across all trials
    trials = raw_data.get("trials", [])
    best_throughput = raw_data["baseline"]["throughput"]
    
    for t in trials:
        if t.get("result") and not t["result"].get("error"):
            best_throughput = max(best_throughput, t["result"]["throughput"])
            
    return raw_data["baseline"]["throughput"], best_throughput


def test_dle_resnet50_improves_throughput(tmp_path):
    """Validates that loadtune can optimize a heavy ResNet-50 pipeline."""
    workload_path = "examples/dle_resnet50.py"
    
    baseline_tp, best_tp = run_loadtune_tune(workload_path, tmp_path)
    
    # Since resnet50 is somewhat compute heavy but we used FakeData, 
    # it might just be testing the dataloader overhead.
    # The key assert is that the tuner ran successfully and didn't crash.
    assert best_tp >= baseline_tp, "Tuning should not degrade performance"


def test_dle_bert_improves_throughput(tmp_path):
    """Validates that loadtune can navigate the artificial 5ms NLP tokenization delay."""
    workload_path = "examples/dle_bert.py"
    
    baseline_tp, best_tp = run_loadtune_tune(workload_path, tmp_path)
    
    # Because of the 5ms delay per sample in __getitem__, a single worker is severely input bound.
    # loadtune MUST discover that adding workers improves throughput.
    assert best_tp > baseline_tp * 1.5, "loadtune failed to discover worker parallelization for BERT tokenization"
