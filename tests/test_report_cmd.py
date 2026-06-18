import argparse
import json
from pathlib import Path
from loadtune.cli import cmd_report

def test_cmd_report_generates_outputs(tmp_path: Path):
    raw_data = {
        "baseline": {
            "workload": "test_workload",
            "knobs": {"num_workers": 0},
            "device": "cpu",
            "num_cpus": 4,
            "steps": 10,
            "batch_size": 32,
            "total_s": 1.0,
            "data_wait_s": 0.5,
            "compute_s": 0.5,
            "startup_s": 0.1,
            "throughput": 100.0,
            "data_wait_frac": 0.5,
            "step_time_p50_ms": 100.0,
            "step_time_p90_ms": 110.0,
            "cpu_util_mean": 50.0,
            "step_data_wait_ms": [50.0]*10,
            "step_compute_ms": [50.0]*10,
            "losses": [1.0]*10,
            "error": None,
            "repeats": 1,
            "throughput_min": None,
            "throughput_max": None
        },
        "trials": [
            {
                "knobs": {"num_workers": 2},
                "reason": "Test trial",
                "result": {
                    "workload": "test_workload",
                    "knobs": {"num_workers": 2},
                    "device": "cpu",
                    "num_cpus": 4,
                    "steps": 10,
                    "batch_size": 32,
                    "total_s": 0.5,
                    "data_wait_s": 0.0,
                    "compute_s": 0.5,
                    "startup_s": 0.1,
                    "throughput": 200.0,
                    "data_wait_frac": 0.0,
                    "step_time_p50_ms": 50.0,
                    "step_time_p90_ms": 50.0,
                    "cpu_util_mean": 80.0,
                    "step_data_wait_ms": [0.0]*10,
                    "step_compute_ms": [50.0]*10,
                    "losses": [1.0]*10,
                    "error": None,
                    "repeats": 1,
                    "throughput_min": None,
                    "throughput_max": None
                }
            }
        ],
        "brain_name": "heuristic",
        "narrative": "This is a test narrative."
    }

    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(raw_data))

    out_file = tmp_path / "report.md"

    args = argparse.Namespace(
        raw_file=str(raw_file),
        out=str(out_file),
        plots=False,
        html=True
    )

    ret = cmd_report(args)
    assert ret == 0

    assert out_file.exists()
    md_content = out_file.read_text()
    assert "This is a test narrative" in md_content
    assert "100.0 samples/s" in md_content
    assert "200.0 samples/s" in md_content

    html_file = tmp_path / "report.html"
    assert html_file.exists()
    html_content = html_file.read_text()
    assert "This is a test narrative" in html_content
    assert "100.0" in html_content
