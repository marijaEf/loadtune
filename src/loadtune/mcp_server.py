import json
import sys
from typing import List, Dict, Any, Optional

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("mcp package is required. pip install mcp")
    sys.exit(1)

from .experiment import run_trial, run_trials, Trial
from .knobs import Knobs
from .profiler import ProfileResult

mcp = FastMCP("loadtune")

@mcp.tool()
def profile_workload(
    workload_path: str, 
    steps: int = 100, 
    workers: int = 0, 
    batch_size: Optional[int] = None
) -> str:
    """Profiles a PyTorch training workload to diagnose data wait vs compute bottlenecks.
    
    Args:
        workload_path: Absolute or relative path to the Python file containing get_workload()
        steps: Number of steps to profile
        workers: Baseline num_workers for the DataLoader
        batch_size: Optional batch size override
        
    Returns:
        JSON string containing the profiling trace and performance metrics.
    """
    knobs = Knobs(num_workers=workers, batch_size=batch_size)
    try:
        result = run_trial(workload_path, knobs, steps=steps, warmup=10)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def trial_configurations(
    workload_path: str, 
    configs: List[Dict[str, Any]], 
    steps: int = 100
) -> str:
    """Runs isolated trials of proposed configurations to measure throughput.
    
    Args:
        workload_path: Path to the Python workload file
        configs: List of dictionaries. Each dict should have a "knobs" dict (e.g. {"num_workers": 2}) and a "reason" string.
        steps: Number of steps to trial
        
    Returns:
        JSON string containing the throughput results of each trial.
    """
    trials = []
    for c in configs:
        knob_dict = c.get("knobs", {})
        knobs = Knobs.from_dict(knob_dict)
        trials.append(Trial(knobs=knobs, reason=c.get("reason", "MCP Trial")))
        
    results_summary = []
    
    def on_progress(i: int, n: int, t: Trial) -> None:
        pass # We could log this if needed

    try:
        run_trials(workload_path, trials, steps=steps, warmup=10, on_progress=on_progress, fast=True)
        for t in trials:
            res = {
                "knobs": t.knobs.to_dict(),
                "reason": t.reason,
            }
            if t.result:
                res["throughput"] = t.result.throughput
                res["data_wait_frac"] = t.result.data_wait_frac
                res["error"] = t.result.error
            results_summary.append(res)
        return json.dumps(results_summary, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

def main():
    mcp.run()

if __name__ == "__main__":
    main()
