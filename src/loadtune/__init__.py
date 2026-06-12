"""loadtune — agentic profiler and tuner for ML training workloads.

Profiles a training loop, splits step time into data-wait vs compute,
diagnoses input-pipeline bottlenecks, and runs short experiments to find
better DataLoader / training configs (e.g. num_workers=2 instead of 4).
"""

__version__ = "0.2.0"

from .knobs import Knobs, KNOB_SPECS
from .workload import Workload
from .profiler import ProfileResult, profile_session

__all__ = ["Knobs", "KNOB_SPECS", "Workload", "ProfileResult", "profile_session"]
