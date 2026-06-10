"""Subprocess entry point: run one profiled trial and print JSON to stdout.

Each trial runs in a fresh process so worker pools, MPS allocator state, and
file caches from one config cannot pollute the next measurement.

Usage (internal): python -m loadtune._trial <workload.py> '<knobs json>' <steps> <warmup>
"""

from __future__ import annotations

import json
import sys
import traceback


def main() -> int:
    workload_path, knobs_json, steps, warmup = (
        sys.argv[1],
        sys.argv[2],
        int(sys.argv[3]),
        int(sys.argv[4]),
    )
    from .knobs import Knobs
    from .profiler import profile_session
    from .workload import load_workload

    try:
        wl = load_workload(workload_path)
        knobs = Knobs.from_dict(json.loads(knobs_json))
        result = profile_session(wl, knobs, steps=steps, warmup=warmup)
        print("LOADTUNE_RESULT " + json.dumps(result.to_dict()))
        return 0
    except Exception:
        err = {"error": traceback.format_exc(limit=5)}
        print("LOADTUNE_RESULT " + json.dumps(err))
        return 1


if __name__ == "__main__":
    sys.exit(main())
