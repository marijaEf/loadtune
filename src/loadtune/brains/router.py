from __future__ import annotations

import os

from ..profiler import ProfileResult
from .heuristic import HeuristicBrain
from .llm import LLMBrain
from .google import GoogleBrain

def get_brain_for_profile(baseline: ProfileResult) -> str:
    """Deterministic routing logic.
    
    Returns the name of the brain/route to use:
    - 'heuristic': for simple bottlenecks or environments where LLMs add little value.
    - 'agent': for complex/compute-bound bottlenecks needing deep reasoning and code edits.
    - 'google'/'llm': for standard tuning if agents are not enabled.
    """
    
    # Rule 1: If it's heavily input-bound but data wait is relatively low (<15%), 
    # it's a simple fix (likely just bumping workers by 1 or 2). The heuristic brain 
    # handles this instantly for free.
    if baseline.data_wait_frac < 0.15:
        return "heuristic"
        
    # Rule 2: If we are compute bound (data wait fraction is tiny), the standard 
    # tuning loop (which only adjusts DataLoader knobs) is useless. We must escalate 
    # to the Multi-Agent orchestrator which can edit code (e.g. inject torch.compile).
    if baseline.data_wait_frac < 0.05 and baseline.device != "mps":
        # Note: We skip agents on MPS for now since torch.compile/amp are CUDA focused.
        return "agent"

    # Default fallback: If it's a massive input bottleneck that might involve 
    # complex thread/worker interactions, route to a GenAI brain.
    if os.environ.get("GEMINI_API_KEY"):
        return "google"
    elif os.environ.get("ANTHROPIC_API_KEY"):
        return "llm"
        
    return "heuristic"
