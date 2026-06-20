# loadtune 🚀

**An agentic profiler and tuner for ML workloads.**

`loadtune` is a deterministic, hardware-aware AI agent that autonomously runs micro-experiments to find the optimal system configuration for your ML pipeline, ensuring your GPUs are never sitting idle.

## The Problem: GPU Starvation

GPUs are incredibly fast, but they often sit idle waiting for the CPU to decode and augment data. 
Tuning PyTorch DataLoaders (`num_workers`, `pin_memory`, `prefetch_factor`, thread constraints) is tedious, undocumented, and hardware-dependent. Guessing these configurations often leads to silent **GPU starvation**, where expensive instances (like A100s) waste 80% of their time waiting for data.

`loadtune` replaces guesswork with empirical measurement. It profiles your code, detects the exact bottleneck (Input-Bound vs Compute-Bound), and tunes the hardware mechanics to maximize samples per second.

### Supported Hardware

`loadtune` is hardware-aware and adjusts its heuristics based on your accelerator:
- **NVIDIA GPUs (CUDA):** Full support. Automatically tracks GPU memory utilization (`--auto-batch`), tests asynchronous transfers (`pin_memory`, `non_blocking`), and handles CUDA OOMs.
- **Apple Silicon (MPS):** Full pipeline tuning. `loadtune` recognizes the unified memory architecture (e.g., skips `pin_memory` as it's a no-op on Mac) and accurately synchronizes the MPS stream for honest compute timings.
- **CPUs:** Full pipeline tuning. Automatically limits `torch.set_num_threads` to prevent contention between DataLoader workers and the main process.

---

## Real-World Results

`loadtune` autonomously found these speedups in under 2 minutes of tuning:

- **NVIDIA A100 (Food101 Vision):** 90% data-wait (Input-bound). Scaled workers and pinned memory. **5.65x speedup** (147 → 830 samples/s).
- **Colab T4 (Lightning CNN):** 95.7% data-wait (Highly input-bound). Handled framework overhead perfectly. **4.25x speedup** (1,477 → 6,279 samples/s).
- **Colab T4 (HuggingFace DistilBERT):** 5.9% data-wait (Compute-bound). Recognized the edge case and applied a mild nudge (`workers=2`, `non_blocking`) for a free **1.06x speedup** without wasting time testing massive worker counts.
- **Apple M2 Pro (Synthetic Vision):** Unified memory constraints. **2.11x speedup** (199 → 421 samples/s).

---

## Getting Started

### 1. Setup

```bash
pip install loadtune
```
Optional dependencies for framework integrations:
```bash
pip install "loadtune[lightning]"  # PyTorch Lightning
pip install "loadtune[nlp]"        # HuggingFace Transformers
pip install "loadtune[all]"        # Everything
```

### 2. How to Run (Three Scenarios)

#### Scenario A: PyTorch Lightning & HuggingFace (Zero Boilerplate)
If you use a high-level framework, `loadtune` extracts the components for you.

**PyTorch Lightning:**
```python
# workloads/my_lightning.py
from loadtune import from_lightning

# ... define module and datamodule ...
def get_workload():
    return from_lightning(my_lightning_module, datamodule=my_datamodule, batch_size=64)
```

**HuggingFace Transformers:**
```python
# workloads/my_hf.py
from loadtune import from_hf_trainer

# ... define model and dataset ...
def get_workload():
    return from_hf_trainer(model, dataset, tokenizer=tokenizer, batch_size=32)
```

Run via CLI using fast-mode (in-process trials to avoid framework import overhead):
```bash
loadtune tune workloads/my_lightning.py --fast
```

#### Scenario B: Native PyTorch (`Workload` API)
If writing custom PyTorch loops, define a `Workload` dataclass that tells `loadtune` how to build your dataset, model, and execute a single training step. See `workloads/synthetic_bottleneck.py` for a full example.

#### Scenario C: The Python API (Notebooks & CI)
You can profile and tune directly from Python scripts without using the CLI:
```python
from loadtune import tune
from loadtune.workload import load_workload

# Load and tune a workload autonomously
workload = load_workload("workloads/my_workload.py")
result = tune(workload, steps=50, max_trials=6, auto_batch=True)

print(f"Best Config: {result.best.knobs.label()} — {result.speedup:.2f}x baseline")
print(result.diagnosis)
```

---

## Advanced Features

- **Auto-Batching (`--auto-batch`)**: If you are compute-bound but your GPU memory utilization is low, `loadtune` autonomously proposes batch-size doubling until you hit ~80% VRAM utilization. Catches OOMs gracefully.
- **Auto-Apply (`--apply`)**: Generates a `loadtune_apply.py` code snippet containing the best configuration found so you can easily import it into your project.
- **Fast Mode (`--fast`)**: Runs trials in-process instead of spawning fresh subprocesses. Drastically reduces trial startup overhead for massive models.
- **Loss Parity Check**: Dynamically verifies that semantics-changing configurations (like precision or batch size) don't break mathematical convergence.

---

## Next Steps: The Vision

`loadtune` is evolving into the definitive **Agentic SRE (Site Reliability Engineer)** for Machine Learning, split into two core disciplines:

### 1. `loadtune train` (Currently Complete)
**Goal: Optimizing GPU utilization during R&D and model training.**
- ✅ Phase 1: Input-pipeline tuning & worker sweeping.
- ✅ Phase 2: Cloud GPU & asynchronous memory evaluation.
- ✅ Phase 3: GPU memory profiling & automatic batch-scaling.
- ✅ Phase 4: Framework adapters (Lightning, HuggingFace).

### 2. `loadtune serve` (Upcoming Phase 5)
**Goal: Optimizing server costs in production inference workloads.**
- **The Challenge:** Maximize throughput (Requests/sec) without violating strict latency SLAs (e.g., p99 latency < 100ms).
- **The Strategy:** Agentic tuning of inference engines (vLLM, Triton, TorchServe).
- **The Knobs:** Autonomously tuning dynamic batching windows, KV-cache block sizes, quantization precision, and maximum concurrency limits based on synthetic HTTP traffic profiling.

## License
MIT
