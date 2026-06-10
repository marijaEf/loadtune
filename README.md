# loadtune

**Agentic profiler & tuner for ML training workloads.** loadtune profiles your training loop, splits every step into *data wait* vs *compute*, diagnoses input-pipeline bottlenecks, and runs short isolated experiments to find a better config — for example proposing `num_workers=2` instead of `num_workers=4` when the extra workers are pure overhead.

Born out of a master-thesis pain: hand-tuning dataloaders is slow, boring, and machine-specific. So it got automated.

## How it works

```
baseline profile ──▶ brain (diagnosis + experiment plan) ──▶ trials ──▶ report
     │                      │                                  │
     │              heuristic rules or                 each trial runs in a
     │              Claude API reasoning               fresh subprocess for
     └─ data-wait vs compute split,                    clean measurements
        CPU util, step-time percentiles
```

Every proposed config is **verified by measurement**, never just suggested. The report shows throughput per trial, speedup vs baseline, and why each config was tried.

## Install

```bash
pip install -e ".[vision]"        # core + torchvision workloads
pip install -e ".[all]"           # + Claude API brain + NLP workload
```

## Quickstart (no downloads needed)

```bash
# 1. Profile the deliberately input-bound synthetic workload
loadtune profile workloads/synthetic_bottleneck.py --steps 50

# 2. Let the agent tune it
loadtune tune workloads/synthetic_bottleneck.py --steps 50 --out report.md
```

Then the real one:

```bash
loadtune tune workloads/resnet50_cifar.py --steps 100 --out resnet_report.md
```

## Choosing the brain

```bash
loadtune tune <workload> --brain heuristic   # deterministic rules, free, offline
loadtune tune <workload> --brain llm         # Claude reasons over the profile
loadtune tune <workload> --brain auto        # llm if ANTHROPIC_API_KEY is set
```

The LLM brain (set `ANTHROPIC_API_KEY`) receives the baseline profile and hardware context, returns a diagnosis and an experiment plan as JSON, and is sandboxed by guardrails (invalid configs are clamped; any API failure falls back to heuristics). `LOADTUNE_LLM_MODEL` overrides the model.

## Writing your own workload

A workload is one Python file with a `get_workload()` function — loadtune owns the DataLoader so it can tune it; you supply dataset, model, and a train step:

```python
from loadtune import Workload

def get_workload() -> Workload:
    return Workload(
        name="my_model",
        make_dataset=...,            # () -> Dataset
        make_model=...,              # () -> nn.Module
        make_optimizer=...,          # (model) -> Optimizer
        train_step=...,              # (model, opt, batch, device) -> loss
        default_batch_size=32,
    )
```

## Knobs tuned in v1

`num_workers`, `prefetch_factor`, `persistent_workers`, `pin_memory` (CUDA only — it's a no-op on Apple Silicon's unified memory), and optionally `batch_size`.

## Notes for Apple Silicon (MPS)

Developed on an M2 Pro. Data-wait measurements use `torch.mps.synchronize()` so compute timings are honest. CPU-side augmentation competes with the GPU for unified-memory bandwidth, which makes dataloader bottlenecks *more* pronounced on Macs — and the wins correspondingly larger. AMP and `torch.compile` knobs are out of scope on MPS for now (limited backend support).

## Roadmap

- [ ] Phase 1: TorchBench-derived workloads on Apple Silicon (this repo)
- [ ] Phase 2: NVIDIA DeepLearningExamples on cloud GPUs — agent vs expert-tuned configs
- [ ] AMP / `torch.compile` knobs on CUDA
- [ ] Accuracy-parity check (fixed-step loss comparison) in every report
- [ ] Auto-apply: patch the recommended config into the user's script

## License

MIT
