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

Every proposed config is **verified by measurement**, never just suggested. The report shows throughput per trial, speedup vs baseline, why each config was tried, and charts of where each step's time goes.

## The tuning methodology

![Structured tuning loop](docs/tuning-loop.svg)

Many knobs can affect training throughput, but searching their joint space is intractable and unnecessary. loadtune follows a **bottleneck-driven loop** instead:

1. **Profile** — a short instrumented run splits every step into *data wait* (accelerator idle) vs *compute*.
2. **Classify the bottleneck** — input-bound, compute-bound, or transfer/memory-bound. Only the matching knob family enters the search; tuning `torch.compile` on an input-bound job is wasted effort.
3. **Trial candidates** — short, subprocess-isolated runs measure each proposal. Within a family, independent knobs get coordinate descent; interacting knobs (e.g. `num_workers` × intra-op threads) are trialed jointly — where the LLM brain prunes the grid by reasoning about the hardware.
4. **Adopt the cheapest config within noise tolerance** of the best throughput — `num_workers=2` beats `num_workers=8` when they're statistically tied.
5. **Re-profile and repeat** — removing one bottleneck exposes the next (a job that was 49% input-bound becomes compute-bound once workers saturate the pipeline). Stop when expected gain no longer justifies trial cost. Math-preserving knobs need only throughput; semantics-changing knobs (batch size, AMP) additionally gate on a fixed-step loss-parity check.

v0.1 implements one iteration of this loop for the input-pipeline knob family; further families and multi-round tuning are on the roadmap.

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

## Results (Phase 1: Apple M2 Pro, 10 cores, MPS)

**Synthetic input-bound workload** (small CNN, deliberately heavy CPU augmentation) — the stress case:

| | baseline `workers=0` | tuned |
|---|---|---|
| throughput | 3,994 samples/s | **8,642 samples/s (2.16x)** |
| data wait | 48% of step time | 2.5% |

**ResNet-50 on CIFAR-10** (224px, heavy augmentation) — the realistic case:

| | baseline `workers=0` | tuned `workers=2` |
|---|---|---|
| throughput | 68.4 samples/s | **78.2 samples/s (1.14x)** |
| data wait | 10.8% of step time | 0.1% |

The ResNet result is the validating one: the profile measured an 11% data-wait fraction, which bounds the input-side speedup at ~1.12x — the agent proposed a single cheap trial, claimed the full ceiling, and stopped instead of sweeping knobs that cannot help.

![ResNet-50 step timeline, baseline vs tuned](results/resnet_report_timeline.png)

*The red band is time the accelerator spends idle, waiting for data. Two DataLoader workers remove it entirely; what remains is pure compute.*

A replication note: on the synthetic workload, single-trial differences between configs on the ~8,600 samples/s plateau (±4%) did not replicate across runs — only the 2x worker effect is robust. Headline numbers should come from repeated measurements (`--repeats`).

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

`num_workers`, `prefetch_factor`, `persistent_workers`, `pin_memory` (CUDA only — it's a no-op on Apple Silicon's unified memory), `num_threads` (intra-op threads, trialed jointly with worker counts since they compete for cores), and optionally `batch_size`.

The heuristic brain's rules, beyond the worker sweep: a **CPU-saturation guard** (input-bound + cores maxed → more workers can't help; the diagnosis says so instead of proposing futile trials), a **jitter rule** (p90/p50 step time ≥ 1.5 with active workers → deeper prefetch to absorb stragglers), and **worker×thread pairing** (4+ workers → paired trial capping main-process intra-op threads to the leftover cores).

## Notes for Apple Silicon (MPS)

Developed on an M2 Pro. Data-wait measurements use `torch.mps.synchronize()` so compute timings are honest. CPU-side augmentation competes with the GPU for unified-memory bandwidth, which makes dataloader bottlenecks *more* pronounced on Macs — and the wins correspondingly larger. AMP and `torch.compile` knobs are out of scope on MPS for now (limited backend support).

## Roadmap

- [ ] Phase 1: TorchBench-derived workloads on Apple Silicon (this repo)
- [ ] Phase 2: NVIDIA DeepLearningExamples on cloud GPUs — agent vs expert-tuned configs
- [x] Repeated measurements: `--repeats N` reports median throughput with min–max spread
- [ ] Multi-round tuning: re-profile after adoption, switch knob families as the bottleneck moves
- [ ] Compute-bound family: AMP, `torch.compile`, `channels_last`, fused optimizers (CUDA)
- [ ] Joint knobs: `num_workers` × `torch.set_num_threads`, `non_blocking` copies
- [ ] Accuracy-parity check (fixed-step loss comparison) for semantics-changing knobs
- [ ] Auto-apply: patch the recommended config into the user's script

## License

MIT
