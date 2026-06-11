# loadtune report — synthetic_bottleneck

*2026-06-11 21:29 · device `mps` · 10 CPUs · brain `heuristic`*

## Diagnosis

The workload is **input-bound**: 49% of step time is spent waiting for the DataLoader. Mean CPU utilisation during the run was 7.0%.

## Baseline

- config: `workers=0`
- throughput: **3840.1 samples/s**
- data wait: 49.0% of step time (0.41s of 0.83s over 50 steps)
- step time p50/p90: 16.3 / 19.0 ms
- dataloader startup: 0.00s

## Trials

| config | throughput (samples/s) | vs baseline | data wait | proposed because |
|---|---|---|---|---|
| `workers=1, persistent` | 7759.0 | 2.02x | 10.0% | data_wait_frac=49% ≥ 20%: input-bound, trying num_workers=1 |
| `workers=2, persistent` | 8703.9 | 2.27x | 2.7% | data_wait_frac=49% ≥ 20%: input-bound, trying num_workers=2 |
| `workers=4, persistent` | 8466.0 | 2.20x | 2.9% | data_wait_frac=49% ≥ 20%: input-bound, trying num_workers=4 |
| `workers=5, persistent` | 8582.4 | 2.23x | 2.6% | data_wait_frac=49% ≥ 20%: input-bound, trying num_workers=5 |
| `workers=8, persistent` | 8600.2 | 2.24x | 2.6% | data_wait_frac=49% ≥ 20%: input-bound, trying num_workers=8 |

## Verdict

**Recommended config: `workers=2, persistent` — 2.27x baseline throughput** (3840.1 → 8703.9 samples/s).
