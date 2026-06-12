# loadtune report — synthetic_bottleneck

*2026-06-12 13:41 · device `mps` · 10 CPUs · brain `heuristic`*

## Diagnosis

The workload is **input-bound**: 48% of step time is spent waiting for the DataLoader. Mean CPU utilisation during the run was 8.7%.

## Baseline

- config: `workers=0`
- throughput: **3993.8 samples/s**
- data wait: 48.4% of step time (1.55s of 3.21s over 200 steps)
- step time p50/p90: 16.0 / 16.4 ms
- dataloader startup: 0.00s

## Trials

| config | throughput (samples/s) | vs baseline | data wait | proposed because |
|---|---|---|---|---|
| `workers=1, persistent` | 7871.6 | 1.97x | 9.1% | data_wait_frac=48% ≥ 20%: input-bound, trying num_workers=1 |
| `workers=2, persistent` | 8297.4 | 2.08x | 2.8% | data_wait_frac=48% ≥ 20%: input-bound, trying num_workers=2 |
| `workers=4, persistent` | 8560.4 | 2.14x | 2.7% | data_wait_frac=48% ≥ 20%: input-bound, trying num_workers=4 |
| `workers=4, persistent, threads=6` | 8641.7 | 2.16x | 2.5% | workers=4 claim cores: cap intra-op threads at 6 to avoid contention |
| `workers=5, persistent` | 8628.1 | 2.16x | 2.6% | data_wait_frac=48% ≥ 20%: input-bound, trying num_workers=5 |
| `workers=5, persistent, threads=5` | 8603.4 | 2.15x | 2.6% | workers=5 claim cores: cap intra-op threads at 5 to avoid contention |
| `workers=8, persistent` | 8596.8 | 2.15x | 2.4% | data_wait_frac=48% ≥ 20%: input-bound, trying num_workers=8 |
| `workers=8, persistent, threads=2` | 8592.9 | 2.15x | 2.5% | workers=8 claim cores: cap intra-op threads at 2 to avoid contention |

## Charts

![report_breakdown.png](report_breakdown.png)

![report_throughput.png](report_throughput.png)

![report_timeline.png](report_timeline.png)


## Verdict

**Recommended config: `workers=4, persistent, threads=6` — 2.16x baseline throughput** (3993.8 → 8641.7 samples/s).
