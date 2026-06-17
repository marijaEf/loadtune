---
name: loadtune
description: Agentic profiler & tuner for PyTorch workloads. Use this skill when the user asks to debug PyTorch DataLoader performance or optimize training throughput.
---

# Loadtune Agent Skill

`loadtune` is a deterministic profiling and tuning tool for PyTorch training workloads. As an AI agent, you can use `loadtune` to diagnose whether a training script is input-bound (Data Wait) or compute-bound, test different DataLoader/training configurations autonomously, and apply the optimal configuration to the user's code.

## Workflow

1. **Initial Profile**:
   Run the `loadtune profile` CLI to get a baseline trace of the workload:
   ```bash
   loadtune profile <path_to_workload.py>
   ```
   The command will output a JSON object containing profiling metrics:
   ```json
   {
     "throughput": 150.5,
     "data_wait_frac": 0.45,
     "cpu_util_mean": 25.0,
     "device": "cuda:0",
     "step_losses": [...]
   }
   ```
   - Pay close attention to `data_wait_frac`. If it is > 0.15 (15%), the workload is bottlenecked by the CPU/DataLoader. If it is low, the workload is compute-bound.

2. **Propose and Test Configurations**:
   Instead of guessing, you can autonomously run trials using the `--knobs` parameter. Provide a JSON string with the exact knobs you wish to test.
   
   Available knobs:
   - `num_workers` (int)
   - `prefetch_factor` (int)
   - `persistent_workers` (bool)
   - `pin_memory` (bool)
   - `num_threads` (int)
   - `compile` (bool)
   - `amp` (bool)
   - `non_blocking` (bool)

   Test your hypothesis:
   ```bash
   loadtune profile <path_to_workload.py> --knobs '{"num_workers": 4, "persistent_workers": true, "amp": true}'
   ```
   *Note: Always use valid JSON for `--knobs`. Do not use Python booleans (`True`/`False`), use JSON booleans (`true`/`false`).*

3. **Verify Loss Parity**:
   When testing optimizations like `amp=true` (Automatic Mixed Precision), always compare the `step_losses` of the trial against the baseline `step_losses`. Make sure the optimization didn't diverge the loss curve significantly. 

4. **Apply to User's Code**:
   Once you've found a configuration that significantly improves `throughput` without breaking loss parity, use your editing tools to modify the user's workload file directly.
   - For DataLoader knobs (`num_workers`, `prefetch_factor`, etc.), edit the `DataLoader` instantiation.
   - For `amp=true`, wrap the forward pass and loss computation in `torch.autocast(device_type="cuda")`.
   - For `compile=true`, wrap the model in `torch.compile(model)`.
   - For `non_blocking=true`, add `non_blocking=True` to `.to(device)` calls.

Alternatively, you can let `loadtune`'s deterministic brain find the optimal configuration for you by running:
```bash
loadtune tune <path_to_workload.py> --apply
```
This will automatically profile, explore, and generate a recommended code patch.

## Best Practices
- Never modify the user's code *before* profiling. Always get a baseline first.
- If a trial returns an `error` in the JSON, discard that configuration and try a less aggressive one (e.g. lowering `num_workers` if OOM occurs).
