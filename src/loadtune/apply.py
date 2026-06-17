"""Auto-apply functionality for generating configuration snippets."""

from __future__ import annotations

from pathlib import Path
from .knobs import Knobs

def generate_apply_snippet(knobs: Knobs, out_dir: Path) -> Path:
    """Generate a Python snippet that applies the recommended knobs."""
    out_file = out_dir / "loadtune_apply.py"
    
    lines = [
        '"""loadtune recommended configuration."""',
        "",
        "import torch",
        "",
        "def get_dataloader_kwargs() -> dict:",
        "    kwargs = {",
        f"        'num_workers': {knobs.num_workers},",
        f"        'pin_memory': {knobs.pin_memory},",
        "    }",
    ]
    
    if knobs.num_workers > 0:
        lines.append(f"    kwargs['persistent_workers'] = {knobs.persistent_workers}")
        if knobs.prefetch_factor is not None:
            lines.append(f"    kwargs['prefetch_factor'] = {knobs.prefetch_factor}")
            
    lines.extend([
        "    return kwargs",
        "",
        "def apply_global_configs() -> None:",
    ])
    
    if knobs.num_threads is not None:
        lines.append(f"    torch.set_num_threads({knobs.num_threads})")
    else:
        lines.append("    pass  # No global configs to apply")
        
    lines.extend([
        "",
        "def apply_model_configs(model: torch.nn.Module) -> torch.nn.Module:",
    ])
    
    if getattr(knobs, "compile", False):
        lines.append("    return torch.compile(model)")
    else:
        lines.append("    return model  # No model configs to apply")

    # Add comments/guidance for training loop modifications (AMP & non_blocking)
    loop_guidance = []
    if getattr(knobs, "non_blocking", False):
        loop_guidance.append(
            "# [non_blocking=True] active: Ensure you pass non_blocking=True to your tensor .to() calls, e.g.:\n"
            "#   inputs = inputs.to(device, non_blocking=True)\n"
            "#   targets = targets.to(device, non_blocking=True)"
        )
    if getattr(knobs, "amp", False):
        loop_guidance.append(
            "# [AMP] active: Wrap your training step forward pass & loss computation in autocast, e.g.:\n"
            "#   with torch.autocast(device_type=device.type):\n"
            "#       outputs = model(inputs)\n"
            "#       loss = criterion(outputs, targets)"
        )

    if loop_guidance:
        lines.extend([
            "",
            '"""',
            "TRAINING LOOP OPTIMIZATIONS:",
            "",
            *loop_guidance,
            '"""',
        ])
        
    out_file.write_text("\n".join(lines) + "\n")
    return out_file
