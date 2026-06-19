"""PyTorch Lightning adapter for loadtune.

Bridges a LightningModule (+ optional LightningDataModule) to loadtune's
Workload interface so you can profile and tune Lightning workloads without
writing boilerplate.

Usage::

    import lightning as L
    from loadtune import from_lightning, tune

    class MyModel(L.LightningModule):
        ...

    workload = from_lightning(MyModel(), batch_size=64)
    result = tune(workload, steps=50)
"""

from __future__ import annotations

from typing import Any, Optional

from ..workload import Workload


def from_lightning(
    module: Any,
    datamodule: Any = None,
    batch_size: int = 32,
) -> Workload:
    """Create a loadtune Workload from a PyTorch Lightning module.

    Parameters
    ----------
    module : LightningModule
        A Lightning module with ``training_step`` and ``configure_optimizers``.
    datamodule : LightningDataModule, optional
        If provided, the training dataset is extracted from
        ``datamodule.train_dataloader().dataset``. Otherwise, falls back to
        ``module.train_dataloader().dataset``.
    batch_size : int
        Default batch size for the workload.

    Returns
    -------
    Workload
        A standard loadtune Workload ready for ``profile_session`` or ``tune``.

    Notes
    -----
    Lightning callbacks, hooks, gradient clipping, and LR schedulers are
    intentionally skipped. loadtune measures raw data pipeline throughput —
    framework overhead from callbacks is out of scope.
    """
    try:
        import lightning  # noqa: F401
    except ImportError:
        try:
            import pytorch_lightning  # noqa: F401
        except ImportError:
            raise ImportError(
                "from_lightning requires PyTorch Lightning: "
                "pip install lightning"
            )

    # --- Extract dataset ---
    def make_dataset():
        if datamodule is not None:
            if hasattr(datamodule, "setup"):
                datamodule.setup(stage="fit")
            dl = datamodule.train_dataloader()
        elif hasattr(module, "train_dataloader"):
            dl = module.train_dataloader()
        else:
            raise ValueError(
                "Cannot extract dataset: provide a LightningDataModule or "
                "implement train_dataloader() on the module."
            )
        return dl.dataset

    # --- Extract collate_fn ---
    collate_fn = None
    try:
        if datamodule is not None:
            if hasattr(datamodule, "setup"):
                datamodule.setup(stage="fit")
            dl = datamodule.train_dataloader()
        elif hasattr(module, "train_dataloader"):
            dl = module.train_dataloader()
        else:
            dl = None
        if dl is not None and hasattr(dl, "collate_fn"):
            fn = dl.collate_fn
            # Only keep non-default collate functions
            from torch.utils.data.dataloader import default_collate
            if fn is not default_collate:
                collate_fn = fn
    except Exception:
        pass  # safe to skip — collate_fn is optional

    # --- Model factory ---
    def make_model():
        return module

    # --- Optimizer factory ---
    def make_optimizer(model):
        result = model.configure_optimizers()
        # configure_optimizers can return many formats:
        #   optimizer, (optimizers, schedulers), dict, etc.
        if isinstance(result, tuple):
            optimizers = result[0]
            if isinstance(optimizers, list):
                return optimizers[0]
            return optimizers
        if isinstance(result, dict):
            return result["optimizer"]
        if isinstance(result, list):
            return result[0]
        return result  # assume it's a single optimizer

    # --- Training step wrapper ---
    # Lightning's training_step returns a loss but does NOT call
    # backward() or optimizer.step() — the Trainer does that.
    def train_step(model, optimizer, batch, device):
        # Move batch to device
        if isinstance(batch, (list, tuple)):
            batch = [
                b.to(device, non_blocking=True) if hasattr(b, "to") else b
                for b in batch
            ]
        elif isinstance(batch, dict):
            batch = {
                k: v.to(device, non_blocking=True) if hasattr(v, "to") else v
                for k, v in batch.items()
            }
        elif hasattr(batch, "to"):
            batch = batch.to(device, non_blocking=True)

        loss = model.training_step(batch, batch_idx=0)

        # Handle dict returns (e.g., {"loss": tensor})
        if isinstance(loss, dict):
            loss = loss["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return loss

    name = module.__class__.__name__
    return Workload(
        name=f"lightning_{name}",
        make_dataset=make_dataset,
        make_model=make_model,
        make_optimizer=make_optimizer,
        train_step=train_step,
        default_batch_size=batch_size,
        collate_fn=collate_fn,
        description=f"Lightning adapter for {name}",
    )
