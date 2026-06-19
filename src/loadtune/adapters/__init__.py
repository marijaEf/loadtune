"""Framework adapters for loadtune.

Optional adapters that bridge PyTorch Lightning and HuggingFace Trainer
to loadtune's Workload interface. Each adapter extracts the dataset,
model, optimizer, and train_step from framework objects and returns a
standard Workload — the profiler and tuning pipeline don't change at all.

These imports are lazy — the frameworks are not required unless you
actually call the adapter functions.
"""

from __future__ import annotations


def from_lightning(module, datamodule=None, batch_size=32):
    """Create a loadtune Workload from a PyTorch Lightning module.

    See :func:`loadtune.adapters.lightning.from_lightning` for full docs.
    """
    from .lightning import from_lightning as _from_lightning
    return _from_lightning(module, datamodule=datamodule, batch_size=batch_size)


def from_hf_trainer(model, dataset, tokenizer=None, data_collator=None, batch_size=16):
    """Create a loadtune Workload from HuggingFace components.

    See :func:`loadtune.adapters.hf.from_hf_trainer` for full docs.
    """
    from .hf import from_hf_trainer as _from_hf_trainer
    return _from_hf_trainer(
        model, dataset, tokenizer=tokenizer,
        data_collator=data_collator, batch_size=batch_size,
    )
