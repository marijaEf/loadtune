"""HuggingFace Transformers adapter for loadtune.

Bridges a HuggingFace PreTrainedModel + dataset to loadtune's Workload
interface so you can profile and tune HF workloads without writing
boilerplate.

Usage::

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from datasets import load_dataset
    from loadtune import from_hf_trainer, tune

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    dataset = load_dataset("glue", "sst2", split="train")
    dataset = dataset.map(lambda x: tokenizer(x["sentence"], truncation=True, padding="max_length"), batched=True)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    workload = from_hf_trainer(model, dataset, batch_size=32)
    result = tune(workload, steps=50)
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from ..workload import Workload


def from_hf_trainer(
    model: Any,
    dataset: Any,
    tokenizer: Any = None,
    data_collator: Any = None,
    batch_size: int = 16,
) -> Workload:
    """Create a loadtune Workload from HuggingFace components.

    Parameters
    ----------
    model : PreTrainedModel
        A HuggingFace model (subclass of nn.Module) that returns an object
        with a ``.loss`` attribute when called with ``labels`` in the input.
    dataset : Dataset
        A torch-compatible dataset (e.g., HuggingFace ``datasets.Dataset``
        with ``set_format("torch")``).
    tokenizer : PreTrainedTokenizer, optional
        Used to build a default ``DataCollatorWithPadding`` if no
        ``data_collator`` is provided.
    data_collator : callable, optional
        Custom collation function. If ``None`` and ``tokenizer`` is given,
        uses ``DataCollatorWithPadding(tokenizer)``.
    batch_size : int
        Default batch size for the workload.

    Returns
    -------
    Workload
        A standard loadtune Workload ready for ``profile_session`` or ``tune``.
    """
    try:
        import transformers  # noqa: F401
    except ImportError:
        raise ImportError(
            "from_hf_trainer requires HuggingFace Transformers: "
            "pip install transformers"
        )

    # --- Resolve collate_fn ---
    collate_fn = data_collator
    if collate_fn is None and tokenizer is not None:
        try:
            from transformers import DataCollatorWithPadding
            collate_fn = DataCollatorWithPadding(tokenizer)
        except ImportError:
            pass

    # --- Dataset factory ---
    def make_dataset():
        return dataset

    # --- Model factory ---
    def make_model():
        return model

    # --- Optimizer factory ---
    def make_optimizer(m):
        # Match HF Trainer's default: AdamW with lr=5e-5
        return torch.optim.AdamW(m.parameters(), lr=5e-5, weight_decay=0.01)

    # --- Training step wrapper ---
    # HF models expect dict-style batches and return ModelOutput with .loss
    def train_step(m, optimizer, batch, device):
        # Move dict batch to device
        if isinstance(batch, dict):
            batch = {
                k: v.to(device, non_blocking=True) if hasattr(v, "to") else v
                for k, v in batch.items()
            }
        elif isinstance(batch, (list, tuple)):
            batch = [
                b.to(device, non_blocking=True) if hasattr(b, "to") else b
                for b in batch
            ]

        # HF models accept **kwargs for dict-style inputs
        if isinstance(batch, dict):
            # Rename 'label' -> 'labels' if needed (common HF dataset quirk)
            if "label" in batch and "labels" not in batch:
                batch["labels"] = batch.pop("label")
            outputs = m(**batch)
        else:
            outputs = m(*batch)

        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return loss

    name = model.__class__.__name__
    return Workload(
        name=f"hf_{name}",
        make_dataset=make_dataset,
        make_model=make_model,
        make_optimizer=make_optimizer,
        train_step=train_step,
        default_batch_size=batch_size,
        collate_fn=collate_fn,
        description=f"HuggingFace adapter for {name}",
    )
