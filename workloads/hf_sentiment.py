"""HuggingFace sentiment workload — demonstrates the from_hf_trainer adapter.

Uses DistilBERT for binary sentiment classification on SST-2.
Requires: pip install transformers datasets
"""

import torch

try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
    )
    from datasets import load_dataset
except ImportError as e:
    raise ImportError(
        "this workload needs transformers + datasets: "
        "pip install transformers datasets"
    ) from e

from loadtune import Workload
from loadtune.adapters.hf import from_hf_trainer

MODEL_NAME = "distilbert-base-uncased"


def get_workload() -> Workload:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2,
    )

    # Load and tokenize SST-2
    dataset = load_dataset("nyu-mll/glue", "sst2", split="train[:2000]")
    dataset = dataset.map(
        lambda x: tokenizer(
            x["sentence"], truncation=True, padding="max_length", max_length=128,
        ),
        batched=True,
    )
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    collator = DataCollatorWithPadding(tokenizer)

    return from_hf_trainer(
        model, dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        batch_size=32,
    )
