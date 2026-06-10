"""DistilBERT fine-tuning on SST-2 — text-side demo (tokenize-on-the-fly).

Needs the nlp extra: pip install "loadtune[nlp]"
Tokenization happens in __getitem__ on purpose: it is the text equivalent of
CPU-side augmentation and is what makes the workload worker-sensitive.
"""

import torch

from loadtune import Workload

try:
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError as e:  # pragma: no cover
    raise ImportError('this workload needs the nlp extra: pip install "loadtune[nlp]"') from e

MODEL = "distilbert-base-uncased"
MAX_LEN = 128


class TokenizeOnTheFly(torch.utils.data.Dataset):
    def __init__(self):
        self.ds = load_dataset("glue", "sst2", split="train")
        self.tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        row = self.ds[idx]
        enc = self.tok(
            row["sentence"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        return (
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
            row["label"],
        )


def make_model():
    return AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)


def train_step(model, optimizer, batch, device):
    input_ids, attention_mask, labels = (t.to(device) for t in batch)
    optimizer.zero_grad(set_to_none=True)
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    out.loss.backward()
    optimizer.step()
    return out.loss


def get_workload() -> Workload:
    return Workload(
        name="distilbert_sst2",
        make_dataset=TokenizeOnTheFly,
        make_model=make_model,
        make_optimizer=lambda m: torch.optim.AdamW(m.parameters(), lr=2e-5),
        train_step=train_step,
        default_batch_size=32,
        description="DistilBERT on SST-2 with on-the-fly tokenization.",
    )
