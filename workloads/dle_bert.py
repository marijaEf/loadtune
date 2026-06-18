"""NVIDIA DeepLearningExamples BERT Mock.

Simulates NLP bottlenecks (tokenization overhead) and transformer compute.
Designed to test loadtune's ability to navigate mixed compute/data-wait bottlenecks.
"""

import time
import torch
import torch.nn as nn
from loadtune import Workload

try:
    from transformers import BertConfig, BertModel
except ImportError as e:
    raise ImportError("this workload needs transformers: pip install transformers") from e

class FakeTokenizationDataset(torch.utils.data.Dataset):
    def __init__(self, size=1000, seq_len=128, vocab_size=30000):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Simulate heavy CPU-side text preprocessing and tokenization delay
        time.sleep(0.005) # 5ms artificial delay per sample!
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        attention_mask = torch.ones(self.seq_len, dtype=torch.long)
        labels = torch.randint(0, self.vocab_size, (self.seq_len,))
        return input_ids, attention_mask, labels


def make_dataset():
    return FakeTokenizationDataset()

def make_model() -> nn.Module:
    # Small BERT to fit easily into memory during fast tests
    config = BertConfig(
        vocab_size=30000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,
    )
    return BertModel(config)

def train_step(model, optimizer, batch, device):
    input_ids, attention_mask, labels = batch
    input_ids = input_ids.to(device, non_blocking=True)
    attention_mask = attention_mask.to(device, non_blocking=True)
    
    optimizer.zero_grad(set_to_none=True)
    
    # Forward
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Dummy loss
    loss = outputs.last_hidden_state.mean()
    
    # Backward
    loss.backward()
    optimizer.step()
    return loss

def get_workload() -> Workload:
    return Workload(
        name="dle_bert_mock",
        make_dataset=make_dataset,
        make_model=make_model,
        train_step=train_step,
        default_batch_size=32,
        description="NVIDIA DLE BERT Mock (Input-heavy tokenization, Transformer compute)"
    )
