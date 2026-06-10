"""Synthetic input-bound workload — no downloads, instant demo.

A small CNN paired with a dataset whose __getitem__ burns CPU (simulating
heavy augmentation / decode). With num_workers=0 this is heavily input-bound;
the right worker count should give a large, easily reproducible speedup.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from loadtune import Workload

IMG = 64
CLASSES = 10


class SlowAugmentDataset(Dataset):
    """Random 'images' with deliberately expensive CPU-side preprocessing."""

    def __init__(self, n: int = 50_000):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(idx)
        img = rng.random((3, IMG, IMG), dtype=np.float32)
        # Simulated augmentation chain: blur-ish convolutions + normalize,
        # all in numpy on the CPU (the classic thesis-code pattern).
        for _ in range(4):
            img = (img + np.roll(img, 1, axis=1) + np.roll(img, 1, axis=2)) / 3.0
        img = (img - img.mean()) / (img.std() + 1e-6)
        label = idx % CLASSES
        return torch.from_numpy(img.copy()), label


def make_model() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * (IMG // 4) ** 2, CLASSES),
    )


def train_step(model, optimizer, batch, device):
    x, y = batch
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()
    return loss


def get_workload() -> Workload:
    return Workload(
        name="synthetic_bottleneck",
        make_dataset=SlowAugmentDataset,
        make_model=make_model,
        make_optimizer=lambda m: torch.optim.SGD(m.parameters(), lr=0.01),
        train_step=train_step,
        default_batch_size=64,
        description="Small CNN with deliberately slow CPU-side augmentation.",
    )
