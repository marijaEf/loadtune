"""Lightning CIFAR-10 workload — demonstrates the from_lightning adapter.

Uses a minimal LightningModule with a simple CNN for CIFAR-10 classification.
Requires: pip install lightning torchvision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import lightning as L
except ImportError:
    import pytorch_lightning as L  # type: ignore

try:
    import torchvision
    from torchvision import transforms
except ImportError as e:
    raise ImportError("this workload needs torchvision: pip install torchvision") from e

from loadtune import Workload
from loadtune.adapters.lightning import from_lightning


class SimpleCNN(L.LightningModule):
    """Minimal CNN for CIFAR-10."""

    def __init__(self, lr: float = 0.01):
        super().__init__()
        self.lr = lr
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        return F.cross_entropy(logits, y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        aug = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        self.train_ds = torchvision.datasets.CIFAR10(
            self.data_dir, train=True, download=True, transform=aug,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
        )


def get_workload() -> Workload:
    # Pre-download CIFAR-10 before loadtune spawns subprocess trials.
    torchvision.datasets.CIFAR10("./data", train=True, download=True)

    module = SimpleCNN()
    datamodule = CIFAR10DataModule()
    return from_lightning(module, datamodule=datamodule, batch_size=64)
