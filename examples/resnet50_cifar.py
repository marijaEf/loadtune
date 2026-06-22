"""ResNet-50 on CIFAR-10 with heavy augmentation — the flagship demo.

CIFAR-10 downloads automatically (~170 MB) to ./data on first run.
Augmentation is intentionally rich (the realistic research-code case), which
makes the input pipeline worker-count sensitive on Apple Silicon.
"""

import torch
import torch.nn as nn

from loadtune import Workload

try:
    import torchvision
    from torchvision import transforms
except ImportError as e:  # pragma: no cover
    raise ImportError("this workload needs torchvision: pip install torchvision") from e

DATA_DIR = "./data"


def make_dataset():
    aug = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
            ),
            transforms.RandomErasing(p=0.25),
        ]
    )
    return torchvision.datasets.CIFAR10(
        DATA_DIR, train=True, download=True, transform=aug
    )


def make_model() -> nn.Module:
    model = torchvision.models.resnet50(weights=None, num_classes=10)
    return model


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
        name="resnet50_cifar10",
        make_dataset=make_dataset,
        make_model=make_model,
        make_optimizer=lambda m: torch.optim.SGD(
            m.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
        ),
        train_step=train_step,
        default_batch_size=32,
        description="ResNet-50, CIFAR-10 upscaled to 224px with heavy augmentation.",
    )
