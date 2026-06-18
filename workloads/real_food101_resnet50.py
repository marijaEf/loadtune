"""Real data ResNet-50 workload (Food101).

Downloads ~5GB of real JPG images on first run. 
This is an excellent proxy for the ImageNet benchmark, as it tests real 
disk I/O, real JPEG decoding overhead, and a full ImageNet-style 
augmentation pipeline.
"""

import torch
import torch.nn as nn
from loadtune import Workload

try:
    import torchvision
    from torchvision import transforms
except ImportError as e:
    raise ImportError("this workload needs torchvision: pip install torchvision") from e

DATA_DIR = "./data"

def make_dataset():
    aug = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # Food101 automatically downloads 5GB on first run and extracts it
    return torchvision.datasets.Food101(DATA_DIR, split="train", download=True, transform=aug)

def make_model() -> nn.Module:
    # Use standard resnet50 for the 101 classes
    return torchvision.models.resnet50(weights=None, num_classes=101)

def train_step(model, optimizer, batch, device):
    x, y = batch
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()
    return loss

def get_workload() -> Workload:
    return Workload(
        name="real_food101_resnet50",
        make_dataset=make_dataset,
        make_model=make_model,
        make_optimizer=lambda m: torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4),
        train_step=train_step,
        default_batch_size=128,
        description="Real 5GB JPG Dataset (Food101) with ImageNet-style augmentations"
    )
