"""NVIDIA DeepLearningExamples ResNet-50 Mock.

Simulates the intense compute and data-augmentation pipeline used in the
official NVIDIA ResNet-50 training script.
Designed to be used for testing loadtune on Cloud GPUs.
"""

import torch
import torch.nn as nn
from loadtune import Workload

try:
    import torchvision
    from torchvision import transforms
except ImportError as e:
    raise ImportError("this workload needs torchvision: pip install torchvision") from e

def make_dataset():
    # Mimic NVIDIA's heavy augmentation pipeline
    aug = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # Use FakeData to avoid downloading gigabytes just for the cloud benchmark!
    return torchvision.datasets.FakeData(size=5000, image_size=(3, 224, 224), num_classes=1000, transform=aug)

def make_model() -> nn.Module:
    # Use standard resnet50
    return torchvision.models.resnet50(weights=None, num_classes=1000)

def train_step(model, optimizer, batch, device):
    x, y = batch
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    
    # Forward
    out = model(x)
    loss = nn.functional.cross_entropy(out, y)
    
    # Backward
    loss.backward()
    optimizer.step()
    return loss

def get_workload() -> Workload:
    return Workload(
        name="dle_resnet50_mock",
        make_dataset=make_dataset,
        make_model=make_model,
        make_optimizer=lambda m: torch.optim.SGD(m.parameters(), lr=0.1),
        train_step=train_step,
        default_batch_size=64,
        description="NVIDIA DLE ResNet50 Mock (Compute-heavy, AMP friendly)"
    )
