"""Tests for Epic 4: Framework Adapters and Python API."""

from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from loadtune import Workload, profile, tune, TuneResult
from loadtune.knobs import Knobs
from loadtune.profiler import ProfileResult


# ===================================================================
# Helper: tiny model + dataset for testing
# ===================================================================

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


def _tiny_dataset():
    return torch.utils.data.TensorDataset(
        torch.randn(100, 4),
        torch.randint(0, 2, (100,)),
    )


def _tiny_workload():
    def train_step(model, optimizer, batch, device):
        x, y = batch
        x, y = x.to(device), y.to(device)
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss

    return Workload(
        name="tiny_test",
        make_dataset=_tiny_dataset,
        make_model=TinyModel,
        make_optimizer=lambda m: torch.optim.SGD(m.parameters(), lr=0.01),
        train_step=train_step,
        default_batch_size=16,
    )


# ===================================================================
# 1. Python API: profile()
# ===================================================================

class TestProfileAPI:
    def test_profile_returns_result(self):
        w = _tiny_workload()
        result = profile(w, steps=5, warmup=2)
        assert isinstance(result, ProfileResult)
        assert result.error is None
        assert result.throughput > 0
        assert result.steps == 5

    def test_profile_with_knobs(self):
        w = _tiny_workload()
        result = profile(w, knobs=Knobs(num_workers=0), steps=3, warmup=1)
        assert result.error is None


# ===================================================================
# 2. Python API: tune()
# ===================================================================

class TestTuneAPI:
    def test_tune_returns_tune_result(self):
        w = _tiny_workload()
        result = tune(w, steps=5, warmup=2, max_trials=2, verbose=False)
        assert isinstance(result, TuneResult)
        assert isinstance(result.baseline, ProfileResult)
        assert isinstance(result.trials, list)
        assert isinstance(result.diagnosis, str)

    def test_tune_speedup_property(self):
        w = _tiny_workload()
        result = tune(w, steps=5, warmup=2, max_trials=1, verbose=False)
        assert result.speedup >= 0  # could be 1.0 if no improvement

    def test_tune_verbose_output(self, capsys):
        w = _tiny_workload()
        tune(w, steps=3, warmup=1, max_trials=1, verbose=True)
        captured = capsys.readouterr()
        assert "[loadtune]" in captured.out


# ===================================================================
# 3. Lightning adapter (mock-based — no Lightning dependency required)
# ===================================================================

class TestLightningAdapterMock:
    """Test the Lightning adapter with mock objects (no real Lightning needed)."""

    def test_from_lightning_returns_workload(self):
        # Mock a LightningModule
        module = MagicMock()
        module.__class__.__name__ = "MockLightningModel"
        module.training_step = MagicMock(return_value=torch.tensor(1.0, requires_grad=True))
        module.configure_optimizers = MagicMock(
            return_value=torch.optim.SGD([torch.nn.Parameter(torch.randn(2))], lr=0.01)
        )

        # Mock a DataModule
        mock_dataset = _tiny_dataset()
        mock_loader = torch.utils.data.DataLoader(mock_dataset, batch_size=8)
        datamodule = MagicMock()
        datamodule.train_dataloader = MagicMock(return_value=mock_loader)

        with patch.dict("sys.modules", {"lightning": MagicMock()}):
            from loadtune.adapters.lightning import from_lightning
            w = from_lightning(module, datamodule=datamodule, batch_size=16)

        assert isinstance(w, Workload)
        assert w.name == "lightning_MockLightningModel"
        assert w.default_batch_size == 16
        assert callable(w.make_dataset)
        assert callable(w.make_model)
        assert callable(w.make_optimizer)
        assert callable(w.train_step)

    def test_lightning_train_step_calls_backward(self):
        """Verify the wrapped train_step calls backward + step."""
        loss_tensor = torch.tensor(2.0, requires_grad=True)
        module = MagicMock()
        module.__class__.__name__ = "TestModel"
        module.training_step = MagicMock(return_value=loss_tensor)
        module.configure_optimizers = MagicMock(
            return_value=torch.optim.SGD([torch.nn.Parameter(torch.randn(2))], lr=0.01)
        )

        mock_dataset = _tiny_dataset()
        mock_loader = torch.utils.data.DataLoader(mock_dataset, batch_size=8)
        datamodule = MagicMock()
        datamodule.train_dataloader = MagicMock(return_value=mock_loader)

        with patch.dict("sys.modules", {"lightning": MagicMock()}):
            from loadtune.adapters.lightning import from_lightning
            w = from_lightning(module, datamodule=datamodule)

        # Create mock optimizer
        optimizer = MagicMock()
        batch = (torch.randn(4, 4), torch.randint(0, 2, (4,)))
        device = torch.device("cpu")

        # The train_step should call training_step, then backward, step
        w.train_step(module, optimizer, batch, device)
        module.training_step.assert_called_once()
        optimizer.zero_grad.assert_called_once()
        optimizer.step.assert_called_once()


# ===================================================================
# 4. HuggingFace adapter (mock-based)
# ===================================================================

class TestHFAdapterMock:
    """Test the HF adapter with mock objects (no real transformers needed)."""

    def test_from_hf_returns_workload(self):
        model = MagicMock(spec=nn.Module)
        model.__class__.__name__ = "MockBertModel"
        model.parameters = MagicMock(return_value=iter([torch.nn.Parameter(torch.randn(2))]))
        dataset = _tiny_dataset()

        with patch.dict("sys.modules", {"transformers": MagicMock()}):
            from loadtune.adapters.hf import from_hf_trainer
            w = from_hf_trainer(model, dataset, batch_size=8)

        assert isinstance(w, Workload)
        assert w.name == "hf_MockBertModel"
        assert w.default_batch_size == 8
        assert callable(w.make_dataset)
        assert callable(w.make_model)
        assert callable(w.make_optimizer)
        assert callable(w.train_step)

    def test_hf_train_step_handles_dict_batch(self):
        """Verify the train_step correctly handles dict-style HF batches."""
        mock_output = MagicMock()
        mock_output.loss = torch.tensor(1.5, requires_grad=True)

        model = MagicMock(spec=nn.Module)
        model.__class__.__name__ = "MockBert"
        model.return_value = mock_output
        model.parameters = MagicMock(return_value=iter([torch.nn.Parameter(torch.randn(2))]))

        dataset = _tiny_dataset()

        with patch.dict("sys.modules", {"transformers": MagicMock()}):
            from loadtune.adapters.hf import from_hf_trainer
            w = from_hf_trainer(model, dataset)

        optimizer = MagicMock()
        dict_batch = {
            "input_ids": torch.randint(0, 100, (4, 16)),
            "attention_mask": torch.ones(4, 16),
            "label": torch.randint(0, 2, (4,)),  # 'label' not 'labels'
        }
        device = torch.device("cpu")

        w.train_step(model, optimizer, dict_batch, device)
        # Model should have been called with **batch (after label rename)
        model.assert_called_once()
        call_kwargs = model.call_args[1]
        assert "labels" in call_kwargs, "Should rename 'label' to 'labels'"
        assert "label" not in call_kwargs

    def test_hf_collator_passed_through(self):
        model = MagicMock(spec=nn.Module)
        model.__class__.__name__ = "MockBert"
        model.parameters = MagicMock(return_value=iter([torch.nn.Parameter(torch.randn(2))]))

        dataset = _tiny_dataset()
        custom_collator = MagicMock()

        with patch.dict("sys.modules", {"transformers": MagicMock()}):
            from loadtune.adapters.hf import from_hf_trainer
            w = from_hf_trainer(model, dataset, data_collator=custom_collator)

        assert w.collate_fn is custom_collator


# ===================================================================
# 5. Workload interface validation
# ===================================================================

class TestWorkloadInterface:
    """Verify any adapter-produced Workload has all required fields."""

    def test_workload_has_required_fields(self):
        w = _tiny_workload()
        assert hasattr(w, "name")
        assert hasattr(w, "make_dataset")
        assert hasattr(w, "make_model")
        assert hasattr(w, "make_optimizer")
        assert hasattr(w, "train_step")
        assert hasattr(w, "default_batch_size")

    def test_workload_dataset_is_callable(self):
        w = _tiny_workload()
        ds = w.make_dataset()
        assert hasattr(ds, "__len__")
        assert hasattr(ds, "__getitem__")

    def test_workload_model_is_module(self):
        w = _tiny_workload()
        model = w.make_model()
        assert isinstance(model, nn.Module)
