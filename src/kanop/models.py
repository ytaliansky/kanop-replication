"""Neural continuation-value regressors.

The paper's MLP baselines use fully connected networks:
- American put: ``[1, 32, 32, 1]``
- Asian-American call: ``[2, 32, 32, 1]``

This module implements the MLP adapter used by the generic LSMC engine. KAN is
left as an explicit placeholder for the next implementation step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only in environments without torch.
    torch = None
    nn = None

Array = np.ndarray
ActivationName = Literal["silu", "relu", "tanh"]


def _require_torch():
    if torch is None or nn is None:
        raise ImportError("TorchMLPRegressor requires PyTorch. Install the project with the torch extra.")


def _as_2d_float_array(x: Array) -> Array:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x[:, None]
    if x.ndim == 2:
        return x
    raise ValueError("expected x with shape (n_samples,) or (n_samples, n_features)")


def _activation(name: ActivationName):
    _require_torch()
    if name == "silu":
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"unsupported activation: {name}")


def _build_mlp(layer_widths: tuple[int, ...], activation: ActivationName):
    _require_torch()
    if len(layer_widths) < 2:
        raise ValueError("layer_widths must contain at least input and output widths")
    if any(width <= 0 for width in layer_widths):
        raise ValueError("all layer widths must be positive")
    if layer_widths[-1] != 1:
        raise ValueError("TorchMLPRegressor expects a scalar output layer")

    layers: list[nn.Module] = []
    for in_width, out_width in zip(layer_widths[:-2], layer_widths[1:-1]):
        layers.append(nn.Linear(in_width, out_width))
        layers.append(_activation(activation))
    layers.append(nn.Linear(layer_widths[-2], layer_widths[-1]))
    return nn.Sequential(*layers)


@dataclass
class TorchMLPRegressor:
    """PyTorch MLP adapter with internal input/target normalization.

    Training uses MSE loss and Adam by default. SiLU is the default activation:
    it is smooth, works well for continuation-value surfaces, and keeps the
    future autograd delta path differentiable almost everywhere.
    """

    layer_widths: tuple[int, ...]
    learning_rate: float = 1e-3
    epochs: int = 200
    batch_size: int = 256
    weight_decay: float = 0.0
    random_seed: int | None = 1234
    device: str = "cpu"
    activation: ActivationName = "silu"
    verbose: bool = False
    optimizer_name: Literal["adam"] = "adam"
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        _require_torch()
        self.layer_widths = tuple(int(width) for width in self.layer_widths)
        self._validate_hyperparameters()
        self.device_ = torch.device(self.device)
        self.model_: nn.Module | None = None
        self.x_mean_: torch.Tensor | None = None
        self.x_std_: torch.Tensor | None = None
        self.y_mean_: torch.Tensor | None = None
        self.y_std_: torch.Tensor | None = None
        self.loss_history_: list[float] = []

    def _validate_hyperparameters(self) -> None:
        if len(self.layer_widths) < 2:
            raise ValueError("layer_widths must contain at least input and output widths")
        if self.layer_widths[-1] != 1:
            raise ValueError("TorchMLPRegressor expects a scalar output layer")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.optimizer_name != "adam":
            raise ValueError("only Adam optimizer is currently supported")

    def fit(self, x: Array, y: Array) -> "TorchMLPRegressor":
        x_np = _as_2d_float_array(x)
        y_np = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        if x_np.shape[0] != y_np.shape[0]:
            raise ValueError("x and y sample counts differ")
        if x_np.shape[1] != self.layer_widths[0]:
            raise ValueError(
                f"input feature count {x_np.shape[1]} does not match layer_widths[0]={self.layer_widths[0]}"
            )

        self._set_random_seed()
        self.model_ = _build_mlp(self.layer_widths, self.activation).to(self.device_)

        x_tensor = torch.as_tensor(x_np, dtype=torch.float32, device=self.device_)
        y_tensor = torch.as_tensor(y_np, dtype=torch.float32, device=self.device_)
        self.x_mean_ = x_tensor.mean(dim=0, keepdim=True)
        self.x_std_ = torch.clamp(x_tensor.std(dim=0, keepdim=True, unbiased=False), min=self.epsilon)
        self.y_mean_ = y_tensor.mean(dim=0, keepdim=True)
        self.y_std_ = torch.clamp(y_tensor.std(dim=0, keepdim=True, unbiased=False), min=self.epsilon)

        x_norm = (x_tensor - self.x_mean_) / self.x_std_
        y_norm = (y_tensor - self.y_mean_) / self.y_std_

        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.MSELoss()
        generator = torch.Generator(device=self.device_)
        if self.random_seed is not None:
            generator.manual_seed(self.random_seed)

        n_samples = x_norm.shape[0]
        self.loss_history_ = []
        self.model_.train()
        for epoch in range(self.epochs):
            permutation = torch.randperm(n_samples, generator=generator, device=self.device_)
            epoch_loss = 0.0
            for start in range(0, n_samples, self.batch_size):
                idx = permutation[start : start + self.batch_size]
                pred = self.model_(x_norm[idx])
                loss = loss_fn(pred, y_norm[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach()) * idx.numel()
            epoch_loss /= n_samples
            self.loss_history_.append(epoch_loss)
            if self.verbose and (epoch == 0 or (epoch + 1) % 50 == 0 or epoch == self.epochs - 1):
                print(f"epoch={epoch + 1} loss={epoch_loss:.6g}")
        self.model_.eval()
        return self

    def predict(self, x: Array) -> Array:
        _require_torch()
        x_np = _as_2d_float_array(x)
        with torch.no_grad():
            x_tensor = torch.as_tensor(x_np, dtype=torch.float32, device=self.device_)
            pred = self.predict_torch(x_tensor)
        return pred.detach().cpu().numpy().reshape(-1)

    def predict_torch(self, x: "torch.Tensor") -> "torch.Tensor":
        """Return denormalized predictions while preserving gradients to ``x``."""
        self._check_fitted()
        if x.ndim == 1:
            x = x[:, None]
        if x.ndim != 2:
            raise ValueError("expected x with shape (n_samples,) or (n_samples, n_features)")
        if x.shape[1] != self.layer_widths[0]:
            raise ValueError(
                f"input feature count {x.shape[1]} does not match layer_widths[0]={self.layer_widths[0]}"
            )

        x_model = x.to(device=self.device_, dtype=torch.float32)
        x_mean = self.x_mean_.to(device=x_model.device, dtype=x_model.dtype)
        x_std = self.x_std_.to(device=x_model.device, dtype=x_model.dtype)
        y_mean = self.y_mean_.to(device=x_model.device, dtype=x_model.dtype)
        y_std = self.y_std_.to(device=x_model.device, dtype=x_model.dtype)
        x_norm = (x_model - x_mean) / x_std
        y_norm = self.model_(x_norm)
        return (y_norm * y_std + y_mean).reshape(-1)

    def _set_random_seed(self) -> None:
        if self.random_seed is None:
            return
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)

    def _check_fitted(self) -> None:
        if (
            self.model_ is None
            or self.x_mean_ is None
            or self.x_std_ is None
            or self.y_mean_ is None
            or self.y_std_ is None
        ):
            raise RuntimeError("TorchMLPRegressor must be fitted before prediction")


class TorchKANRegressor:
    """TODO: PyTorch KAN regression adapter with fit/predict methods."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("KAN will be implemented after the MLP baseline is verified.")
