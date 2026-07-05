"""Autograd-based delta estimators for continuation-value models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - exercised only without torch installed.
    torch = None

Array = np.ndarray
OptionType = Literal["put"]


class TorchContinuationModel(Protocol):
    def predict_torch(self, x: "torch.Tensor") -> "torch.Tensor":
        ...


@dataclass(frozen=True)
class DeltaDiagnostics:
    delta: float
    model_price_from_t1: float
    mean_continuation: float
    mean_intrinsic: float
    n_paths: int


def first_step_standard_normals_from_paths(
    paths: Array,
    *,
    s0: float,
    sigma: float,
    r: float,
    q: float,
    dt: float,
) -> Array:
    """Recover first-step standard-normal shocks from simulated GBM paths."""
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 2 or paths.shape[1] < 2:
        raise ValueError("paths must have shape (n_paths, n_steps + 1) with at least one step")
    if s0 <= 0.0:
        raise ValueError("s0 must be positive")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    s1 = paths[:, 1]
    drift = (r - q - 0.5 * sigma**2) * dt
    return (np.log(s1 / s0) - drift) / (sigma * np.sqrt(dt))


def delta_from_t1_continuation_model(
    model: TorchContinuationModel,
    z_t1: Array,
    s0: float,
    strike: float,
    sigma: float,
    r: float,
    q: float,
    dt: float,
    option_type: OptionType = "put",
    device: str = "cpu",
    return_diagnostics: bool = False,
) -> float | DeltaDiagnostics:
    """Estimate delta by differentiating the paper-style t1 value formula.

    The first simulated step is reconstructed in PyTorch as
    ``S_t1 = S0 * exp((r - q - 0.5 sigma^2) dt + sigma sqrt(dt) Z)`` using the
    supplied first-step shocks. The trained t1 continuation model supplies
    ``F_hat(S_t1)``. For an American put, the t1 path value is
    ``max(F_hat(S_t1), max(K - S_t1, 0))`` and the returned delta is the
    derivative of the discounted mean value with respect to ``S0``.
    """
    if torch is None:
        raise ImportError("delta_from_t1_continuation_model requires PyTorch")
    if option_type != "put":
        raise ValueError("only American put delta is currently implemented")
    if s0 <= 0.0:
        raise ValueError("s0 must be positive")
    if strike <= 0.0:
        raise ValueError("strike must be positive")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    device_ = torch.device(device)
    z = torch.as_tensor(np.asarray(z_t1, dtype=np.float32).reshape(-1), dtype=torch.float32, device=device_)
    if z.numel() == 0:
        raise ValueError("z_t1 must contain at least one shock")

    s0_tensor = torch.tensor(float(s0), dtype=torch.float32, device=device_, requires_grad=True)
    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * z
    s_t1 = s0_tensor * torch.exp(torch.as_tensor(drift, dtype=torch.float32, device=device_) + diffusion)
    intrinsic = torch.clamp(float(strike) - s_t1, min=0.0)
    continuation = model.predict_torch(s_t1[:, None])
    value_t1 = torch.maximum(continuation.reshape(-1), intrinsic)
    discount = torch.exp(torch.as_tensor(-r * dt, dtype=torch.float32, device=value_t1.device))
    price_t0 = discount * torch.mean(value_t1)
    price_t0.backward()

    if s0_tensor.grad is None:
        raise RuntimeError("autograd did not produce a delta")

    diagnostics = DeltaDiagnostics(
        delta=float(s0_tensor.grad.detach().cpu()),
        model_price_from_t1=float(price_t0.detach().cpu()),
        mean_continuation=float(torch.mean(continuation.detach()).cpu()),
        mean_intrinsic=float(torch.mean(intrinsic.detach()).cpu()),
        n_paths=int(z.numel()),
    )
    if return_diagnostics:
        return diagnostics
    return diagnostics.delta
