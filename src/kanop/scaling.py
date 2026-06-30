"""Feature scaling helpers for basis-regression experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Array = np.ndarray
ScalingMode = Literal["raw", "S_over_K", "S_over_S0", "standardized"]


@dataclass(frozen=True)
class BasisInputScaler:
    """Scale state inputs before applying fixed polynomial bases.

    The paper does not specify whether Laguerre/Hermite inputs are raw prices,
    moneyness-like ratios, or standardized values. The default ``raw`` mode keeps
    the current behavior. Non-raw modes are exposed for reproducibility sweeps
    because scaling materially affects conditioning of high-order basis terms.
    """

    mode: ScalingMode = "raw"
    s0: float | None = None
    strike: float | None = None

    def transform(self, x: Array) -> Array:
        x = np.asarray(x, dtype=float)
        if self.mode == "raw":
            return x
        if self.mode == "S_over_K":
            if self.strike is None:
                raise ValueError("strike is required for S_over_K scaling")
            return x / self.strike
        if self.mode == "S_over_S0":
            if self.s0 is None:
                raise ValueError("s0 is required for S_over_S0 scaling")
            return x / self.s0
        if self.mode == "standardized":
            mean = np.mean(x, axis=0, keepdims=True)
            std = np.std(x, axis=0, keepdims=True)
            return (x - mean) / np.where(std > 0.0, std, 1.0)
        raise ValueError(f"unknown basis scaling mode: {self.mode}")
