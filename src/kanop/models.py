"""Placeholders for neural regressors to be added next.

The paper compares:
- MLP [1, 32, 32, 1] with 100,000 paths for the American put.
- MLP [2, 32, 32, 1] with 100,000 paths for the Asian-American call.
- KANOP [1, 3, 1] with 10,000 paths for the American put.
- KANOP [2, 5, 1] with 10,000 paths for the Asian-American call.

These classes are intentionally not implemented in the baseline skeleton because
first verifying the LSMC simulation/regression engine is the safest path.
"""

from __future__ import annotations


class TorchMLPRegressor:
    """TODO: PyTorch MLP regression adapter with fit/predict methods."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Add after baseline OLS experiments are verified.")


class TorchKANRegressor:
    """TODO: PyTorch KAN regression adapter with fit/predict methods."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Add after baseline OLS experiments are verified.")
