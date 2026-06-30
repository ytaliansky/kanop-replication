"""Payoff and state-feature helpers."""

from __future__ import annotations

import numpy as np


def put_intrinsic(stock: np.ndarray, strike: float) -> np.ndarray:
    return np.maximum(strike - stock, 0.0)


def call_intrinsic(stock: np.ndarray, strike: float) -> np.ndarray:
    return np.maximum(stock - strike, 0.0)


def twap_paths(paths: np.ndarray, include_initial: bool = True) -> np.ndarray:
    """Return running arithmetic average/TWAP features.

    Args:
        paths: shape (n_paths, n_steps + 1).
        include_initial: if True, average observations from t0 through tk. If
            False, average observations from t1 through tk. The Asian-American
            table in the KANOP paper is much closer to the exclude-initial
            convention, so the experiment script uses ``include_initial=False``.

    For k=0 with ``include_initial=False``, the function returns S0 simply as a
    placeholder. The LSMC engine does not exercise at k=0.
    """
    if paths.ndim != 2:
        raise ValueError("paths must be 2D with shape (n_paths, n_steps + 1)")

    if include_initial:
        cumsum = np.cumsum(paths, axis=1)
        counts = np.arange(1, paths.shape[1] + 1, dtype=float)
        return cumsum / counts

    out = np.empty_like(paths, dtype=float)
    out[:, 0] = paths[:, 0]
    cumsum = np.cumsum(paths[:, 1:], axis=1)
    counts = np.arange(1, paths.shape[1], dtype=float)
    out[:, 1:] = cumsum / counts
    return out


def asian_call_intrinsic_from_twap(twap: np.ndarray, strike: float) -> np.ndarray:
    return np.maximum(twap - strike, 0.0)
