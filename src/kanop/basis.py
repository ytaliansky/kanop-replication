"""Basis functions for conventional LSMC regressions.

The paper uses:
- Weighted Laguerre polynomials for the American put baseline.
- Hermite polynomials for comparison in the American put baseline.
- Unweighted Laguerre polynomials up to total degree 4, including cross-products,
  for the Asian-American call baseline, giving 15 regressors in two dimensions.
"""

from __future__ import annotations

import numpy as np


def _as_2d_column(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x[:, None]
    if x.ndim == 2 and x.shape[1] == 1:
        return x
    raise ValueError("expected a 1D array or a 2D single-column array")


def laguerre_values(x: np.ndarray, max_order: int) -> np.ndarray:
    """Return L_0(x), ..., L_max_order(x) using the paper's recurrence."""
    x = _as_2d_column(x)[:, 0]
    if max_order < 0:
        raise ValueError("max_order must be non-negative")
    values = np.empty((x.shape[0], max_order + 1), dtype=float)
    values[:, 0] = 1.0
    if max_order >= 1:
        values[:, 1] = 1.0 - x
    for p in range(2, max_order + 1):
        values[:, p] = ((2 * p - 1 - x) * values[:, p - 1] - (p - 1) * values[:, p - 2]) / p
    return values


def weighted_laguerre_basis(x: np.ndarray, max_order: int = 5) -> np.ndarray:
    """Weighted Laguerre basis e^{-x/2} L_p(x), p=0..max_order."""
    x_col = _as_2d_column(x)[:, 0]
    return np.exp(-0.5 * x_col)[:, None] * laguerre_values(x_col, max_order)


def laguerre_basis(x: np.ndarray, max_order: int = 5) -> np.ndarray:
    return laguerre_values(x, max_order)


def hermite_values(x: np.ndarray, max_order: int) -> np.ndarray:
    """Return H_0(x), ..., H_max_order(x) using the paper's recurrence."""
    x = _as_2d_column(x)[:, 0]
    if max_order < 0:
        raise ValueError("max_order must be non-negative")
    values = np.empty((x.shape[0], max_order + 1), dtype=float)
    values[:, 0] = 1.0
    if max_order >= 1:
        values[:, 1] = 2.0 * x
    for p in range(2, max_order + 1):
        values[:, p] = 2.0 * x * values[:, p - 1] - 2.0 * (p - 1) * values[:, p - 2]
    return values


def hermite_basis(x: np.ndarray, max_order: int = 5) -> np.ndarray:
    return hermite_values(x, max_order)


def polynomial_basis(x: np.ndarray, max_order: int = 5, include_intercept: bool = True) -> np.ndarray:
    """Simple monomial basis: 1, x, x^2, ..."""
    x = _as_2d_column(x)[:, 0]
    start = 0 if include_intercept else 1
    powers = [x**p for p in range(start, max_order + 1)]
    return np.column_stack(powers)


def laguerre_total_degree_cross_basis(x: np.ndarray, max_total_order: int = 4) -> np.ndarray:
    """2D Laguerre cross-product basis with total degree <= max_total_order.

    For max_total_order=4 and inputs [S_t, TWAP_t], the number of regressors is
    (4+1)(4+2)/2 = 15, matching the paper.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("expected x with shape (n_samples, 2)")
    l0 = laguerre_values(x[:, 0], max_total_order)
    l1 = laguerre_values(x[:, 1], max_total_order)

    features = []
    for i in range(max_total_order + 1):
        for j in range(max_total_order + 1 - i):
            features.append(l0[:, i] * l1[:, j])
    return np.column_stack(features)
