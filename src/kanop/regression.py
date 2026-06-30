"""Regression adapters used by the LSMC engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

Array = np.ndarray


class Regressor(Protocol):
    def fit(self, x: Array, y: Array) -> "Regressor": ...
    def predict(self, x: Array) -> Array: ...


BasisFunction = Callable[[Array], Array]


@dataclass
class OLSBasisRegressor:
    """Ordinary least-squares regression over a supplied basis map."""

    basis_fn: BasisFunction
    ridge: float = 0.0
    coef_: Array | None = None

    def fit(self, x: Array, y: Array) -> "OLSBasisRegressor":
        phi = self.basis_fn(x)
        y = np.asarray(y, dtype=float).reshape(-1)
        if phi.shape[0] != y.shape[0]:
            raise ValueError("x and y sample counts differ")

        if self.ridge > 0:
            gram = phi.T @ phi
            penalty = self.ridge * np.eye(gram.shape[0])
            self.coef_ = np.linalg.solve(gram + penalty, phi.T @ y)
        else:
            self.coef_, *_ = np.linalg.lstsq(phi, y, rcond=None)
        return self

    def predict(self, x: Array) -> Array:
        if self.coef_ is None:
            raise RuntimeError("regressor must be fitted before predict")
        phi = self.basis_fn(x)
        return phi @ self.coef_


def make_ols_factory(basis_fn: BasisFunction, ridge: float = 0.0) -> Callable[[], OLSBasisRegressor]:
    """Factory function because LSMC trains a new regressor at each time step."""
    return lambda: OLSBasisRegressor(basis_fn=basis_fn, ridge=ridge)
