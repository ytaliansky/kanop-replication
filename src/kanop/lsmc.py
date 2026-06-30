"""Generic Least-Squares Monte Carlo engine.

This file implements the backward-induction structure that KANOP keeps intact.
KANOP's only major algorithmic change is to replace the OLS basis regressor with
a KAN regressor at each exercise date.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .regression import Regressor

Array = np.ndarray
FeatureFunction = Callable[[Array, int], Array]
IntrinsicFunction = Callable[[Array, int], Array]
RegressorFactory = Callable[[], Regressor]


@dataclass
class TimeStepFit:
    step: int
    time: float
    regressor: Regressor
    fit_all_paths: bool
    fit_mask: Array
    x_all: Array
    x_train: Array
    y_train: Array
    y_all: Array
    continuation_pred: Array
    intrinsic: Array
    exercise: Array
    cashflow_time: Array


@dataclass
class LSMCResult:
    price: float
    n_steps: int
    fit_all_paths: bool
    cashflow: Array
    exercise_step: Array
    exercise_time: Array
    discounted_cashflow: Array
    fits: list[TimeStepFit] = field(default_factory=list)

    @property
    def exercise_frequency(self) -> float:
        """Fraction of paths exercised before maturity."""
        return float(np.mean(self.exercise_step < self.n_steps))


def lsmc_price(
    paths: Array,
    times: Array,
    r: float,
    intrinsic_fn: IntrinsicFunction,
    feature_fn: FeatureFunction,
    regressor_factory: RegressorFactory,
    fit_all_paths: bool = True,
    store_fits: bool = False,
) -> LSMCResult:
    """Price an American/Bermudan-style option by LSMC backward induction.

    Args:
        paths: simulated paths, shape (n_paths, n_steps + 1).
        times: exercise-time grid, shape (n_steps + 1,).
        r: continuously compounded risk-free rate.
        intrinsic_fn: returns intrinsic values at a step.
        feature_fn: returns regression state/features at a step.
        regressor_factory: creates a fresh model per time step.
        fit_all_paths: if False, fit only paths currently ITM. The KANOP paper
            appears to use all paths in some continuation fits, especially for
            delta-related diagnostics, but ITM-only fitting materially changes
            prices. Experiment scripts expose and record this choice.
        store_fits: store trained regressors, full and masked state values,
            regression targets, fit masks, continuation predictions, exercise
            decisions, and cashflow-time diagnostics.
    """
    paths = np.asarray(paths, dtype=float)
    times = np.asarray(times, dtype=float)
    if paths.ndim != 2:
        raise ValueError("paths must have shape (n_paths, n_steps + 1)")
    n_paths, n_cols = paths.shape
    n_steps = n_cols - 1
    if times.shape != (n_steps + 1,):
        raise ValueError("times shape must match paths")

    # Terminal payoff and terminal exercise time.
    cashflow = intrinsic_fn(paths, n_steps).astype(float).copy()
    exercise_step = np.full(n_paths, n_steps, dtype=int)
    fits: list[TimeStepFit] = []

    # Move backward from t_{K-1} to t_1. t_0 is valuation time, not an exercise date.
    for k in range(n_steps - 1, 0, -1):
        x_all = feature_fn(paths, k)
        intrinsic = intrinsic_fn(paths, k)

        # Discount already-decided future cashflow back to current time t_k.
        future_times = times[exercise_step]
        y_all = cashflow * np.exp(-r * (future_times - times[k]))

        if fit_all_paths:
            fit_mask = np.ones(n_paths, dtype=bool)
        else:
            fit_mask = intrinsic > 0.0

        # Degenerate fallback: if no paths are fit-eligible, continuation is +inf
        # so no path exercises at this time.
        if not np.any(fit_mask):
            continuation = np.full(n_paths, np.inf)
            exercise = np.zeros(n_paths, dtype=bool)
            regressor = regressor_factory()
            x_train = x_all[fit_mask]
            y_train = y_all[fit_mask]
        else:
            regressor = regressor_factory()
            x_train = x_all[fit_mask]
            y_train = y_all[fit_mask]
            regressor.fit(x_train, y_train)
            continuation = np.asarray(regressor.predict(x_all), dtype=float).reshape(-1)
            exercise = (intrinsic > 0.0) & (intrinsic > continuation)

        cashflow[exercise] = intrinsic[exercise]
        exercise_step[exercise] = k

        if store_fits:
            fits.append(
                TimeStepFit(
                    step=k,
                    time=float(times[k]),
                    regressor=regressor,
                    fit_all_paths=fit_all_paths,
                    fit_mask=fit_mask.copy(),
                    x_all=x_all.copy(),
                    x_train=x_train.copy(),
                    y_train=y_train.copy(),
                    y_all=y_all.copy(),
                    continuation_pred=continuation.copy(),
                    intrinsic=intrinsic.copy(),
                    exercise=exercise.copy(),
                    cashflow_time=times[exercise_step].copy(),
                )
            )

    exercise_time = times[exercise_step]
    discounted = cashflow * np.exp(-r * exercise_time)
    return LSMCResult(
        price=float(np.mean(discounted)),
        n_steps=n_steps,
        fit_all_paths=fit_all_paths,
        cashflow=cashflow,
        exercise_step=exercise_step,
        exercise_time=exercise_time,
        discounted_cashflow=discounted,
        fits=fits,
    )
