"""Risk-neutral path simulation."""

from __future__ import annotations

import numpy as np


def make_time_grid(maturity_years: float, n_steps: int) -> np.ndarray:
    """Return a grid from 0 to maturity inclusive, shape (n_steps + 1,)."""
    if maturity_years <= 0:
        raise ValueError("maturity_years must be positive")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    return np.linspace(0.0, maturity_years, n_steps + 1)


def simulate_gbm_paths(
    s0: float,
    maturity_years: float,
    r: float,
    sigma: float,
    n_steps: int,
    n_paths: int,
    q: float = 0.0,
    seed: int | None = None,
    antithetic: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate risk-neutral geometric Brownian motion paths.

    Dynamics under Q:
        dS_t / S_t = (r - q) dt + sigma dW_t

    Returns:
        paths: shape (n_paths, n_steps + 1)
        times: shape (n_steps + 1,)
    """
    if s0 <= 0:
        raise ValueError("s0 must be positive")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")

    rng = np.random.default_rng(seed)
    times = make_time_grid(maturity_years, n_steps)
    dt = maturity_years / n_steps

    if antithetic:
        half = (n_paths + 1) // 2
        z_half = rng.standard_normal((half, n_steps))
        z = np.vstack([z_half, -z_half])[:n_paths]
    else:
        z = rng.standard_normal((n_paths, n_steps))

    increments = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.cumsum(increments, axis=1)
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = s0
    paths[:, 1:] = s0 * np.exp(log_paths)
    return paths, times
