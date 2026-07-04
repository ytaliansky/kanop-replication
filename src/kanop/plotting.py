"""Plotting helpers for continuation-value diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .black_scholes import bs_price
from .lsmc import LSMCResult


def american_put_true_continuation(
    stock: np.ndarray,
    *,
    strike: float,
    maturity_years: float,
    time: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> np.ndarray:
    """Black-Scholes European put continuation benchmark at time ``time``."""
    stock = np.asarray(stock, dtype=float)
    tau = maturity_years - time
    if tau <= 0.0:
        return np.maximum(strike - stock, 0.0)
    return np.array([bs_price(s, strike, tau, r, sigma, option_type="put", q=q) for s in stock])


def _fit_for_step(result: LSMCResult, step: int):
    matches = [fit for fit in result.fits if fit.step == step]
    if not matches:
        raise ValueError(f"no stored diagnostic fit for step {step}")
    return matches[0]


def stock_price_grid_at_step(paths: np.ndarray, step: int, n_points: int = 300) -> np.ndarray:
    """Return a sorted stock-price grid covering the simulated range at ``step``."""
    s_at_step = np.asarray(paths[:, step], dtype=float)
    lo = float(np.min(s_at_step))
    hi = float(np.max(s_at_step))
    if lo == hi:
        pad = max(abs(lo) * 0.01, 1e-6)
        lo -= pad
        hi += pad
    return np.linspace(lo, hi, n_points)


def plot_american_put_continuation_step(
    *,
    paths: np.ndarray,
    times: np.ndarray,
    fits_by_name: dict[str, LSMCResult],
    step: int,
    strike: float,
    maturity_years: float,
    r: float,
    sigma: float,
    q: float,
    output_path: str | Path,
    feature_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    feature_transforms_by_name: dict[str, Callable[[np.ndarray], np.ndarray] | None] | None = None,
    reference_label: str = "Black-Scholes continuation",
    n_grid: int = 300,
) -> Path:
    """Save a paper-style continuation plot for one exercise step."""
    grid = stock_price_grid_at_step(paths, step, n_points=n_grid)
    true_vals = american_put_true_continuation(
        grid,
        strike=strike,
        maturity_years=maturity_years,
        time=float(times[step]),
        r=r,
        sigma=sigma,
        q=q,
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(grid, true_vals, label=reference_label, color="black", linewidth=2.0)

    for name, result in fits_by_name.items():
        fit = _fit_for_step(result, step)
        x_grid = grid[:, None]
        transform = feature_transform
        if feature_transforms_by_name is not None and name in feature_transforms_by_name:
            transform = feature_transforms_by_name[name]
        if transform is not None:
            x_grid = transform(x_grid)
        pred = np.asarray(fit.regressor.predict(x_grid), dtype=float).reshape(-1)
        ax.plot(grid, pred, label=name, linewidth=1.8)

    ax.set_title(f"American put continuation at t{step}")
    ax.set_xlabel("Stock price")
    ax.set_ylabel("Continuation value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_american_put_continuation_steps(
    *,
    paths: np.ndarray,
    times: np.ndarray,
    fits_by_name: dict[str, LSMCResult],
    strike: float,
    maturity_years: float,
    r: float,
    sigma: float,
    q: float,
    steps_to_plot: tuple[int, ...],
    output_dir: str | Path,
    feature_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    feature_transforms_by_name: dict[str, Callable[[np.ndarray], np.ndarray] | None] | None = None,
    reference_label: str = "Black-Scholes continuation",
    filename_template: str = "american_put_continuation_baselines_t{step}.png",
) -> list[Path]:
    """Save one continuation plot per requested exercise step."""
    output_dir = Path(output_dir)
    return [
        plot_american_put_continuation_step(
            paths=paths,
            times=times,
            fits_by_name=fits_by_name,
            step=step,
            strike=strike,
            maturity_years=maturity_years,
            r=r,
            sigma=sigma,
            q=q,
            output_path=output_dir / filename_template.format(step=step),
            feature_transform=feature_transform,
            feature_transforms_by_name=feature_transforms_by_name,
            reference_label=reference_label,
        )
        for step in steps_to_plot
    ]


def plot_american_put_continuation(
    paths: np.ndarray,
    times: np.ndarray,
    fits_by_name: dict[str, object],
    strike: float,
    maturity_years: float,
    r: float,
    sigma: float,
    q: float,
    steps_to_plot: tuple[int, ...],
    output_path: str,
    feature_transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> None:
    """Plot fitted continuation curves against Black-Scholes European put curves.

    This stacked diagnostic is retained for compatibility. Prefer
    ``plot_american_put_continuation_steps`` for paper-style single-step files.
    """
    # One figure with one panel per requested time step. This helper is for quick
    # diagnostics. For paper-quality plots, create separate figures per model/time.
    n = len(steps_to_plot)
    fig, axes = plt.subplots(n, 1, figsize=(8, 3.5 * n), squeeze=False)

    for ax, step in zip(axes[:, 0], steps_to_plot):
        s_at_step = paths[:, step]
        grid = np.linspace(np.percentile(s_at_step, 1), np.percentile(s_at_step, 99), 250)
        bs_vals = american_put_true_continuation(
            grid,
            strike=strike,
            maturity_years=maturity_years,
            time=float(times[step]),
            r=r,
            sigma=sigma,
            q=q,
        )
        ax.plot(grid, bs_vals, label="European put target")

        for name, result in fits_by_name.items():
            matching = [fit for fit in result.fits if fit.step == step]
            if not matching:
                continue
            fit = matching[0]
            x_grid = grid[:, None]
            if feature_transform is not None:
                x_grid = feature_transform(x_grid)
            pred = fit.regressor.predict(x_grid)
            ax.plot(grid, pred, label=name)

        ax.set_title(f"Continuation approximation at step t_{step}")
        ax.set_xlabel("Stock price")
        ax.set_ylabel("Continuation value")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
