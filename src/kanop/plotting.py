"""Plotting helpers for continuation-value diagnostics."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .black_scholes import bs_price


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
) -> None:
    """Plot fitted continuation curves against Black-Scholes European put curves."""
    # One figure with one panel per requested time step. This helper is for quick
    # diagnostics. For paper-quality plots, create separate figures per model/time.
    n = len(steps_to_plot)
    fig, axes = plt.subplots(n, 1, figsize=(8, 3.5 * n), squeeze=False)

    for ax, step in zip(axes[:, 0], steps_to_plot):
        s_at_step = paths[:, step]
        grid = np.linspace(np.percentile(s_at_step, 1), np.percentile(s_at_step, 99), 250)
        tau = maturity_years - times[step]
        bs_vals = np.array([
            bs_price(s, strike, tau, r, sigma, option_type="put", q=q) for s in grid
        ])
        ax.plot(grid, bs_vals, label="European put target")

        for name, result in fits_by_name.items():
            matching = [fit for fit in result.fits if fit.step == step]
            if not matching:
                continue
            fit = matching[0]
            pred = fit.regressor.predict(grid[:, None])
            ax.plot(grid, pred, label=name)

        ax.set_title(f"Continuation approximation at step t_{step}")
        ax.set_xlabel("Stock price")
        ax.set_ylabel("Continuation value")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
