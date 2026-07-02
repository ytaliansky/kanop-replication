"""Tiny smoke test for the PyTorch KAN-style regressor.

This is not a paper-scale KANOP experiment. It verifies that the self-contained
piecewise-linear spline-edge KAN can fit a smooth function and run inside the
existing LSMC engine.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from kanop.config import AmericanPutConfig
from kanop.lsmc import lsmc_price
from kanop.models import TorchKANRegressor
from kanop.payoffs import put_intrinsic
from kanop.simulation import simulate_gbm_paths


def fit_smooth_function() -> float:
    x = np.linspace(-2.0, 2.0, 160, dtype=np.float32)[:, None]
    y = np.sin(x[:, 0]) + 0.1 * x[:, 0]
    regressor = TorchKANRegressor(
        widths=(1, 3, 1),
        grid_size=21,
        learning_rate=5e-3,
        epochs=300,
        batch_size=64,
        random_seed=1234,
    ).fit(x, y)
    pred = regressor.predict(x)
    return float(np.mean((pred - y) ** 2))


def run_lsmc_smoke():
    cfg = AmericanPutConfig(n_paths=500, n_days=8)
    paths, times = simulate_gbm_paths(
        cfg.s0,
        cfg.maturity_years,
        cfg.r,
        cfg.sigma,
        cfg.n_steps,
        cfg.n_paths,
        q=cfg.q,
        seed=1234,
    )

    def intrinsic(paths_array, step):
        return put_intrinsic(paths_array[:, step], cfg.strike)

    def feature(paths_array, step):
        return paths_array[:, step : step + 1]

    return lsmc_price(
        paths,
        times,
        cfg.r,
        intrinsic,
        feature,
        lambda: TorchKANRegressor(
            widths=(1, 3, 1),
            grid_size=15,
            learning_rate=5e-3,
            epochs=10,
            batch_size=128,
            random_seed=1234,
        ),
        fit_all_paths=True,
        store_fits=True,
    )


def main() -> None:
    mse = fit_smooth_function()
    result = run_lsmc_smoke()
    print(f"KAN smoke function-fit MSE: {mse:.6f}")
    print(f"KAN smoke LSMC price: {result.price:.6f}")
    print(f"Stored diagnostic fits: {len(result.fits)}")
    print(f"First diagnostic regressor: {type(result.fits[0].regressor).__name__}")


if __name__ == "__main__":
    main()
