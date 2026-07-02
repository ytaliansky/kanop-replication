"""Tiny smoke test for the PyTorch MLP regressor inside LSMC.

This is intentionally not a paper-replication run. It uses very few paths and
epochs to verify that the neural regressor satisfies the continuation-regressor
interface without making the full MLP baseline expensive.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kanop.config import AmericanPutConfig
from kanop.lsmc import lsmc_price
from kanop.models import TorchMLPRegressor
from kanop.payoffs import put_intrinsic
from kanop.simulation import simulate_gbm_paths


def main() -> None:
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

    result = lsmc_price(
        paths,
        times,
        cfg.r,
        intrinsic,
        feature,
        lambda: TorchMLPRegressor(
            layer_widths=(1, 16, 16, 1),
            learning_rate=5e-3,
            epochs=10,
            batch_size=128,
            random_seed=1234,
        ),
        fit_all_paths=True,
        store_fits=True,
    )
    print(f"MLP smoke LSMC price: {result.price:.6f}")
    print(f"Stored diagnostic fits: {len(result.fits)}")


if __name__ == "__main__":
    main()
