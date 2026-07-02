import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kanop.config import AmericanPutConfig
from kanop.lsmc import lsmc_price
from kanop.models import TorchMLPRegressor
from kanop.payoffs import put_intrinsic
from kanop.simulation import simulate_gbm_paths


def test_torch_mlp_regressor_fits_simple_1d_function():
    x = np.linspace(-1.0, 1.0, 160, dtype=np.float32)[:, None]
    y = x[:, 0] ** 2 + 0.25 * x[:, 0]

    regressor = TorchMLPRegressor(
        layer_widths=(1, 16, 16, 1),
        learning_rate=5e-3,
        epochs=350,
        batch_size=64,
        random_seed=11,
    ).fit(x, y)

    pred = regressor.predict(x)
    assert pred.shape == (x.shape[0],)
    assert np.mean((pred - y) ** 2) < 5e-3


def test_torch_mlp_regressor_fits_simple_2d_function():
    grid = np.linspace(-1.0, 1.0, 14, dtype=np.float32)
    x0, x1 = np.meshgrid(grid, grid)
    x = np.column_stack([x0.reshape(-1), x1.reshape(-1)]).astype(np.float32)
    y = 0.5 * x[:, 0] ** 2 + x[:, 1] - 0.25 * x[:, 0] * x[:, 1]

    regressor = TorchMLPRegressor(
        layer_widths=(2, 24, 24, 1),
        learning_rate=5e-3,
        epochs=450,
        batch_size=64,
        random_seed=17,
    ).fit(x, y)

    pred = regressor.predict(x)
    assert pred.shape == (x.shape[0],)
    assert np.mean((pred - y) ** 2) < 1e-2


def test_torch_mlp_predict_torch_shape_and_gradients():
    x = np.linspace(-1.0, 1.0, 80, dtype=np.float32)[:, None]
    y = np.sin(x[:, 0])
    regressor = TorchMLPRegressor(
        layer_widths=(1, 12, 12, 1),
        learning_rate=5e-3,
        epochs=250,
        batch_size=40,
        random_seed=23,
    ).fit(x, y)

    x_torch = torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float32, requires_grad=True)
    pred = regressor.predict_torch(x_torch)
    assert pred.shape == (3,)
    assert isinstance(pred, torch.Tensor)

    pred.sum().backward()
    assert x_torch.grad is not None
    assert x_torch.grad.shape == x_torch.shape
    assert torch.isfinite(x_torch.grad).all()
    assert torch.any(torch.abs(x_torch.grad) > 0.0)


def test_torch_mlp_regressor_runs_inside_lsmc_smoke():
    cfg = AmericanPutConfig(n_paths=120, n_days=4)
    paths, times = simulate_gbm_paths(
        cfg.s0,
        cfg.maturity_years,
        cfg.r,
        cfg.sigma,
        cfg.n_steps,
        cfg.n_paths,
        q=cfg.q,
        seed=5,
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
            layer_widths=(1, 8, 1),
            learning_rate=1e-2,
            epochs=3,
            batch_size=64,
            random_seed=31,
        ),
        fit_all_paths=True,
        store_fits=True,
    )

    assert result.price >= 0.0
    assert result.n_steps == cfg.n_steps
    assert len(result.fits) == cfg.n_steps - 1
    assert result.fits[0].continuation_pred.shape == (cfg.n_paths,)
