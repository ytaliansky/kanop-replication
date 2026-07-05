import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kanop.delta import (
    DeltaDiagnostics,
    delta_from_t1_continuation_model,
    first_step_standard_normals_from_paths,
)
from kanop.models import TorchKANRegressor, TorchMLPRegressor


class MockPutLikeContinuation:
    def __init__(self, strike: float) -> None:
        self.strike = strike

    def predict_torch(self, x):
        return 0.6 * torch.clamp(self.strike - x[:, 0], min=0.0)


def test_delta_returns_finite_negative_value_for_mock_put_like_model():
    diagnostics = delta_from_t1_continuation_model(
        MockPutLikeContinuation(strike=4.0),
        z_t1=np.array([-0.5, 0.0, 0.5], dtype=np.float32),
        s0=4.0,
        strike=4.0,
        sigma=0.20,
        r=0.0,
        q=0.0,
        dt=1 / 252,
        return_diagnostics=True,
    )

    assert isinstance(diagnostics, DeltaDiagnostics)
    assert np.isfinite(diagnostics.delta)
    assert diagnostics.delta < 0.0
    assert diagnostics.n_paths == 3


def test_first_step_standard_normals_are_recovered_from_paths():
    s0 = 4.0
    sigma = 0.2
    r = 0.04
    q = 0.0
    dt = 1 / 252
    z = np.array([-1.0, 0.0, 1.0])
    s1 = s0 * np.exp((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    paths = np.column_stack([np.full_like(s1, s0), s1])

    recovered = first_step_standard_normals_from_paths(paths, s0=s0, sigma=sigma, r=r, q=q, dt=dt)

    assert np.allclose(recovered, z)


def test_mlp_predict_torch_supports_full_delta_path():
    x = np.linspace(3.5, 4.5, 80, dtype=np.float32)[:, None]
    y = np.maximum(4.0 - x[:, 0], 0.0)
    regressor = TorchMLPRegressor(
        layer_widths=(1, 8, 1),
        learning_rate=5e-3,
        epochs=20,
        batch_size=40,
        random_seed=1,
    ).fit(x, y)

    delta = delta_from_t1_continuation_model(
        regressor,
        z_t1=np.linspace(-1.0, 1.0, 12),
        s0=4.0,
        strike=4.0,
        sigma=0.20,
        r=0.0,
        q=0.0,
        dt=1 / 252,
    )

    assert np.isfinite(delta)


def test_kan_predict_torch_supports_full_delta_path():
    x = np.linspace(3.5, 4.5, 80, dtype=np.float32)[:, None]
    y = np.maximum(4.0 - x[:, 0], 0.0)
    regressor = TorchKANRegressor(
        widths=(1, 3, 1),
        grid_size=7,
        learning_rate=5e-3,
        epochs=20,
        batch_size=40,
        random_seed=2,
    ).fit(x, y)

    delta = delta_from_t1_continuation_model(
        regressor,
        z_t1=np.linspace(-1.0, 1.0, 12),
        s0=4.0,
        strike=4.0,
        sigma=0.20,
        r=0.0,
        q=0.0,
        dt=1 / 252,
    )

    assert np.isfinite(delta)
