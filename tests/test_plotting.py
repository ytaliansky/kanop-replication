import matplotlib
import numpy as np

matplotlib.use("Agg")

from kanop.basis import hermite_basis, weighted_laguerre_basis
from kanop.config import AmericanPutConfig
from kanop.lsmc import lsmc_price
from kanop.payoffs import put_intrinsic
from kanop.plotting import american_put_true_continuation, plot_american_put_continuation_step
from kanop.regression import make_ols_factory
from kanop.simulation import simulate_gbm_paths


def test_american_put_true_continuation_is_sensible():
    cfg = AmericanPutConfig()
    stock = np.array([3.8, 4.0, 4.2])

    values = american_put_true_continuation(
        stock,
        strike=cfg.strike,
        maturity_years=cfg.maturity_years,
        time=cfg.maturity_years / 2.0,
        r=cfg.r,
        sigma=cfg.sigma,
        q=cfg.q,
    )

    assert values.shape == stock.shape
    assert np.all(values >= 0.0)
    assert values[0] > values[1] > values[2]


def test_american_put_continuation_plot_writes_file(tmp_path):
    cfg = AmericanPutConfig(n_paths=300, n_days=8)
    paths, times = simulate_gbm_paths(
        cfg.s0,
        cfg.maturity_years,
        cfg.r,
        cfg.sigma,
        cfg.n_steps,
        cfg.n_paths,
        q=cfg.q,
        seed=7,
    )

    def intrinsic(paths_array, step):
        return put_intrinsic(paths_array[:, step], cfg.strike)

    def feature(paths_array, step):
        return paths_array[:, step : step + 1]

    weighted = lsmc_price(
        paths,
        times,
        cfg.r,
        intrinsic,
        feature,
        make_ols_factory(lambda x: weighted_laguerre_basis(x, max_order=3)),
        fit_all_paths=True,
        store_fits=True,
    )
    hermite = lsmc_price(
        paths,
        times,
        cfg.r,
        intrinsic,
        feature,
        make_ols_factory(lambda x: hermite_basis(x, max_order=3)),
        fit_all_paths=True,
        store_fits=True,
    )

    output_path = tmp_path / "continuation.png"
    written = plot_american_put_continuation_step(
        paths=paths,
        times=times,
        fits_by_name={"Weighted Laguerre": weighted, "Hermite": hermite},
        step=4,
        strike=cfg.strike,
        maturity_years=cfg.maturity_years,
        r=cfg.r,
        sigma=cfg.sigma,
        q=cfg.q,
        output_path=output_path,
    )

    assert written == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
