from kanop.basis import weighted_laguerre_basis
from kanop.config import AmericanPutConfig
from kanop.lsmc import lsmc_price
from kanop.payoffs import put_intrinsic
from kanop.regression import make_ols_factory
from kanop.simulation import simulate_gbm_paths


def _american_put_setup(n_paths=1000):
    cfg = AmericanPutConfig(n_paths=n_paths)
    paths, times = simulate_gbm_paths(
        cfg.s0,
        cfg.maturity_years,
        cfg.r,
        cfg.sigma,
        cfg.n_steps,
        cfg.n_paths,
        q=cfg.q,
        seed=42,
    )

    def intrinsic(paths_array, step):
        return put_intrinsic(paths_array[:, step], cfg.strike)

    def feature(paths_array, step):
        return paths_array[:, step : step + 1]

    return cfg, paths, times, intrinsic, feature


def test_lsmc_runs_on_american_put():
    cfg, paths, times, intrinsic, feature = _american_put_setup()

    result = lsmc_price(
        paths,
        times,
        cfg.r,
        intrinsic,
        feature,
        make_ols_factory(lambda x: weighted_laguerre_basis(x, max_order=5)),
        fit_all_paths=True,
    )
    assert result.price >= 0
    assert result.exercise_step.shape == (cfg.n_paths,)
    assert result.n_steps == cfg.n_steps
    assert result.exercise_time.shape == (cfg.n_paths,)


def test_lsmc_fit_all_paths_and_itm_only_both_run():
    cfg, paths, times, intrinsic, feature = _american_put_setup()

    results = [
        lsmc_price(
            paths,
            times,
            cfg.r,
            intrinsic,
            feature,
            make_ols_factory(lambda x: weighted_laguerre_basis(x, max_order=5)),
            fit_all_paths=fit_all_paths,
        )
        for fit_all_paths in (True, False)
    ]

    for result, fit_all_paths in zip(results, (True, False)):
        assert result.price >= 0
        assert result.n_steps == cfg.n_steps
        assert result.fit_all_paths is fit_all_paths
        assert 0.0 <= result.exercise_frequency <= 1.0


def test_lsmc_store_fits_keeps_step_diagnostics():
    cfg, paths, times, intrinsic, feature = _american_put_setup(n_paths=200)

    result = lsmc_price(
        paths,
        times,
        cfg.r,
        intrinsic,
        feature,
        make_ols_factory(lambda x: weighted_laguerre_basis(x, max_order=5)),
        fit_all_paths=True,
        store_fits=True,
    )

    assert result.fits
    first_fit = result.fits[0]
    assert first_fit.fit_all_paths is True
    assert first_fit.fit_mask.shape == (cfg.n_paths,)
    assert first_fit.x_all.shape == (cfg.n_paths, 1)
    assert first_fit.x_train.shape == (cfg.n_paths, 1)
    assert first_fit.y_all.shape == (cfg.n_paths,)
    assert first_fit.cashflow_time.shape == (cfg.n_paths,)
