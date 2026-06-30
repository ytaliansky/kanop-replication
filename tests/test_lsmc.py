from kanop.basis import weighted_laguerre_basis
from kanop.config import AmericanPutConfig
from kanop.lsmc import lsmc_price
from kanop.payoffs import put_intrinsic
from kanop.regression import make_ols_factory
from kanop.simulation import simulate_gbm_paths


def test_lsmc_runs_on_american_put():
    cfg = AmericanPutConfig(n_paths=1000)
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
