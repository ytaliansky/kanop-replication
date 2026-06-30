"""Run the paper's American put baseline experiments with OLS basis models.

This reproduces the non-neural starting point:
- Weighted Laguerre model, first six orders L_0..L_5 with exp(-x/2) weighting.
- Hermite model, first six orders H_0..H_5.

Next steps are to plug in MLP and KAN regressors through the same lsmc_price API.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from kanop.basis import hermite_basis, weighted_laguerre_basis
from kanop.black_scholes import bs_delta, bs_price
from kanop.config import AmericanPutConfig
from kanop.lsmc import lsmc_price
from kanop.metrics import absolute_error, relative_error
from kanop.payoffs import put_intrinsic
from kanop.plotting import plot_american_put_continuation
from kanop.regression import make_ols_factory
from kanop.simulation import simulate_gbm_paths

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


def main(seed: int = 1234) -> None:
    cfg = AmericanPutConfig()
    paths, times = simulate_gbm_paths(
        s0=cfg.s0,
        maturity_years=cfg.maturity_years,
        r=cfg.r,
        q=cfg.q,
        sigma=cfg.sigma,
        n_steps=cfg.n_steps,
        n_paths=cfg.n_paths,
        seed=seed,
    )

    def intrinsic(paths_array, step: int):
        return put_intrinsic(paths_array[:, step], cfg.strike)

    def feature(paths_array, step: int):
        return paths_array[:, step : step + 1]

    weighted_laguerre = lsmc_price(
        paths=paths,
        times=times,
        r=cfg.r,
        intrinsic_fn=intrinsic,
        feature_fn=feature,
        regressor_factory=make_ols_factory(lambda x: weighted_laguerre_basis(x, max_order=5)),
        fit_all_paths=True,
        store_fits=True,
    )

    hermite = lsmc_price(
        paths=paths,
        times=times,
        r=cfg.r,
        intrinsic_fn=intrinsic,
        feature_fn=feature,
        regressor_factory=make_ols_factory(lambda x: hermite_basis(x, max_order=5)),
        fit_all_paths=True,
        store_fits=True,
    )

    bs_p = bs_price(cfg.s0, cfg.strike, cfg.maturity_years, cfg.r, cfg.sigma, "put", q=cfg.q)
    bs_d = bs_delta(cfg.s0, cfg.strike, cfg.maturity_years, cfg.r, cfg.sigma, "put", q=cfg.q)

    rows = [
        {
            "model": "Black-Scholes exact formula",
            "price": bs_p,
            "price_target_used": cfg.paper_bs_price,
            "abs_error_vs_paper_price": absolute_error(bs_p, cfg.paper_bs_price),
            "rel_error_vs_paper_price": relative_error(bs_p, cfg.paper_bs_price),
            "delta_exact_formula": bs_d,
            "paper_delta_target": cfg.paper_bs_delta,
        },
        {
            "model": "Weighted Laguerre LSMC",
            "price": weighted_laguerre.price,
            "price_target_used": cfg.paper_bs_price,
            "abs_error_vs_paper_price": absolute_error(weighted_laguerre.price, cfg.paper_bs_price),
            "rel_error_vs_paper_price": relative_error(weighted_laguerre.price, cfg.paper_bs_price),
            "delta_exact_formula": None,
            "paper_delta_target": cfg.paper_bs_delta,
        },
        {
            "model": "Hermite LSMC",
            "price": hermite.price,
            "price_target_used": cfg.paper_bs_price,
            "abs_error_vs_paper_price": absolute_error(hermite.price, cfg.paper_bs_price),
            "rel_error_vs_paper_price": relative_error(hermite.price, cfg.paper_bs_price),
            "delta_exact_formula": None,
            "paper_delta_target": cfg.paper_bs_delta,
        },
    ]

    out = pd.DataFrame(rows)
    out_path = RESULTS / "american_put_baselines.csv"
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"\nWrote {out_path}")

    plot_path = FIGURES / "american_put_continuation_laguerre_hermite.png"
    plot_american_put_continuation(
        paths=paths,
        times=times,
        fits_by_name={
            "Weighted Laguerre": weighted_laguerre,
            "Hermite": hermite,
        },
        strike=cfg.strike,
        maturity_years=cfg.maturity_years,
        r=cfg.r,
        sigma=cfg.sigma,
        q=cfg.q,
        steps_to_plot=(49, 25, 1),
        output_path=str(plot_path),
    )
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
