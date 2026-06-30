"""Run the paper's American put baseline experiments with OLS basis models.

This reproduces the non-neural starting point:
- Weighted Laguerre model, first six orders L_0..L_5 with exp(-x/2) weighting.
- Hermite model, first six orders H_0..H_5.

Next steps are to plug in MLP and KAN regressors through the same lsmc_price API.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from kanop.basis import hermite_basis, weighted_laguerre_basis
from kanop.black_scholes import bs_delta, bs_price
from kanop.config import AmericanPutConfig
from kanop.lsmc import lsmc_price
from kanop.metrics import absolute_error, relative_error
from kanop.payoffs import put_intrinsic
from kanop.regression import make_ols_factory
from kanop.scaling import BasisInputScaler, ScalingMode
from kanop.simulation import simulate_gbm_paths

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


def american_put_result_row(
    *,
    model: str,
    seed: int,
    n_paths: int,
    fit_all_paths: bool,
    basis_scaling: str,
    price: float,
    black_scholes_target: float,
    paper_model_price: float,
    runtime_seconds: float,
) -> dict[str, float | int | str | bool]:
    return {
        "model": model,
        "seed": seed,
        "n_paths": n_paths,
        "fit_all_paths": fit_all_paths,
        "basis_scaling": basis_scaling,
        "price": price,
        "black_scholes_target": black_scholes_target,
        "paper_model_price": paper_model_price,
        "abs_error_vs_bs": absolute_error(price, black_scholes_target),
        "rel_error_vs_bs": relative_error(price, black_scholes_target),
        "difference_from_paper_model": price - paper_model_price,
        "runtime_seconds": runtime_seconds,
    }


def run_baselines(
    seed: int = 1234,
    fit_all_paths: bool = True,
    basis_scaling: ScalingMode = "raw",
    store_fits: bool = True,
) -> tuple[pd.DataFrame, dict[str, object], object, object]:
    cfg = AmericanPutConfig()
    scaler = BasisInputScaler(mode=basis_scaling, s0=cfg.s0, strike=cfg.strike)
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
        return scaler.transform(paths_array[:, step : step + 1])

    start = time.perf_counter()
    weighted_laguerre = lsmc_price(
        paths=paths,
        times=times,
        r=cfg.r,
        intrinsic_fn=intrinsic,
        feature_fn=feature,
        regressor_factory=make_ols_factory(lambda x: weighted_laguerre_basis(x, max_order=5)),
        fit_all_paths=fit_all_paths,
        store_fits=store_fits,
    )
    weighted_laguerre_seconds = time.perf_counter() - start

    start = time.perf_counter()
    hermite = lsmc_price(
        paths=paths,
        times=times,
        r=cfg.r,
        intrinsic_fn=intrinsic,
        feature_fn=feature,
        regressor_factory=make_ols_factory(lambda x: hermite_basis(x, max_order=5)),
        fit_all_paths=fit_all_paths,
        store_fits=store_fits,
    )
    hermite_seconds = time.perf_counter() - start

    bs_p = bs_price(cfg.s0, cfg.strike, cfg.maturity_years, cfg.r, cfg.sigma, "put", q=cfg.q)
    rows = [
        american_put_result_row(
            model="Weighted Laguerre LSMC",
            seed=seed,
            n_paths=cfg.n_paths,
            fit_all_paths=fit_all_paths,
            basis_scaling=basis_scaling,
            price=weighted_laguerre.price,
            black_scholes_target=bs_p,
            paper_model_price=cfg.paper_model_targets["weighted_laguerre"].price,
            runtime_seconds=weighted_laguerre_seconds,
        ),
        american_put_result_row(
            model="Hermite LSMC",
            seed=seed,
            n_paths=cfg.n_paths,
            fit_all_paths=fit_all_paths,
            basis_scaling=basis_scaling,
            price=hermite.price,
            black_scholes_target=bs_p,
            paper_model_price=cfg.paper_model_targets["hermite"].price,
            runtime_seconds=hermite_seconds,
        ),
    ]

    metadata = {
        "paths": paths,
        "times": times,
        "config": cfg,
        "scaler": scaler,
        "black_scholes_delta_exact": bs_delta(
            cfg.s0,
            cfg.strike,
            cfg.maturity_years,
            cfg.r,
            cfg.sigma,
            "put",
            q=cfg.q,
        ),
    }
    return pd.DataFrame(rows), metadata, weighted_laguerre, hermite


def main(
    seed: int = 1234,
    fit_all_paths: bool = True,
    basis_scaling: ScalingMode = "raw",
    write_plot: bool = True,
) -> None:
    out, metadata, weighted_laguerre, hermite = run_baselines(
        seed=seed,
        fit_all_paths=fit_all_paths,
        basis_scaling=basis_scaling,
        store_fits=True,
    )
    out_path = RESULTS / "american_put_baselines.csv"
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"\nWrote {out_path}")

    if not write_plot:
        return

    from kanop.plotting import plot_american_put_continuation

    plot_path = FIGURES / "american_put_continuation_laguerre_hermite.png"
    cfg = metadata["config"]
    paths = metadata["paths"]
    times = metadata["times"]
    scaler = metadata["scaler"]
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
        feature_transform=scaler.transform,
    )
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--basis-scaling", choices=["raw", "S_over_K", "S_over_S0", "standardized"], default="raw")
    parser.add_argument(
        "--fit-itm-only",
        action="store_true",
        help=(
            "Fit continuation regressions only on in-the-money paths. "
            "Default fits all paths, which appears closer to some paper diagnostics."
        ),
    )
    parser.add_argument("--skip-plot", action="store_true", help="Write the CSV table without generating a figure.")
    args = parser.parse_args()
    main(
        seed=args.seed,
        fit_all_paths=not args.fit_itm_only,
        basis_scaling=args.basis_scaling,
        write_plot=not args.skip_plot,
    )
