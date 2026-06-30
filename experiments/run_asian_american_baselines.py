"""Run Asian-American call baseline experiment with Laguerre cross-product LSMC."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from kanop.basis import laguerre_total_degree_cross_basis
from kanop.config import ASIAN_AMERICAN_CASES
from kanop.lsmc import lsmc_price
from kanop.metrics import absolute_error, relative_error
from kanop.payoffs import asian_call_intrinsic_from_twap, twap_paths
from kanop.regression import make_ols_factory
from kanop.scaling import BasisInputScaler, ScalingMode
from kanop.simulation import simulate_gbm_paths

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def asian_american_result_row(
    *,
    case_id: int,
    strike: float,
    weeks: int,
    sigma: float,
    model: str,
    price: float,
    paper_model_price: float,
    target_asian_american_price: float,
    runtime_seconds: float,
    seed: int,
    n_paths: int,
    fit_all_paths: bool,
    basis_scaling: str,
) -> dict[str, float | int | str | bool]:
    return {
        "case_id": case_id,
        "K": strike,
        "weeks": weeks,
        "sigma": sigma,
        "model": model,
        "price": price,
        "paper_model_price": paper_model_price,
        "target_asian_american_price": target_asian_american_price,
        "abs_error_vs_target": absolute_error(price, target_asian_american_price),
        "rel_error_vs_target": relative_error(price, target_asian_american_price),
        "difference_from_paper_model": price - paper_model_price,
        "runtime_seconds": runtime_seconds,
        "seed": seed,
        "n_paths": n_paths,
        "fit_all_paths": fit_all_paths,
        "basis_scaling": basis_scaling,
    }


def run_case(
    case_index: int,
    seed: int = 1234,
    fit_all_paths: bool = True,
    basis_scaling: ScalingMode = "raw",
) -> dict[str, float | int | str | bool]:
    cfg = ASIAN_AMERICAN_CASES[case_index]
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
    twap = twap_paths(paths, include_initial=False)

    def intrinsic(paths_array, step: int):
        # paths_array is not used directly because TWAP is precomputed from the same paths.
        return asian_call_intrinsic_from_twap(twap[:, step], cfg.strike)

    def feature(paths_array, step: int):
        raw_features = pd.DataFrame({
            "S": paths_array[:, step],
            "TWAP": twap[:, step],
        }).to_numpy()
        return scaler.transform(raw_features)

    start = time.perf_counter()
    laguerre = lsmc_price(
        paths=paths,
        times=times,
        r=cfg.r,
        intrinsic_fn=intrinsic,
        feature_fn=feature,
        regressor_factory=make_ols_factory(lambda x: laguerre_total_degree_cross_basis(x, max_total_order=4)),
        fit_all_paths=fit_all_paths,
        store_fits=False,
    )
    runtime_seconds = time.perf_counter() - start

    return asian_american_result_row(
        case_id=case_index + 1,
        strike=cfg.strike,
        weeks=cfg.n_weeks,
        sigma=cfg.sigma,
        model="Laguerre LSMC",
        price=laguerre.price,
        paper_model_price=cfg.paper_model_targets["laguerre"].price,
        target_asian_american_price=cfg.paper_model_targets["asian_american"].price,
        runtime_seconds=runtime_seconds,
        seed=seed,
        n_paths=cfg.n_paths,
        fit_all_paths=fit_all_paths,
        basis_scaling=basis_scaling,
    )


def main(seed: int = 1234, fit_all_paths: bool = True, basis_scaling: ScalingMode = "raw") -> None:
    rows = [
        run_case(i, seed=seed, fit_all_paths=fit_all_paths, basis_scaling=basis_scaling)
        for i in range(len(ASIAN_AMERICAN_CASES))
    ]
    out = pd.DataFrame(rows)
    out_path = RESULTS / "asian_american_laguerre_baselines.csv"
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--basis-scaling", choices=["raw", "S_over_K", "S_over_S0", "standardized"], default="raw")
    parser.add_argument(
        "--fit-itm-only",
        action="store_true",
        help=(
            "Fit continuation regressions only on in-the-money paths. "
            "Default fits all paths, matching the current baseline convention."
        ),
    )
    args = parser.parse_args()
    main(seed=args.seed, fit_all_paths=not args.fit_itm_only, basis_scaling=args.basis_scaling)
