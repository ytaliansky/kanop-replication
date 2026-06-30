"""Run Asian-American call baseline experiment with Laguerre cross-product LSMC."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kanop.basis import laguerre_total_degree_cross_basis
from kanop.config import ASIAN_AMERICAN_CASES
from kanop.lsmc import lsmc_price
from kanop.metrics import absolute_error, relative_error
from kanop.payoffs import asian_call_intrinsic_from_twap, twap_paths
from kanop.regression import make_ols_factory
from kanop.simulation import simulate_gbm_paths

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def run_case(case_index: int, seed: int = 1234) -> dict[str, float | int | str]:
    cfg = ASIAN_AMERICAN_CASES[case_index]
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
        return pd.DataFrame({
            "S": paths_array[:, step],
            "TWAP": twap[:, step],
        }).to_numpy()

    laguerre = lsmc_price(
        paths=paths,
        times=times,
        r=cfg.r,
        intrinsic_fn=intrinsic,
        feature_fn=feature,
        regressor_factory=make_ols_factory(lambda x: laguerre_total_degree_cross_basis(x, max_total_order=4)),
        fit_all_paths=True,
        store_fits=False,
    )

    # The paper's Eurasian price is calculated as discounted terminal Asian payoff.
    terminal_payoff = asian_call_intrinsic_from_twap(twap[:, -1], cfg.strike)
    eurasian_mc = float((terminal_payoff * np.exp(-cfg.r * cfg.maturity_years)).mean())

    target = cfg.paper_asian_american_price
    return {
        "case": case_index + 1,
        "S0": cfg.s0,
        "K": cfg.strike,
        "weeks": cfg.n_weeks,
        "sigma": cfg.sigma,
        "paper_eurasian_price": cfg.paper_eurasian_price,
        "replicated_eurasian_mc": eurasian_mc,
        "paper_asian_american_target": target,
        "laguerre_lsmc_price": laguerre.price,
        "abs_error_vs_paper_target": absolute_error(laguerre.price, target) if target is not None else None,
        "rel_error_vs_paper_target": relative_error(laguerre.price, target) if target is not None else None,
    }


def main(seed: int = 1234) -> None:
    rows = [run_case(i, seed=seed) for i in range(len(ASIAN_AMERICAN_CASES))]
    out = pd.DataFrame(rows)
    out_path = RESULTS / "asian_american_laguerre_baselines.csv"
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
