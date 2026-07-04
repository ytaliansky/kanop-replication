"""Run American put LSMC models at multiple risk-free rates.

This experiment leaves the paper's r=0 replication unchanged and adds a
rate-sensitivity check. For r>0, the American put's early-exercise feature is
economically meaningful, so the Cox-Ross-Rubinstein American put price is the
main benchmark. The European Black-Scholes price is reported as a lower-bound
reference.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

from kanop.basis import hermite_basis, weighted_laguerre_basis
from kanop.binomial import american_put_binomial_price
from kanop.black_scholes import bs_price
from kanop.config import AmericanPutConfig
from kanop.lsmc import LSMCResult, lsmc_price
from kanop.models import TorchKANRegressor, TorchMLPRegressor
from kanop.payoffs import put_intrinsic
from kanop.regression import make_ols_factory
from kanop.scaling import BasisInputScaler, ScalingMode
from kanop.simulation import simulate_gbm_paths

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

DEFAULT_KAN_ARCHITECTURE = (1, 3, 1)
DEFAULT_MLP_ARCHITECTURE = (1, 32, 32, 1)


def configure_matplotlib_cache() -> None:
    """Point Matplotlib/font caches at writable project-local directories."""
    mpl_config = ROOT / ".matplotlib_cache"
    xdg_cache = ROOT / ".cache"
    fallback_home = xdg_cache / "home"
    for path in (mpl_config, xdg_cache, fallback_home):
        path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))
    os.environ["HOME"] = str(fallback_home)


def architecture_label(architecture: tuple[int, ...]) -> str:
    return "[" + ", ".join(str(width) for width in architecture) + "]"


def exercise_diagnostics(result: LSMCResult) -> dict[str, float | int]:
    early_mask = result.exercise_step < result.n_steps
    early_count = int(np.sum(early_mask))
    if early_count:
        average_exercise_time = float(np.mean(result.exercise_time[early_mask]))
    else:
        average_exercise_time = float("nan")
    return {
        "early_exercise_count": early_count,
        "early_exercise_fraction": float(np.mean(early_mask)),
        "average_exercise_time": average_exercise_time,
    }


def rate_sensitivity_row(
    *,
    r: float,
    model: str,
    seed: int,
    n_paths: int,
    fit_all_paths: bool,
    result: LSMCResult,
    european_bs_price: float,
    american_binomial_price: float,
    runtime_seconds: float,
    architecture: tuple[int, ...] | None = None,
    epochs: int | None = None,
    learning_rate: float | None = None,
    batch_size: int | None = None,
    weight_decay: float | None = None,
    grid_size: int | None = None,
    device: str | None = None,
    basis_scaling: str | None = None,
) -> dict[str, float | int | str | bool | None]:
    row = {
        "r": r,
        "model": model,
        "seed": seed,
        "n_paths": n_paths,
        "fit_all_paths": fit_all_paths,
        "price": result.price,
        "european_bs_price": european_bs_price,
        "american_binomial_price": american_binomial_price,
        "error_vs_american_binomial": result.price - american_binomial_price,
        "difference_vs_european_bs": result.price - european_bs_price,
        **exercise_diagnostics(result),
        "runtime_seconds": runtime_seconds,
        "architecture": architecture_label(architecture) if architecture is not None else None,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "grid_size": grid_size,
        "device": device,
        "basis_scaling": basis_scaling,
    }
    return row


def run_rate_case(
    *,
    r: float,
    seed: int,
    n_paths: int,
    fit_all_paths: bool,
    kan_epochs: int,
    mlp_epochs: int,
    learning_rate: float,
    batch_size: int,
    weight_decay: float,
    grid_size: int,
    device: str,
    include_mlp: bool,
    binomial_steps: int,
    basis_scaling: ScalingMode = "raw",
) -> tuple[list[dict[str, float | int | str | bool | None]], dict[str, object], dict[str, LSMCResult]]:
    base_cfg = AmericanPutConfig()
    cfg = AmericanPutConfig(n_paths=n_paths, r=r)
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
    scaler = BasisInputScaler(mode=basis_scaling, s0=cfg.s0, strike=cfg.strike)

    def intrinsic(paths_array, step: int):
        return put_intrinsic(paths_array[:, step], cfg.strike)

    def raw_feature(paths_array, step: int):
        return paths_array[:, step : step + 1]

    def basis_feature(paths_array, step: int):
        return scaler.transform(raw_feature(paths_array, step))

    european_bs = bs_price(cfg.s0, cfg.strike, cfg.maturity_years, cfg.r, cfg.sigma, "put", q=cfg.q)
    american_binomial = american_put_binomial_price(
        cfg.s0,
        cfg.strike,
        cfg.maturity_years,
        cfg.sigma,
        cfg.r,
        q=cfg.q,
        n_steps=binomial_steps,
    )

    rows: list[dict[str, float | int | str | bool | None]] = []
    results: dict[str, LSMCResult] = {}

    basis_specs = (
        ("Weighted Laguerre LSMC", lambda x: weighted_laguerre_basis(x, max_order=5)),
        ("Hermite LSMC", lambda x: hermite_basis(x, max_order=5)),
    )
    for model_name, basis_fn in basis_specs:
        start = time.perf_counter()
        result = lsmc_price(
            paths=paths,
            times=times,
            r=cfg.r,
            intrinsic_fn=intrinsic,
            feature_fn=basis_feature,
            regressor_factory=make_ols_factory(basis_fn),
            fit_all_paths=fit_all_paths,
            store_fits=True,
        )
        runtime = time.perf_counter() - start
        results[model_name] = result
        rows.append(
            rate_sensitivity_row(
                r=r,
                model=model_name,
                seed=seed,
                n_paths=n_paths,
                fit_all_paths=fit_all_paths,
                result=result,
                european_bs_price=european_bs,
                american_binomial_price=american_binomial,
                runtime_seconds=runtime,
                basis_scaling=basis_scaling,
            )
        )

    if include_mlp:
        start = time.perf_counter()
        mlp_result = lsmc_price(
            paths=paths,
            times=times,
            r=cfg.r,
            intrinsic_fn=intrinsic,
            feature_fn=raw_feature,
            regressor_factory=lambda: TorchMLPRegressor(
                layer_widths=DEFAULT_MLP_ARCHITECTURE,
                learning_rate=learning_rate,
                epochs=mlp_epochs,
                batch_size=batch_size,
                weight_decay=weight_decay,
                random_seed=seed,
                device=device,
            ),
            fit_all_paths=fit_all_paths,
            store_fits=True,
        )
        runtime = time.perf_counter() - start
        results["MLP LSMC"] = mlp_result
        rows.append(
            rate_sensitivity_row(
                r=r,
                model="MLP LSMC",
                seed=seed,
                n_paths=n_paths,
                fit_all_paths=fit_all_paths,
                result=mlp_result,
                european_bs_price=european_bs,
                american_binomial_price=american_binomial,
                runtime_seconds=runtime,
                architecture=DEFAULT_MLP_ARCHITECTURE,
                epochs=mlp_epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                weight_decay=weight_decay,
                device=device,
            )
        )

    start = time.perf_counter()
    kanop_result = lsmc_price(
        paths=paths,
        times=times,
        r=cfg.r,
        intrinsic_fn=intrinsic,
        feature_fn=raw_feature,
        regressor_factory=lambda: TorchKANRegressor(
            widths=DEFAULT_KAN_ARCHITECTURE,
            grid_size=grid_size,
            learning_rate=learning_rate,
            epochs=kan_epochs,
            batch_size=batch_size,
            weight_decay=weight_decay,
            random_seed=seed,
            device=device,
        ),
        fit_all_paths=fit_all_paths,
        store_fits=True,
    )
    runtime = time.perf_counter() - start
    results["KANOP LSMC"] = kanop_result
    rows.append(
        rate_sensitivity_row(
            r=r,
            model="KANOP LSMC",
            seed=seed,
            n_paths=n_paths,
            fit_all_paths=fit_all_paths,
            result=kanop_result,
            european_bs_price=european_bs,
            american_binomial_price=american_binomial,
            runtime_seconds=runtime,
            architecture=DEFAULT_KAN_ARCHITECTURE,
            epochs=kan_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            weight_decay=weight_decay,
            grid_size=grid_size,
            device=device,
        )
    )

    metadata = {
        "config": cfg,
        "base_config": base_cfg,
        "paths": paths,
        "times": times,
        "scaler": scaler,
        "european_bs_price": european_bs,
        "american_binomial_price": american_binomial,
        "binomial_steps": binomial_steps,
    }
    return rows, metadata, results


def run_rate_sensitivity(
    *,
    seed: int = 1234,
    n_paths: int = 10_000,
    rates: tuple[float, ...] = (0.0, 0.04),
    fit_all_paths: bool = True,
    kan_epochs: int = 200,
    mlp_epochs: int = 50,
    learning_rate: float = 5e-3,
    batch_size: int = 2048,
    weight_decay: float = 0.0,
    grid_size: int = 10,
    device: str = "cpu",
    include_mlp: bool = False,
    binomial_steps: int = 1000,
    write_plots: bool = True,
) -> tuple[pd.DataFrame, dict[float, dict[str, object]], dict[float, dict[str, LSMCResult]], list[Path]]:
    all_rows: list[dict[str, float | int | str | bool | None]] = []
    metadata_by_rate: dict[float, dict[str, object]] = {}
    results_by_rate: dict[float, dict[str, LSMCResult]] = {}

    for rate in rates:
        rows, metadata, results = run_rate_case(
            r=rate,
            seed=seed,
            n_paths=n_paths,
            fit_all_paths=fit_all_paths,
            kan_epochs=kan_epochs,
            mlp_epochs=mlp_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            weight_decay=weight_decay,
            grid_size=grid_size,
            device=device,
            include_mlp=include_mlp,
            binomial_steps=binomial_steps,
        )
        all_rows.extend(rows)
        metadata_by_rate[rate] = metadata
        results_by_rate[rate] = results

    out = pd.DataFrame(all_rows)
    output_path = RESULTS / "american_put_rate_sensitivity.csv"
    out.to_csv(output_path, index=False)

    plot_paths: list[Path] = []
    if write_plots and 0.04 in metadata_by_rate:
        plot_paths = write_rate_004_plot(metadata_by_rate[0.04], results_by_rate[0.04])

    return out, metadata_by_rate, results_by_rate, plot_paths


def write_rate_004_plot(metadata: dict[str, object], results: dict[str, LSMCResult]) -> list[Path]:
    configure_matplotlib_cache()
    from kanop.plotting import plot_american_put_continuation_steps

    cfg = metadata["config"]
    scaler = metadata["scaler"]
    fits = {
        "Weighted Laguerre": results["Weighted Laguerre LSMC"],
        "Hermite": results["Hermite LSMC"],
        "KANOP": results["KANOP LSMC"],
    }
    transforms = {
        "Weighted Laguerre": scaler.transform,
        "Hermite": scaler.transform,
        "KANOP": None,
    }
    if "MLP LSMC" in results:
        fits["MLP"] = results["MLP LSMC"]
        transforms["MLP"] = None

    return plot_american_put_continuation_steps(
        paths=metadata["paths"],
        times=metadata["times"],
        fits_by_name=fits,
        strike=cfg.strike,
        maturity_years=cfg.maturity_years,
        r=cfg.r,
        sigma=cfg.sigma,
        q=cfg.q,
        steps_to_plot=(25,),
        output_dir=FIGURES,
        feature_transforms_by_name=transforms,
        reference_label="European BS reference",
        filename_template="american_put_r004_continuation_all_t{step}.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-paths", type=int, default=10_000)
    parser.add_argument("--rates", type=float, nargs="+", default=[0.0, 0.04])
    parser.add_argument("--kan-epochs", type=int, default=200)
    parser.add_argument("--mlp-epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--include-mlp", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--binomial-steps", type=int, default=1000)
    parser.add_argument(
        "--fit-itm-only",
        action="store_true",
        help="Fit continuation regressions only on in-the-money paths. Default fits all paths.",
    )
    args = parser.parse_args()

    out, metadata_by_rate, results_by_rate, plot_paths = run_rate_sensitivity(
        seed=args.seed,
        n_paths=args.n_paths,
        rates=tuple(args.rates),
        fit_all_paths=not args.fit_itm_only,
        kan_epochs=args.kan_epochs,
        mlp_epochs=args.mlp_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        grid_size=args.grid_size,
        device=args.device,
        include_mlp=args.include_mlp,
        binomial_steps=args.binomial_steps,
        write_plots=not args.skip_plots,
    )

    print(out.to_string(index=False))
    print(f"\nWrote {RESULTS / 'american_put_rate_sensitivity.csv'}")
    for rate in args.rates:
        metadata = metadata_by_rate[rate]
        print(
            f"r={rate:.4f}: European BS={metadata['european_bs_price']:.8f}, "
            f"American binomial={metadata['american_binomial_price']:.8f}"
        )
        print(f"r={rate:.4f}: stored diagnostic fits per model:")
        for model, result in results_by_rate[rate].items():
            print(f"  {model}: {len(result.fits)}")
    for plot_path in plot_paths:
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
