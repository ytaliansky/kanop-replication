"""Run the American put MLP continuation-value baseline.

This script integrates ``TorchMLPRegressor`` into the same LSMC backward
induction pipeline used by the fixed-basis baselines. It is intentionally
limited to the American put MLP case; KANOP, delta estimation, and Asian-American
MLP runs are later milestones.
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

import pandas as pd

from kanop.black_scholes import bs_price
from kanop.config import AmericanPutConfig
from kanop.lsmc import LSMCResult, lsmc_price
from kanop.metrics import absolute_error, relative_error
from kanop.models import TorchMLPRegressor
from kanop.payoffs import put_intrinsic
from kanop.simulation import simulate_gbm_paths

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
PLOT_STEPS = (49, 25, 1)
DEFAULT_ARCHITECTURE = (1, 32, 32, 1)


def parse_architecture(value: str) -> tuple[int, ...]:
    widths = tuple(int(part.strip()) for part in value.replace("[", "").replace("]", "").split(",") if part.strip())
    if len(widths) < 2:
        raise argparse.ArgumentTypeError("architecture must contain at least input and output widths")
    if widths[0] != 1 or widths[-1] != 1:
        raise argparse.ArgumentTypeError("American put MLP architecture must start with 1 and end with 1")
    return widths


def architecture_label(architecture: tuple[int, ...]) -> str:
    return "[" + ", ".join(str(width) for width in architecture) + "]"


def american_put_mlp_result_row(
    *,
    seed: int,
    n_paths: int,
    fit_all_paths: bool,
    architecture: tuple[int, ...],
    epochs: int,
    learning_rate: float,
    batch_size: int,
    price: float,
    black_scholes_target: float,
    paper_model_price: float,
    runtime_seconds: float,
) -> dict[str, float | int | str | bool]:
    return {
        "model": "MLP LSMC",
        "seed": seed,
        "n_paths": n_paths,
        "fit_all_paths": fit_all_paths,
        "architecture": architecture_label(architecture),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "price": price,
        "black_scholes_target": black_scholes_target,
        "paper_model_price": paper_model_price,
        "abs_error_vs_bs": absolute_error(price, black_scholes_target),
        "rel_error_vs_bs": relative_error(price, black_scholes_target),
        "difference_from_paper_model": price - paper_model_price,
        "runtime_seconds": runtime_seconds,
    }


def run_mlp_experiment(
    *,
    seed: int = 1234,
    n_paths: int | None = None,
    architecture: tuple[int, ...] = DEFAULT_ARCHITECTURE,
    epochs: int = 200,
    learning_rate: float = 1e-3,
    batch_size: int = 256,
    weight_decay: float = 0.0,
    device: str = "cpu",
    fit_all_paths: bool = True,
) -> tuple[pd.DataFrame, dict[str, object], LSMCResult]:
    cfg = AmericanPutConfig(n_paths=n_paths or AmericanPutConfig().mlp_n_paths)
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

    start = time.perf_counter()
    mlp_result = lsmc_price(
        paths=paths,
        times=times,
        r=cfg.r,
        intrinsic_fn=intrinsic,
        feature_fn=feature,
        regressor_factory=lambda: TorchMLPRegressor(
            layer_widths=architecture,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            weight_decay=weight_decay,
            random_seed=seed,
            device=device,
        ),
        fit_all_paths=fit_all_paths,
        store_fits=True,
    )
    runtime_seconds = time.perf_counter() - start

    bs_p = bs_price(cfg.s0, cfg.strike, cfg.maturity_years, cfg.r, cfg.sigma, "put", q=cfg.q)
    row = american_put_mlp_result_row(
        seed=seed,
        n_paths=cfg.n_paths,
        fit_all_paths=fit_all_paths,
        architecture=architecture,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        price=mlp_result.price,
        black_scholes_target=bs_p,
        paper_model_price=cfg.paper_model_targets["mlp"].price,
        runtime_seconds=runtime_seconds,
    )
    metadata = {
        "paths": paths,
        "times": times,
        "config": cfg,
        "architecture": architecture,
        "weight_decay": weight_decay,
        "device": device,
    }
    return pd.DataFrame([row]), metadata, mlp_result


def write_results(out: pd.DataFrame, output_path: str | Path = RESULTS / "american_put_mlp.csv") -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return output_path


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


def write_plots(metadata: dict[str, object], mlp_result: LSMCResult) -> list[Path]:
    configure_matplotlib_cache()
    from kanop.plotting import plot_american_put_continuation_steps

    cfg = metadata["config"]
    return plot_american_put_continuation_steps(
        paths=metadata["paths"],
        times=metadata["times"],
        fits_by_name={"MLP": mlp_result},
        strike=cfg.strike,
        maturity_years=cfg.maturity_years,
        r=cfg.r,
        sigma=cfg.sigma,
        q=cfg.q,
        steps_to_plot=PLOT_STEPS,
        output_dir=FIGURES,
        filename_template="american_put_continuation_mlp_t{step}.png",
    )


def main(
    *,
    seed: int,
    n_paths: int,
    architecture: tuple[int, ...],
    epochs: int,
    learning_rate: float,
    batch_size: int,
    weight_decay: float,
    device: str,
    fit_all_paths: bool,
    write_plot: bool,
) -> None:
    out, metadata, mlp_result = run_mlp_experiment(
        seed=seed,
        n_paths=n_paths,
        architecture=architecture,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        weight_decay=weight_decay,
        device=device,
        fit_all_paths=fit_all_paths,
    )
    out_path = write_results(out)
    print(out.to_string(index=False))
    print(f"\nWrote {out_path}")
    print(f"Stored diagnostic fits: {len(mlp_result.fits)}")

    if write_plot:
        for plot_path in write_plots(metadata, mlp_result):
            print(f"Wrote {plot_path}")


if __name__ == "__main__":
    cfg = AmericanPutConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-paths", type=int, default=cfg.mlp_n_paths)
    parser.add_argument("--architecture", type=parse_architecture, default=DEFAULT_ARCHITECTURE)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--fit-itm-only",
        action="store_true",
        help="Fit continuation regressions only on in-the-money paths. Default fits all paths.",
    )
    parser.add_argument("--skip-plot", action="store_true", help="Write the CSV table without generating figures.")
    args = parser.parse_args()
    main(
        seed=args.seed,
        n_paths=args.n_paths,
        architecture=args.architecture,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        device=args.device,
        fit_all_paths=not args.fit_itm_only,
        write_plot=not args.skip_plot,
    )
