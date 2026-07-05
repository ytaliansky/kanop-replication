"""Compare American put autograd deltas for neural LSMC models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from experiments.run_american_put_kanop import (
    DEFAULT_ARCHITECTURE as DEFAULT_KAN_ARCHITECTURE,
    run_kanop_experiment,
)
from experiments.run_american_put_mlp import (
    DEFAULT_ARCHITECTURE as DEFAULT_MLP_ARCHITECTURE,
    run_mlp_experiment,
)
from kanop.config import AmericanPutConfig
from kanop.metrics import absolute_error

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def comparison_row(
    *,
    source_row: pd.Series,
    paper_model_delta: float,
    grid_size: int | None = None,
) -> dict[str, float | int | str | None]:
    estimated_delta = float(source_row["estimated_delta"])
    exact_delta = float(source_row["black_scholes_exact_delta"])
    return {
        "model": source_row["model"],
        "r": float(source_row["r"]),
        "seed": int(source_row["seed"]),
        "n_paths": int(source_row["n_paths"]),
        "epochs": int(source_row["epochs"]),
        "learning_rate": float(source_row["learning_rate"]),
        "grid_size": grid_size,
        "price": float(source_row["price"]),
        "estimated_delta": estimated_delta,
        "black_scholes_exact_delta": exact_delta,
        "paper_delta_target": float(source_row["paper_delta_target"]),
        "paper_model_delta": paper_model_delta,
        "abs_error_vs_exact_bs_delta": absolute_error(estimated_delta, exact_delta),
        "abs_error_vs_paper_model_delta": absolute_error(estimated_delta, paper_model_delta),
        "runtime_seconds": float(source_row["runtime_seconds"]),
    }


def run_delta_comparison(
    *,
    seed: int = 1234,
    kan_n_paths: int | None = None,
    mlp_n_paths: int | None = None,
    kan_epochs: int = 200,
    mlp_epochs: int = 200,
    kan_learning_rate: float = 5e-3,
    mlp_learning_rate: float = 1e-3,
    batch_size: int = 2048,
    weight_decay: float = 0.0,
    grid_size: int = 10,
    device: str = "cpu",
    fit_all_paths: bool = True,
    r: float = AmericanPutConfig().r,
) -> pd.DataFrame:
    cfg = AmericanPutConfig(r=r)
    kan_df, _, _ = run_kanop_experiment(
        seed=seed,
        n_paths=kan_n_paths or cfg.kanop_n_paths,
        architecture=DEFAULT_KAN_ARCHITECTURE,
        grid_size=grid_size,
        epochs=kan_epochs,
        learning_rate=kan_learning_rate,
        batch_size=batch_size,
        weight_decay=weight_decay,
        device=device,
        fit_all_paths=fit_all_paths,
        r=r,
        compute_delta=True,
    )
    mlp_df, _, _ = run_mlp_experiment(
        seed=seed,
        n_paths=mlp_n_paths or cfg.mlp_n_paths,
        architecture=DEFAULT_MLP_ARCHITECTURE,
        epochs=mlp_epochs,
        learning_rate=mlp_learning_rate,
        batch_size=batch_size,
        weight_decay=weight_decay,
        device=device,
        fit_all_paths=fit_all_paths,
        r=r,
        compute_delta=True,
    )

    rows = [
        comparison_row(
            source_row=kan_df.iloc[0],
            paper_model_delta=cfg.paper_model_targets["kanop"].delta,
            grid_size=grid_size,
        ),
        comparison_row(
            source_row=mlp_df.iloc[0],
            paper_model_delta=cfg.paper_model_targets["mlp"].delta,
        ),
    ]
    out = pd.DataFrame(rows)
    out_path = RESULTS / "american_put_delta_comparison.csv"
    out.to_csv(out_path, index=False)
    return out


def main() -> None:
    cfg = AmericanPutConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--kan-n-paths", type=int, default=cfg.kanop_n_paths)
    parser.add_argument("--mlp-n-paths", type=int, default=cfg.mlp_n_paths)
    parser.add_argument("--kan-epochs", type=int, default=200)
    parser.add_argument("--mlp-epochs", type=int, default=200)
    parser.add_argument("--kan-learning-rate", type=float, default=5e-3)
    parser.add_argument("--mlp-learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--r", type=float, default=cfg.r)
    parser.add_argument(
        "--fit-itm-only",
        action="store_true",
        help="Fit continuation regressions only on in-the-money paths. Default fits all paths.",
    )
    args = parser.parse_args()

    out = run_delta_comparison(
        seed=args.seed,
        kan_n_paths=args.kan_n_paths,
        mlp_n_paths=args.mlp_n_paths,
        kan_epochs=args.kan_epochs,
        mlp_epochs=args.mlp_epochs,
        kan_learning_rate=args.kan_learning_rate,
        mlp_learning_rate=args.mlp_learning_rate,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        grid_size=args.grid_size,
        device=args.device,
        fit_all_paths=not args.fit_itm_only,
        r=args.r,
    )
    print(out.to_string(index=False))
    print(f"\nWrote {RESULTS / 'american_put_delta_comparison.csv'}")


if __name__ == "__main__":
    main()
