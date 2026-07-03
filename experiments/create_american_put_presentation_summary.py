"""Create presentation-ready American put summary artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from kanop.config import AmericanPutConfig

from experiments.run_american_put_baselines import configure_matplotlib_cache, run_baselines
from experiments.run_american_put_kanop import run_kanop_experiment
from experiments.run_american_put_mlp import run_mlp_experiment

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SUMMARY_PATH = RESULTS / "american_put_presentation_summary.csv"
PLOT_PATH = FIGURES / "american_put_continuation_all_t25.png"


SUMMARY_COLUMNS = [
    "model",
    "price",
    "black_scholes_target",
    "paper_model_price",
    "abs_error_vs_bs",
    "difference_from_paper_model",
    "runtime_seconds",
]


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _first_row(df: pd.DataFrame | None) -> pd.Series | None:
    if df is None or df.empty:
        return None
    return df.iloc[0]


def build_summary_rows(
    baseline_df: pd.DataFrame,
    mlp_df: pd.DataFrame | None,
    kanop_df: pd.DataFrame,
) -> pd.DataFrame:
    cfg = AmericanPutConfig()
    bs_target = float(baseline_df.iloc[0]["black_scholes_target"])
    paper_bs_target = float(cfg.paper_model_targets["black_scholes"].price)
    rows: list[dict[str, float | str | None]] = [
        {
            "model": "Black-Scholes",
            "price": bs_target,
            "black_scholes_target": bs_target,
            "paper_model_price": paper_bs_target,
            "abs_error_vs_bs": 0.0,
            "difference_from_paper_model": bs_target - paper_bs_target,
            "runtime_seconds": 0.0,
        }
    ]

    for _, row in baseline_df.iterrows():
        rows.append(_summary_row(row))

    mlp_row = _first_row(mlp_df)
    if mlp_row is not None:
        rows.append(_summary_row(mlp_row))

    rows.append(_summary_row(kanop_df.iloc[0]))
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _summary_row(row: pd.Series) -> dict[str, float | str | None]:
    return {
        "model": row["model"],
        "price": float(row["price"]),
        "black_scholes_target": float(row["black_scholes_target"]),
        "paper_model_price": float(row["paper_model_price"]),
        "abs_error_vs_bs": float(row["abs_error_vs_bs"]),
        "difference_from_paper_model": float(row["difference_from_paper_model"]),
        "runtime_seconds": float(row["runtime_seconds"]),
    }


def write_combined_plot(
    *,
    baseline_df: pd.DataFrame,
    mlp_df: pd.DataFrame | None,
    kanop_df: pd.DataFrame,
) -> Path:
    baseline_seed = int(baseline_df.iloc[0]["seed"])
    baseline_n_paths = int(baseline_df.iloc[0]["n_paths"])
    baseline_fit_all_paths = bool(baseline_df.iloc[0]["fit_all_paths"])
    baseline_scaling = str(baseline_df.iloc[0].get("basis_scaling", "raw"))
    _, baseline_meta, weighted_laguerre, hermite = run_baselines(
        seed=baseline_seed,
        n_paths=baseline_n_paths,
        fit_all_paths=baseline_fit_all_paths,
        basis_scaling=baseline_scaling,
        store_fits=True,
    )

    fits_by_name = {
        "Weighted Laguerre": weighted_laguerre,
        "Hermite": hermite,
    }

    mlp_row = _first_row(mlp_df)
    if mlp_row is not None:
        _, _, mlp_result = run_mlp_experiment(
            seed=int(mlp_row["seed"]),
            n_paths=int(mlp_row["n_paths"]),
            epochs=int(mlp_row["epochs"]),
            learning_rate=float(mlp_row["learning_rate"]),
            batch_size=int(mlp_row["batch_size"]),
            fit_all_paths=bool(mlp_row["fit_all_paths"]),
        )
        fits_by_name["MLP"] = mlp_result

    kanop_row = kanop_df.iloc[0]
    _, _, kanop_result = run_kanop_experiment(
        seed=int(kanop_row["seed"]),
        n_paths=int(kanop_row["n_paths"]),
        grid_size=int(kanop_row["grid_size"]),
        epochs=int(kanop_row["epochs"]),
        learning_rate=float(kanop_row["learning_rate"]),
        batch_size=int(kanop_row["batch_size"]),
        fit_all_paths=bool(kanop_row["fit_all_paths"]),
    )
    fits_by_name["KANOP"] = kanop_result

    configure_matplotlib_cache()
    from kanop.plotting import plot_american_put_continuation_step

    cfg = baseline_meta["config"]
    return plot_american_put_continuation_step(
        paths=baseline_meta["paths"],
        times=baseline_meta["times"],
        fits_by_name=fits_by_name,
        step=25,
        strike=cfg.strike,
        maturity_years=cfg.maturity_years,
        r=cfg.r,
        sigma=cfg.sigma,
        q=cfg.q,
        output_path=PLOT_PATH,
        feature_transforms_by_name={
            "Weighted Laguerre": baseline_meta["scaler"].transform,
            "Hermite": baseline_meta["scaler"].transform,
            "MLP": None,
            "KANOP": None,
        },
    )


def main() -> None:
    baseline_df = pd.read_csv(RESULTS / "american_put_baselines.csv")
    mlp_df = _read_csv_if_exists(RESULTS / "american_put_mlp.csv")
    kanop_df = pd.read_csv(RESULTS / "american_put_kanop.csv")

    summary = build_summary_rows(baseline_df, mlp_df, kanop_df)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    plot_path = write_combined_plot(baseline_df=baseline_df, mlp_df=mlp_df, kanop_df=kanop_df)

    print(summary.to_string(index=False))
    print(f"\nWrote {SUMMARY_PATH}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
