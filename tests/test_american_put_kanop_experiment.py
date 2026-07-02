import pandas as pd
import pytest

pytest.importorskip("torch")

from experiments.run_american_put_kanop import (
    DEFAULT_ARCHITECTURE,
    american_put_kanop_result_row,
    run_kanop_experiment,
    write_results,
)
from kanop.models import TorchKANRegressor


EXPECTED_COLUMNS = [
    "model",
    "seed",
    "n_paths",
    "fit_all_paths",
    "architecture",
    "grid_size",
    "epochs",
    "learning_rate",
    "batch_size",
    "price",
    "black_scholes_target",
    "paper_model_price",
    "abs_error_vs_bs",
    "rel_error_vs_bs",
    "difference_from_paper_model",
    "runtime_seconds",
]


def test_american_put_kanop_result_row_has_expected_columns():
    row = american_put_kanop_result_row(
        seed=1234,
        n_paths=10_000,
        fit_all_paths=True,
        architecture=DEFAULT_ARCHITECTURE,
        grid_size=10,
        epochs=200,
        learning_rate=5e-3,
        batch_size=2048,
        price=0.142,
        black_scholes_target=0.1421,
        paper_model_price=0.1427,
        runtime_seconds=1.0,
    )

    assert list(row.keys()) == EXPECTED_COLUMNS
    assert row["model"] == "KANOP LSMC"
    assert row["architecture"] == "[1, 3, 1]"
    assert row["grid_size"] == 10
    assert row["paper_model_price"] == 0.1427
    assert row["difference_from_paper_model"] == row["price"] - 0.1427


def test_american_put_kanop_experiment_tiny_smoke_has_diagnostics():
    out, metadata, result = run_kanop_experiment(
        seed=1234,
        n_paths=80,
        architecture=(1, 3, 1),
        grid_size=6,
        epochs=1,
        learning_rate=1e-2,
        batch_size=80,
        fit_all_paths=True,
    )

    assert list(out.columns) == EXPECTED_COLUMNS
    assert out.loc[0, "n_paths"] == 80
    assert out.loc[0, "architecture"] == "[1, 3, 1]"
    assert out.loc[0, "grid_size"] == 6
    assert result.price >= 0.0
    assert len(result.fits) == metadata["config"].n_steps - 1
    assert isinstance(result.fits[0].regressor, TorchKANRegressor)
    assert result.fits[0].continuation_pred.shape == (80,)


def test_american_put_kanop_results_csv_has_expected_columns(tmp_path):
    row = american_put_kanop_result_row(
        seed=1,
        n_paths=10,
        fit_all_paths=False,
        architecture=(1, 3, 1),
        grid_size=8,
        epochs=1,
        learning_rate=1e-2,
        batch_size=10,
        price=0.1,
        black_scholes_target=0.1421,
        paper_model_price=0.1427,
        runtime_seconds=0.2,
    )
    out_path = write_results(pd.DataFrame([row]), tmp_path / "american_put_kanop.csv")

    loaded = pd.read_csv(out_path)
    assert list(loaded.columns) == EXPECTED_COLUMNS
    assert loaded.loc[0, "model"] == "KANOP LSMC"
