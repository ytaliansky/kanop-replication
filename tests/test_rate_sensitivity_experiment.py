from pathlib import Path

from experiments.run_american_put_rate_sensitivity import (
    exercise_diagnostics,
    run_rate_sensitivity,
)


EXPECTED_COLUMNS = {
    "r",
    "model",
    "seed",
    "n_paths",
    "fit_all_paths",
    "price",
    "european_bs_price",
    "american_binomial_price",
    "error_vs_american_binomial",
    "difference_vs_european_bs",
    "early_exercise_count",
    "early_exercise_fraction",
    "average_exercise_time",
    "runtime_seconds",
    "architecture",
    "epochs",
    "learning_rate",
    "batch_size",
    "weight_decay",
    "grid_size",
    "device",
    "basis_scaling",
}


def test_rate_sensitivity_tiny_smoke_run_writes_expected_columns():
    out, metadata_by_rate, results_by_rate, plot_paths = run_rate_sensitivity(
        seed=7,
        n_paths=128,
        rates=(0.04,),
        kan_epochs=1,
        mlp_epochs=1,
        learning_rate=1e-3,
        batch_size=64,
        grid_size=5,
        include_mlp=False,
        binomial_steps=100,
        write_plots=False,
    )

    assert EXPECTED_COLUMNS.issubset(set(out.columns))
    assert set(out["model"]) == {"Weighted Laguerre LSMC", "Hermite LSMC", "KANOP LSMC"}
    assert plot_paths == []
    assert metadata_by_rate[0.04]["american_binomial_price"] >= metadata_by_rate[0.04]["european_bs_price"]
    assert Path("results/american_put_rate_sensitivity.csv").exists()

    kanop = results_by_rate[0.04]["KANOP LSMC"]
    assert len(kanop.fits) == kanop.n_steps - 1
    diagnostics = exercise_diagnostics(kanop)
    assert 0.0 <= diagnostics["early_exercise_fraction"] <= 1.0
