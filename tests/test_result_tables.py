from experiments.run_american_put_baselines import american_put_result_row
from experiments.run_asian_american_baselines import asian_american_result_row


def test_american_put_row_uses_model_specific_paper_target():
    row = american_put_result_row(
        model="Weighted Laguerre LSMC",
        seed=1234,
        n_paths=10_000,
        fit_all_paths=True,
        basis_scaling="raw",
        price=0.1389,
        black_scholes_target=0.1421,
        paper_model_price=0.1395,
        runtime_seconds=1.25,
    )

    assert row["black_scholes_target"] == 0.1421
    assert row["paper_model_price"] == 0.1395
    assert row["difference_from_paper_model"] == row["price"] - 0.1395
    assert row["abs_error_vs_bs"] == abs(row["price"] - 0.1421)
    assert row["basis_scaling"] == "raw"


def test_asian_american_row_keeps_laguerre_target_separate():
    row = asian_american_result_row(
        case_id=1,
        strike=100.0,
        weeks=13,
        sigma=0.15,
        model="Laguerre LSMC",
        price=2.2527,
        paper_model_price=2.2750,
        target_asian_american_price=2.3210,
        runtime_seconds=0.5,
        seed=1234,
        n_paths=10_000,
        fit_all_paths=True,
        basis_scaling="raw",
    )

    assert row["paper_model_price"] == 2.2750
    assert row["target_asian_american_price"] == 2.3210
    assert row["difference_from_paper_model"] == row["price"] - 2.2750
    assert row["abs_error_vs_target"] == abs(row["price"] - 2.3210)
    assert row["fit_all_paths"] is True
