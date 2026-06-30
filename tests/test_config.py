from kanop.config import ASIAN_AMERICAN_CASES, AmericanPutConfig


def test_american_put_config_contains_all_paper_targets():
    cfg = AmericanPutConfig()

    assert cfg.paper_bs_price == 0.1421
    assert cfg.paper_bs_delta == -0.5000
    assert cfg.n_paths == 10_000
    assert cfg.mlp_n_paths == 100_000
    assert cfg.kanop_n_paths == 10_000

    assert cfg.paper_model_targets["weighted_laguerre"].price == 0.1395
    assert cfg.paper_model_targets["weighted_laguerre"].delta == -0.4876
    assert cfg.paper_model_targets["hermite"].price == 0.1407
    assert cfg.paper_model_targets["hermite"].delta == -0.4899
    assert cfg.paper_model_targets["mlp"].price == 0.1384
    assert cfg.paper_model_targets["mlp"].delta == -0.4976
    assert cfg.paper_model_targets["kanop"].price == 0.1427
    assert cfg.paper_model_targets["kanop"].delta == -0.4970


def test_asian_american_config_contains_all_paper_targets():
    expected = [
        (2.1638, 2.3210, 2.2750, 2.2601, 2.3216),
        (3.3621, 3.6500, 3.5716, 3.6134, 3.6589),
        (4.7659, 5.2660, 5.0719, 5.1422, 5.2382),
        (2.6628, 2.8580, 2.7162, 2.7943, 2.8309),
    ]

    for cfg, targets in zip(ASIAN_AMERICAN_CASES, expected):
        eurasian, asian_american, laguerre, mlp, kanop = targets
        assert cfg.n_paths == 10_000
        assert cfg.mlp_n_paths == 100_000
        assert cfg.kanop_n_paths == 10_000
        assert cfg.paper_model_targets["eurasian"].price == eurasian
        assert cfg.paper_model_targets["asian_american"].price == asian_american
        assert cfg.paper_model_targets["laguerre"].price == laguerre
        assert cfg.paper_model_targets["mlp"].price == mlp
        assert cfg.paper_model_targets["kanop"].price == kanop
