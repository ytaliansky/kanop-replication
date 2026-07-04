from kanop.binomial import american_put_binomial_price, european_put_binomial_price
from kanop.black_scholes import bs_price
from kanop.config import AmericanPutConfig


def test_american_put_binomial_above_european_black_scholes_when_rates_positive():
    cfg = AmericanPutConfig(r=0.04)
    european = bs_price(cfg.s0, cfg.strike, cfg.maturity_years, cfg.r, cfg.sigma, "put", q=cfg.q)
    american = american_put_binomial_price(
        cfg.s0,
        cfg.strike,
        cfg.maturity_years,
        cfg.sigma,
        cfg.r,
        q=cfg.q,
        n_steps=1000,
    )
    assert american >= european


def test_american_put_binomial_close_to_european_black_scholes_at_zero_rates():
    cfg = AmericanPutConfig()
    european = bs_price(cfg.s0, cfg.strike, cfg.maturity_years, cfg.r, cfg.sigma, "put", q=cfg.q)
    american = american_put_binomial_price(
        cfg.s0,
        cfg.strike,
        cfg.maturity_years,
        cfg.sigma,
        cfg.r,
        q=cfg.q,
        n_steps=1000,
    )
    assert abs(american - european) < 5e-4


def test_binomial_price_is_stable_as_steps_increase():
    cfg = AmericanPutConfig(r=0.04)
    price_500 = american_put_binomial_price(
        cfg.s0,
        cfg.strike,
        cfg.maturity_years,
        cfg.sigma,
        cfg.r,
        q=cfg.q,
        n_steps=500,
    )
    price_1000 = american_put_binomial_price(
        cfg.s0,
        cfg.strike,
        cfg.maturity_years,
        cfg.sigma,
        cfg.r,
        q=cfg.q,
        n_steps=1000,
    )
    assert abs(price_1000 - price_500) < 2e-3


def test_european_binomial_matches_black_scholes_reasonably():
    cfg = AmericanPutConfig(r=0.04)
    expected = bs_price(cfg.s0, cfg.strike, cfg.maturity_years, cfg.r, cfg.sigma, "put", q=cfg.q)
    actual = european_put_binomial_price(
        cfg.s0,
        cfg.strike,
        cfg.maturity_years,
        cfg.sigma,
        cfg.r,
        q=cfg.q,
        n_steps=1000,
    )
    assert abs(actual - expected) < 5e-4
