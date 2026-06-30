from kanop.black_scholes import bs_delta, bs_price
from kanop.config import AmericanPutConfig


def test_american_put_paper_price_scale():
    cfg = AmericanPutConfig()
    price = bs_price(cfg.s0, cfg.strike, cfg.maturity_years, cfg.r, cfg.sigma, "put", q=cfg.q)
    assert abs(price - cfg.paper_bs_price) < 5e-4


def test_black_scholes_put_call_parity_at_zero_rates_atm():
    cfg = AmericanPutConfig()
    call = bs_price(cfg.s0, cfg.strike, cfg.maturity_years, cfg.r, cfg.sigma, "call", q=cfg.q)
    put = bs_price(cfg.s0, cfg.strike, cfg.maturity_years, cfg.r, cfg.sigma, "put", q=cfg.q)
    assert abs(call - put) < 1e-12


def test_exact_put_delta_is_formula_not_paper_rounded_target():
    cfg = AmericanPutConfig()
    delta = bs_delta(cfg.s0, cfg.strike, cfg.maturity_years, cfg.r, cfg.sigma, "put", q=cfg.q)
    assert -0.51 < delta < -0.47
