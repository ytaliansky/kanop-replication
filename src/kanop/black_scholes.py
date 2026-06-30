"""Black-Scholes benchmark formulas."""

from __future__ import annotations

import math
from typing import Literal

from scipy.stats import norm

OptionType = Literal["call", "put"]


def _d1_d2(s0: float, strike: float, maturity: float, r: float, sigma: float, q: float = 0.0) -> tuple[float, float]:
    if s0 <= 0:
        raise ValueError("s0 must be positive")
    if strike <= 0:
        raise ValueError("strike must be positive")
    if maturity <= 0:
        raise ValueError("maturity must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    sqrt_t = math.sqrt(maturity)
    d1 = (math.log(s0 / strike) + (r - q + 0.5 * sigma**2) * maturity) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return d1, d2


def bs_price(
    s0: float,
    strike: float,
    maturity: float,
    r: float,
    sigma: float,
    option_type: OptionType,
    q: float = 0.0,
) -> float:
    """Return Black-Scholes price for a European call or put."""
    d1, d2 = _d1_d2(s0, strike, maturity, r, sigma, q)
    disc_r = math.exp(-r * maturity)
    disc_q = math.exp(-q * maturity)

    if option_type == "call":
        return s0 * disc_q * norm.cdf(d1) - strike * disc_r * norm.cdf(d2)
    if option_type == "put":
        return strike * disc_r * norm.cdf(-d2) - s0 * disc_q * norm.cdf(-d1)
    raise ValueError(f"unknown option_type: {option_type}")


def bs_delta(
    s0: float,
    strike: float,
    maturity: float,
    r: float,
    sigma: float,
    option_type: OptionType,
    q: float = 0.0,
) -> float:
    """Return Black-Scholes delta for a European call or put."""
    d1, _ = _d1_d2(s0, strike, maturity, r, sigma, q)
    disc_q = math.exp(-q * maturity)
    if option_type == "call":
        return disc_q * norm.cdf(d1)
    if option_type == "put":
        return disc_q * (norm.cdf(d1) - 1.0)
    raise ValueError(f"unknown option_type: {option_type}")
