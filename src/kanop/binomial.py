"""Binomial-tree option pricing benchmarks."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

OptionType = Literal["put"]


def _validate_crr_inputs(
    s0: float,
    strike: float,
    maturity: float,
    sigma: float,
    n_steps: int,
) -> None:
    if s0 <= 0.0:
        raise ValueError("s0 must be positive")
    if strike <= 0.0:
        raise ValueError("strike must be positive")
    if maturity <= 0.0:
        raise ValueError("maturity must be positive")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")


def _crr_parameters(
    maturity: float,
    sigma: float,
    r: float,
    q: float,
    n_steps: int,
) -> tuple[float, float, float, float, float]:
    dt = maturity / n_steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    growth = math.exp((r - q) * dt)
    p = (growth - d) / (u - d)
    if not 0.0 <= p <= 1.0:
        raise ValueError(
            "CRR risk-neutral probability is outside [0, 1]; "
            "increase n_steps or check r, q, sigma, and maturity"
        )
    discount = math.exp(-r * dt)
    return dt, u, d, p, discount


def european_put_binomial_price(
    s0: float,
    strike: float,
    maturity: float,
    sigma: float,
    r: float,
    q: float = 0.0,
    n_steps: int = 1000,
) -> float:
    """Return a European put price using a Cox-Ross-Rubinstein tree."""
    _validate_crr_inputs(s0, strike, maturity, sigma, n_steps)
    _, u, d, p, discount = _crr_parameters(maturity, sigma, r, q, n_steps)

    j = np.arange(n_steps + 1)
    stock = s0 * (u**j) * (d ** (n_steps - j))
    values = np.maximum(strike - stock, 0.0)
    for _ in range(n_steps - 1, -1, -1):
        values = discount * (p * values[1:] + (1.0 - p) * values[:-1])
    return float(values[0])


def american_put_binomial_price(
    s0: float,
    strike: float,
    maturity: float,
    sigma: float,
    r: float,
    q: float = 0.0,
    n_steps: int = 1000,
) -> float:
    """Return an American put price using a Cox-Ross-Rubinstein tree."""
    _validate_crr_inputs(s0, strike, maturity, sigma, n_steps)
    _, u, d, p, discount = _crr_parameters(maturity, sigma, r, q, n_steps)

    j = np.arange(n_steps + 1)
    stock = s0 * (u**j) * (d ** (n_steps - j))
    values = np.maximum(strike - stock, 0.0)

    for step in range(n_steps - 1, -1, -1):
        continuation = discount * (p * values[1:] + (1.0 - p) * values[:-1])
        j = np.arange(step + 1)
        stock = s0 * (u**j) * (d ** (step - j))
        exercise = np.maximum(strike - stock, 0.0)
        values = np.maximum(exercise, continuation)
    return float(values[0])
