"""Experiment constants from the KANOP paper.

The paper's American put price target of 0.1421 is reproduced with T=50/252.
The paper reports delta target as -0.5000, even though exact Black-Scholes put
Delta with T=50/252 and sigma=0.20 is slightly above -0.5 in magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass

TRADING_DAYS_PER_YEAR = 252
WEEKS_PER_YEAR = 52


@dataclass(frozen=True)
class AmericanPutConfig:
    s0: float = 4.0
    strike: float = 4.0
    n_days: int = 50
    sigma: float = 0.20
    r: float = 0.0
    q: float = 0.0
    n_paths: int = 10_000
    paper_bs_price: float = 0.1421
    paper_bs_delta: float = -0.5000

    @property
    def maturity_years(self) -> float:
        return self.n_days / TRADING_DAYS_PER_YEAR

    @property
    def n_steps(self) -> int:
        return self.n_days


@dataclass(frozen=True)
class AsianAmericanConfig:
    s0: float
    strike: float
    n_weeks: int
    sigma: float
    r: float = 0.05
    q: float = 0.0
    n_paths: int = 10_000
    paper_eurasian_price: float | None = None
    paper_asian_american_price: float | None = None

    @property
    def maturity_years(self) -> float:
        return self.n_weeks / WEEKS_PER_YEAR

    @property
    def n_steps(self) -> int:
        return self.n_weeks


ASIAN_AMERICAN_CASES: tuple[AsianAmericanConfig, ...] = (
    AsianAmericanConfig(
        s0=100.0,
        strike=100.0,
        n_weeks=13,
        sigma=0.15,
        paper_eurasian_price=2.1638,
        paper_asian_american_price=2.3210,
    ),
    AsianAmericanConfig(
        s0=100.0,
        strike=100.0,
        n_weeks=13,
        sigma=0.25,
        paper_eurasian_price=3.3621,
        paper_asian_american_price=3.6500,
    ),
    AsianAmericanConfig(
        s0=100.0,
        strike=100.0,
        n_weeks=26,
        sigma=0.25,
        paper_eurasian_price=4.7659,
        paper_asian_american_price=5.2660,
    ),
    AsianAmericanConfig(
        s0=100.0,
        strike=105.0,
        n_weeks=26,
        sigma=0.25,
        paper_eurasian_price=2.6628,
        paper_asian_american_price=2.8580,
    ),
)
