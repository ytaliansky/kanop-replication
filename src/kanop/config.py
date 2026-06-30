"""Experiment constants from the KANOP paper.

The paper's American put price target of 0.1421 is reproduced with T=50/252.
The paper reports delta target as -0.5000, even though exact Black-Scholes put
Delta with T=50/252 and sigma=0.20 is slightly above -0.5 in magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass, field

TRADING_DAYS_PER_YEAR = 252
WEEKS_PER_YEAR = 52
LSMC_BASELINE_PATHS = 10_000
NEURAL_MLP_PATHS = 100_000
KANOP_PATHS = 10_000


@dataclass(frozen=True)
class PaperModelTarget:
    price: float | None
    delta: float | None = None


@dataclass(frozen=True)
class AmericanPutConfig:
    s0: float = 4.0
    strike: float = 4.0
    n_days: int = 50
    sigma: float = 0.20
    r: float = 0.0
    q: float = 0.0
    n_paths: int = LSMC_BASELINE_PATHS
    mlp_n_paths: int = NEURAL_MLP_PATHS
    kanop_n_paths: int = KANOP_PATHS
    paper_bs_price: float = 0.1421
    paper_bs_delta: float = -0.5000
    paper_model_targets: dict[str, PaperModelTarget] = field(
        default_factory=lambda: {
            "black_scholes": PaperModelTarget(price=0.1421, delta=-0.5000),
            "weighted_laguerre": PaperModelTarget(price=0.1395, delta=-0.4876),
            "hermite": PaperModelTarget(price=0.1407, delta=-0.4899),
            "mlp": PaperModelTarget(price=0.1384, delta=-0.4976),
            "kanop": PaperModelTarget(price=0.1427, delta=-0.4970),
        }
    )

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
    n_paths: int = LSMC_BASELINE_PATHS
    mlp_n_paths: int = NEURAL_MLP_PATHS
    kanop_n_paths: int = KANOP_PATHS
    paper_eurasian_price: float | None = None
    paper_asian_american_price: float | None = None
    paper_laguerre_price: float | None = None
    paper_mlp_price: float | None = None
    paper_kanop_price: float | None = None

    @property
    def paper_model_targets(self) -> dict[str, PaperModelTarget]:
        return {
            "eurasian": PaperModelTarget(price=self.paper_eurasian_price),
            "asian_american": PaperModelTarget(price=self.paper_asian_american_price),
            "laguerre": PaperModelTarget(price=self.paper_laguerre_price),
            "mlp": PaperModelTarget(price=self.paper_mlp_price),
            "kanop": PaperModelTarget(price=self.paper_kanop_price),
        }

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
        paper_laguerre_price=2.2750,
        paper_mlp_price=2.2601,
        paper_kanop_price=2.3216,
    ),
    AsianAmericanConfig(
        s0=100.0,
        strike=100.0,
        n_weeks=13,
        sigma=0.25,
        paper_eurasian_price=3.3621,
        paper_asian_american_price=3.6500,
        paper_laguerre_price=3.5716,
        paper_mlp_price=3.6134,
        paper_kanop_price=3.6589,
    ),
    AsianAmericanConfig(
        s0=100.0,
        strike=100.0,
        n_weeks=26,
        sigma=0.25,
        paper_eurasian_price=4.7659,
        paper_asian_american_price=5.2660,
        paper_laguerre_price=5.0719,
        paper_mlp_price=5.1422,
        paper_kanop_price=5.2382,
    ),
    AsianAmericanConfig(
        s0=100.0,
        strike=105.0,
        n_weeks=26,
        sigma=0.25,
        paper_eurasian_price=2.6628,
        paper_asian_american_price=2.8580,
        paper_laguerre_price=2.7162,
        paper_mlp_price=2.7943,
        paper_kanop_price=2.8309,
    ),
)
