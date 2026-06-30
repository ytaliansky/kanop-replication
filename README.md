# KANOP Replication Skeleton

This repository is a Python project skeleton for recreating the experiments in:

**KANOP: A Data-Efficient Option Pricing Model using Kolmogorov-Arnold Networks**

The goal is to first reproduce the paper's LSMC baselines and benchmarks, then add the MLP and KANOP regressors.

## What is implemented now

- Risk-neutral GBM path simulation
- Black-Scholes price and delta benchmark
- Standard payoff functions
- TWAP path feature construction
- Polynomial, Laguerre, weighted Laguerre, and Hermite basis functions
- Generic LSMC backward-induction engine
- American put baseline experiment with weighted Laguerre and Hermite regressors
- Asian-American call baseline experiment with Laguerre cross-product regressors
- Result-table helpers and basic tests

## What is intentionally left as next-step TODOs

- PyTorch MLP regressor matching the paper's `[1, 32, 32, 1]` and `[2, 32, 32, 1]` models
- PyTorch KAN regressor matching the paper's `[1, 3, 1]` and `[2, 5, 1]` KANOP models
- Autograd-based delta calculation for trained neural regressors
- Exact continuation-value plots matching the paper's page-7 figures

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run tests

```bash
pytest
```

## Run American put baseline experiment

```bash
python experiments/run_american_put_baselines.py
```

This writes:

```text
results/american_put_baselines.csv
figures/american_put_continuation_laguerre_hermite.png
```

## Run Asian-American baseline experiment

```bash
python experiments/run_asian_american_baselines.py
```

This writes:

```text
results/asian_american_laguerre_baselines.csv
```

## Paper experiment parameters captured here

### American put

- `S0 = 4.0`
- `K = 4.0`
- `T = 50 trading days`
- `sigma = 20% annualized`
- `r = 0%`
- `q = 0%`
- `paths = 10,000`
- daily exercise
- Black-Scholes paper target price: `0.1421`
- Black-Scholes paper target delta: `-0.5000`

The Black-Scholes price of about `0.1421` corresponds to using `50 / 252` years.
The paper reports the delta target as `-0.5000`; exact Black-Scholes put delta at `T=50/252` is slightly different, so this project stores both exact formula outputs and paper-reported targets.

### Asian-American call cases

All cases use:

- `S0 = 100`
- `r = 5%`
- weekly exercise
- `paths = 10,000` for Laguerre/KANOP baselines
- Asian average convention in the experiment script: average from first monitoring date onward, excluding `S0`, because this better matches the paper's Eurasian benchmark table

| Case | K | Weeks | sigma | Paper Asian-American target |
|---|---:|---:|---:|---:|
| 1 | 100 | 13 | 0.15 | 2.3210 |
| 2 | 100 | 13 | 0.25 | 3.6500 |
| 3 | 100 | 26 | 0.25 | 5.2660 |
| 4 | 105 | 26 | 0.25 | 2.8580 |

## Notes on reproducibility

The paper provides model structures and reported outputs, but not all training hyperparameters, random seeds, spline-grid details, optimizer settings, or normalization rules. Exact numeric reproduction may therefore require tuning after the MLP/KAN components are added.
