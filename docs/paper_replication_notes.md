# Paper replication notes

## KANOP algorithmic target

The paper keeps the LSMC backward-induction algorithm and replaces the fixed-basis OLS continuation-value regression with a KAN model.

At each step `k = K-1, ..., 1`:

1. Compute discounted future cashflows for each simulated path.
2. Fit continuation value `F_hat(omega; t_k)` from the current state variables.
3. Exercise if intrinsic value is positive and greater than fitted continuation value.
4. Update the cashflow and exercise time.

## American put replication target

Parameters:

```text
S0 = 4.0
K = 4.0
T = 50 trading days = 50 / 252 years
sigma = 0.20
r = 0.00
q = 0.00
paths = 10,000
```

Paper table:

| Model | Price | Delta |
|---|---:|---:|
| Black-Scholes | 0.1421 | -0.5000 |
| Weighted Laguerre | 0.1395 | -0.4876 |
| Hermite | 0.1407 | -0.4899 |
| MLP | 0.1384 | -0.4976 |
| KANOP | 0.1427 | -0.4970 |

Implemented starting point:

- Weighted Laguerre with `L_0..L_5`, weighted by `exp(-x/2)`.
- Hermite with `H_0..H_5`.
- Reusable PyTorch MLP regressor with Adam/MSE training, SiLU activation, and
  internal input/target normalization.
- Reusable PyTorch KAN-style regressor with learnable piecewise-linear spline
  edges, internal normalization, and differentiable `predict_torch`. This is a
  minimal self-contained KAN implementation with a TODO for cubic B-splines.
- American put MLP LSMC runner with default architecture `[1, 32, 32, 1]`.
- American put KANOP LSMC runner with default architecture `[1, 3, 1]`.

## Asian-American call replication target

Parameters vary by case. Use Laguerre cross-products up to total degree 4 in `[S_t, TWAP_t]`, which creates 15 regressors.

| Case | K | Weeks | sigma | Eurasian | Asian-American | Laguerre | MLP | KANOP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100 | 13 | 0.15 | 2.1638 | 2.3210 | 2.2750 | 2.2601 | 2.3216 |
| 2 | 100 | 13 | 0.25 | 3.3621 | 3.6500 | 3.5716 | 3.6134 | 3.6589 |
| 3 | 100 | 26 | 0.25 | 4.7659 | 5.2660 | 5.0719 | 5.1422 | 5.2382 |
| 4 | 105 | 26 | 0.25 | 2.6628 | 2.8580 | 2.7162 | 2.7943 | 2.8309 |

## Next implementation milestones

1. Verify that the path simulator reproduces terminal European/Eurasian MC prices.
2. Tune whether regressions should use all paths or only ITM paths for each experiment.
3. Add Asian-American MLP experiment runner and tune MLP hyperparameters/seeds.
4. Tune American put KANOP hyperparameters/seeds and compare spline choices.
5. Add autograd delta at `t_1`.
6. Add seed-stability runs.

## Reproducibility switches

The baseline scripts default to fitting continuation regressions on all paths.
This is consistent with parts of the paper's continuation/delta discussion, but
ITM-only fitting is a standard LSMC variant and materially changes prices. Use
`--fit-itm-only` to run that variant; the result CSVs record the setting.

Fixed-basis inputs default to raw state variables, preserving the initial
implementation. Because high-order Laguerre/Hermite terms are sensitive to input
scale, scripts also expose `--basis-scaling raw|S_over_K|S_over_S0|standardized`.
The result CSVs record the chosen scaling mode.

## KANOP implementation caveats

The paper does not specify random seed, optimizer, epochs, learning rate, KAN
grid size, spline order, or normalization. This repository's first KANOP runner
uses a self-contained piecewise-linear spline-edge KAN rather than an external
or original cubic/B-spline KAN implementation. Differences from the reported
KANOP price should therefore be interpreted as implementation and
hyperparameter gaps, not as calibrated paper reproduction.
