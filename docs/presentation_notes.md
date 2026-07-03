# American Put Replication Presentation Notes

## 5-minute slide outline

1. Problem and setup
   - Replicate the KANOP American put case under risk-neutral GBM.
   - Parameters: `S0=4`, `K=4`, `T=50/252`, `sigma=20%`, `r=q=0`, daily exercise.
   - Benchmark: Black-Scholes European put price near `0.1421`.

2. LSMC pipeline
   - Simulate GBM paths.
   - Run backward induction from `t49` to `t1`.
   - At each step, fit continuation value from current stock state.
   - Exercise when intrinsic value exceeds fitted continuation value.

3. Model comparison
   - Fixed bases: Weighted Laguerre and Hermite.
   - Neural baseline: MLP `[1, 32, 32, 1]`.
   - KANOP-style model: piecewise-linear spline-edge KAN `[1, 3, 1]`.

4. Key results
   - Black-Scholes exact target: `0.142115`.
   - Weighted Laguerre: `0.138970`, error vs BS `0.003146`.
   - Hermite: `0.138809`, error vs BS `0.003306`.
   - MLP: `0.140668`, error vs BS `0.001447`.
   - KANOP: `0.141839`, error vs BS `0.000276`.
   - Current KANOP is closest to the Black-Scholes target in this run.

5. Continuation-value diagnostic
   - Show `figures/american_put_continuation_all_t25.png`.
   - The plot overlays true Black-Scholes continuation with fitted continuation curves.
   - The diagnostic makes clear whether the price improvement comes from a better continuation approximation.

## Caveats

- The paper does not specify random seed, optimizer details, epochs, batch size, KAN grid size, spline order, or normalization.
- The repository KAN is a self-contained piecewise-linear spline-edge model, not the original paper's likely B-spline KAN implementation.
- MLP row shown here uses the available practical run (`10,000` paths, `50` epochs), not the paper's `100,000`-path MLP setting.
- Delta estimates are not implemented yet, so reported paper deltas are not reproduced here.
- Exact Black-Scholes formula delta differs from the paper's rounded `-0.5000` target.

## Future research extensions

- Add autograd-based delta estimation for MLP and KANOP continuation models.
- Run seed-stability studies for all learned regressors.
- Upgrade KAN edges from piecewise-linear splines to cubic B-splines and compare price/continuation fits.
- Implement full Asian-American MLP and KANOP experiments with `[S_t, TWAP_t]` inputs.
- Add hyperparameter sweeps for KAN grid size, epochs, learning rate, and fit-all-paths vs ITM-only masking.
- Produce paper-style continuation plots across all reported exercise dates and model variants.
