# Conformalized quantile regression with empirical copula

This experiment applies rolling-origin conformal calibration to the raw
conditional quantile models before reusing the same empirical copula. It tests
whether calibrated conditional marginals improve distributional forecasts and
the realised cost/risk tradeoff.

## Calibration contract

All corrections use only 210 out-of-sample 2024 days. They are fitted
separately for PV, price, local hour and the four central intervals
P05--P95, P10--P90, P20--P80 and P35--P65. For a lower quantile level `alpha`,
the nonconformity score is:

```text
score[i] = max(
    q_hat[alpha](x[i]) - y[i],
    y[i] - q_hat[1 - alpha](x[i]),
    0
)
```

The finite-sample conformal score quantile expands the corresponding lower and
upper endpoints symmetrically. P50 is unchanged. The resulting grid is
monotonically rearranged; PV is clipped to 0--10 kWh. No 2025 observation
affects a correction. The empirical copula ranks are unchanged, isolating the
effect of marginal calibration.

## Forecast result

| Generator | Target | P10--P90 coverage | Mean width | CRPS |
|---|---|---:|---:|---:|
| Raw quantile-copula | PV | 72.73% | 1.552 kWh | **0.3709** |
| CQR-copula | PV | **90.90%** | 1.926 kWh | 0.3714 |
| Centred residual bootstrap | PV | 84.10% | 1.695 kWh | 0.4060 |
| Raw quantile-copula | Price | 41.16% | 35.131 EUR/MWh | 16.5781 |
| CQR-copula | Price | **87.14%** | 81.385 EUR/MWh | 15.2472 |
| Centred residual bootstrap | Price | 80.03% | 60.486 EUR/MWh | **15.0209** |

CQR solves the dangerous undercoverage of the raw price model and reduces its
CRPS substantially. It overshoots the nominal 80% coverage for both targets.
For price, the 2024 hourly P10/P90 corrections are often 16--27 EUR/MWh per
side, producing wider intervals than the residual bootstrap. This is a useful
calibration result rather than an automatic modelling improvement: conformal
coverage can be conservative under a different future distribution, and this
version uses hour-conditioned scores rather than a season- or
regime-conditioned calibration window.

## Economic result

The day-ahead nomination continues to use the frozen point PV forecast, so the
comparison changes only the stochastic schedule. The realised intraday
deviation component is consequently identical across day-ahead-only variants.

| Strategy | Total cost, EUR | Change vs deterministic DA, EUR | Realised CVaR95, EUR/day | Maximum day, EUR | Battery discharge, kWh |
|---|---:|---:|---:|---:|---:|
| Deterministic day-ahead | 3,039.83 | 0.00 | 31.862 | 46.323 | 3,423.12 |
| CQR-copula SAA | 3,044.67 | +4.84 | 31.660 | 49.567 | 3,742.24 |
| CVaR, λ = 0.05 | 3,046.35 | +6.52 | 31.659 | 49.567 | 3,869.37 |
| CVaR, λ = 0.10 | 3,050.11 | +10.28 | **31.564** | 48.233 | 4,042.98 |
| CVaR, λ = 0.25 | 3,060.05 | +20.22 | 31.598 | 48.240 | 4,422.63 |
| CVaR, λ = 0.50 | 3,073.79 | +33.96 | 31.675 | 48.240 | 4,865.56 |
| CVaR, λ = 1.00 | 3,092.70 | +52.87 | 31.819 | 48.207 | 5,467.63 |

CQR modestly improves the raw quantile-copula tail result: its best observed
point is λ = 0.10, with a 0.299 EUR/day CVaR95 reduction for a 10.28 EUR
period premium. The paired weekly block-bootstrap 95% interval for that tail
reduction is -0.500 to 1.656 EUR/day, so it is not statistically stable.

The centred residual-bootstrap with λ = 0.10 remains better: total cost
3,046.35 EUR, CVaR95 31.431 EUR/day and maximum day 45.533 EUR. CQR's wider
tails cause more battery discharge and a larger insurance premium, while the
realised maximum remains above deterministic day-ahead. Stronger risk weights
make this tradeoff worse.

The practical conclusion for V1 is therefore:

1. empirical copula dependence is implemented correctly;
2. raw conditional price quantiles are seriously underdispersed;
3. hourly CQR repairs coverage but is too conservative;
4. centred joint residual bootstrap remains the preferred simple stochastic
   baseline for the available data.

A later calibration refinement could use recent rolling windows, season or
price-volatility regimes, or standardized/nonconformity scores. It should only
be pursued if improving conditional uncertainty is itself an important project
goal; the present experiment already demonstrates the full quantile, copula,
conformal and stochastic-optimization workflow.

## Reproduction

```bash
energy run-cqr-copula-experiment --project-root .
energy run-cqr-copula-economic-backtest --project-root .
```

Forecast outputs are stored in `artifacts/experiments/cqr_copula_v1/` and the
economic ledger in
`artifacts/experiments/cqr_copula_economic_backtest_v1/`.
