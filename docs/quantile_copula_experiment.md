# Conditional quantile regression and empirical copula V1

This experiment replaces the unconditional residual-bootstrap marginals with
conditional PV and day-ahead-price quantiles. It retains a simple empirical
dependence model and evaluates both forecast calibration and realised 2025
economics.

## Method

PV and price use the same feature contracts as their frozen point models. One
CatBoost `MultiQuantile` model per target estimates:

```text
Q05, Q10, Q20, Q35, Q50, Q65, Q80, Q90, Q95
```

The 2024 training procedure is expanding-window and rolling-origin. For every
out-of-sample day, the factual error relative to P50 is ranked separately by
target and local hour. One historical day is one complete vector of 32 ranks:
16 PV ranks and 16 price ranks. Sampling a whole vector preserves empirical
dependence between targets and across the working day. Each sampled rank is
then passed through the new day's conditional quantile curve. Ranks outside
P05--P95 are clipped to the supported endpoints.

CatBoost does not enforce non-crossing MultiQuantile outputs. The raw 2025
forecasts cross on 31.4% of PV rows and 53.6% of price rows, so monotone
rearrangement sorts each row before interpolation. This restores valid
quantile ordering but does not calibrate interval width.

## Forecast and scenario result

The untouched evaluation covers 261 days and 4,176 target hours, with 500
scenarios per day.

| Generator | Target | Point/P50 MAE | Scenario-mean MAE | P10--P90 coverage | Width | CRPS |
|---|---|---:|---:|---:|---:|---:|
| Residual bootstrap, centred | PV, kWh | 0.548 | 0.585 | 84.10% | 1.695 | 0.4060 |
| Quantile + copula | PV, kWh | **0.521** | **0.553** | 72.73% | 1.552 | **0.3709** |
| Residual bootstrap, centred | Price, EUR/MWh | 20.223 | **20.290** | 80.03% | 60.486 | **15.0209** |
| Quantile + copula | Price, EUR/MWh | **20.066** | 21.731 | **41.16%** | 35.131 | 16.5781 |

PV improves in MAE and CRPS but its interval is somewhat too narrow. The price
marginal fails calibration badly. For example, the nominal price P10 contains
36.2% of facts below it, while nominal P90 contains only 77.2%. The central
interval is therefore compressed from both sides. This is not a copula defect:
sampling reproduces the empirical rank-correlation matrix with RMSE 0.010 and
maximum absolute error 0.034. A copula can preserve dependence, but it cannot
repair miscalibrated marginals.

## Economic replay

For a clean comparison, the submitted day-ahead position continues to use the
same frozen RMSE point-PV forecast as deterministic and residual-bootstrap
strategies. Quantile-copula scenarios affect the shared load and discharge
schedule, without changing the centre used for nomination. All strategies are
then settled on the same factual day-ahead and intraday prices.

| Strategy | Total cost, EUR | Change vs deterministic DA, EUR | Realised CVaR95, EUR/day | Maximum day, EUR | Battery discharge, kWh |
|---|---:|---:|---:|---:|---:|
| Deterministic day-ahead | 3,039.83 | 0.00 | 31.862 | 46.323 | 3,423.12 |
| Quantile-copula SAA | 3,043.19 | +3.35 | 31.644 | 49.567 | 3,750.56 |
| CVaR, λ = 0.05 | 3,045.77 | +5.93 | 31.673 | 49.567 | 3,816.30 |
| CVaR, λ = 0.10 | 3,047.18 | +7.35 | 31.673 | 49.604 | 3,874.70 |
| CVaR, λ = 0.25 | 3,051.06 | +11.23 | **31.611** | 48.240 | 4,080.00 |
| CVaR, λ = 0.50 | 3,057.31 | +17.48 | 31.640 | 48.207 | 4,300.64 |
| CVaR, λ = 1.00 | 3,067.83 | +28.00 | 31.811 | 48.207 | 4,487.19 |

The realised tail response is weak and non-monotonic. The best observed
CVaR95 is at λ = 0.25, a reduction of 0.251 EUR per tail day for an
11.23 EUR total premium. Its paired weekly block-bootstrap 95% interval for
tail reduction is -0.523 to 1.605 EUR/day and includes zero. Stronger risk
weights increase battery use without providing consistent factual protection.

The centred residual-bootstrap result is better: at λ = 0.10 it costs
3,046.35 EUR, has CVaR95 of 31.431 EUR/day and a maximum day of 45.533 EUR.
The raw quantile-copula model therefore does not replace the bootstrap
baseline. The result points directly to the next planned variant: conformal
calibration of the conditional quantiles before applying the same empirical
copula. That experiment should test whether conditional marginals can retain
the PV CRPS gain while restoring price-tail coverage.

## Reproduction and artifacts

```bash
energy run-quantile-copula-experiment --project-root .
energy run-quantile-copula-economic-backtest --project-root .
```

Forecast artifacts are written to `artifacts/experiments/quantile_copula_v1/`.
They include rolling-origin quantiles, the empirical copula library, 2025
quantile forecasts, scenario bands, calibration by quantile, crossing rates
and dependence audits.

Economic artifacts are written to
`artifacts/experiments/quantile_copula_economic_backtest_v1/` and contain the
daily ledger, hourly decisions, scenario objectives, coverage, configuration
and paired block-bootstrap uncertainty intervals.
