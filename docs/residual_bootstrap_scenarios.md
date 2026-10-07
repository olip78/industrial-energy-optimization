# Joint residual-bootstrap scenarios

The first probabilistic forecasting experiment constructs paired PV and
day-ahead-price trajectories for the 16-hour controllable window, local hours
06:00 through 21:00. It does not yet solve the stochastic optimization problem;
it validates the scenario generator before it becomes an SAA input.

## Residual library

The library is built from expanding-window out-of-sample forecasts in 2024.
After an initial 84 common delivery days, the existing PV and spatial-weather
price CatBoost models are refit every 28 delivery days. Each predicted day
contributes one complete block

$$
R_d =
\left(
\varepsilon^{PV}_{d,6:21},
\varepsilon^{DA}_{d,6:21}
\right).
$$

The resulting library contains 210 complete paired days. PV residuals are
converted from the source profile's W to the reference site's scaled hourly
kWh. Price residuals remain in EUR/MWh.

For a new day, one source day is sampled with replacement and both complete
residual paths are added to the point forecasts. PV is clipped to the physical
range 0--10 kWh per hour. Price is left unbounded because negative market
prices are valid. Sampling retains the source date for auditability and only
allows dates strictly earlier than the forecast day.

## Untouched 2025 evaluation

The generator is evaluated on 261 common delivery days from 1 January through
30 September 2025, using 500 scenarios per day. The nominal interval is
P10--P90, so its target marginal coverage is 80%.

| Residual treatment | Target | Point MAE | Scenario-mean MAE | P10--P90 coverage | Mean width | CRPS |
|---|---|---:|---:|---:|---:|---:|
| Raw | PV, kWh | 0.548 | 0.578 | 81.35% | 1.642 kWh | 0.4053 |
| Raw | Price, EUR/MWh | 20.223 | 20.939 | 79.36% | 60.486 EUR/MWh | 15.0684 |
| Hour-centred | PV, kWh | 0.548 | 0.585 | 84.10% | 1.695 kWh | 0.4060 |
| Hour-centred | Price, EUR/MWh | 20.223 | 20.290 | 80.03% | 60.486 EUR/MWh | 15.0209 |

Both versions give credible first-order uncertainty intervals. Raw residuals
work slightly better for PV. Hour-centering is preferable for price because it
prevents 2024 hour-specific bias from shifting the 2025 scenario mean. The
choice can be target-specific without damaging dependence: subtracting a
constant by target/hour leaves the residual correlation structure unchanged.

Sampling 10,000 blocks reproduced the training correlation matrix with RMSE
0.0146 and maximum absolute error 0.0607. This verifies the implementation,
although dependence preservation is expected from whole-block sampling.

The main limitation is that this V1 bootstrap is unconditional. Its interval
width depends on local hour but not on the new day's predicted weather or
generation level. Hour-level coverage ranges more widely than pooled coverage,
especially around the morning PV ramp and some price hours. Conditional
bootstrap, quantile-copula, and CQR-copula variants address that limitation.

## Commands and artifacts

```bash
energy run-residual-bootstrap-experiment --project-root .

energy run-residual-bootstrap-experiment \
  --project-root . \
  --center-residuals \
  --artifact-name joint_residual_bootstrap_centered_v1
```

Each artifact directory contains the rolling-origin residual library,
out-of-sample prediction bands, a fully auditable example scenario batch,
metrics by target and hour, residual diagnostics, and correlation audits.
