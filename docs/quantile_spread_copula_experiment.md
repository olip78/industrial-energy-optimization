# Quantile regression and empirical copula V2

This experiment produces the joint day-ahead scenarios required by the final
stochastic formulation. It models three uncertain hourly quantities over the
06:00--21:00 operating window:

- PV generation, in kWh;
- the day-ahead auction price, in EUR/MWh;
- the intraday spread `intraday price - day-ahead price`, in EUR/MWh.

The intraday price in scenario `s` is reconstructed as

```math
P^{ID}_{s,h} = P^{DA}_{s,h} + \Delta P_{s,h}.
```

This is a forecasting and scenario-generation experiment. It does not run MPC
or allow scenario-specific control decisions. A later stochastic day-ahead
optimizer must choose one common schedule before the scenario is known.

## Information and validation contract

The PV and day-ahead price marginal forecasts are the rolling-origin
MultiQuantile CatBoost forecasts from `quantile_copula_v1`. The added spread
model uses the same leakage-safe features available before the day-ahead
auction. Realised day-ahead and intraday prices are targets only.

The empirical copula is fitted from 210 rolling-origin 2024 days. One sampled
historical rank row supplies all 48 ranks for a scenario: three targets times
16 operating hours. This preserves the observed cross-hour and cross-target
rank dependence rather than sampling each hour independently. Only copula rows
strictly earlier than the simulated delivery day are eligible.

Raw CatBoost quantiles are monotonically rearranged because MultiQuantile can
produce crossings. For the new spread marginal, crossings occurred in 77.17%
of rolling-origin 2024 rows and 52.97% of 2025 rows before rearrangement. This
is a model diagnostic, not data leakage.

Hourly central-interval conformal corrections are estimated solely from the
rolling-origin 2024 errors. The untouched evaluation period is 1 January
through 30 September 2025, containing 261 complete delivery days and 4,176
hourly observations.

## 2025 result

The nominal P10--P90 coverage is 80%.

| Generator | Target | Point MAE | Point RMSE | P10--P90 coverage | Mean width | CRPS |
|---|---|---:|---:|---:|---:|---:|
| Raw quantiles | PV | 0.521 kWh | 1.030 kWh | 72.73% | 1.552 kWh | 0.3709 |
| CQR | PV | 0.521 kWh | 1.030 kWh | 90.90% | 1.926 kWh | 0.3714 |
| Raw quantiles | Day-ahead price | 20.066 EUR/MWh | 30.770 EUR/MWh | 41.16% | 35.131 EUR/MWh | 16.5781 |
| CQR | Day-ahead price | 20.066 EUR/MWh | 30.770 EUR/MWh | 87.14% | 81.385 EUR/MWh | 15.2472 |
| Raw quantiles | Intraday spread | 15.204 EUR/MWh | 24.382 EUR/MWh | 54.98% | 36.129 EUR/MWh | 12.3308 |
| CQR | Intraday spread | 15.204 EUR/MWh | 24.382 EUR/MWh | 87.69% | 70.228 EUR/MWh | 11.8452 |

The point metrics are unchanged by CQR because the median is left untouched.
Raw price and spread intervals are materially underdispersed. CQR repairs this
and improves CRPS for both market targets, at the cost of conservative and
wider intervals. PV CQR is also conservative and does not improve CRPS, so the
economic backtest should compare both raw and calibrated generators rather
than assuming that wider coverage is automatically better.

The calibrated P10--P90 coverage by hour is approximately 84--97% for PV,
82--94% for day-ahead price and 79--94% for the intraday spread. The weakest
spread coverage occurs around 13:00--14:00; the widest price and spread ranges
occur around 19:00--20:00.

## Reproduction

```bash
energy run-quantile-spread-copula-experiment \
  --project-root . \
  --source-artifact-name quantile_copula_v1 \
  --iterations 500 \
  --n-scenarios 500
```

Outputs are written to
`artifacts/experiments/quantile_spread_copula_v2/`. The principal files are:

- `test_raw_quantile_forecasts.parquet` and
  `test_calibrated_quantile_forecasts.parquet`;
- `empirical_copula_library.parquet`;
- `raw_prediction_bands.parquet` and `cqr_prediction_bands.parquet`;
- `example_scenario_batch.parquet`;
- `metrics.csv`, `quantile_calibration.csv`,
  `conformal_corrections.csv`, `dependence_summary.csv` and
  `spread_quantile_crossing.csv`.

The next economic stage must hold the selected day-ahead load, battery and
curtailment plan fixed when it is settled against actual 2025 values. It must
not add MPC or re-optimise the plan after observing a scenario.
