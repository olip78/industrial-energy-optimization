# Intraday PV weather-refresh evaluation

## Decision

Use the most recently available ECMWF IFS weather values in place of the stale
day-ahead weather fields in the direct PV residual model. Keep the frozen
day-ahead PV prediction as the anchor and retain the factual PV/residual lags.
Refit the residual model at a regular boundary using only completed delivery
days.

The change improves the locked point-forecast holdout. Its incremental economic
effect in the present reference scenario is positive but very small and not yet
statistically conclusive.

## Pipeline audit

The point-in-time joins are valid:

- every selected IFS run was published no later than the MPC decision time;
- every forecast target is strictly later than the decision time;
- factual PV lags use observations strictly before the decision hour;
- the day-ahead base forecast remains the model fitted on 2024 data.

Observed weather is deliberately excluded from the intraday feature set. The
historical "actual weather" table is retrospective data and was not archived
with a trustworthy real-time availability boundary; using it would risk target
leakage and would create a feature that the live service could not reproduce.
The deployable model therefore uses the latest on-time IFS forecast together
with factual PV observations available before the decision hour. Real-time
station observations can be added later if the ingestion pipeline records both
measurement time and availability time.

The previous experiment had two material weaknesses.

First, the weather archives are asymmetric. The 2024 archive starts on
2024-03-14 and contains 293 runs, all initialized at 00 UTC. The 2025 archive
contains 1,064 runs through 2025-09-30 and exposes the 00/06/12/18 UTC cycles.
A residual model fitted only on 2024 therefore never learned how to use the
fresh daytime cycles seen in 2025.

Second, the previous feature set included the old ICON level, new IFS level and
their exact difference at the same time. This is not target leakage, but it is
an unnecessary deterministic redundancy for a small tree-training sample. The
validated default now replaces the stale weather levels with the fresh levels
and adds snapshot lead and age. `revision` and legacy modes remain available
for controlled experiments.

The weather refresh itself contains useful physical information. On 2025
solar-active rows, shortwave-radiation MAE falls from approximately 67 to
58 W/m² during the 08:00-11:00 decision window and from 62 to 54 W/m² during
12:00-15:00. Direct radiation, diffuse radiation, cloud cover and wind speed
also improve on average.

## Validation design

The day-ahead PV head is fitted once on 2024 and then frozen. Its 2024 residual
training targets use cross-fitted head predictions, which avoids unrealistically
small in-sample residuals.

The MPC residual model uses an expanding monthly training window:

1. begin with the eligible 2024 rows;
2. at each 2025 month boundary, add only completed earlier 2025 days;
3. generate predictions for the new month;
4. use 2025 Q2 to select the weather representation;
5. evaluate the selected representation once on the locked 2025 Q3 holdout.

Q2 selected `rolling_updated_weather_replace` by MAE, with RMSE as the
tie-break. The legacy all-features variant happened to score slightly better on
Q3, but it was not selected before the holdout and is therefore not promoted.

### Locked Q3 point metrics

| Variant | MAE, W | RMSE, W | P95 absolute error, W |
|---|---:|---:|---:|
| Frozen day-ahead | 17.63 | 32.93 | 71.82 |
| Frozen 2024 MPC, original weather | 17.49 | 33.24 | 74.57 |
| Rolling MPC, original weather | 16.98 | 32.78 | 70.36 |
| **Rolling MPC, fresh IFS replacing stale weather** | **16.46** | **31.31** | **69.71** |

Relative to the rolling original-weather model, fresh IFS reduces MAE by
0.52 W (3.1%) and RMSE by 1.47 W (4.5%). Relative to the frozen 2024 MPC
model, the combined online adaptation and weather refresh reduce MAE by
1.04 W (5.9%).

The gain is concentrated at leads 2-5 hours: the MAE reductions are about
0.60, 1.08, 0.86 and 0.79 W respectively. At lead 1 the reduction is only
0.08 W because the newest factual PV residual already carries most of the
short-horizon signal. This lead profile helps explain why the aggregate
forecast improvement produces only a small economic change in receding-horizon
control, where only the first action is executed before the next optimization.

## Economic MPC replay

The economic comparison uses the same frozen day-ahead plan, price forecasts,
realized market data, grid tariff, battery model and Q3 dates for every row.
Only the PV trajectory supplied to MPC changes.

| Q3 policy | Total cost, EUR | MPC saving vs day-ahead-only, EUR | Share of remaining day-ahead-to-Oracle gap |
|---|---:|---:|---:|
| Day-ahead-only | 2,456.02 | 0.00 | 0.0% |
| Frozen 2024 MPC | 2,445.30 | 10.72 | 39.35% |
| Rolling MPC, original weather | 2,444.97 | 11.04 | 40.56% |
| **Rolling MPC, fresh IFS** | **2,444.74** | **11.28** | **41.41%** |
| Oracle | 2,428.78 | — | 100.0% |

Fresh weather saves EUR 0.23 over the rolling original-weather MPC on the
91-day holdout, or about 2.1% of MPC's incremental value. The 7-day circular
block-bootstrap 95% interval for mean daily savings is approximately
[-0.00013, 0.00658] EUR and includes zero. This result supports the feature on
forecast quality and correct economic direction; it does not yet establish a
material economic gain.

The small economic response is plausible. The day-ahead nomination is already
fixed, forecast errors are small after scaling, and MPC can change only the
remaining load, battery and curtailment decisions. Many daily optimization
solutions stay in the same active region even when the PV trajectory becomes
slightly more accurate.

## Reproduction

Run the rolling forecast evaluation:

```bash
python -m energy.cli run-pv-mpc-weather-refresh-evaluation \
  --project-root /Users/andreichekunov/andrei/machine_learning/2026/energy
```

Run the Q3 economic replay with the selected weather-refresh artifact:

```bash
python -m energy.cli run-dynamic-charge-economic-backtest \
  --project-root /Users/andreichekunov/andrei/machine_learning/2026/energy \
  --start 2025-07-01 \
  --end 2025-09-30 \
  --artifact-name economic_backtest_v4_pv_mpc_updated_weather_q3 \
  --pv-mpc-prediction-path artifacts/experiments/pv_mpc_weather_refresh_rolling_v1/pv_mpc_predictions_selected_updated_weather.parquet
```

Primary artifacts:

- `artifacts/experiments/pv_mpc_weather_refresh_rolling_v1/metrics.csv`;
- `artifacts/experiments/pv_mpc_weather_refresh_rolling_v1/holdout_lead_metrics.csv`;
- `artifacts/experiments/pv_mpc_weather_refresh_rolling_v1/feature_importance.csv`;
- `artifacts/experiments/pv_mpc_weather_refresh_rolling_v1/economic_comparison_q3.csv`;
- `artifacts/experiments/pv_mpc_weather_refresh_rolling_v1/audit_and_economic_summary.json`;
- `artifacts/experiments/economic_backtest_v4_pv_mpc_updated_weather_q3/`.

## Limitations

- The experiment covers one PV installation and only one Q3 holdout.
- The 2024 weather archive does not contain the daytime IFS cycles available in
  2025, so the first three months of 2025 act as an online warm-up.
- IFS and the day-ahead ICON input are different forecast systems. Their
  difference mixes forecast revision with provider-specific bias.
- Monthly refitting is part of the evaluated policy. A deployment must schedule
  that refit and monitor weather-cycle coverage and feature drift.
