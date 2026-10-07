# Frozen temporal backtest: train on 2024, evaluate on 2025

This is the project-level forecast evaluation that follows model development.
It does not tune models on 2025.  It refits the fixed V1 configurations on all
eligible 2024 data and then evaluates them on the future 2025 period.

Run it after building both annual PV datasets and the spatial day-ahead-price
feature table:

```bash
energy build-training-data --project-root . --year 2024
energy build-training-data --project-root . --year 2025
energy build-day-ahead-price-spatial-data --project-root .
energy run-temporal-backtest-2025 --project-root .
```

The command writes outputs to:

```text
artifacts/experiments/temporal_backtest_v1/2024_to_2025/
```

## What is evaluated

### Day-ahead PV

The registered-style CatBoost day-ahead PV forecast is fitted on all 2024 rows
from local 06:00 through 21:00.  It uses only calendar fields, deterministic
solar geometry and the archived day-ahead weather forecast.  The results report
both all those delivery hours and the solar-active subset, defined as
clear-sky GHI of at least 25 W/m².

### MPC PV corrections

The MPC portion compares the same 2025 `(decision hour, future target hour)`
pairs for four forecasts:

1. frozen day-ahead baseline;
2. latest-residual persistence;
3. direct CatBoost residual correction;
4. MIMO MLP residual trajectory correction;
5. MIMO LSTM residual trajectory correction.

The day-ahead baseline used inside this experiment is fitted once on all
solar-active 2024 rows.  For residual-model training only, it is cross-fitted
inside 2024: an out-of-fold head forecast supplies a realistic historical
residual.  This prevents an in-sample head model from making residuals
artificially small.  No 2025 outcome is used in those predictions or in model
training.

At each 2025 MPC decision time the correction models see only the frozen
head forecast and factual PV from solar-active hours strictly before the
decision.  There is still no newer intraday weather snapshot in V1.

The MLP and LSTM use their fixed compact architectures and a fixed number of
epochs.  The number is recorded in `summary.json`; it is not selected from
2025 metrics.

### Day-ahead prices

The existing chronological day-ahead-price experiment is rerun with 2024 as
training and 1 January–30 September 2025 as the test period.  The source
becomes 15-minute after September 2025, whereas the V1 price model uses
hourly auction data; the later period is deliberately excluded rather than
resampled or mixed with the hourly experiment.

## Outputs

- `metrics.csv` — compact metrics for all PV and price variants;
- `pv_mpc_metrics_by_lead_hour.csv` — MPC MAE/RMSE by remaining lead hour;
- `pv_day_ahead_hourly_predictions.parquet` — day-ahead PV predictions;
- `pv_mpc_trajectory_predictions.parquet` — all MPC decision/target forecasts;
- `summary.json` — inputs, temporal contract and evaluation limits.

PV metrics are in watts. `nmae_train_observed_peak` and
`nrmse_train_observed_peak` use the maximum observed 2024 PV target as a fixed
normalisation reference.  Price metrics remain in EUR/MWh.

This backtest evaluates forecast plausibility and temporal generalisation.  It
does not yet assess economic value.  The next stage is the market and battery
simulator, where deterministic and stochastic decision policies can be
compared using the same frozen out-of-sample forecasts.

## Completed result: 2024 training, 2025 holdout

The following run used `n_lags=4`, 500 CatBoost iterations and the fixed
300-epoch neural configurations.  It did not inspect 2025 during fitting,
scaling or model selection.

### PV day-ahead forecast

| Evaluation slice | Rows | MAE, W | RMSE, W | nMAE / observed 2024 peak |
| --- | ---: | ---: | ---: | ---: |
| Local 06:00–21:00 delivery hours | 5,824 | 26.17 | 51.35 | 5.32% |
| Solar-active hours | 3,942 | 37.06 | 62.03 | 7.53% |

### PV forecasts used by MPC

| Variant | Decision/target pairs | MAE, W | RMSE, W | Change in MAE vs MPC baseline |
| --- | ---: | ---: | ---: | ---: |
| Frozen day-ahead baseline | 17,034 | 17.44 | 33.15 | — |
| Latest-residual persistence | 17,034 | 44.65 | 68.59 | 156.0% worse |
| Direct CatBoost residual correction | 17,034 | **16.41** | **31.70** | **+5.9%** |
| MIMO MLP residual correction | 17,034 | 17.28 | 33.11 | +0.9% |
| MIMO LSTM residual correction | 17,034 | 17.08 | 33.73 | +2.1% |

The direct CatBoost correction is therefore the V1 point forecast to carry
into the MPC simulator.  The two neural trajectory models are retained as
completed research comparisons: they run under the correct information set but
do not justify added operational complexity on this one-site data.

The two PV day-ahead rows above must not be compared directly with the MPC
baseline.  The former scores each local 06:00–21:00 delivery hour once and
uses the general deployable day-ahead feature contract.  The latter scores
only solar-active `(decision hour, future target hour)` pairs, which weights
later low-generation target hours differently, and uses the active-hour head
forecast that was part of the residual-model experiment.

### Day-ahead price forecast, January–September 2025

| Variant | MAE, EUR/MWh | RMSE, EUR/MWh | Mean daily Spearman |
| --- | ---: | ---: | ---: |
| D-1 same-hour persistence | 25.57 | 40.12 | 0.796 |
| Price history + calendar | 20.36 | 30.95 | 0.876 |
| Price history + local weather | 19.33 | 30.06 | 0.878 |
| Price history + spatial weather | **17.43** | **27.31** | **0.912** |

The spatial-weather price model improves MAE by 31.8% relative to persistence.
The price result ends in September because the raw series changes from hourly
to 15-minute data on 1 October 2025.
