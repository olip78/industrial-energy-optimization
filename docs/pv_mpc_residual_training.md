# Direct residual PV model for MPC

`energy train-pv-mpc-residual` trains the pointwise CatBoost correction model
used by the MPC loop. The model does not replace the day-ahead PV forecast. It
predicts the residual of that frozen forecast for one future solar-active hour.
The MPC process calls it once for every remaining target hour.

For a delivery hour `t`, the frozen day-ahead forecast is `y_hat_da[t]` and the
actual PV output is `y[t]`. The residual target is:

```text
residual[t] = y[t] - y_hat_da[t]
```

At local decision hour `tau`, the model can use factual PV only from completed
hours before `tau`. It receives the most recent solar-active actual PV values,
their corresponding day-ahead forecasts and residuals, plus the target hour's
day-ahead forecast, target weather forecast and solar geometry. It predicts:

```text
residual_hat[tau, t]
pv_hat_mpc[tau, t] = max(y_hat_da[t] + residual_hat[tau, t], 0)
```

The training data contains one row per `(decision hour, future target hour)`.
The original day-ahead prediction for a historical training row is out-of-fold.
This matters: fitting it in-sample would make historic residuals artificially
small and train a residual model that cannot work in operation.

## Solar-active hours and early-day lags

The model dynamically selects an active interval rather than a fixed
08:00–18:00 clock window:

```text
clear-sky GHI >= 25 W/m²
```

Clear-sky GHI is deterministic from time and the provisional Pforzheim proxy
coordinate, so it is available at decision time. The interval is longer in
summer and shorter in winter. At the start of a solar day, unavailable lag
values are represented by `NaN` and an explicit availability flag.

## Train and evaluate

Install the optional training stack and configure MLflow as described in
`docs/pv_training_and_mlflow.md`.

```bash
energy train-pv-mpc-residual \
  --project-root . \
  --year 2024 \
  --n-lags 4
```

The application evaluates the full two-stage process with season-balanced
whole-week outer folds. Each fold compares three forecasts on the same MPC
decision / target pairs:

1. `baseline`: frozen day-ahead PV forecast;
2. `persistence`: day-ahead forecast plus the latest observed residual;
3. `direct_residual`: day-ahead forecast plus the CatBoost correction.

The MLflow run records mean and standard deviation of MAE and RMSE across
outer folds, the feature contract, week assignments, feature importance and
the final model.

## Register a model version

```bash
energy train-pv-mpc-residual \
  --project-root . \
  --year 2024 \
  --n-lags 4 \
  --register \
  --registered-model-name energy-pv-mpc-residual
```

This creates a version in MLflow Model Registry but assigns no alias. The
residual model must be deployed with the compatible frozen day-ahead forecast
contract; `day_ahead_prediction_w` is an input to the residual model, not a
prediction it recreates itself.

## Intraday weather update (V1.1)

The optional ECMWF IFS snapshot layer gives the residual model a forecast that
was published during delivery, while preserving the original day-ahead head
forecast. Start training after the IFS archive begins and pass the decision /
target snapshot table:

```bash
energy train-pv-mpc-residual \
  --project-root . --year 2024 \
  --train-start-date 2024-03-14 \
  --weather-snapshot-path \
    data/features/mpc_pv_weather_ifs_snapshots/mpc_pv_weather_ifs_snapshots_2024.parquet
```

The model replaces only target weather features with the latest snapshot that
was available before the decision. It adds snapshot lead and age features, and
keeps actual PV/residual lags strictly before the decision. The data source,
availability rule and download commands are documented in
[intraday_weather_ifs.md](intraday_weather_ifs.md).
