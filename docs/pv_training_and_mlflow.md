# Day-ahead PV training application and MLflow

`energy train-pv-day-ahead` is the reproducible training entry point for the
model that can be used at the day-ahead decision point. It trains one CatBoost
regressor from forecast weather, calendar fields and deterministic solar
geometry. Actual weather and the Oracle proxy are deliberately excluded: they
are useful for diagnostics, but unavailable when a real day-ahead forecast is
made.

The command reads `data/features/day_ahead_pv/day_ahead_pv_<year>.parquet` by
default. It retains local hours 06:00–21:00, derives solar elevation, azimuth,
clear-sky GHI and the forecast clear-sky index, then uses whole local-calendar
weeks for a season-balanced 70% / 15% / 15% train, validation and test split.
Early stopping uses only validation data. The test partition is evaluated once,
after the final model is refit on the train and validation partitions.

`nmae_observed_peak` and `nrmse_observed_peak` use the observed maximum PV
power in the selected dataset as their denominator. They are reporting metrics,
not percentages of installed capacity.

## Local setup

Install the application with the optional training dependencies:

```bash
python -m pip install -e '.[train,dev]'
```

Run a local tracking server in a separate terminal:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Copy the environment template and load it in the terminal that runs training:

```bash
cp config/mlflow.env.example .env
source .env
```

The application also works without a running server. If `MLFLOW_TRACKING_URI`
is absent, it creates a local SQLite tracking database at `mlflow/mlflow.db`.

## Train and record a run

```bash
energy train-pv-day-ahead --project-root . --year 2024
```

Each MLflow run records the training configuration, validation and held-out
test metrics, the feature contract, the exact week allocation and the CatBoost
model. The feature contract also states the required non-negative prediction
post-processing: use `max(prediction_w, 0)` before a forecast enters the
optimisation layer.

## Register a version

After reviewing the held-out metrics, create a version in MLflow Model Registry:

```bash
energy train-pv-day-ahead \
  --project-root . \
  --year 2024 \
  --register \
  --registered-model-name energy-pv-day-ahead
```

The command creates a model version under `energy-pv-day-ahead`; it does not
assign an alias such as `candidate` or `champion`. That promotion should remain
a separate, reviewable decision after a backtest.

## Remote MLflow server

Set `MLFLOW_TRACKING_URI` to the server address. If its administrator enables
Basic authentication, set `MLFLOW_TRACKING_USERNAME` and
`MLFLOW_TRACKING_PASSWORD`. If it uses bearer-token authentication, set
`MLFLOW_TRACKING_TOKEN`. Keep those values in the ignored `.env` file or in the
deployment secret manager, never in source code or MLflow parameters.

## MPC residual model

The PV residual corrector has its own training entry point and Model Registry
flow. It requires a frozen day-ahead prediction as an input at serving time;
see [pv_mpc_residual_training.md](pv_mpc_residual_training.md) for its
point-in-time contract and commands.
