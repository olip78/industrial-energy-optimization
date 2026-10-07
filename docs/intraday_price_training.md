# Intraday-price training application and MLflow

`energy train-intraday-price` packages the fourth point forecaster for the MPC
loop. It trains a CatBoost model on the realised intraday-minus-day-ahead
spread using the separate V2 feature table. The raw registered model outputs a
spread in EUR/MWh; it does not recreate the day-ahead curve.

At an MPC decision time `tau`, the decision layer combines that raw output with
the known auction-cleared day-ahead price only for the next delivery hour:

```text
lead_hours = 1:  intraday_hat = day_ahead_price + 0.70 * spread_hat
lead_hours >= 2: intraday_hat = day_ahead_price
```

Thus every hourly MPC run receives one newly corrected price, reoptimises the
whole remaining battery plan, and retains the stable day-ahead curve for the
distant horizon.

## Training and validation contract

The default run uses the enriched V2 table
`data/features/intraday_price/intraday_price_mpc_v2_2024_2025.parquet` and
trains on 2024. January–September form the development period; Q4 is the
chronological validation period. It records both next-hour and full remaining
horizon MAE/RMSE. After validation, the model is refit on all eligible 2024
rows.

The configuration is fixed from the feature experiment:

- CatBoost Huber loss with `delta=10`, depth 4, 500 trees, learning rate 0.05
  and `l2_leaf_reg=50`;
- correction weight 0.70, selected by Q4 next-hour MAE;
- all time-dependent input features have explicit availability checks;
- target is a realised hourly delivery-period average, so no future target
  value can be a feature.

## Train locally

Install and configure MLflow as described in
[`pv_training_and_mlflow.md`](pv_training_and_mlflow.md), then run:

```bash
energy train-intraday-price --project-root . --year 2024
```

The run logs parameters, validation metrics, feature and split contracts,
feature importance, the CatBoost model and an input/output signature. If
`MLFLOW_TRACKING_URI` is not configured, it uses the project-local SQLite
store at `mlflow/mlflow.db`.

## Register a version

```bash
energy train-intraday-price \
  --project-root . \
  --year 2024 \
  --register \
  --registered-model-name energy-intraday-price
```

The command creates an MLflow Model Registry version but does not assign a
`champion` alias. Promotion belongs to the economic MPC backtest, where this
model will be compared with pure day-ahead and the unshrunk one-hour correction.
