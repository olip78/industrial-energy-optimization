# Day-ahead price experiment: price history and spatial weather

This is the first deliberately compact price-forecasting experiment for the
DE-LU day-ahead control problem. Its purpose is to produce an honest forecast
suitable for ranking expensive and cheap delivery hours in the optimiser, not
to reproduce a commercial market-forecasting system.

## Prediction contract

For every local delivery period τ in a delivery day `D`, the model predicts the
cleared day-ahead price from information available at **D-1 11:00
Europe/Berlin**. The target auction outcome is treated as available at D-1
13:00 local, so it is inaccessible to the feature builder.

The experiment keeps only normal 24-hour local delivery days. DST transition
days are excluded, as are the quarter-hourly price products from October 2025
onwards. This is intentional for the hourly V1 optimisation problem; it does
not average quarter-hourly auction products into artificial hourly prices.

The spatial ICON archive has no complete fixed 24/48-hour layers before 18
February 2024. That boundary is also applied to the price table.

## Dataset

The materialised table is:

```text
data/features/day_ahead_price_spatial/
  day_ahead_price_spatial_2024-02-18_2025-09-30.parquet
```

It has 13,392 rows, representing 558 complete delivery days. Every row keeps
`as_of_utc`, `target_available_at_utc`,
`price_history_available_at_utc`, and the selected forecast availability
timestamp for local weather plus every spatial location. The test suite checks
that all feature timestamps are at or before `as_of_utc`, while the price
target remains later.

### Feature groups

| Group | Contents | Count |
| --- | --- | ---: |
| Price and calendar | Delivery-hour calendar, German national holidays, full previous-day curve, same-hour D-1/D-2/D-7/D-14 lags, four-week same-hour mean, and shifted recent-price statistics | 49 |
| Local weather | Existing Pforzheim ICON proxy: temperature, radiation, wind at 10 m, cloud cover, selected lead | 7 |
| Spatial weather | Six selected ICON variables across ten representative locations | 60 |

`price_prev_day_h00` through `price_prev_day_h23` are cleared prices for the
whole preceding delivery-day curve. They were available before the next day's
decision cutoff and are not leakage.

## Evaluation design

The test interval is a single untouched chronological holdout: **1 January to
30 September 2025**. Training uses only eligible 2024 rows. It contains 7,104
training rows and the holdout contains 6,288 rows.

The CatBoost hyperparameters were fixed before running the holdout: RMSE loss,
500 trees, depth 6, learning rate 0.05, and `l2_leaf_reg=10`. No early stopping
or feature selection uses 2025 labels. This is a first backtest, rather than a
hyperparameter search.

The variants are:

1. `persistence_d1`: yesterday's cleared price in the same delivery hour;
2. `price_only_catboost`: calendar and price-history features;
3. `local_weather_catboost`: price features plus the single Pforzheim proxy;
4. `spatial_weather_catboost`: price features plus ten-location spatial ICON
   weather. It includes a Pforzheim-like south-west point, but uses wind at
   120 m consistently across the spatial layer.

## First chronological result

| Variant | MAE, EUR/MWh | RMSE, EUR/MWh | Mean daily Spearman | Top-quartile overlap | Bottom-quartile overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Persistence D-1 | 25.57 | 40.12 | 0.796 | 0.704 | 0.781 |
| Price-only CatBoost | 20.36 | 30.95 | 0.876 | 0.783 | 0.823 |
| Local-weather CatBoost | 19.33 | 30.06 | 0.878 | 0.782 | 0.821 |
| Spatial-weather CatBoost | **17.43** | **27.31** | **0.912** | **0.809** | **0.858** |

Spatial weather reduces MAE by 14.4% relative to the price-only model and by
31.8% relative to persistence on this fixed holdout. The local proxy improves
level accuracy modestly, while the geographic weather field gives a material
additional gain.

The most important spatial weather features in the winning model are wind at
120 m in Lower Saxony, the North Sea offshore point, and Mecklenburg. This is
consistent with the expected price effect of German wind generation; it is a
useful plausibility check, not a causal claim.

## Limits and next checks

The result is encouraging but not sufficient to call the model production
ready. The weather values come from a retrospective fixed-lead archive rather
than the complete original forecast run selected at every historical cutoff.
There is also only one final 2025 holdout and no forecasted German residual
load, wind/PV generation, fuel, or CO2 features yet.

The next modelling step is a small rolling-origin evaluation within the same
feature contract, followed by an optional audited residual-load forecast
extension. Economic evaluation should ultimately compare the optimiser's
realised cost under each price forecast, rather than select a model from MAE
alone.

## Reproduce

```bash
python scripts/collect_spatial_weather_data.py .
energy build-day-ahead-price-spatial-data --project-root .
energy run-day-ahead-price-experiment --project-root .
```

The backtest writes `metrics.csv`, `hourly_predictions.parquet`, feature
importances and the frozen configuration under:

```text
artifacts/experiments/day_ahead_price_spatial_v1/
```
