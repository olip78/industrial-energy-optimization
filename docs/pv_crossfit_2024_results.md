# PV nested cross-fitting: 2024 baseline

## Setup

- 8,277 usable hourly PV observations;
- five outer and five inner `SeasonalWeekKFold` folds, grouped by full local weeks;
- all folds contain observations from winter, spring, summer and autumn;
- `HistGradientBoostingRegressor` with 100 iterations is used for both the Oracle and forecast models;
- the Oracle learns PV power from actual weather, then produces a non-negative proxy from forecast weather;
- the deployable forecast model is trained on forecast weather, calendar features and the inner out-of-fold Oracle proxy.

The comparison also includes an identical forecast model without the Oracle proxy.

## Pooled out-of-fold results

| Model | MAE, W | RMSE, W |
|---|---:|---:|
| Oracle with actual weather | 18.51 | 43.79 |
| Forecast-weather baseline | 18.48 | 43.31 |
| Forecast weather + Oracle proxy | 18.87 | 44.10 |

The Oracle proxy did not improve this first model: compared with the forecast-weather baseline, MAE rose by 0.39 W and RMSE by 0.79 W. The proxy is therefore not a champion feature in the baseline configuration.

This does not invalidate the cross-fitting design. It suggests that the forecast model can already learn most of the useful transformation from weather to PV power, while the Oracle proxy adds a correlated and slightly noisy duplicate of those weather features.

## Interpretation and next checks

All-hour metrics include nighttime zero generation, so they should be supplemented with active-generation or daylight-only metrics before comparing more advanced models. Useful next experiments are:

1. compare the proxy feature with a residual model, where the forecast model learns the correction to the Oracle proxy;
2. tune the Oracle and forecast models independently rather than using one baseline configuration for both;
3. add solar-geometry features and evaluate errors separately for low, medium and high production;
4. keep the same nested folds while trying a different model family.

The row-level predictions and per-fold values are stored in `artifacts/pv_crossfit/2024/`.
