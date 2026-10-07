# Spatial IFS forecasts for the intraday-price model

This data layer extends the intraday DE-LU price work with forecasts available
after the day-ahead auction. It is intentionally separate from the local
Pforzheim IFS archive used by the PV MPC model.

## Geography and source

The archive uses the ten representative weather-regime coordinates already
defined in `config/spatial_weather_locations_v1.json` for the day-ahead price
model: North Sea, Schleswig-Holstein, Lower Saxony, Mecklenburg, Ruhr,
Central Germany, Brandenburg, Saxony, Bavaria and Pforzheim. They are model
coordinates, not claims about individual power-plant locations.

Each ECMWF IFS HRES initialisation is downloaded once with all ten coordinates
in one request. The table contains temperature, global/direct/diffuse
radiation, wind speed at 120 metres and cloud cover.

## Point-in-time contract

For an intraday-price decision at `as_of_utc`, a forecast value may be used
only when:

```text
run_init_utc + 6 hours <= as_of_utc < valid_time_utc
```

The six-hour delay is deliberately conservative. The raw table preserves both
the model initialisation and the assumed publication time, so a later feature
builder can audit every input used by the one-hour spread forecast.

## Commands

```bash
# The archive has daily 00 UTC IFS runs in 2024.
energy collect-ecmwf-ifs-spatial-weather \
  --project-root . \
  --start 2024-03-14 --end 2024-12-31 \
  --run-hours 0

# Keep all available updates for forward operation in 2025.
energy collect-ecmwf-ifs-spatial-weather \
  --project-root . \
  --start 2025-01-01 --end 2025-09-30 \
  --run-hours 0 6 12 18

energy materialize-ecmwf-ifs-spatial-weather --project-root .
```

Raw JSON is stored under
`data/raw/weather_forecasts_ecmwf_ifs_spatial_single_runs/runs/`. The
normalised long table is
`data/processed/weather_forecast_hourly_ecmwf_ifs_spatial_single_runs.parquet`;
the manifest is `data/metadata/ecmwf_ifs_spatial_intraday_weather_manifest.json`.

## Completed archive coverage

The resumable collection was completed through 30 September 2025 and then
materialised from the entire 2024--2025 raw cache:

- 1,385 scheduled run records are cached;
- 1,377 runs are available and 8 are explicit provider-side gaps;
- every available run contains 48 forecast hours for all 10 locations;
- the processed table contains 660,960 rows with no duplicate
  `(location_id, run_init_utc, valid_time_utc)` keys;
- the first available run is 14 March 2024 at 00:00 UTC and the last is
  30 September 2025 at 18:00 UTC.

Open-Meteo reports the following runs as `modelRunUnavailable`: 4 August 2025
at 18:00 UTC; 5 August at 00:00 and 06:00; 6 August at 00:00; 7 August at
18:00; 8 August at 00:00 and 12:00; and 9 August at 00:00. These absences are
persisted in the raw cache and manifest so later resumes do not retry them or
silently impute a forecast that did not exist in the provider archive.

The downloader accepts both Open-Meteo representations of this condition:
the older HTTP 400 response and the current plain-text `modelRunUnavailable`
message returned with HTTP 200. All other malformed or transient responses
still fail or use bounded retry rather than being mislabeled as archive gaps.

## 2024 first experiment

The intraday-price model predicts the next-hour spread
`intraday_price - day_ahead_price`. Its future weather inputs describe the
revision of market-wide expectations, rather than weather at the PV site.

`run-intraday-price-spatial-weather-experiment` was added as a reproducible
2024 prototype. It trains through 30 September and uses October--December as
a strictly later chronological validation period. All variants use exactly the
same 6,655 next-hour rows; only their input features differ.

For every decision/target pair, it derives, for each of the six weather
variables:

- the IFS forecast published by 11:00 Europe/Berlin on the day before delivery;
- the latest IFS forecast published by `as_of_utc`; and
- their same-provider revision, `latest IFS - day-ahead IFS`.

Each view is represented by the mean and standard deviation across the ten
locations, yielding 36 weather features: 12 day-ahead levels, 12 latest levels
and 12 revisions. The model is restricted to `lead_hours == 1`, which is the
only horizon at which the current MPC policy applies the intraday correction.
The experiment also evaluates a compact 12-feature challenger consisting of
the revisions only. This avoids asking CatBoost to infer a difference from two
highly correlated sets of absolute weather levels.

Run it after materialising the archive:

```bash
energy run-intraday-price-spatial-weather-experiment --project-root .
```

The command writes metrics, per-row predictions, feature importances and the
complete point-in-time feature contract to
`artifacts/experiments/intraday_price_spatial_ifs_2024/`.

The first result should be treated as a model-selection result, not as a
deployment claim: it has no untouched 2025 weather-complete holdout. On 2024
Q4, the full spatial-weather challenger was essentially tied with the
price-only correction (MAE 12.385 vs 12.380 EUR/MWh). The compact
revision-only challenger was similarly close but did not improve it
(12.398/47.072 MAE/RMSE versus 12.380/47.007). The project therefore retains
the price-only model as the V1 production candidate. The artifacts preserve
both challengers for future feature work and a 2025 out-of-sample comparison.

## Complementary randomized whole-week cross-validation

The chronological Q4 split is the deployment-relevant protocol. To check
whether its result was particular to that quarter, the project also provides
season-balanced randomized three-fold cross-validation by complete local
calendar weeks:

```bash
energy run-intraday-price-spatial-weather-week-cv --project-root . --n-splits 3
```

Every row still uses only weather forecasts published before its own decision
time, and no calendar week appears in both the train and test side of a fold.
The method is useful for comparing feature sets across the available seasons.
It is not a replacement for chronological validation because a fold's fitted
parameters can use weeks that occur after one of its test weeks.

The first three-fold run produced pooled out-of-fold results on all 6,655
rows: price-only MAE/RMSE was 12.460/55.238 EUR/MWh; the spatial-IFS
challenger was 12.525/55.420; the compact revision-only version was
12.476/55.313. Thus the extra weather was slightly worse under both evaluation
protocols. The result does not show a material deterioration, but it gives no
evidence to promote either weather version.
