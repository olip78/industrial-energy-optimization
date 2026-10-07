# Spatial forecast-weather dataset

`spatial-weather-v1` is a separate data layer for DE-LU day-ahead price
experiments. It extends the existing single Pforzheim PV-weather proxy with ten
representative weather-regime coordinates: North Sea offshore, northern and
north-eastern wind regions, western and central Germany, eastern Germany,
Saxony, Bavaria, and the existing south-west proxy.

The coordinates are modelling locations, not claims about weather stations,
renewable assets, or population-weighted regions. Their roles are defined in
[`config/spatial_weather_locations_v1.json`](../config/spatial_weather_locations_v1.json).

## Source and point-in-time contract

The collector uses Open-Meteo's `previous-runs` archive with the DWD ICON
seamless model. Each row has fixed 24- and 48-hour-lead forecast values for:

- temperature at 2 m;
- shortwave, direct, and diffuse radiation;
- wind speed at 120 m;
- total cloud cover.

The processed table is:

```text
data/processed/weather_forecast_hourly_icon_spatial_lead_24_48.csv.gz
```

It contains `location_id`, requested/model coordinates, `valid_time_utc`,
`weather_fcst24_*`, `weather_fcst48_*`, and a separate inferred availability
timestamp for each forecast layer. A downstream model may use a value only
when its `*_available_at_utc <= as_of_utc`.

The price training table keeps the selected `local_weather_available_at_utc` and one `spatial_<location_id>_available_at_utc` audit field per location. These timestamp fields are not model features; they make the point-in-time check reproducible from the materialised Parquet.

The dataset is forecast data, not ERA5 or observed weather. It is still a
retrospective fixed-lead archive, rather than an exact reconstruction of every
historical ICON run. That limitation is recorded in the manifest and must stay
visible in the price-model documentation.

The raw collection window begins on 1 January 2024, but the upstream ICON
previous-runs archive does not populate both fixed-lead layers until mid-February
2024. Under this project’s `D-1 11:00 Europe/Berlin` convention, the first
complete spatial-weather delivery day is **18 February 2024**. Future price
experiments must start from that day or explicitly handle the missing early rows.

## Reproduce

```bash
python scripts/collect_spatial_weather_data.py .
```

The collection is resumable per month. It saves original API responses under
`data/raw/weather_forecasts_icon_spatial_previous_runs_monthly/`, a quality
summary and reproducibility metadata under `data/metadata/`, and the processed
CSV above.

## Planned price features

The first price experiment should compare a price-only model, the existing
single-point weather proxy, and this spatial data layer. Start with raw
per-location features. Do not fit PCA on the complete data before temporal
validation; it may be introduced inside each training fold only if the raw
features prove unstable.
