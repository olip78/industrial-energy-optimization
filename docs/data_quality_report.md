# Data quality and coverage

The collected core data covers 1 January 2024 through 31 December 2025.

## Confirmed UTC intersection

Actual ERA5 weather, archived ICON forecasts and DE-LU day-ahead prices share **17,543 hourly UTC timestamps**, from `2024-01-01T00:00:00Z` to `2025-12-31T22:00:00Z`. The corresponding monthly counts are in `data/processed/coverage_by_month.csv`.

## PV measurements

MPVBench contributes 70,176 records at 15-minute resolution for five measured PV profiles. `1a` is the proposed working profile. It has 3 fully zero days and a maximum measured power of 533.61 W.

The source timestamps have **no timezone or UTC offset**. They must be retained as `source_time` until the source convention is confirmed. Consequently, the project does not yet claim exact interval-by-interval joins between PV output and UTC weather/prices.

Three dates are fully zero simultaneously for profiles 1a, 2a and 2b: 2024-05-10, 2024-07-07, 2025-07-23. Treat them as likely common collection/processing outages and exclude them from PV model evaluation by a quality flag. Profile 1b has 147 all-zero days and profile 1c has 313; neither is a primary evaluation profile.

## Weather

`weather_actual_hourly_era5.csv.gz` is factual weather **reanalysis**, not measurements next to the panels. It provides solar radiation, direct and diffuse radiation, temperature, wind and cloud cover at an approximate Pforzheim point (48.89, 8.70).

`weather_forecast_hourly_icon_lead_24_48.csv.gz` provides the same variables as historical forecasts at fixed 24- and 48-hour lead times. The row time is the forecast target time. Each forecast field name ends in `previous_day1` or `previous_day2`. This supports an honest fixed-lead day-ahead experiment; it is not a complete single forecast run available at one selected decision time.

## Prices

`prices_day_ahead_de_lu.csv.gz` contains historical DE-LU day-ahead auction outcomes in EUR/MWh. The resolution is hourly before 1 October 2025 and 15-minute afterwards. They are outcomes after auction clearing; they must not be used as a pre-auction price forecast. They can be used as realised settlement prices in a simulated day-ahead strategy.

## Required modelling decisions still open

1. Confirm or deliberately assume the MPVBench timezone and interval convention before joining PV and weather.
2. Choose the exact market decision time and map it to the 24-hour or 48-hour forecast lead layer.
3. Mark the PV scale-up from a 0.6–0.8 kW balcony system to an industrial scenario as synthetic.
4. Define battery and demand parameters separately; no battery telemetry is present in this dataset.
