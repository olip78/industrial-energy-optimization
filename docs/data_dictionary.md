# Data dictionary

## `data/processed/pv_power_15min.csv.gz`

| Column | Meaning |
|---|---|
| `source_time` | PV source timestamp; timezone currently unknown |
| `1a`, `1b`, `1c`, `2a`, `2b` | Measured active PV power in W |
| `source_timezone` | Always `unknown` in this first collection |
| `interval_minutes` | 15 |

Use profile `1a` for the first experiment. Exclude 2024-05-10, 2024-07-07 and 2025-07-23 from PV forecast evaluation until their common zero outputs are explained.

## `data/processed/pv_power_15min_assumed_europe_berlin.csv.gz`

This is a separate join-ready copy; the original PV table remains unchanged.

| Additional column | Meaning |
|---|---|
| `valid_time_utc` | UTC conversion under the provisional `Europe/Berlin` assumption |
| `timezone_assumption` | The chosen working convention |
| `timezone_conversion_status` | `converted` or `unresolved_dst_transition` |
| `pv_quality_status` | `usable` or `suspected_common_source_outage` |

Sixteen rows around daylight-saving transitions have blank `valid_time_utc`; do not fill or reuse them in exact time joins.

## `data/processed/weather_actual_hourly_era5.csv.gz`

`valid_time_utc` is the hourly UTC timestamp. Weather values are ERA5 reanalysis for the approximate Pforzheim grid cell.

| Column | Unit |
|---|---|
| `temperature_2m` | °C |
| `shortwave_radiation`, `direct_radiation`, `diffuse_radiation` | W/m² |
| `wind_speed_10m` | km/h |
| `cloud_cover` | % |

## `data/processed/weather_forecast_hourly_icon_lead_24_48.csv.gz`

`valid_time_utc` is the weather target timestamp. Every feature occurs twice: suffix `previous_day1` is a forecast 24 hours before target; suffix `previous_day2` is 48 hours before target. Units match the actual-weather table.

These are fixed-lead historical forecasts. They can answer, for example, “how much PV would we have forecast for tomorrow’s 13:00?” They are not a full forecast run received at one particular intraday decision timestamp.

## `data/processed/prices_day_ahead_de_lu.csv.gz`

| Column | Meaning |
|---|---|
| `valid_time_utc` | Start of the delivery interval in UTC |
| `day_ahead_price_eur_per_mwh` | Cleared day-ahead price in EUR/MWh |

The series is hourly through September 2025 and 15-minute after 1 October 2025. Do not aggregate or forward-fill it without recording which market period the experiment uses.

## Time convention

Internal weather and price timestamps are UTC. Market delivery days and production deadlines should later be defined in `Europe/Berlin`. PV is intentionally not aligned yet because the source time zone is undocumented.
