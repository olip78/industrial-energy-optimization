"""Create transparent coverage and quality summaries for the collected Energy data."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(sys.argv[1]).resolve()
RAW = PROJECT / "data" / "raw"
PROCESSED = PROJECT / "data" / "processed"
METADATA = PROJECT / "data" / "metadata"
DOCS = PROJECT / "docs"


def summary(frame: pd.DataFrame, timestamp: str) -> dict:
    dates = pd.to_datetime(frame[timestamp], utc=True, errors="coerce")
    return {
        "rows": int(len(frame)),
        "first_timestamp": str(dates.min()),
        "last_timestamp": str(dates.max()),
        "duplicate_timestamps": int(dates.duplicated().sum()),
        "missing_by_column": {key: int(value) for key, value in frame.isna().sum().items()},
    }


def utc_strings(frame: pd.DataFrame, timestamp: str) -> set[str]:
    values = pd.to_datetime(frame[timestamp], utc=True, errors="coerce")
    return set(values.dropna().dt.strftime("%Y-%m-%dT%H:%M:%SZ"))


def main() -> None:
    pv = pd.read_csv(PROCESSED / "pv_power_15min.csv.gz")
    actual = pd.read_csv(PROCESSED / "weather_actual_hourly_era5.csv.gz")
    forecast = pd.read_csv(PROCESSED / "weather_forecast_hourly_icon_lead_24_48.csv.gz")
    price = pd.read_csv(PROCESSED / "prices_day_ahead_de_lu.csv.gz")

    pv_times = pd.to_datetime(pv["source_time"], errors="coerce")
    profiles = [name for name in ("1a", "1b", "2a", "2b", "1c") if name in pv]
    daily = pv.set_index(pv_times)[profiles].resample("D").max()
    common_zero_days = daily.index[(daily[["1a", "2a", "2b"]] == 0).all(axis=1)].strftime("%Y-%m-%d").tolist()
    pv_profile_summary = {
        profile: {
            "nulls": int(pv[profile].isna().sum()),
            "min_w": float(pv[profile].min()),
            "max_w": float(pv[profile].max()),
            "all_zero_days": int((daily[profile] == 0).sum()),
            "last_positive_source_time": str(pv.loc[pv[profile] > 0, "source_time"].iloc[-1]),
        }
        for profile in profiles
    }

    actual_index = utc_strings(actual, "valid_time_utc")
    forecast_index = utc_strings(forecast, "valid_time_utc")
    price_index = utc_strings(price, "valid_time_utc")
    common_hourly = actual_index & forecast_index & price_index
    monthly = pd.DataFrame({
        "month_utc": pd.date_range("2024-01-01", "2025-12-01", freq="MS", tz="UTC").strftime("%Y-%m"),
    })
    for name, frame, column in [
        ("weather_actual_hours", actual, "valid_time_utc"),
        ("weather_forecast_hours", forecast, "valid_time_utc"),
        ("day_ahead_price_hours", price, "valid_time_utc"),
    ]:
        count = pd.to_datetime(frame[column], utc=True).dt.strftime("%Y-%m").value_counts()
        monthly[name] = monthly["month_utc"].map(count).fillna(0).astype(int)
    pv_month_counts = pv_times.dt.strftime("%Y-%m").value_counts()
    monthly["pv_15min_intervals_source_clock"] = monthly["month_utc"].map(pv_month_counts).fillna(0).astype(int)
    monthly.to_csv(PROCESSED / "coverage_by_month.csv", index=False)

    price_steps = pd.to_datetime(price["valid_time_utc"], utc=True).sort_values().diff().dropna().dt.total_seconds().value_counts()
    report = {
        "pv": {
            "rows": int(len(pv)),
            "first_source_time": str(pv_times.min()),
            "last_source_time": str(pv_times.max()),
            "source_timezone": "unknown; source timestamps must not be joined to UTC weather or prices without an explicit assumption",
            "profiles": pv_profile_summary,
            "common_all_zero_days_for_1a_2a_2b": common_zero_days,
        },
        "actual_weather_era5": summary(actual, "valid_time_utc"),
        "forecast_weather_icon_previous_runs": summary(forecast, "valid_time_utc"),
        "day_ahead_prices_de_lu": {
            **summary(price, "valid_time_utc"),
            "interval_step_seconds": {str(int(step)): int(count) for step, count in price_steps.items()},
        },
        "common_hourly_utc_coverage": {"hours": int(len(common_hourly)), "first": min(common_hourly), "last": max(common_hourly)},
    }
    (METADATA / "data_quality_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    (DOCS / "data_quality_report.md").write_text(f"""# Data quality and coverage

The collected core data covers 1 January 2024 through 31 December 2025.

## Confirmed UTC intersection

Actual ERA5 weather, archived ICON forecasts and DE-LU day-ahead prices share **{len(common_hourly):,} hourly UTC timestamps**, from `{min(common_hourly)}` to `{max(common_hourly)}`. The corresponding monthly counts are in `data/processed/coverage_by_month.csv`.

## PV measurements

MPVBench contributes {len(pv):,} records at 15-minute resolution for five measured PV profiles. `1a` is the proposed working profile. It has {pv_profile_summary['1a']['all_zero_days']} fully zero days and a maximum measured power of {pv_profile_summary['1a']['max_w']:.2f} W.

The source timestamps have **no timezone or UTC offset**. They must be retained as `source_time` until the source convention is confirmed. Consequently, the project does not yet claim exact interval-by-interval joins between PV output and UTC weather/prices.

Three dates are fully zero simultaneously for profiles 1a, 2a and 2b: {', '.join(common_zero_days)}. Treat them as likely common collection/processing outages and exclude them from PV model evaluation by a quality flag. Profile 1b has {pv_profile_summary['1b']['all_zero_days']} all-zero days and profile 1c has {pv_profile_summary['1c']['all_zero_days']}; neither is a primary evaluation profile.

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
""")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
