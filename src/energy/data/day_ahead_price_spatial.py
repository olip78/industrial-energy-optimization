"""Leakage-safe day-ahead price features with spatial ICON forecasts.

The builder treats one calendar day as the forecasting unit.  Every feature is
known at the declared D-1 11:00 Europe/Berlin decision time, and price history
is represented by a complete previous-day curve plus selected same-hour lags.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

LOCAL_TZ = "Europe/Berlin"
UTC = "UTC"
DEFAULT_START = date(2024, 2, 18)
DEFAULT_END = date(2025, 9, 30)
SPATIAL_VARIABLES = (
    "temperature_2m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "wind_speed_120m",
    "cloud_cover",
)
LOCAL_VARIABLES = (
    "temperature_2m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "wind_speed_10m",
    "cloud_cover",
)


@dataclass(frozen=True)
class DayAheadPriceSpatialBuildResult:
    dataset_path: Path
    manifest_path: Path
    rows: int
    feature_groups: dict[str, list[str]]

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_path": str(self.dataset_path),
            "manifest_path": str(self.manifest_path),
            "rows": self.rows,
            "feature_groups": self.feature_groups,
        }


def _as_utc(value: pd.Timestamp) -> pd.Timestamp:
    if value.tzinfo is None:
        return value.tz_localize(UTC)
    return value.tz_convert(UTC)


def _easter_sunday(year: int) -> date:
    """Gregorian computus; sufficient for the German national holiday calendar."""

    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day_of_month = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day_of_month)


def _german_national_holidays(year: int) -> set[date]:
    easter = _easter_sunday(year)
    return {
        date(year, 1, 1),
        easter - timedelta(days=2),
        easter + timedelta(days=1),
        date(year, 5, 1),
        easter + timedelta(days=39),
        easter + timedelta(days=50),
        date(year, 10, 3),
        date(year, 12, 25),
        date(year, 12, 26),
    }


def _is_holiday(day: date) -> bool:
    return day in _german_national_holidays(day.year)


def _decision_time(day: date) -> pd.Timestamp:
    return pd.Timestamp(day, tz=LOCAL_TZ) - pd.Timedelta(days=1) + pd.Timedelta(hours=11)


def _target_price_available_at(day: date) -> pd.Timestamp:
    """Project convention: the auction outcome is available at D-1 13:00 local."""

    return pd.Timestamp(day, tz=LOCAL_TZ) - pd.Timedelta(days=1) + pd.Timedelta(hours=13)


def _price_history_available_at(day: date) -> pd.Timestamp:
    """D-1 delivery curve was cleared at D-2 13:00 local under the convention."""

    return pd.Timestamp(day, tz=LOCAL_TZ) - pd.Timedelta(days=2) + pd.Timedelta(hours=13)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_regular_price_days(processed: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices = pd.read_csv(processed / "prices_day_ahead_de_lu.csv.gz")
    prices["valid_time_utc"] = pd.to_datetime(prices["valid_time_utc"], utc=True)
    prices["day_ahead_price_eur_per_mwh"] = pd.to_numeric(
        prices["day_ahead_price_eur_per_mwh"], errors="coerce"
    )
    prices = prices.dropna(subset=["valid_time_utc", "day_ahead_price_eur_per_mwh"])
    local = prices["valid_time_utc"].dt.tz_convert(LOCAL_TZ)
    prices["delivery_date_local"] = local.dt.date
    prices["hour_local"] = local.dt.hour.astype("int8")
    prices = prices.sort_values("valid_time_utc").drop_duplicates("valid_time_utc")

    daily_shape = prices.groupby("delivery_date_local").agg(
        periods=("hour_local", "size"), unique_hours=("hour_local", "nunique")
    )
    regular_days = daily_shape.index[
        (daily_shape["periods"] == 24) & (daily_shape["unique_hours"] == 24)
    ]
    regular = prices.loc[prices["delivery_date_local"].isin(regular_days)].copy()
    curves = regular.pivot(
        index="delivery_date_local",
        columns="hour_local",
        values="day_ahead_price_eur_per_mwh",
    ).reindex(columns=range(24)).sort_index()
    times = regular.pivot(
        index="delivery_date_local", columns="hour_local", values="valid_time_utc"
    ).reindex(columns=range(24)).sort_index()
    return regular, curves, times


def _price_calendar_row(day: date, hour: int) -> dict[str, int | float]:
    day_of_week = day.weekday()
    day_of_year = day.timetuple().tm_yday
    return {
        "hour_local": hour,
        "day_of_week": day_of_week,
        "month": day.month,
        "day_of_year": day_of_year,
        "is_weekend": int(day_of_week >= 5),
        "is_holiday_de": int(_is_holiday(day)),
        "is_pre_holiday_de": int(_is_holiday(day + timedelta(days=1))),
        "is_post_holiday_de": int(_is_holiday(day - timedelta(days=1))),
        "hour_sin": math.sin(2.0 * math.pi * hour / 24.0),
        "hour_cos": math.cos(2.0 * math.pi * hour / 24.0),
        "day_of_year_sin": math.sin(2.0 * math.pi * day_of_year / 366.0),
        "day_of_year_cos": math.cos(2.0 * math.pi * day_of_year / 366.0),
    }


def _build_price_rows(
    curves: pd.DataFrame,
    times: pd.DataFrame,
    start: date,
    end: date,
) -> tuple[pd.DataFrame, list[str]]:
    available_days = set(curves.index)
    records: list[dict[str, object]] = []
    for delivery_day in sorted(day for day in available_days if start <= day <= end):
        history_days = {delivery_day - timedelta(days=lag) for lag in range(1, 8)}
        history_days.update(delivery_day - timedelta(days=lag) for lag in (14, 21, 28))
        if not history_days.issubset(available_days):
            continue
        previous_curve = curves.loc[delivery_day - timedelta(days=1)]
        target_curve = curves.loc[delivery_day]
        weekly_values = np.concatenate(
            [curves.loc[delivery_day - timedelta(days=lag)].to_numpy() for lag in range(1, 8)]
        )
        previous_stats = {
            "price_prev_day_mean": float(previous_curve.mean()),
            "price_prev_day_std": float(previous_curve.std(ddof=0)),
            "price_prev_day_min": float(previous_curve.min()),
            "price_prev_day_max": float(previous_curve.max()),
            "price_prev_day_range": float(previous_curve.max() - previous_curve.min()),
            "price_prev_day_negative_share": float((previous_curve < 0).mean()),
            "price_recent_7d_mean": float(weekly_values.mean()),
            "price_recent_7d_std": float(weekly_values.std(ddof=0)),
        }
        as_of = _as_utc(_decision_time(delivery_day))
        target_available = _as_utc(_target_price_available_at(delivery_day))
        history_available = _as_utc(_price_history_available_at(delivery_day))
        for hour in range(24):
            row: dict[str, object] = {
                "as_of_utc": as_of,
                "valid_time_utc": _as_utc(times.loc[delivery_day, hour]),
                "delivery_date_local": delivery_day.isoformat(),
                "day_ahead_price_eur_per_mwh": float(target_curve.loc[hour]),
                "target_available_at_utc": target_available,
                "price_history_available_at_utc": history_available,
                **_price_calendar_row(delivery_day, hour),
                **previous_stats,
            }
            for previous_hour in range(24):
                row[f"price_prev_day_h{previous_hour:02d}"] = float(previous_curve.loc[previous_hour])
            row["price_lag_d1_same_hour"] = float(previous_curve.loc[hour])
            for lag in (2, 7, 14):
                row[f"price_lag_d{lag}_same_hour"] = float(
                    curves.loc[delivery_day - timedelta(days=lag), hour]
                )
            row["price_same_hour_4w_mean"] = float(
                np.mean(
                    [
                        curves.loc[delivery_day - timedelta(days=lag), hour]
                        for lag in (7, 14, 21, 28)
                    ]
                )
            )
            records.append(row)

    frame = pd.DataFrame(records).sort_values("valid_time_utc").reset_index(drop=True)
    calendar_features = [
        "hour_local",
        "day_of_week",
        "month",
        "day_of_year",
        "is_weekend",
        "is_holiday_de",
        "is_pre_holiday_de",
        "is_post_holiday_de",
        "hour_sin",
        "hour_cos",
        "day_of_year_sin",
        "day_of_year_cos",
    ]
    price_features = [
        *calendar_features,
        *(f"price_prev_day_h{hour:02d}" for hour in range(24)),
        "price_lag_d1_same_hour",
        "price_lag_d2_same_hour",
        "price_lag_d7_same_hour",
        "price_lag_d14_same_hour",
        "price_same_hour_4w_mean",
        "price_prev_day_mean",
        "price_prev_day_std",
        "price_prev_day_min",
        "price_prev_day_max",
        "price_prev_day_range",
        "price_prev_day_negative_share",
        "price_recent_7d_mean",
        "price_recent_7d_std",
    ]
    return frame, price_features


def _select_local_weather(frame: pd.DataFrame, processed: Path) -> tuple[pd.DataFrame, list[str]]:
    weather = pd.read_csv(processed / "weather_forecast_hourly_icon_lead_24_48.csv.gz")
    weather["valid_time_utc"] = pd.to_datetime(weather["valid_time_utc"], utc=True)
    needed = [
        f"{variable}_previous_day{lead_day}"
        for variable in LOCAL_VARIABLES
        for lead_day in (1, 2)
    ]
    weather = weather[["valid_time_utc", *needed]].copy()
    merged = frame.merge(weather, on="valid_time_utc", how="left", validate="one_to_one")
    available24 = merged["valid_time_utc"] - pd.Timedelta(hours=24) <= merged["as_of_utc"]
    available48 = merged["valid_time_utc"] - pd.Timedelta(hours=48) <= merged["as_of_utc"]
    features: list[str] = []
    for variable in LOCAL_VARIABLES:
        feature = f"local_weather_{variable}"
        first = merged[f"{variable}_previous_day1"]
        second = merged[f"{variable}_previous_day2"]
        merged[feature] = first.where(available24 & first.notna(), second.where(available48))
        features.append(feature)
    merged["local_weather_forecast_lead_hours"] = np.where(available24, 24, 48).astype("int8")
    merged["local_weather_available_at_utc"] = (
        merged["valid_time_utc"] - pd.Timedelta(hours=24)
    ).where(available24, merged["valid_time_utc"] - pd.Timedelta(hours=48))
    features.append("local_weather_forecast_lead_hours")
    if merged[features].isna().any().any():
        raise ValueError("Local weather is incomplete after the selected fixed-lead availability rule")
    if not (merged["local_weather_available_at_utc"] <= merged["as_of_utc"]).all():
        raise AssertionError("Local weather is newer than the decision cutoff")
    return merged.drop(columns=needed), features


def _select_spatial_weather(
    frame: pd.DataFrame,
    processed: Path,
    locations: list[dict[str, object]],
) -> tuple[pd.DataFrame, list[str]]:
    weather = pd.read_csv(processed / "weather_forecast_hourly_icon_spatial_lead_24_48.csv.gz")
    weather["valid_time_utc"] = pd.to_datetime(weather["valid_time_utc"], utc=True)
    result = frame.copy()
    features: list[str] = []
    for location in locations:
        location_id = str(location["location_id"])
        values24 = [f"weather_fcst24_{variable}" for variable in SPATIAL_VARIABLES]
        values48 = [f"weather_fcst48_{variable}" for variable in SPATIAL_VARIABLES]
        subset = weather.loc[
            weather["location_id"] == location_id,
            [
                "valid_time_utc",
                "weather_fcst24_available_at_utc",
                "weather_fcst48_available_at_utc",
                *values24,
                *values48,
            ],
        ].copy()
        if len(subset) != weather["valid_time_utc"].nunique():
            raise ValueError(f"Spatial weather has an incomplete or duplicate time grid for {location_id}")
        suffix = f"__{location_id}"
        subset = subset.rename(columns={column: f"{column}{suffix}" for column in subset if column != "valid_time_utc"})
        result = result.merge(subset, on="valid_time_utc", how="left", validate="one_to_one")
        complete24 = result[[f"{column}{suffix}" for column in values24]].notna().all(axis=1)
        complete48 = result[[f"{column}{suffix}" for column in values48]].notna().all(axis=1)
        available24 = complete24 & (
            pd.to_datetime(result[f"weather_fcst24_available_at_utc{suffix}"], utc=True)
            <= result["as_of_utc"]
        )
        available48 = complete48 & (
            pd.to_datetime(result[f"weather_fcst48_available_at_utc{suffix}"], utc=True)
            <= result["as_of_utc"]
        )
        for variable in SPATIAL_VARIABLES:
            feature = f"spatial_{location_id}_{variable}"
            result[feature] = result[f"weather_fcst24_{variable}{suffix}"].where(
                available24,
                result[f"weather_fcst48_{variable}{suffix}"].where(available48),
            )
            features.append(feature)
        result[f"spatial_{location_id}_forecast_lead_hours"] = np.where(
            available24, 24, 48
        ).astype("int8")
        result[f"spatial_{location_id}_available_at_utc"] = (
            result["valid_time_utc"] - pd.Timedelta(hours=24)
        ).where(available24, result["valid_time_utc"] - pd.Timedelta(hours=48))
        if not (result[f"spatial_{location_id}_available_at_utc"] <= result["as_of_utc"]).all():
            raise AssertionError(f"Spatial weather is newer than the decision cutoff: {location_id}")
        temporary = [column for column in result if column.endswith(suffix)]
        result = result.drop(columns=temporary)
    if result[features].isna().any().any():
        raise ValueError("Spatial weather is incomplete after the selected fixed-lead availability rule")
    return result, features


def build_day_ahead_price_spatial_dataset(
    project_root: str | Path,
    *,
    start: date = DEFAULT_START,
    end: date = DEFAULT_END,
) -> DayAheadPriceSpatialBuildResult:
    """Materialize the dedicated feature table used by price-model experiments."""

    if end < start:
        raise ValueError("end must not be before start")
    root = Path(project_root).resolve()
    processed = root / "data" / "processed"
    config = json.loads((root / "config" / "spatial_weather_locations_v1.json").read_text())
    locations = config["locations"]
    regular_prices, curves, times = _load_regular_price_days(processed)
    frame, price_features = _build_price_rows(curves, times, start, end)
    frame, local_features = _select_local_weather(frame, processed)
    frame, spatial_features = _select_spatial_weather(frame, processed, locations)

    if not (frame["price_history_available_at_utc"] <= frame["as_of_utc"]).all():
        raise AssertionError("A price-history feature is newer than its decision cutoff")
    if not (frame["target_available_at_utc"] > frame["as_of_utc"]).all():
        raise AssertionError("Price target must not be available at its decision cutoff")
    weather_availability_columns = [
        "local_weather_available_at_utc",
        *(f"spatial_{location['location_id']}_available_at_utc" for location in locations),
    ]
    if not (frame[weather_availability_columns].le(frame["as_of_utc"], axis=0)).all().all():
        raise AssertionError("A selected weather feature is newer than the decision cutoff")
    if frame.duplicated("valid_time_utc").any():
        raise AssertionError("Target table has duplicate delivery periods")

    frame["feature_version"] = "da-price-spatial-v1"
    frame["data_snapshot_id"] = "price-2024-2025-spatial-weather-v1"
    output_dir = root / "data" / "features" / "day_ahead_price_spatial"
    metadata_dir = root / "data" / "metadata"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / f"day_ahead_price_spatial_{start.isoformat()}_{end.isoformat()}.parquet"
    frame.to_parquet(dataset_path, index=False)

    feature_groups = {
        "price_only": price_features,
        "local_weather": local_features,
        "spatial_weather": spatial_features,
    }
    manifest = {
        "feature_version": "da-price-spatial-v1",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target": "day_ahead_price_eur_per_mwh",
        "target_window_local": {"start": start.isoformat(), "end": end.isoformat()},
        "decision_time": "D-1 11:00 Europe/Berlin",
        "price_target_availability_assumption": "D-1 13:00 Europe/Berlin",
        "regular_day_policy": "Only local delivery days with exactly one period for every hour 00..23 are retained; DST transition days and post-2025-09 15-minute products are excluded.",
        "rows": int(len(frame)),
        "delivery_days": int(frame["delivery_date_local"].nunique()),
        "first_delivery_day": str(frame["delivery_date_local"].min()),
        "last_delivery_day": str(frame["delivery_date_local"].max()),
        "feature_groups": feature_groups,
        "weather_availability_audit_columns": weather_availability_columns,
        "spatial_weather_locations": config,
        "source_files": {
            "prices": "data/processed/prices_day_ahead_de_lu.csv.gz",
            "local_weather": "data/processed/weather_forecast_hourly_icon_lead_24_48.csv.gz",
            "spatial_weather": "data/processed/weather_forecast_hourly_icon_spatial_lead_24_48.csv.gz",
        },
        "regular_price_days_available": int(len(regular_prices["delivery_date_local"].unique())),
        "dataset_sha256": _sha256(dataset_path),
    }
    manifest_path = metadata_dir / "day_ahead_price_spatial_v1_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return DayAheadPriceSpatialBuildResult(dataset_path, manifest_path, len(frame), feature_groups)
