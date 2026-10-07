"""Collect an auditable multi-location archive of ICON forecast weather.

The table is intentionally separate from the single Pforzheim PV weather source.
It holds fixed 24- and 48-hour-lead forecast values for ten representative
DE-LU weather-regime locations.  It does not contain realised weather.

Open-Meteo's previous-runs archive exposes fixed lead-time forecasts.  For the
project's day-ahead feature contract, their inferred availability times are
``valid_time - lead_hours``.  A later feature builder must still enforce
``available_at_utc <= as_of_utc`` for every selected feature.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import certifi
import pandas as pd

API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
DEFAULT_START = date(2024, 1, 1)
DEFAULT_END = date(2025, 12, 31)
MODEL = "icon_seamless"
LEADS = (24, 48)
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

VARIABLES = (
    "temperature_2m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "wind_speed_120m",
    "cloud_cover",
)


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def monthly_windows(start: date, end: date) -> list[tuple[date, date]]:
    windows: list[tuple[date, date]] = []
    current = start.replace(day=1)
    while current <= end:
        next_month = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
        windows.append((max(current, start), min(next_month - timedelta(days=1), end)))
        current = next_month
    return windows


def fetch_json(url: str, attempts: int = 4) -> Any:
    request = Request(url, headers={"User-Agent": "energy-pet-project-data-collection/0.2"})
    for attempt in range(attempts):
        try:
            with urlopen(request, timeout=120, context=SSL_CONTEXT) as response:
                return json.loads(response.read())
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            if attempt == attempts - 1:
                raise RuntimeError(f"Request failed after {attempts} attempts: {url}") from error
            time.sleep(2.0 * (attempt + 1))
    raise AssertionError("unreachable")


def _query_url(locations: list[dict[str, Any]], period_start: date, period_end: date) -> str:
    hourly = [
        f"{variable}_previous_day{lead_days}"
        for variable in VARIABLES
        for lead_days in (1, 2)
    ]
    params = {
        "latitude": ",".join(str(location["latitude"]) for location in locations),
        "longitude": ",".join(str(location["longitude"]) for location in locations),
        "start_date": period_start.isoformat(),
        "end_date": period_end.isoformat(),
        "hourly": ",".join(hourly),
        "models": MODEL,
        "timezone": "UTC",
    }
    return f"{API_URL}?{urlencode(params)}"


def _read_locations(project_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config_path = project_root / "config" / "spatial_weather_locations_v1.json"
    payload = json.loads(config_path.read_text())
    locations = payload.get("locations", [])
    if not locations or len({location["location_id"] for location in locations}) != len(locations):
        raise ValueError("Location configuration must contain unique non-empty location_id values")
    return payload, locations


def _normalise_month(
    payload: Any,
    locations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    responses = payload if isinstance(payload, list) else [payload]
    if len(responses) != len(locations):
        raise RuntimeError(
            f"Expected {len(locations)} location responses, received {len(responses)}"
        )

    rows: list[dict[str, Any]] = []
    for location, response in zip(locations, responses, strict=True):
        if not isinstance(response, dict) or "hourly" not in response:
            raise RuntimeError(f"Unexpected API response for {location['location_id']}")
        hourly = response["hourly"]
        times = hourly.get("time", [])
        expected = [
            f"{variable}_previous_day{lead_days}"
            for variable in VARIABLES
            for lead_days in (1, 2)
        ]
        missing = [column for column in expected if column not in hourly]
        if missing:
            raise RuntimeError(f"Missing fields for {location['location_id']}: {missing}")
        for index, valid_time in enumerate(times):
            row: dict[str, Any] = {
                "location_id": location["location_id"],
                "location_role": location["role"],
                "latitude_requested": location["latitude"],
                "longitude_requested": location["longitude"],
                "latitude_model": response.get("latitude"),
                "longitude_model": response.get("longitude"),
                "elevation_model_m": response.get("elevation"),
                "valid_time_utc": pd.Timestamp(valid_time, tz="UTC"),
                "weather_model": MODEL,
            }
            for variable in VARIABLES:
                row[f"weather_fcst24_{variable}"] = hourly[f"{variable}_previous_day1"][index]
                row[f"weather_fcst48_{variable}"] = hourly[f"{variable}_previous_day2"][index]
            rows.append(row)
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _materialise_processed(
    raw_directory: Path,
    processed_path: Path,
    locations: list[dict[str, Any]],
    start: date,
    end: date,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(raw_directory.glob("*.json")):
        rows.extend(_normalise_month(json.loads(path.read_text()), locations))
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No spatial forecast rows were materialised")
    frame["valid_time_utc"] = pd.to_datetime(frame["valid_time_utc"], utc=True)
    start_utc = pd.Timestamp(start, tz="UTC")
    end_utc = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
    frame = frame.loc[frame["valid_time_utc"].between(start_utc, end_utc)].copy()
    frame["weather_fcst24_available_at_utc"] = (
        frame["valid_time_utc"] - pd.Timedelta(hours=24)
    )
    frame["weather_fcst48_available_at_utc"] = (
        frame["valid_time_utc"] - pd.Timedelta(hours=48)
    )
    frame = frame.sort_values(["location_id", "valid_time_utc"]).reset_index(drop=True)
    for timestamp_column in (
        "valid_time_utc",
        "weather_fcst24_available_at_utc",
        "weather_fcst48_available_at_utc",
    ):
        frame[timestamp_column] = frame[timestamp_column].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(processed_path, index=False, compression="gzip")
    return frame


def _quality_summary(frame: pd.DataFrame, locations: list[dict[str, Any]]) -> dict[str, Any]:
    value_columns = [
        f"weather_fcst{lead}_{variable}"
        for lead in LEADS
        for variable in VARIABLES
    ]
    summary: dict[str, Any] = {}
    for location in locations:
        location_id = location["location_id"]
        subset = frame.loc[frame["location_id"] == location_id]
        summary[location_id] = {
            "rows": int(len(subset)),
            "first_valid_time_utc": subset["valid_time_utc"].min(),
            "last_valid_time_utc": subset["valid_time_utc"].max(),
            "duplicate_location_time_rows": int(subset.duplicated(["location_id", "valid_time_utc"]).sum()),
            "null_values": {column: int(subset[column].isna().sum()) for column in value_columns},
        }
    return summary


def _day_ahead_availability_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Audit the project D-1 11:00 Europe/Berlin availability convention."""

    valid_time = pd.to_datetime(frame["valid_time_utc"], utc=True)
    as_of = (
        valid_time.dt.tz_convert("Europe/Berlin").dt.normalize()
        - pd.DateOffset(days=1)
        + pd.Timedelta(hours=11)
    ).dt.tz_convert("UTC")
    columns_24 = [f"weather_fcst24_{variable}" for variable in VARIABLES]
    columns_48 = [f"weather_fcst48_{variable}" for variable in VARIABLES]
    available_24 = frame[columns_24].notna().all(axis=1) & (
        pd.to_datetime(frame["weather_fcst24_available_at_utc"], utc=True) <= as_of
    )
    available_48 = frame[columns_48].notna().all(axis=1) & (
        pd.to_datetime(frame["weather_fcst48_available_at_utc"], utc=True) <= as_of
    )
    usable = available_24 | available_48
    delivery_day = valid_time.dt.tz_convert("Europe/Berlin").dt.date
    by_day = pd.DataFrame({"delivery_day": delivery_day, "usable": usable}).groupby(
        "delivery_day", sort=True
    )["usable"].all()
    complete_days = by_day.loc[by_day]
    return {
        "as_of_convention": "D-1 11:00 Europe/Berlin",
        "row_availability_rate": float(usable.mean()),
        "delivery_days_total": int(len(by_day)),
        "complete_delivery_days": int(len(complete_days)),
        "first_complete_delivery_day": str(complete_days.index.min()) if len(complete_days) else None,
        "last_incomplete_delivery_day": str(by_day.loc[~by_day].index.max()) if (~by_day).any() else None,
    }


def collect(project_root: Path, start: date, end: date, refresh: bool = False) -> dict[str, Any]:
    config, locations = _read_locations(project_root)
    raw_directory = project_root / "data" / "raw" / "weather_forecasts_icon_spatial_previous_runs_monthly"
    processed_path = project_root / "data" / "processed" / "weather_forecast_hourly_icon_spatial_lead_24_48.csv.gz"
    metadata_directory = project_root / "data" / "metadata"
    raw_directory.mkdir(parents=True, exist_ok=True)
    metadata_directory.mkdir(parents=True, exist_ok=True)

    windows = monthly_windows(start, end)
    pending = [
        (period_start, period_end)
        for period_start, period_end in windows
        if refresh or not (raw_directory / f"{period_start:%Y-%m}.json").exists()
    ]
    print(f"Spatial weather months to fetch: {len(pending)}/{len(windows)}", flush=True)
    failures: dict[str, str] = {}
    for number, (period_start, period_end) in enumerate(pending, start=1):
        try:
            payload = fetch_json(_query_url(locations, period_start, period_end))
            _normalise_month(payload, locations)  # Validate before persisting raw data.
            _write_json(raw_directory / f"{period_start:%Y-%m}.json", payload)
        except Exception as error:  # Continue to expose all failed windows in one run.
            failures[period_start.isoformat()] = str(error)
        print(
            f"Spatial weather month {number}/{len(pending)}: {period_start:%Y-%m}; failures: {len(failures)}",
            flush=True,
        )
        time.sleep(0.5)

    _write_json(metadata_directory / "spatial_weather_v1_download_failures.json", failures)
    if failures:
        raise RuntimeError(f"Spatial forecast collection failed for {len(failures)} month(s)")

    frame = _materialise_processed(raw_directory, processed_path, locations, start, end)
    manifest = {
        "data_version": "spatial-weather-v1",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {
            "provider": "Open-Meteo",
            "endpoint": API_URL,
            "model": MODEL,
            "mode": "previous-runs archive",
        },
        "time_window": {"start": start.isoformat(), "end": end.isoformat()},
        "locations": config,
        "variables": list(VARIABLES),
        "lead_hours": list(LEADS),
        "availability_contract": {
            "weather_fcst24_available_at_utc": "valid_time_utc - 24 hours",
            "weather_fcst48_available_at_utc": "valid_time_utc - 48 hours",
            "selection_rule": "A consumer must require available_at_utc <= its own as_of_utc.",
            "not_included": "Realised weather, ERA5 reanalysis, and a full per-run forecast trajectory.",
        },
        "processed_file": {
            "path": str(processed_path.relative_to(project_root)),
            "sha256": _sha256(processed_path),
            "rows": int(len(frame)),
        },
        "quality": {
            "per_location": _quality_summary(frame, locations),
            "day_ahead_feature_availability": _day_ahead_availability_summary(frame),
        },
    }
    manifest_path = metadata_directory / "spatial_weather_v1_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "processed_path": str(processed_path),
        "manifest_path": str(manifest_path),
        "rows": len(frame),
        "locations": len(locations),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--start", type=parse_date, default=DEFAULT_START)
    parser.add_argument("--end", type=parse_date, default=DEFAULT_END)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if args.end < args.start:
        raise SystemExit("--end must not be before --start")
    result = collect(args.project_root.resolve(), args.start, args.end, args.refresh)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
