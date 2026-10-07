"""Spatial ECMWF IFS forecast-run archive for the intraday DE-LU price model.

The day-ahead price model uses ten representative German weather-regime
coordinates.  This module retrieves the same coordinates from Open-Meteo's
ECMWF IFS HRES *Single Runs* archive, preserving the run initialisation time.
It deliberately writes to a separate archive from the local PV weather data.
"""

from __future__ import annotations

import json
import ssl
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from energy.data.intraday_weather import (
    ECMWF_ARCHIVE_START,
    ECMWF_SINGLE_RUNS_ENDPOINT,
    ECMWF_SINGLE_RUNS_MODEL,
    RUN_HOURS_UTC,
    UTC,
    _run_schedule,
)


SPATIAL_LOCATION_CONFIG = Path("config/spatial_weather_locations_v1.json")
SPATIAL_WEATHER_VARIABLES = (
    "temperature_2m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "wind_speed_120m",
    "cloud_cover",
)
RAW_RELATIVE_PATH = Path("data/raw/weather_forecasts_ecmwf_ifs_spatial_single_runs")
PROCESSED_RELATIVE_PATH = Path(
    "data/processed/weather_forecast_hourly_ecmwf_ifs_spatial_single_runs.parquet"
)
MANIFEST_RELATIVE_PATH = Path("data/metadata/ecmwf_ifs_spatial_intraday_weather_manifest.json")


class _ModelRunUnavailableError(RuntimeError):
    """Open-Meteo returned a successful HTTP response for an absent model run."""


@dataclass(frozen=True)
class SpatialEcmwfIfsArchiveConfig:
    """Parameters for a resumable multi-coordinate IFS archive build."""

    project_root: Path
    start: date = ECMWF_ARCHIVE_START
    end: date = date(2025, 9, 30)
    forecast_hours: int = 48
    publication_delay_hours: int = 6
    run_hours_utc: tuple[int, ...] = RUN_HOURS_UTC
    locations_path: Path | None = None
    max_workers: int = 1


@dataclass(frozen=True)
class SpatialEcmwfIfsArchiveResult:
    raw_directory: Path
    processed_path: Path
    requested_runs: int
    downloaded_runs: int
    cached_runs: int
    unavailable_runs: int
    locations: int
    hourly_rows: int

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["raw_directory"] = str(self.raw_directory)
        result["processed_path"] = str(self.processed_path)
        return result


@dataclass(frozen=True)
class SpatialEcmwfIfsMaterializationResult:
    processed_path: Path
    available_runs: int
    locations: int
    hourly_rows: int

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["processed_path"] = str(self.processed_path)
        return result


def collect_spatial_ecmwf_ifs_single_runs(
    config: SpatialEcmwfIfsArchiveConfig,
    *,
    fetcher: Callable[[str], Any] | None = None,
) -> SpatialEcmwfIfsArchiveResult:
    """Download each IFS run once, requesting all German coordinates together."""

    _validate_config(config)
    root = config.project_root.expanduser().resolve()
    locations = _load_locations(root, config.locations_path)
    raw_directory = root / RAW_RELATIVE_PATH
    runs = list(_run_schedule(config.start, config.end, config.run_hours_utc))
    cached = _cached_run_keys(raw_directory)
    cached_before = len(cached.intersection({_run_key(run) for run in runs}))
    download = fetcher or _download_json
    downloaded = 0

    missing_runs = [run_init for run_init in runs if _run_key(run_init) not in cached]
    for run_init, response, status, error_message in _download_missing_runs(
        missing_runs, locations, config, download
    ):
        _write_record(
            raw_directory,
            run_init,
            {
                "run_init_utc": run_init.isoformat(),
                "available_at_utc": (
                    run_init + pd.Timedelta(hours=config.publication_delay_hours)
                ).isoformat(),
                "status": status,
                "error": error_message,
                "request": {
                    "endpoint": ECMWF_SINGLE_RUNS_ENDPOINT,
                    "model": ECMWF_SINGLE_RUNS_MODEL,
                    "locations": locations,
                    "forecast_hours": config.forecast_hours,
                    "weather_variables": list(SPATIAL_WEATHER_VARIABLES),
                },
                "response": response,
            },
        )
        cached.add(_run_key(run_init))
        downloaded += int(status == "available")

    table = _normalise_requested_records(raw_directory, config)
    processed = root / PROCESSED_RELATIVE_PATH
    processed.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(processed, index=False)
    _write_manifest(root, raw_directory, processed, table, locations)
    return SpatialEcmwfIfsArchiveResult(
        raw_directory=raw_directory,
        processed_path=processed,
        requested_runs=len(runs),
        downloaded_runs=downloaded,
        cached_runs=cached_before,
        unavailable_runs=_count_status(raw_directory, "unavailable"),
        locations=len(locations),
        hourly_rows=len(table),
    )


def materialize_spatial_ecmwf_ifs_archive(
    project_root: str | Path,
    *,
    locations_path: Path | None = None,
) -> SpatialEcmwfIfsMaterializationResult:
    """Combine all successfully cached spatial IFS run responses."""

    root = Path(project_root).expanduser().resolve()
    raw_directory = root / RAW_RELATIVE_PATH
    records = [record for record in _raw_records(raw_directory) if record.get("status") == "available"]
    if not records:
        raise FileNotFoundError(f"No cached spatial IFS runs found under {raw_directory}")
    table = pd.concat([_response_to_frame(record) for record in records], ignore_index=True)
    _validate_table(table)
    table = table.sort_values(["location_id", "run_init_utc", "valid_time_utc"]).reset_index(drop=True)
    processed = root / PROCESSED_RELATIVE_PATH
    processed.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(processed, index=False)
    locations = _load_locations(root, locations_path)
    _write_manifest(root, raw_directory, processed, table, locations)
    return SpatialEcmwfIfsMaterializationResult(
        processed_path=processed,
        available_runs=int(table["run_init_utc"].nunique()),
        locations=int(table["location_id"].nunique()),
        hourly_rows=len(table),
    )


def _load_locations(root: Path, configured_path: Path | None) -> list[dict[str, object]]:
    path = configured_path or root / SPATIAL_LOCATION_CONFIG
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text())
    locations = payload.get("locations")
    if not isinstance(locations, list) or not locations:
        raise ValueError(f"Spatial location config has no locations: {path}")
    required = {"location_id", "latitude", "longitude", "role"}
    if any(not required.issubset(location) for location in locations):
        raise ValueError(f"Spatial location config is incomplete: {path}")
    identifiers = [str(location["location_id"]) for location in locations]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Spatial location IDs must be unique")
    return locations


def _validate_config(config: SpatialEcmwfIfsArchiveConfig) -> None:
    if config.start < ECMWF_ARCHIVE_START:
        raise ValueError(f"ECMWF IFS single-run archive starts on {ECMWF_ARCHIVE_START.isoformat()}")
    if config.end < config.start:
        raise ValueError("end must be on or after start")
    if config.forecast_hours < 30:
        raise ValueError("forecast_hours must cover the next-day delivery horizon")
    if config.publication_delay_hours < 6:
        raise ValueError("publication_delay_hours must be at least six hours")
    if not config.run_hours_utc or any(hour not in RUN_HOURS_UTC for hour in config.run_hours_utc):
        raise ValueError(f"run_hours_utc must be a non-empty subset of {RUN_HOURS_UTC}")
    if not 1 <= config.max_workers <= 2:
        raise ValueError("max_workers must be between one and two")


def _download_missing_runs(
    runs: list[pd.Timestamp],
    locations: list[dict[str, object]],
    config: SpatialEcmwfIfsArchiveConfig,
    downloader: Callable[[str], Any],
) -> Iterable[tuple[pd.Timestamp, Any, str, str | None]]:
    if config.max_workers == 1:
        for run_init in runs:
            yield _download_one_run(run_init, locations, config, downloader)
        return
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = {
            executor.submit(_download_one_run, run_init, locations, config, downloader): run_init
            for run_init in runs
        }
        for future in as_completed(futures):
            yield future.result()


def _download_one_run(
    run_init: pd.Timestamp,
    locations: list[dict[str, object]],
    config: SpatialEcmwfIfsArchiveConfig,
    downloader: Callable[[str], Any],
) -> tuple[pd.Timestamp, Any, str, str | None]:
    for attempt in range(5):
        try:
            response = downloader(_request_url(run_init, locations, config))
            return run_init, response, "available", None
        except _ModelRunUnavailableError as error:
            # The Single Runs API sometimes reports a missing archive run as a
            # plain-text HTTP 200 response instead of its older HTTP 400 JSON
            # response. Persist the gap so a resumable collection advances to
            # the next run rather than retrying the same permanent absence.
            return run_init, None, "unavailable", str(error)
        except HTTPError as error:
            if error.code == 400:
                return run_init, None, "unavailable", error.read().decode("utf-8", errors="replace")
            if error.code not in {429, 500, 502, 503, 504} or attempt == 4:
                raise
            time.sleep(5 * (2**attempt))
        except (URLError, TimeoutError, json.JSONDecodeError):
            if attempt == 4:
                raise
            time.sleep(5 * (2**attempt))
    raise AssertionError("unreachable retry state")


def _request_url(
    run_init: pd.Timestamp,
    locations: list[dict[str, object]],
    config: SpatialEcmwfIfsArchiveConfig,
) -> str:
    query = urlencode(
        {
            "latitude": ",".join(str(location["latitude"]) for location in locations),
            "longitude": ",".join(str(location["longitude"]) for location in locations),
            "hourly": ",".join(SPATIAL_WEATHER_VARIABLES),
            "models": ECMWF_SINGLE_RUNS_MODEL,
            "run": run_init.strftime("%Y-%m-%dT%H:%M"),
            "forecast_hours": config.forecast_hours,
            "timezone": "GMT",
        }
    )
    return f"{ECMWF_SINGLE_RUNS_ENDPOINT}?{query}"


def _download_json(url: str) -> Any:
    try:
        import certifi
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise RuntimeError("Install certifi before downloading Open-Meteo data") from error
    context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(url, timeout=60, context=context) as response:  # nosec B310 - fixed public HTTPS endpoint
        body = response.read().decode("utf-8")
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        if "modelRunUnavailable" in body:
            raise _ModelRunUnavailableError(body) from None
        raise


def _run_key(run_init: pd.Timestamp) -> str:
    return run_init.strftime("%Y-%m-%dT%H:%M:%S%z")


def _raw_path(raw_directory: Path, run_init: pd.Timestamp) -> Path:
    return raw_directory / "runs" / f"ecmwf_ifs_spatial_{run_init.strftime('%Y%m%dT%H%MZ')}.json"


def _write_record(raw_directory: Path, run_init: pd.Timestamp, record: dict[str, Any]) -> None:
    path = _raw_path(raw_directory, run_init)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, separators=(",", ":")))
    temporary.replace(path)


def _raw_records(raw_directory: Path) -> Iterable[dict[str, Any]]:
    for path in sorted((raw_directory / "runs").glob("ecmwf_ifs_spatial_*.json")):
        yield json.loads(path.read_text())


def _cached_run_keys(raw_directory: Path) -> set[str]:
    return {
        _run_key(pd.Timestamp(record["run_init_utc"]).tz_convert(UTC))
        for record in _raw_records(raw_directory)
    }


def _normalise_requested_records(raw_directory: Path, config: SpatialEcmwfIfsArchiveConfig) -> pd.DataFrame:
    requested = {_run_key(value) for value in _run_schedule(config.start, config.end, config.run_hours_utc)}
    rows: list[pd.DataFrame] = []
    seen: set[str] = set()
    for record in _raw_records(raw_directory):
        key = _run_key(pd.Timestamp(record["run_init_utc"]).tz_convert(UTC))
        if key not in requested or key in seen:
            continue
        seen.add(key)
        if record.get("status") == "available":
            rows.append(_response_to_frame(record))
    missing = requested.difference(seen)
    if missing:
        raise RuntimeError(f"The spatial IFS archive is incomplete: {len(missing)} runs are absent")
    if not rows:
        raise RuntimeError("The requested spatial IFS archive has no successful runs")
    table = pd.concat(rows, ignore_index=True)
    _validate_table(table)
    return table.sort_values(["location_id", "run_init_utc", "valid_time_utc"]).reset_index(drop=True)


def _response_to_frame(record: dict[str, Any]) -> pd.DataFrame:
    locations = record["request"]["locations"]
    payload = record["response"]
    responses = payload if isinstance(payload, list) else [payload]
    if len(responses) != len(locations):
        raise ValueError("Open-Meteo spatial response count differs from requested location count")
    frames: list[pd.DataFrame] = []
    for location, response in zip(locations, responses, strict=True):
        hourly = response.get("hourly", {})
        times = hourly.get("time")
        if not isinstance(times, list) or not times:
            raise ValueError("Open-Meteo spatial response has no hourly time values")
        data: dict[str, Any] = {
            "location_id": str(location["location_id"]),
            "location_role": str(location["role"]),
            "latitude": float(location["latitude"]),
            "longitude": float(location["longitude"]),
            "run_init_utc": pd.Timestamp(record["run_init_utc"]),
            "available_at_utc": pd.Timestamp(record["available_at_utc"]),
            "valid_time_utc": pd.to_datetime(times, utc=True),
        }
        for variable in SPATIAL_WEATHER_VARIABLES:
            values = hourly.get(variable)
            if not isinstance(values, list) or len(values) != len(times):
                raise ValueError(f"Open-Meteo spatial response has invalid {variable} values")
            data[f"weather_forecast_{variable}"] = pd.to_numeric(
                pd.Series(values), errors="coerce"
            ).to_numpy(dtype=float)
        frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True)


def _validate_table(table: pd.DataFrame) -> None:
    weather_columns = [f"weather_forecast_{value}" for value in SPATIAL_WEATHER_VARIABLES]
    required = {"location_id", "run_init_utc", "available_at_utc", "valid_time_utc", *weather_columns}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise KeyError(f"Spatial IFS table is missing columns: {missing}")
    if table.duplicated(["location_id", "run_init_utc", "valid_time_utc"]).any():
        raise ValueError("Spatial IFS table contains duplicate location/run/time rows")


def _count_status(raw_directory: Path, status: str) -> int:
    return sum(record.get("status") == status for record in _raw_records(raw_directory))


def _write_manifest(
    root: Path,
    raw_directory: Path,
    processed: Path,
    table: pd.DataFrame,
    locations: list[dict[str, object]],
) -> None:
    path = root / MANIFEST_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "provider": "Open-Meteo Single Runs API",
                "endpoint": ECMWF_SINGLE_RUNS_ENDPOINT,
                "model": "ECMWF IFS HRES via ecmwf_ifs",
                "locations": locations,
                "variables": list(SPATIAL_WEATHER_VARIABLES),
                "availability_contract": "run_init_utc + 6 hours <= intraday_price as_of_utc",
                "raw_directory": str(raw_directory.relative_to(root)),
                "processed_path": str(processed.relative_to(root)),
                "available_runs": int(table["run_init_utc"].nunique()),
                "unavailable_runs": _count_status(raw_directory, "unavailable"),
                "locations_count": int(table["location_id"].nunique()),
                "hourly_rows": len(table),
                "null_values": {
                    column: int(table[column].isna().sum())
                    for column in [f"weather_forecast_{value}" for value in SPATIAL_WEATHER_VARIABLES]
                },
                "first_run_init_utc": table["run_init_utc"].min().isoformat(),
                "last_run_init_utc": table["run_init_utc"].max().isoformat(),
            },
            indent=2,
        )
    )
