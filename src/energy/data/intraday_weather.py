"""Archived ECMWF forecast runs for leakage-safe intraday PV forecasting.

The day-ahead project tables contain fixed-lead DWD ICON values.  They are
appropriate for the D-1 decision but cannot reconstruct a weather update that
was visible during delivery.  This module keeps a separate, appendable archive
of ECMWF IFS HRES runs and materialises the exact weather snapshot selected at
each MPC decision time.

The archive starts on 2024-03-14.  A six-hour run-publication delay is used as
a conservative point-in-time boundary even though the provider states a
typical 4--6 hour delay for global ECMWF output.
"""

from __future__ import annotations

import json
import ssl
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd


UTC = "UTC"
LOCAL_TZ = "Europe/Berlin"
ECMWF_SINGLE_RUNS_ENDPOINT = "https://single-runs-api.open-meteo.com/v1/forecast"
ECMWF_SINGLE_RUNS_MODEL = "ecmwf_ifs"
ECMWF_ARCHIVE_START = date(2024, 3, 14)
RUN_HOURS_UTC = (0, 6, 12, 18)
WEATHER_VARIABLES = (
    "temperature_2m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "wind_speed_10m",
    "cloud_cover",
)


@dataclass(frozen=True)
class EcmwfIfsArchiveConfig:
    """Parameters for collecting only the forecast runs needed by MPC."""

    project_root: Path
    start: date = ECMWF_ARCHIVE_START
    end: date = date(2025, 12, 31)
    latitude: float = 48.89
    longitude: float = 8.70
    forecast_hours: int = 48
    publication_delay_hours: int = 6
    run_hours_utc: tuple[int, ...] = RUN_HOURS_UTC


@dataclass(frozen=True)
class EcmwfIfsArchiveResult:
    """Paths and coverage returned by a resumable archive collection."""

    raw_directory: Path
    processed_path: Path
    requested_runs: int
    downloaded_runs: int
    cached_runs: int
    unavailable_runs: int
    hourly_rows: int

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "raw_directory": str(self.raw_directory),
            "processed_path": str(self.processed_path),
        }


@dataclass(frozen=True)
class MpcWeatherSnapshotResult:
    """Materialised decision-time weather rows for one delivery year."""

    output_path: Path
    rows: int
    delivery_days: int
    first_delivery_day: str | None
    last_delivery_day: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "output_path": str(self.output_path),
            "rows": self.rows,
            "delivery_days": self.delivery_days,
            "first_delivery_day": self.first_delivery_day,
            "last_delivery_day": self.last_delivery_day,
        }


@dataclass(frozen=True)
class EcmwfIfsMaterializationResult:
    """Combined normalised archive built from all locally cached run files."""

    processed_path: Path
    available_runs: int
    hourly_rows: int

    def to_dict(self) -> dict[str, object]:
        return {
            "processed_path": str(self.processed_path),
            "available_runs": self.available_runs,
            "hourly_rows": self.hourly_rows,
        }


def collect_ecmwf_ifs_single_runs(
    config: EcmwfIfsArchiveConfig,
    *,
    fetcher: Callable[[str], dict[str, Any]] | None = None,
) -> EcmwfIfsArchiveResult:
    """Download and cache ECMWF IFS runs, then write one normalised Parquet table.

    Raw responses are stored as one atomically-written JSON file per run.
    Re-running the command skips an already cached run, so a transient provider
    failure does not lose completed work or require a fresh full download.
    """

    _validate_archive_config(config)
    root = config.project_root.expanduser().resolve()
    raw_directory = root / "data" / "raw" / "weather_forecasts_ecmwf_ifs_single_runs"
    raw_directory.mkdir(parents=True, exist_ok=True)
    processed_path = root / "data" / "processed" / "weather_forecast_hourly_ecmwf_ifs_single_runs.parquet"
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    runs = list(_run_schedule(config.start, config.end, config.run_hours_utc))
    cached = _cached_run_keys(raw_directory)
    cached_before_collection = len(cached.intersection({_run_key(run) for run in runs}))
    downloader = fetcher or _download_json
    downloaded = 0
    for run_init in runs:
        run_key = _run_key(run_init)
        if run_key in cached:
            continue
        try:
            response = downloader(_request_url(run_init, config))
            status = "available"
            error_message = None
        except HTTPError as error:
            if error.code != 400:
                raise
            # The upstream archive currently has gaps in the advertised run
            # cadence during early coverage.  Persist the provider's explicit
            # absence so a later resume does not retry the same nonexistent run.
            response = None
            status = "unavailable"
            error_message = error.read().decode("utf-8", errors="replace")
        record = {
            "run_init_utc": run_init.isoformat(),
            "available_at_utc": (run_init + pd.Timedelta(hours=config.publication_delay_hours)).isoformat(),
            "status": status,
            "error": error_message,
            "request": {
                "endpoint": ECMWF_SINGLE_RUNS_ENDPOINT,
                "model": ECMWF_SINGLE_RUNS_MODEL,
                "latitude": config.latitude,
                "longitude": config.longitude,
                "forecast_hours": config.forecast_hours,
                "weather_variables": list(WEATHER_VARIABLES),
            },
            "response": response,
        }
        _append_raw_record(raw_directory, run_init, record)
        cached.add(run_key)
        downloaded += int(status == "available")

    table = _normalise_raw_archive(raw_directory, config)
    table.to_parquet(processed_path, index=False)
    return EcmwfIfsArchiveResult(
        raw_directory=raw_directory,
        processed_path=processed_path,
        requested_runs=len(runs),
        downloaded_runs=downloaded,
        cached_runs=cached_before_collection,
        unavailable_runs=_count_unavailable_runs(raw_directory, {_run_key(run) for run in runs}),
        hourly_rows=len(table),
    )


def build_mpc_weather_snapshots(
    project_root: str | Path,
    *,
    year: int,
    source_path: str | Path | None = None,
    start: date | None = None,
    end: date | None = None,
) -> MpcWeatherSnapshotResult:
    """Select the newest published IFS run for every MPC decision/target pair.

    The result contains only targets strictly after the decision hour.  This
    mirrors the execution boundary of the direct residual PV model.  It does
    not contain observations or weather values published after ``as_of_utc``.
    """

    root = Path(project_root).expanduser().resolve()
    source = Path(source_path).expanduser().resolve() if source_path else (
        root / "data" / "processed" / "weather_forecast_hourly_ecmwf_ifs_single_runs.parquet"
    )
    if not source.exists():
        raise FileNotFoundError(
            f"ECMWF IFS archive was not found: {source}. "
            "Run `energy collect-ecmwf-ifs-weather` first."
        )
    pv_path = root / "data" / "features" / "day_ahead_pv" / f"day_ahead_pv_{year}.parquet"
    if not pv_path.exists():
        raise FileNotFoundError(
            f"Day-ahead PV rows were not found: {pv_path}. "
            "Run `energy build-training-data` first."
        )

    forecasts = pd.read_parquet(source)
    forecasts = _normalise_forecast_table(forecasts)
    targets = pd.read_parquet(pv_path)
    targets["valid_time_utc"] = pd.to_datetime(targets["valid_time_utc"], utc=True)
    targets = targets.loc[
        pd.to_datetime(targets["delivery_date_local"]).dt.year.eq(year),
        ["delivery_date_local", "valid_time_utc", "hour_local"],
    ].drop_duplicates()
    if start is not None:
        targets = targets.loc[targets["delivery_date_local"] >= start.isoformat()].copy()
    if end is not None:
        targets = targets.loc[targets["delivery_date_local"] <= end.isoformat()].copy()
    if targets.empty:
        raise ValueError("No day-ahead PV rows remain in the requested weather-snapshot range")

    snapshot_rows: list[pd.DataFrame] = []
    run_times = forecasts[["run_init_utc", "available_at_utc"]].drop_duplicates().sort_values("available_at_utc")
    for delivery_day, day_targets in targets.groupby("delivery_date_local", sort=True):
        day_targets = day_targets.sort_values("valid_time_utc")
        if day_targets["hour_local"].nunique() != 24:
            # The rest of V1 excludes non-24-hour local delivery days as well.
            continue
        local_start = pd.Timestamp(delivery_day, tz=LOCAL_TZ)
        for decision_hour in range(24):
            as_of = local_start + pd.DateOffset(hours=decision_hour)
            as_of_utc = as_of.tz_convert(UTC)
            eligible = run_times.loc[run_times["available_at_utc"] <= as_of_utc]
            if eligible.empty:
                continue
            chosen = eligible.iloc[-1]
            future = day_targets.loc[day_targets["valid_time_utc"] > as_of_utc]
            if future.empty:
                continue
            run_forecast = forecasts.loc[
                forecasts["run_init_utc"].eq(chosen["run_init_utc"])
            ]
            rows = future.merge(
                run_forecast,
                on="valid_time_utc",
                how="inner",
                validate="one_to_one",
            )
            if rows.empty:
                continue
            rows = rows.assign(
                as_of_utc=as_of_utc,
                decision_hour_local=np.int8(decision_hour),
                weather_snapshot_lead_hours=(
                    (rows["valid_time_utc"] - rows["run_init_utc"]).dt.total_seconds() / 3600
                ).astype("int16"),
                weather_snapshot_age_hours=(
                    (as_of_utc - rows["available_at_utc"]).dt.total_seconds() / 3600
                ).astype("int16"),
            )
            snapshot_rows.append(rows)

    if not snapshot_rows:
        raise ValueError(f"No MPC weather snapshots can be built for {year}")
    snapshots = pd.concat(snapshot_rows, ignore_index=True)
    weather_columns = [f"weather_forecast_{variable}" for variable in WEATHER_VARIABLES]
    required = [
        "delivery_date_local",
        "valid_time_utc",
        "as_of_utc",
        "decision_hour_local",
        "run_init_utc",
        "available_at_utc",
        "weather_snapshot_lead_hours",
        "weather_snapshot_age_hours",
        *weather_columns,
    ]
    snapshots = snapshots.loc[:, required].rename(
        columns={"run_init_utc": "weather_snapshot_run_init_utc", "available_at_utc": "weather_snapshot_available_at_utc"}
    )
    snapshots = snapshots.dropna(subset=weather_columns).sort_values(
        ["as_of_utc", "valid_time_utc"]
    ).reset_index(drop=True)
    _validate_snapshots(snapshots, weather_columns)

    output_dir = root / "data" / "features" / "mpc_pv_weather_ifs_snapshots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"mpc_pv_weather_ifs_snapshots_{year}.parquet"
    snapshots.to_parquet(output_path, index=False)
    days = snapshots["delivery_date_local"].drop_duplicates().sort_values()
    return MpcWeatherSnapshotResult(
        output_path=output_path,
        rows=len(snapshots),
        delivery_days=len(days),
        first_delivery_day=days.iloc[0] if len(days) else None,
        last_delivery_day=days.iloc[-1] if len(days) else None,
    )


def materialize_ecmwf_ifs_archive(
    project_root: str | Path,
) -> EcmwfIfsMaterializationResult:
    """Combine every atomically cached successful run into the processed table.

    Collection is intentionally chunkable.  This command is the final step
    after collecting different date ranges or run frequencies, and avoids
    replacing the processed table with only the most recently fetched chunk.
    """

    root = Path(project_root).expanduser().resolve()
    raw_directory = root / "data" / "raw" / "weather_forecasts_ecmwf_ifs_single_runs"
    records = [record for record in _raw_records(raw_directory) if record.get("status", "available") == "available"]
    if not records:
        raise FileNotFoundError(
            f"No cached successful IFS runs were found under {raw_directory}."
        )
    table = pd.concat([_response_to_frame(record) for record in records], ignore_index=True)
    if table.duplicated(["run_init_utc", "valid_time_utc"]).any():
        raise ValueError("Cached IFS run files contain duplicate run / valid-time rows")
    table = table.sort_values(["run_init_utc", "valid_time_utc"]).reset_index(drop=True)
    processed_path = root / "data" / "processed" / "weather_forecast_hourly_ecmwf_ifs_single_runs.parquet"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(processed_path, index=False)
    statuses = {
        "available": len(records),
        "unavailable": _count_status(raw_directory, "unavailable"),
    }
    metadata_path = root / "data" / "metadata" / "ecmwf_ifs_intraday_weather_manifest.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "provider": "Open-Meteo Single Runs API",
                "endpoint": ECMWF_SINGLE_RUNS_ENDPOINT,
                "model": "ECMWF IFS HRES via ecmwf_ifs",
                "location": {"latitude": 48.89, "longitude": 8.70},
                "variables": list(WEATHER_VARIABLES),
                "availability_contract": "run_init_utc + 6 hours <= MPC as_of_utc",
                "raw_directory": str(raw_directory.relative_to(root)),
                "processed_path": str(processed_path.relative_to(root)),
                "available_runs": int(table["run_init_utc"].nunique()),
                "unavailable_runs": statuses["unavailable"],
                "hourly_rows": len(table),
                "first_run_init_utc": table["run_init_utc"].min().isoformat(),
                "last_run_init_utc": table["run_init_utc"].max().isoformat(),
            },
            indent=2,
        )
    )
    return EcmwfIfsMaterializationResult(
        processed_path=processed_path,
        available_runs=int(table["run_init_utc"].nunique()),
        hourly_rows=len(table),
    )


def _validate_archive_config(config: EcmwfIfsArchiveConfig) -> None:
    if config.start < ECMWF_ARCHIVE_START:
        raise ValueError(f"ECMWF IFS single-run archive starts on {ECMWF_ARCHIVE_START.isoformat()}")
    if config.end < config.start:
        raise ValueError("end must be on or after start")
    if config.forecast_hours < 30:
        raise ValueError("forecast_hours must cover at least 30 hours for the delivery-day MPC horizon")
    if config.publication_delay_hours < 6:
        raise ValueError("publication_delay_hours must be at least 6 to avoid look-ahead")
    if not config.run_hours_utc or any(hour not in RUN_HOURS_UTC for hour in config.run_hours_utc):
        raise ValueError(f"run_hours_utc must be a non-empty subset of {RUN_HOURS_UTC}")


def _run_schedule(
    start: date,
    end: date,
    run_hours_utc: tuple[int, ...] = RUN_HOURS_UTC,
) -> Iterable[pd.Timestamp]:
    current = pd.Timestamp(start, tz=UTC)
    final = pd.Timestamp(end, tz=UTC)
    while current <= final:
        for hour in run_hours_utc:
            yield current + pd.Timedelta(hours=hour)
        current += pd.Timedelta(days=1)


def _run_key(run_init: pd.Timestamp) -> str:
    return run_init.strftime("%Y-%m-%dT%H:%M:%S%z")


def _request_url(run_init: pd.Timestamp, config: EcmwfIfsArchiveConfig) -> str:
    query = urlencode(
        {
            "latitude": config.latitude,
            "longitude": config.longitude,
            "hourly": ",".join(WEATHER_VARIABLES),
            "models": ECMWF_SINGLE_RUNS_MODEL,
            "run": run_init.strftime("%Y-%m-%dT%H:%M"),
            "forecast_hours": config.forecast_hours,
            "timezone": "GMT",
        }
    )
    return f"{ECMWF_SINGLE_RUNS_ENDPOINT}?{query}"


def _download_json(url: str) -> dict[str, Any]:
    try:
        import certifi
    except ImportError as error:  # pragma: no cover - project dependency
        raise RuntimeError("Install certifi before downloading Open-Meteo data.") from error
    context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(url, timeout=60, context=context) as response:  # nosec B310 - fixed public HTTPS endpoint
        return json.loads(response.read().decode("utf-8"))


def _run_raw_path(raw_directory: Path, run_init: pd.Timestamp) -> Path:
    return raw_directory / "runs" / f"ecmwf_ifs_{run_init.strftime('%Y%m%dT%H%MZ')}.json"


def _append_raw_record(raw_directory: Path, run_init: pd.Timestamp, record: dict[str, Any]) -> None:
    path = _run_raw_path(raw_directory, run_init)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(record, separators=(",", ":")))
    temporary_path.replace(path)


def _raw_records(raw_directory: Path) -> Iterable[dict[str, Any]]:
    for path in sorted((raw_directory / "runs").glob("ecmwf_ifs_*.json")):
        yield json.loads(path.read_text())


def _cached_run_keys(raw_directory: Path) -> set[str]:
    return {
        _run_key(pd.Timestamp(record["run_init_utc"]).tz_convert(UTC))
        for record in _raw_records(raw_directory)
    }


def _normalise_raw_archive(
    raw_directory: Path,
    config: EcmwfIfsArchiveConfig,
) -> pd.DataFrame:
    requested = {
        _run_key(value)
        for value in _run_schedule(config.start, config.end, config.run_hours_utc)
    }
    rows: list[pd.DataFrame] = []
    seen: set[str] = set()
    for record in _raw_records(raw_directory):
        run_init = pd.Timestamp(record["run_init_utc"]).tz_convert(UTC)
        run_key = _run_key(run_init)
        if run_key not in requested or run_key in seen:
            continue
        seen.add(run_key)
        if record.get("status", "available") == "available":
            rows.append(_response_to_frame(record))
    missing = requested.difference(seen)
    if missing:
        raise RuntimeError(f"The raw IFS archive is incomplete: {len(missing)} requested runs are missing")
    if not rows:
        raise RuntimeError("The requested IFS archive has no available runs")
    result = pd.concat(rows, ignore_index=True)
    result = result.sort_values(["run_init_utc", "valid_time_utc"]).reset_index(drop=True)
    return result


def _response_to_frame(record: dict[str, Any]) -> pd.DataFrame:
    response = record["response"]
    hourly = response.get("hourly", {})
    times = hourly.get("time")
    if not isinstance(times, list) or not times:
        raise ValueError("Open-Meteo response has no hourly time values")
    data: dict[str, Any] = {
        "run_init_utc": pd.Timestamp(record["run_init_utc"]),
        "available_at_utc": pd.Timestamp(record["available_at_utc"]),
        "valid_time_utc": pd.to_datetime(times, utc=True),
    }
    for variable in WEATHER_VARIABLES:
        values = hourly.get(variable)
        if not isinstance(values, list) or len(values) != len(times):
            raise ValueError(f"Open-Meteo response has invalid {variable} values")
        data[f"weather_forecast_{variable}"] = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return pd.DataFrame(data)


def _count_unavailable_runs(raw_directory: Path, requested: set[str]) -> int:
    return sum(
        _run_key(pd.Timestamp(record["run_init_utc"]).tz_convert(UTC)) in requested
        and record.get("status") == "unavailable"
        for record in _raw_records(raw_directory)
    )


def _count_status(raw_directory: Path, status: str) -> int:
    return sum(record.get("status", "available") == status for record in _raw_records(raw_directory))


def _normalise_forecast_table(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"run_init_utc", "available_at_utc", "valid_time_utc", *[f"weather_forecast_{name}" for name in WEATHER_VARIABLES]}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"ECMWF IFS archive is missing columns: {missing}")
    result = frame.copy()
    for column in ("run_init_utc", "available_at_utc", "valid_time_utc"):
        result[column] = pd.to_datetime(result[column], utc=True)
    if result.duplicated(["run_init_utc", "valid_time_utc"]).any():
        raise ValueError("ECMWF IFS archive has duplicate run / valid-time rows")
    return result


def _validate_snapshots(frame: pd.DataFrame, weather_columns: list[str]) -> None:
    if frame.empty:
        raise ValueError("MPC weather snapshot table is empty after complete-weather filtering")
    if frame.duplicated(["delivery_date_local", "decision_hour_local", "valid_time_utc"]).any():
        raise ValueError("MPC weather snapshot table has duplicate decision / target rows")
    as_of = pd.to_datetime(frame["as_of_utc"], utc=True)
    available = pd.to_datetime(frame["weather_snapshot_available_at_utc"], utc=True)
    valid = pd.to_datetime(frame["valid_time_utc"], utc=True)
    if not (available <= as_of).all():
        raise AssertionError("A weather snapshot was selected before it was published")
    if not (valid > as_of).all():
        raise AssertionError("A weather snapshot exposes an observed or current target hour")
    if frame.loc[:, weather_columns].isna().any().any():
        raise AssertionError("A selected MPC weather snapshot has a missing weather value")
