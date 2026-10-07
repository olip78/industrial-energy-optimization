"""Build canonical and model-specific point-in-time training datasets."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from energy.data.splitting import add_week_index

LOCAL_TZ = "Europe/Berlin"
UTC = "UTC"
PV_PROFILE = "1a"
DAY_AHEAD_DECISION_HOUR_LOCAL = 11
WEATHER_VARIABLES = (
    "temperature_2m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "wind_speed_10m",
    "cloud_cover",
)
PV_OUTAGE_LOCAL_DATES = {"2024-05-10", "2024-07-07", "2025-07-23"}


@dataclass(frozen=True)
class BuildResult:
    canonical_path: Path
    dataset_paths: dict[str, Path]
    manifest_path: Path
    duckdb_path: Path


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _parse_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _calendar_features(valid_time: pd.Series) -> pd.DataFrame:
    local = valid_time.dt.tz_convert(LOCAL_TZ)
    hour = local.dt.hour.astype("int8")
    day_of_week = local.dt.dayofweek.astype("int8")
    day_of_year = local.dt.dayofyear.astype("int16")
    result = pd.DataFrame(index=valid_time.index)
    result["valid_time_local"] = local.astype(str)
    result["delivery_date_local"] = local.dt.strftime("%Y-%m-%d")
    result["hour_local"] = hour
    result["day_of_week"] = day_of_week
    result["day_of_year"] = day_of_year
    result["month"] = local.dt.month.astype("int8")
    result["is_weekend"] = (day_of_week >= 5).astype("int8")
    result["hour_sin"] = np.sin(2.0 * math.pi * hour / 24.0)
    result["hour_cos"] = np.cos(2.0 * math.pi * hour / 24.0)
    result["day_of_year_sin"] = np.sin(2.0 * math.pi * day_of_year / 366.0)
    result["day_of_year_cos"] = np.cos(2.0 * math.pi * day_of_year / 366.0)
    return result


def _day_ahead_price_available_at(valid_time: pd.Series) -> pd.Series:
    local = valid_time.dt.tz_convert(LOCAL_TZ)
    delivery_midnight = local.dt.normalize()
    previous_day = delivery_midnight - pd.DateOffset(days=1)
    published_local = previous_day + pd.Timedelta(hours=13)
    return published_local.dt.tz_convert(UTC)


def _day_ahead_as_of(valid_time: pd.Series) -> pd.Series:
    """Return the common D-1 decision time for every delivery-day period."""

    local = valid_time.dt.tz_convert(LOCAL_TZ)
    delivery_midnight = local.dt.normalize()
    decision_local = (
        delivery_midnight
        - pd.DateOffset(days=1)
        + pd.Timedelta(hours=DAY_AHEAD_DECISION_HOUR_LOCAL)
    )
    return decision_local.dt.tz_convert(UTC)


def _select_available_day_ahead_weather(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the freshest complete fixed-lead weather layer available at as-of."""

    columns_24 = [f"weather_fcst24_{variable}" for variable in WEATHER_VARIABLES]
    columns_48 = [f"weather_fcst48_{variable}" for variable in WEATHER_VARIABLES]
    available_24 = (
        frame["weather_fcst24_available_at_utc"].le(frame["as_of_utc"])
        & frame[columns_24].notna().all(axis=1)
    )
    available_48 = (
        frame["weather_fcst48_available_at_utc"].le(frame["as_of_utc"])
        & frame[columns_48].notna().all(axis=1)
    )
    use_24 = available_24
    use_48 = ~use_24 & available_48
    for variable in WEATHER_VARIABLES:
        selected = frame[f"weather_fcst24_{variable}"].where(use_24)
        frame[f"weather_forecast_{variable}"] = selected.where(
            use_24,
            frame[f"weather_fcst48_{variable}"].where(use_48),
        )
    frame["weather_forecast_lead_hours"] = pd.Series(
        np.select([use_24, use_48], [24, 48], default=np.nan),
        index=frame.index,
    ).astype("Int8")
    frame["weather_forecast_available_at_utc"] = frame[
        "weather_fcst24_available_at_utc"
    ].where(use_24, frame["weather_fcst48_available_at_utc"].where(use_48))
    return frame


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TrainingDatasetBuilder:
    """Materialize canonical hourly data and leakage-safe training tables."""

    def __init__(self, project_root: str | Path, output_root: str | Path | None = None) -> None:
        self.project_root = Path(project_root).resolve()
        self.processed = self.project_root / "data" / "processed"
        self.output_root = Path(output_root).resolve() if output_root else self.project_root / "data"
        self.curated = self.output_root / "curated"
        self.features = self.output_root / "features"
        self.metadata = self.output_root / "metadata"

    def build(self, year: int) -> BuildResult:
        for directory in (self.curated, self.features, self.metadata):
            directory.mkdir(parents=True, exist_ok=True)

        canonical = self._build_canonical()
        canonical_year = canonical.loc[
            canonical["delivery_date_local"].str.startswith(str(year), na=False)
        ].reset_index(drop=True)
        canonical_path = self.curated / f"canonical_hourly_{year}.parquet"
        canonical_year.to_parquet(canonical_path, index=False)

        day_ahead_pv = self._build_day_ahead_pv(canonical, year)
        datasets = {
            "day_ahead_pv": day_ahead_pv,
            "oracle_pv": self._build_oracle_pv(canonical, day_ahead_pv, year),
            "day_ahead_price": self._build_day_ahead_price(canonical, year),
            "mpc_pv": self._build_mpc_pv(canonical, year),
        }
        paths: dict[str, Path] = {}
        for name, frame in datasets.items():
            directory = self.features / name
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{name}_{year}.parquet"
            frame.to_parquet(path, index=False)
            paths[name] = path

        duckdb_path = self.curated / "energy_training.duckdb"
        self._write_duckdb(duckdb_path, canonical_path, paths, year)
        manifest_path = self.metadata / f"training_datasets_{year}_manifest.json"
        self._write_manifest(manifest_path, canonical_path, paths, datasets, year)
        return BuildResult(canonical_path, paths, manifest_path, duckdb_path)

    def _build_canonical(self) -> pd.DataFrame:
        pv = _read_csv(self.processed / "pv_power_15min_assumed_europe_berlin.csv.gz")
        pv["valid_time_utc"] = _parse_utc(pv["valid_time_utc"])
        pv = pv.dropna(subset=["valid_time_utc"]).copy()
        pv = pv.set_index("valid_time_utc")
        hourly_pv = pv.resample("1h").agg(
            pv_power_mean_w=(PV_PROFILE, "mean"),
            pv_intervals=(PV_PROFILE, "count"),
            pv_quality_status=("pv_quality_status", lambda values: "usable" if (values == "usable").all() else "excluded"),
        )
        hourly_pv["pv_energy_kwh"] = hourly_pv["pv_power_mean_w"] / 1000.0
        hourly_pv["pv_available_at_utc"] = hourly_pv.index + pd.Timedelta(hours=1)
        hourly_pv = hourly_pv.reset_index()

        actual = _read_csv(self.processed / "weather_actual_hourly_era5.csv.gz")
        actual["valid_time_utc"] = _parse_utc(actual["valid_time_utc"])
        actual = actual.rename(
            columns={variable: f"weather_actual_{variable}" for variable in WEATHER_VARIABLES}
        )
        actual["weather_actual_available_at_utc"] = actual["valid_time_utc"] + pd.Timedelta(hours=1)

        forecast = _read_csv(self.processed / "weather_forecast_hourly_icon_lead_24_48.csv.gz")
        forecast["valid_time_utc"] = _parse_utc(forecast["valid_time_utc"])
        keep = ["valid_time_utc"] + [
            f"{variable}_{suffix}"
            for variable in WEATHER_VARIABLES
            for suffix in ("previous_day1", "previous_day2")
        ]
        rename = {
            f"{variable}_previous_day1": f"weather_fcst24_{variable}"
            for variable in WEATHER_VARIABLES
        }
        rename.update(
            {
                f"{variable}_previous_day2": f"weather_fcst48_{variable}"
                for variable in WEATHER_VARIABLES
            }
        )
        forecast = forecast[keep].rename(columns=rename)
        forecast["weather_fcst24_available_at_utc"] = (
            forecast["valid_time_utc"] - pd.Timedelta(hours=24)
        )
        forecast["weather_fcst48_available_at_utc"] = (
            forecast["valid_time_utc"] - pd.Timedelta(hours=48)
        )

        prices = _read_csv(self.processed / "prices_day_ahead_de_lu.csv.gz")
        prices["valid_time_utc"] = _parse_utc(prices["valid_time_utc"])
        prices = prices.sort_values("valid_time_utc")
        price_steps = prices["valid_time_utc"].diff().dt.total_seconds()
        hourly_prices = prices.loc[price_steps.isna() | (price_steps >= 3600)].copy()
        hourly_prices["day_ahead_price_available_at_utc"] = _day_ahead_price_available_at(
            hourly_prices["valid_time_utc"]
        )

        canonical = actual.merge(forecast, on="valid_time_utc", how="outer")
        canonical = canonical.merge(hourly_prices, on="valid_time_utc", how="outer")
        canonical = canonical.merge(hourly_pv, on="valid_time_utc", how="outer")
        canonical = canonical.sort_values("valid_time_utc").reset_index(drop=True)
        calendar = _calendar_features(canonical["valid_time_utc"])
        canonical = pd.concat([canonical, calendar], axis=1)
        canonical["pv_source_profile"] = PV_PROFILE
        canonical["pv_timezone_assumption"] = "Europe/Berlin (provisional)"
        canonical["data_version"] = "energy-hourly-v1"
        local_date = canonical["delivery_date_local"]
        canonical.loc[local_date.isin(PV_OUTAGE_LOCAL_DATES), "pv_quality_status"] = "suspected_source_outage"
        return canonical

    @staticmethod
    def _base_columns() -> list[str]:
        return [
            "as_of_utc",
            "valid_time_utc",
            "valid_time_local",
            "delivery_date_local",
            "horizon_hours",
            "hour_local",
            "day_of_week",
            "day_of_year",
            "month",
            "is_weekend",
            "hour_sin",
            "hour_cos",
            "day_of_year_sin",
            "day_of_year_cos",
            "feature_version",
            "data_snapshot_id",
        ]

    def _build_day_ahead_pv(self, canonical: pd.DataFrame, year: int) -> pd.DataFrame:
        frame = canonical.loc[
            canonical["delivery_date_local"].str.startswith(str(year), na=False)
        ].copy()
        frame["as_of_utc"] = _day_ahead_as_of(frame["valid_time_utc"])
        frame["horizon_hours"] = (
            (frame["valid_time_utc"] - frame["as_of_utc"]).dt.total_seconds() / 3600
        ).astype("int16")
        frame = _select_available_day_ahead_weather(frame)
        frame["feature_version"] = "da-pv-v2"
        frame["data_snapshot_id"] = f"canonical-hourly-{year}-v2"
        frame["target_available_at_utc"] = frame["pv_available_at_utc"]
        forecast_columns = [f"weather_forecast_{variable}" for variable in WEATHER_VARIABLES]
        required = [
            "valid_time_utc",
            "as_of_utc",
            "pv_power_mean_w",
            "pv_energy_kwh",
            *forecast_columns,
        ]
        frame = frame.loc[frame["pv_quality_status"] == "usable"].dropna(subset=required)
        frame = add_week_index(frame)
        frame["forecast_shortwave_is_daylight"] = (
            frame["weather_forecast_shortwave_radiation"] > 0
        ).astype("int8")
        columns = self._base_columns() + ["week_index"] + forecast_columns + [
            "forecast_shortwave_is_daylight",
            "weather_forecast_lead_hours",
            "weather_forecast_available_at_utc",
            "target_available_at_utc",
            "pv_power_mean_w",
            "pv_energy_kwh",
            "pv_quality_status",
            "pv_source_profile",
            "pv_timezone_assumption",
        ]
        return frame[columns].sort_values("valid_time_utc").reset_index(drop=True)

    def _build_day_ahead_price(self, canonical: pd.DataFrame, year: int) -> pd.DataFrame:
        frame = canonical.sort_values("valid_time_utc").copy()
        frame["as_of_utc"] = _day_ahead_as_of(frame["valid_time_utc"])
        frame["horizon_hours"] = (
            (frame["valid_time_utc"] - frame["as_of_utc"]).dt.total_seconds() / 3600
        ).astype("int16")
        frame["price_lag_24h"] = frame["day_ahead_price_eur_per_mwh"].shift(24)
        frame["price_lag_168h"] = frame["day_ahead_price_eur_per_mwh"].shift(168)
        frame["price_lag_336h"] = frame["day_ahead_price_eur_per_mwh"].shift(336)
        frame["price_lag_504h"] = frame["day_ahead_price_eur_per_mwh"].shift(504)
        frame["price_lag_672h"] = frame["day_ahead_price_eur_per_mwh"].shift(672)
        frame["price_same_hour_4w_mean"] = frame[
            ["price_lag_168h", "price_lag_336h", "price_lag_504h", "price_lag_672h"]
        ].mean(axis=1)
        frame["price_lag_24h_available_at_utc"] = frame[
            "day_ahead_price_available_at_utc"
        ].shift(24)
        frame["price_lag_168h_available_at_utc"] = frame[
            "day_ahead_price_available_at_utc"
        ].shift(168)
        frame = _select_available_day_ahead_weather(frame)
        frame["feature_version"] = "da-price-v2"
        frame["data_snapshot_id"] = f"canonical-hourly-{year}-v2"
        frame["target_available_at_utc"] = frame["day_ahead_price_available_at_utc"]
        forecast_columns = [f"weather_forecast_{variable}" for variable in WEATHER_VARIABLES]
        frame = frame.loc[
            frame["delivery_date_local"].str.startswith(str(year), na=False)
        ]
        required = [
            "day_ahead_price_eur_per_mwh",
            "price_lag_24h",
            "price_lag_168h",
            *forecast_columns,
        ]
        frame = frame.dropna(subset=required)
        feature_available = (
            (frame["weather_forecast_available_at_utc"] <= frame["as_of_utc"])
            & (frame["price_lag_24h_available_at_utc"] <= frame["as_of_utc"])
            & (frame["price_lag_168h_available_at_utc"] <= frame["as_of_utc"])
        )
        frame = frame.loc[feature_available]
        columns = self._base_columns() + forecast_columns + [
            "weather_forecast_lead_hours",
            "weather_forecast_available_at_utc",
            "price_lag_24h",
            "price_lag_168h",
            "price_lag_336h",
            "price_lag_504h",
            "price_lag_672h",
            "price_same_hour_4w_mean",
            "price_lag_24h_available_at_utc",
            "price_lag_168h_available_at_utc",
            "target_available_at_utc",
            "day_ahead_price_eur_per_mwh",
        ]
        return frame[columns].sort_values("valid_time_utc").reset_index(drop=True)

    def _build_oracle_pv(
        self,
        canonical: pd.DataFrame,
        day_ahead_pv: pd.DataFrame,
        year: int,
    ) -> pd.DataFrame:
        """Materialize the actual-weather PV view with identical target rows."""

        actual_columns = [f"weather_actual_{variable}" for variable in WEATHER_VARIABLES]
        actual = canonical[
            [
                "valid_time_utc",
                *actual_columns,
                "weather_actual_available_at_utc",
            ]
        ]
        frame = day_ahead_pv.merge(actual, on="valid_time_utc", how="left", validate="one_to_one")
        frame["feature_version"] = "oracle-pv-v1"
        frame["data_snapshot_id"] = f"canonical-hourly-{year}-v2"
        required = ["pv_power_mean_w", "pv_energy_kwh", *actual_columns]
        frame = frame.dropna(subset=required)
        columns = self._base_columns() + ["week_index"] + actual_columns + [
            "weather_actual_available_at_utc",
            "target_available_at_utc",
            "pv_power_mean_w",
            "pv_energy_kwh",
            "pv_quality_status",
            "pv_source_profile",
            "pv_timezone_assumption",
        ]
        return frame[columns].sort_values("valid_time_utc").reset_index(drop=True)

    def _build_mpc_pv(self, canonical: pd.DataFrame, year: int) -> pd.DataFrame:
        hourly = canonical.sort_values("valid_time_utc").copy()
        outputs: list[pd.DataFrame] = []
        for horizon in range(1, 11):
            frame = hourly.copy()
            frame["as_of_utc"] = frame["valid_time_utc"] - pd.Timedelta(hours=horizon)
            frame["horizon_hours"] = horizon
            latest_observed_shift = horizon + 1
            frame["pv_last_observed_w"] = frame["pv_power_mean_w"].shift(latest_observed_shift)
            frame["pv_last_observation_available_at_utc"] = frame["pv_available_at_utc"].shift(
                latest_observed_shift
            )
            frame["pv_target_lag_24h_w"] = frame["pv_power_mean_w"].shift(24)
            frame["pv_target_lag_168h_w"] = frame["pv_power_mean_w"].shift(168)
            frame["feature_version"] = "mpc-pv-v1"
            frame["data_snapshot_id"] = f"canonical-hourly-{year}-v2"
            outputs.append(frame)
        result = pd.concat(outputs, ignore_index=True)
        result = result.loc[
            result["delivery_date_local"].str.startswith(str(year), na=False)
        ]
        result = result.loc[result["pv_quality_status"] == "usable"]
        required = [
            "pv_power_mean_w",
            "pv_energy_kwh",
            "pv_last_observed_w",
            "pv_target_lag_24h_w",
            "pv_target_lag_168h_w",
        ]
        result = result.dropna(subset=required)
        result = result.loc[
            result["pv_last_observation_available_at_utc"] <= result["as_of_utc"]
        ]
        columns = self._base_columns() + [
            "pv_last_observed_w",
            "pv_target_lag_24h_w",
            "pv_target_lag_168h_w",
            "pv_last_observation_available_at_utc",
            "pv_power_mean_w",
            "pv_energy_kwh",
            "pv_quality_status",
            "pv_source_profile",
            "pv_timezone_assumption",
        ]
        return result[columns].sort_values(["as_of_utc", "horizon_hours"]).reset_index(drop=True)

    @staticmethod
    def _write_duckdb(
        path: Path,
        canonical_path: Path,
        datasets: dict[str, Path],
        year: int,
    ) -> None:
        if path.exists():
            path.unlink()
        connection = duckdb.connect(str(path))
        try:
            canonical_sql_path = str(canonical_path).replace("'", "''")
            connection.execute(
                f"CREATE TABLE canonical_hourly_{year} AS "
                f"SELECT * FROM read_parquet('{canonical_sql_path}')"
            )
            for name, dataset_path in datasets.items():
                dataset_sql_path = str(dataset_path).replace("'", "''")
                connection.execute(
                    f"CREATE TABLE {name}_{year} AS "
                    f"SELECT * FROM read_parquet('{dataset_sql_path}')"
                )
        finally:
            connection.close()

    def _write_manifest(
        self,
        path: Path,
        canonical_path: Path,
        paths: dict[str, Path],
        frames: dict[str, pd.DataFrame],
        year: int,
    ) -> None:
        def dataset_entry(name: str, dataset_path: Path, frame: pd.DataFrame) -> dict:
            return {
                "path": str(dataset_path.relative_to(self.output_root)),
                "rows": int(len(frame)),
                "columns": list(frame.columns),
                "first_valid_time_utc": frame["valid_time_utc"].min().isoformat(),
                "last_valid_time_utc": frame["valid_time_utc"].max().isoformat(),
                "sha256": _file_sha256(dataset_path),
            }

        payload = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "year": year,
            "pipeline_version": "1.3.0",
            "canonical": {
                "path": str(canonical_path.relative_to(self.output_root)),
                "sha256": _file_sha256(canonical_path),
            },
            "datasets": {
                name: dataset_entry(name, paths[name], frames[name]) for name in paths
            },
            "unavailable_datasets": {
                "intraday_price": "No audited historical intraday price series is available."
            },
            "assumptions": {
                "pv_profile": PV_PROFILE,
                "pv_timezone": "Europe/Berlin (provisional)",
                "day_ahead_decision_time": "11:00 Europe/Berlin on D-1, one-hour submission buffer",
                "weather_forecast": "freshest point-in-time available ICON fixed-lead layer: previous_day1 (24h) or previous_day2 (48h)",
                "day_ahead_price_publication": "13:00 Europe/Berlin on D-1, modelling assumption used only for availability checks",
                "mpc_horizons_hours": list(range(1, 11)),
                "pv_outage_dates_excluded": sorted(PV_OUTAGE_LOCAL_DATES),
            },
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
