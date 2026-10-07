"""Audited hourly DE-LU intraday-price data and MPC feature construction.

Energy-Charts republishes EPEX SPOT's delivery-hour aggregates.  The V1 target
is the Intraday Continuous Average Price, a realised volume-weighted price for
the delivery hour.  It is suitable for an offline, hourly settlement proxy,
but it is not a time-stamped executable quote or an order-book snapshot.
"""

from __future__ import annotations

import json
import ssl
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np
import pandas as pd


SOURCE_URL_TEMPLATE = (
    "https://www.energy-charts.info/charts/price_spot_market/data/de/year_{year}.json"
)
PROCESSED_RELATIVE_PATH = Path("data/processed/prices_intraday_continuous_de_lu_hourly.csv.gz")
FEATURES_RELATIVE_PATH = Path("data/features/intraday_price/intraday_price_mpc_2024_2025.parquet")
ENRICHED_FEATURES_RELATIVE_PATH = Path(
    "data/features/intraday_price/intraday_price_mpc_v2_2024_2025.parquet"
)
MANIFEST_RELATIVE_PATH = Path("data/metadata/intraday_price_v1_manifest.json")
TARGET_COLUMN = "intraday_continuous_avg_price_eur_per_mwh"
DAY_AHEAD_COLUMN = "day_ahead_price_eur_per_mwh"

_SOURCE_SERIES = {
    "Day Ahead Auction (DE-LU)": DAY_AHEAD_COLUMN,
    "Intraday Continuous Average Price (DE-LU)": TARGET_COLUMN,
    "Intraday Continuous ID1-Price (DE-LU)": "intraday_continuous_id1_price_eur_per_mwh",
    "Intraday Continuous ID3-Price (DE-LU)": "intraday_continuous_id3_price_eur_per_mwh",
}


@dataclass(frozen=True)
class IntradayPriceCollectionResult:
    """Written raw, processed and metadata paths for the source audit."""

    raw_paths: tuple[Path, ...]
    processed_path: Path
    manifest_path: Path
    rows: int

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["raw_paths"] = [str(path) for path in self.raw_paths]
        result["processed_path"] = str(self.processed_path)
        result["manifest_path"] = str(self.manifest_path)
        return result


@dataclass(frozen=True)
class IntradayPriceFeatureResult:
    """Materialised decision/target table and its feature contract."""

    dataset_path: Path
    manifest_path: Path
    rows: int
    feature_names: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["dataset_path"] = str(self.dataset_path)
        result["manifest_path"] = str(self.manifest_path)
        result["feature_names"] = list(self.feature_names)
        return result


def collect_intraday_price_data(
    project_root: Path,
    *,
    years: tuple[int, ...] = (2024, 2025),
) -> IntradayPriceCollectionResult:
    """Download and audit public hourly DE-LU continuous intraday indices."""

    root = project_root.expanduser().resolve()
    raw_dir = root / "data" / "raw"
    processed_path = root / PROCESSED_RELATIVE_PATH
    manifest_path = root / MANIFEST_RELATIVE_PATH
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    raw_paths: list[Path] = []
    frames: list[pd.DataFrame] = []
    source_urls: dict[str, str] = {}
    for year in years:
        url = SOURCE_URL_TEMPLATE.format(year=year)
        payload = _download_json(url)
        raw_path = raw_dir / f"prices_intraday_continuous_de_lu_energy_charts_{year}.json"
        raw_path.write_text(json.dumps(payload))
        raw_paths.append(raw_path)
        source_urls[str(year)] = url
        frames.append(_parse_source_payload(payload, source_url=url))

    combined = pd.concat(frames, ignore_index=True).sort_values("valid_time_utc")
    combined = combined.drop_duplicates("valid_time_utc", keep="last").reset_index(drop=True)
    _validate_hourly_source_frame(combined)
    combined.to_csv(processed_path, index=False, compression="gzip")

    manifest = {
        "dataset_version": "intraday-price-v1",
        "source": {
            "publisher": "Energy-Charts / Fraunhofer ISE",
            "market_origin": "EPEX SPOT as stated in Energy-Charts chart metadata",
            "urls": source_urls,
            "series": {
                "target": "Intraday Continuous Average Price (DE-LU)",
                "supplementary": [
                    "Intraday Continuous ID1-Price (DE-LU)",
                    "Intraday Continuous ID3-Price (DE-LU)",
                ],
                "day_ahead_anchor": "Day Ahead Auction (DE-LU)",
            },
            "resolution": "hourly delivery-period aggregates",
        },
        "rows": int(len(combined)),
        "coverage_utc": {
            "start": combined["valid_time_utc"].min().isoformat(),
            "end": combined["valid_time_utc"].max().isoformat(),
        },
        "target_contract": {
            "target": TARGET_COLUMN,
            "meaning": (
                "Realised volume-weighted continuous intraday average for a delivery hour. "
                "It is a V1 settlement proxy, not an executable quote at MPC decision time."
            ),
            "availability_assumption": (
                "A completed delivery hour is treated as observable at the beginning of the "
                "following hour. This is deliberately conservative and is not a claim about "
                "the public publication latency of Energy-Charts."
            ),
        },
        "quality_checks": {
            "unique_utc_timestamps": True,
            "intraday_target_and_index_series_complete": True,
            "day_ahead_anchor_complete_through": "2025-09-30",
            "hourly_step_seconds": 3600,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return IntradayPriceCollectionResult(
        raw_paths=tuple(raw_paths),
        processed_path=processed_path,
        manifest_path=manifest_path,
        rows=len(combined),
    )


def build_intraday_price_mpc_dataset(
    project_root: Path,
    *,
    source_path: Path | None = None,
    feature_version: str = "v1",
) -> IntradayPriceFeatureResult:
    """Build leakage-safe hourly MPC decision/target rows for intraday price.

    A row is one ``(as_of hour, future delivery hour)`` pair.  It predicts the
    intraday-vs-day-ahead spread.  Only continuous-market delivery prices from
    six completed hours before ``as_of`` are used as intraday observations.
    """

    root = project_root.expanduser().resolve()
    source = source_path or (root / PROCESSED_RELATIVE_PATH)
    source = source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(
            f"Intraday source table was not found: {source}. "
            "Run collect-intraday-price-data first."
        )
    dataset_path = root / _feature_dataset_relative_path(feature_version)
    manifest_path = root / MANIFEST_RELATIVE_PATH
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    source_frame = pd.read_csv(source)
    source_frame["valid_time_utc"] = pd.to_datetime(source_frame["valid_time_utc"], utc=True)
    rows = _make_mpc_rows(source_frame, feature_version=feature_version)
    feature_names = tuple(_feature_names(feature_version))
    _validate_mpc_rows(rows, feature_names)
    rows.to_parquet(dataset_path, index=False)

    existing_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    feature_dataset_metadata = {
        "path": str(dataset_path.relative_to(root)),
        "rows": int(len(rows)),
        "feature_names": list(feature_names),
        "target": "target_intraday_spread_eur_per_mwh",
        "prediction_formula": f"{DAY_AHEAD_COLUMN} + predicted target_intraday_spread_eur_per_mwh",
        "as_of_contract": (
            "At as_of_utc, six completed delivery-hour intraday values ending one hour "
            "before as_of are available. Targets are strictly later delivery hours of the "
            "same local day."
        ),
        "regular_delivery_days_only": True,
        "timezone": "Europe/Berlin",
    }
    existing_manifest.setdefault("feature_datasets", {})[feature_version] = feature_dataset_metadata
    if feature_version == "v1":
        existing_manifest["feature_dataset"] = feature_dataset_metadata
    manifest_path.write_text(json.dumps(existing_manifest, indent=2))
    return IntradayPriceFeatureResult(
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        rows=len(rows),
        feature_names=feature_names,
    )


def _download_json(url: str) -> list[dict[str, Any]]:
    try:
        import certifi
    except ImportError as error:  # pragma: no cover - caller environment dependent
        raise RuntimeError("Install certifi before downloading Energy-Charts data.") from error
    context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(url, timeout=60, context=context) as response:
        payload = json.load(response)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Energy-Charts returned an unexpected payload from {url}")
    return payload


def _parse_source_payload(payload: list[dict[str, Any]], *, source_url: str) -> pd.DataFrame:
    timestamps = payload[0].get("xAxisValues")
    if not isinstance(timestamps, list) or not timestamps:
        raise ValueError(f"Source payload from {source_url} has no xAxisValues")
    series: dict[str, list[float | None]] = {}
    for item in payload:
        name = _english_series_name(item.get("name"))
        if name in _SOURCE_SERIES:
            values = item.get("data")
            if not isinstance(values, list) or len(values) != len(timestamps):
                raise ValueError(f"Series {name!r} is not aligned to source timestamps")
            series[_SOURCE_SERIES[name]] = values
    missing = sorted(set(_SOURCE_SERIES.values()).difference(series))
    if missing:
        raise KeyError(f"Energy-Charts payload is missing required series: {missing}")
    result = pd.DataFrame(
        {
            "valid_time_utc": pd.to_datetime(timestamps, unit="ms", utc=True),
            **series,
        }
    )
    return result


def _english_series_name(value: Any) -> str:
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return str(value[0].get("en"))
    if isinstance(value, dict):
        return str(value.get("en"))
    return str(value)


def _validate_hourly_source_frame(frame: pd.DataFrame) -> None:
    required = {"valid_time_utc", *_SOURCE_SERIES.values()}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Intraday source table is missing columns: {missing}")
    if frame["valid_time_utc"].duplicated().any():
        raise ValueError("Intraday source timestamps are not unique")
    intraday_columns = [
        TARGET_COLUMN,
        "intraday_continuous_id1_price_eur_per_mwh",
        "intraday_continuous_id3_price_eur_per_mwh",
    ]
    if frame.loc[:, intraday_columns].isna().any().any():
        missing_columns = frame.loc[:, intraday_columns].columns[
            frame.loc[:, intraday_columns].isna().any()
        ].tolist()
        raise ValueError(f"Intraday source contains missing required values: {missing_columns}")
    steps = frame["valid_time_utc"].sort_values().diff().dropna().dt.total_seconds()
    if not steps.eq(3600).all():
        raise ValueError("Intraday source must be a continuous hourly series")


def _make_mpc_rows(source: pd.DataFrame, *, feature_version: str = "v1") -> pd.DataFrame:
    _validate_hourly_source_frame(source)
    frame = (
        source.dropna(subset=[DAY_AHEAD_COLUMN, TARGET_COLUMN])
        .copy()
        .sort_values("valid_time_utc")
        .reset_index(drop=True)
    )
    local = frame["valid_time_utc"].dt.tz_convert("Europe/Berlin")
    frame["delivery_date_local"] = local.dt.date
    frame["hour_local"] = local.dt.hour.astype("int8")
    frame["day_of_week"] = local.dt.dayofweek.astype("int8")
    frame["month"] = local.dt.month.astype("int8")
    frame["day_of_year"] = local.dt.dayofyear.astype("int16")
    frame["intraday_spread_eur_per_mwh"] = frame[TARGET_COLUMN] - frame[DAY_AHEAD_COLUMN]

    regular_days = _regular_local_days(frame)
    lookup = {
        (row.delivery_date_local, int(row.hour_local)): row
        for row in frame.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for delivery_day, day in frame.loc[frame["delivery_date_local"].isin(regular_days)].groupby(
        "delivery_date_local", sort=False
    ):
        day = day.sort_values("hour_local")
        for _, decision in day.iloc[:-1].iterrows():
            position = int(decision.name)
            recent = frame.iloc[position - 6 : position]
            if len(recent) != 6 or position < 6:
                continue
            if not _is_hourly_contiguous(recent["valid_time_utc"]):
                continue
            if not (recent["valid_time_utc"] < decision["valid_time_utc"]).all():
                raise AssertionError("A current intraday price entered the feature window")
            historical_window_d7 = _same_clock_history_window(recent, lookup, days_back=7)
            historical_window_d14 = _same_clock_history_window(recent, lookup, days_back=14)
            if historical_window_d7 is None or historical_window_d14 is None:
                continue
            future = day.loc[day["hour_local"] > decision["hour_local"]]
            for _, target in future.iterrows():
                target_date = target["delivery_date_local"]
                target_hour = int(target["hour_local"])
                d1 = lookup.get((target_date - timedelta(days=1), target_hour))
                d7 = lookup.get((target_date - timedelta(days=7), target_hour))
                d14 = lookup.get((target_date - timedelta(days=14), target_hour))
                if d1 is None or d7 is None or d14 is None:
                    continue
                rows.append(
                    _make_feature_row(
                        decision=decision,
                        target=target,
                        day=day,
                        recent=recent,
                        historical_window_d7=historical_window_d7,
                        historical_window_d14=historical_window_d14,
                        d1=d1,
                        d7=d7,
                        d14=d14,
                        feature_version=feature_version,
                    )
                )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No intraday MPC rows were created")
    return result


def _regular_local_days(frame: pd.DataFrame) -> set[date]:
    hours_by_day = frame.groupby("delivery_date_local")["hour_local"].agg(list)
    return {
        delivery_day
        for delivery_day, hours in hours_by_day.items()
        if hours == list(range(24))
    }


def _is_hourly_contiguous(times: pd.Series) -> bool:
    return bool(times.diff().dropna().eq(pd.Timedelta(hours=1)).all())


def _same_clock_history_window(
    recent: pd.DataFrame,
    lookup: dict[tuple[date, int], Any],
    *,
    days_back: int,
) -> list[Any] | None:
    historic: list[Any] = []
    for row in recent.itertuples(index=False):
        matched = lookup.get((row.delivery_date_local - timedelta(days=days_back), int(row.hour_local)))
        if matched is None:
            return None
        historic.append(matched)
    return historic


def _make_feature_row(
    *,
    decision: pd.Series,
    target: pd.Series,
    day: pd.DataFrame,
    recent: pd.DataFrame,
    historical_window_d7: list[Any],
    historical_window_d14: list[Any],
    d1: Any,
    d7: Any,
    d14: Any,
    feature_version: str,
) -> dict[str, Any]:
    recent_prices = recent[TARGET_COLUMN].to_numpy(dtype=float)
    recent_spreads = recent["intraday_spread_eur_per_mwh"].to_numpy(dtype=float)
    day_ahead_curve = day[DAY_AHEAD_COLUMN].to_numpy(dtype=float)
    row: dict[str, Any] = {
        "as_of_utc": decision["valid_time_utc"],
        "target_valid_time_utc": target["valid_time_utc"],
        "target_available_at_utc": target["valid_time_utc"] + pd.Timedelta(hours=1),
        "intraday_history_available_at_utc": decision["valid_time_utc"],
        "day_ahead_price_available_at_utc": _day_ahead_available_at(target["delivery_date_local"]),
        "delivery_date_local": target["delivery_date_local"],
        "decision_hour_local": int(decision["hour_local"]),
        "target_hour_local": int(target["hour_local"]),
        "lead_hours": int(target["hour_local"] - decision["hour_local"]),
        "target_day_of_week": int(target["day_of_week"]),
        "target_month": int(target["month"]),
        "target_day_of_year": int(target["day_of_year"]),
        DAY_AHEAD_COLUMN: float(target[DAY_AHEAD_COLUMN]),
        "day_ahead_day_mean_eur_per_mwh": float(day_ahead_curve.mean()),
        "day_ahead_day_std_eur_per_mwh": float(day_ahead_curve.std(ddof=0)),
        "day_ahead_day_min_eur_per_mwh": float(day_ahead_curve.min()),
        "day_ahead_day_max_eur_per_mwh": float(day_ahead_curve.max()),
        "day_ahead_target_minus_day_mean_eur_per_mwh": float(
            target[DAY_AHEAD_COLUMN] - day_ahead_curve.mean()
        ),
        "intraday_last6_mean_eur_per_mwh": float(recent_prices.mean()),
        "intraday_last6_std_eur_per_mwh": float(recent_prices.std(ddof=0)),
        "intraday_last6_min_eur_per_mwh": float(recent_prices.min()),
        "intraday_last6_max_eur_per_mwh": float(recent_prices.max()),
        "intraday_last6_momentum_eur_per_mwh": float(recent_prices[-1] - recent_prices[-2]),
        "spread_last6_mean_eur_per_mwh": float(recent_spreads.mean()),
        "spread_last6_std_eur_per_mwh": float(recent_spreads.std(ddof=0)),
        "spread_last6_momentum_eur_per_mwh": float(recent_spreads[-1] - recent_spreads[-2]),
        "intraday_d1_same_hour_eur_per_mwh": float(getattr(d1, TARGET_COLUMN)),
        "intraday_d7_same_hour_eur_per_mwh": float(getattr(d7, TARGET_COLUMN)),
        "intraday_d14_same_hour_eur_per_mwh": float(getattr(d14, TARGET_COLUMN)),
        "spread_d1_same_hour_eur_per_mwh": float(getattr(d1, "intraday_spread_eur_per_mwh")),
        "spread_d7_same_hour_eur_per_mwh": float(getattr(d7, "intraday_spread_eur_per_mwh")),
        "spread_d14_same_hour_eur_per_mwh": float(getattr(d14, "intraday_spread_eur_per_mwh")),
        "target_intraday_price_eur_per_mwh": float(target[TARGET_COLUMN]),
        "target_intraday_spread_eur_per_mwh": float(target["intraday_spread_eur_per_mwh"]),
    }
    for lag, (_, observation) in enumerate(recent.iloc[::-1].iterrows(), start=1):
        row[f"intraday_price_lag_{lag}_eur_per_mwh"] = float(observation[TARGET_COLUMN])
        row[f"intraday_spread_lag_{lag}_eur_per_mwh"] = float(
            observation["intraday_spread_eur_per_mwh"]
        )
    historic_prices_d7 = np.asarray(
        [getattr(item, TARGET_COLUMN) for item in historical_window_d7], dtype=float
    )
    historic_spreads_d7 = np.asarray(
        [getattr(item, "intraday_spread_eur_per_mwh") for item in historical_window_d7],
        dtype=float,
    )
    row["intraday_last6_mean_minus_d7_eur_per_mwh"] = float(
        recent_prices.mean() - historic_prices_d7.mean()
    )
    row["spread_last6_mean_minus_d7_eur_per_mwh"] = float(
        recent_spreads.mean() - historic_spreads_d7.mean()
    )
    row["intraday_last6_d7_available"] = 1
    if feature_version == "v2":
        _add_enriched_features(
            row=row,
            decision=decision,
            target=target,
            recent_prices=recent_prices,
            recent_spreads=recent_spreads,
            historic_prices_d7=historic_prices_d7,
            historical_window_d14=historical_window_d14,
        )
    return row


def _day_ahead_available_at(delivery_day: date) -> pd.Timestamp:
    local = pd.Timestamp(delivery_day).tz_localize("Europe/Berlin") - pd.Timedelta(days=1)
    return (local + pd.Timedelta(hours=13)).tz_convert("UTC")


def _feature_dataset_relative_path(feature_version: str) -> Path:
    paths = {
        "v1": FEATURES_RELATIVE_PATH,
        "v2": ENRICHED_FEATURES_RELATIVE_PATH,
    }
    try:
        return paths[feature_version]
    except KeyError as error:
        raise ValueError("feature_version must be 'v1' or 'v2'") from error


def _add_enriched_features(
    *,
    row: dict[str, Any],
    decision: pd.Series,
    target: pd.Series,
    recent_prices: np.ndarray,
    recent_spreads: np.ndarray,
    historic_prices_d7: np.ndarray,
    historical_window_d14: list[Any],
) -> None:
    """Add small-window statistics and local AR(1) features for the V2 trial.

    The AR feature is an ARIMA(1,0,0) forecast estimated only from the six
    completed delivery periods. It takes ``lead_hours + 1`` steps because the
    current delivery period has not completed at ``as_of``.
    """

    historic_prices_d14 = np.asarray(
        [getattr(item, TARGET_COLUMN) for item in historical_window_d14], dtype=float
    )
    steps_from_last_observation = int(target["hour_local"] - decision["hour_local"]) + 1
    price_mean = float(recent_prices.mean())
    spread_mean = float(recent_spreads.mean())
    row.update(
        {
            "intraday_last6_median_eur_per_mwh": float(np.median(recent_prices)),
            "intraday_last6_iqr_eur_per_mwh": float(
                np.quantile(recent_prices, 0.75) - np.quantile(recent_prices, 0.25)
            ),
            "intraday_last6_abs_change_mean_eur_per_mwh": float(
                np.abs(np.diff(recent_prices)).mean()
            ),
            "intraday_last6_slope_eur_per_mwh_per_hour": _linear_slope(recent_prices),
            "intraday_last_price_minus_mean_eur_per_mwh": float(recent_prices[-1] - price_mean),
            "spread_last6_median_eur_per_mwh": float(np.median(recent_spreads)),
            "spread_last6_abs_change_mean_eur_per_mwh": float(
                np.abs(np.diff(recent_spreads)).mean()
            ),
            "spread_last6_slope_eur_per_mwh_per_hour": _linear_slope(recent_spreads),
            "spread_last_value_minus_mean_eur_per_mwh": float(recent_spreads[-1] - spread_mean),
            "intraday_last6_mean_d7_eur_per_mwh": float(historic_prices_d7.mean()),
            "intraday_last6_std_d7_eur_per_mwh": float(historic_prices_d7.std(ddof=0)),
            "intraday_last6_mean_d14_eur_per_mwh": float(historic_prices_d14.mean()),
            "intraday_last6_std_d14_eur_per_mwh": float(historic_prices_d14.std(ddof=0)),
            "intraday_last6_mean_minus_d14_eur_per_mwh": float(price_mean - historic_prices_d14.mean()),
            "intraday_last6_mean_ratio_d7": _ratio_or_sentinel(
                price_mean, float(historic_prices_d7.mean())
            ),
            "intraday_last6_mean_ratio_d14": _ratio_or_sentinel(
                price_mean, float(historic_prices_d14.mean())
            ),
            "intraday_last6_mean_ratio_d7_valid": int(not np.isclose(historic_prices_d7.mean(), 0.0)),
            "intraday_last6_mean_ratio_d14_valid": int(not np.isclose(historic_prices_d14.mean(), 0.0)),
            "intraday_ar1_forecast_eur_per_mwh": _ar1_forecast(
                recent_prices, steps=steps_from_last_observation
            ),
            "spread_ar1_forecast_eur_per_mwh": _ar1_forecast(
                recent_spreads, steps=steps_from_last_observation
            ),
            "day_ahead_target_minus_decision_eur_per_mwh": float(
                target[DAY_AHEAD_COLUMN] - decision[DAY_AHEAD_COLUMN]
            ),
        }
    )


def _linear_slope(values: np.ndarray) -> float:
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, values, deg=1)[0])


def _ratio_or_sentinel(numerator: float, denominator: float) -> float:
    """Retain the requested -99 sentinel for an exactly zero reference mean."""

    return -99.0 if np.isclose(denominator, 0.0) else float(numerator / denominator)


def _ar1_forecast(values: np.ndarray, *, steps: int) -> float:
    """Closed-form ARIMA(1,0,0) forecast with stable coefficient clipping."""

    x = values[:-1]
    y = values[1:]
    x_centered = x - x.mean()
    denominator = float(np.dot(x_centered, x_centered))
    if np.isclose(denominator, 0.0):
        return float(values.mean())
    phi = float(np.dot(x_centered, y - y.mean()) / denominator)
    phi = float(np.clip(phi, -0.98, 0.98))
    intercept = float(y.mean() - phi * x.mean())
    forecast = float(values[-1])
    for _ in range(steps):
        forecast = intercept + phi * forecast
    return forecast


def _feature_names(feature_version: str = "v1") -> list[str]:
    base = [
        "decision_hour_local",
        "target_hour_local",
        "lead_hours",
        "target_day_of_week",
        "target_month",
        "target_day_of_year",
        DAY_AHEAD_COLUMN,
        "day_ahead_day_mean_eur_per_mwh",
        "day_ahead_day_std_eur_per_mwh",
        "day_ahead_day_min_eur_per_mwh",
        "day_ahead_day_max_eur_per_mwh",
        "day_ahead_target_minus_day_mean_eur_per_mwh",
        "intraday_last6_mean_eur_per_mwh",
        "intraday_last6_std_eur_per_mwh",
        "intraday_last6_min_eur_per_mwh",
        "intraday_last6_max_eur_per_mwh",
        "intraday_last6_momentum_eur_per_mwh",
        "spread_last6_mean_eur_per_mwh",
        "spread_last6_std_eur_per_mwh",
        "spread_last6_momentum_eur_per_mwh",
        "intraday_d1_same_hour_eur_per_mwh",
        "intraday_d7_same_hour_eur_per_mwh",
        "intraday_d14_same_hour_eur_per_mwh",
        "spread_d1_same_hour_eur_per_mwh",
        "spread_d7_same_hour_eur_per_mwh",
        "spread_d14_same_hour_eur_per_mwh",
        "intraday_last6_mean_minus_d7_eur_per_mwh",
        "spread_last6_mean_minus_d7_eur_per_mwh",
        "intraday_last6_d7_available",
        *[f"intraday_price_lag_{lag}_eur_per_mwh" for lag in range(1, 7)],
        *[f"intraday_spread_lag_{lag}_eur_per_mwh" for lag in range(1, 7)],
    ]
    if feature_version == "v1":
        return base
    if feature_version != "v2":
        raise ValueError("feature_version must be 'v1' or 'v2'")
    return [
        *base,
        "intraday_last6_median_eur_per_mwh",
        "intraday_last6_iqr_eur_per_mwh",
        "intraday_last6_abs_change_mean_eur_per_mwh",
        "intraday_last6_slope_eur_per_mwh_per_hour",
        "intraday_last_price_minus_mean_eur_per_mwh",
        "spread_last6_median_eur_per_mwh",
        "spread_last6_abs_change_mean_eur_per_mwh",
        "spread_last6_slope_eur_per_mwh_per_hour",
        "spread_last_value_minus_mean_eur_per_mwh",
        "intraday_last6_mean_d7_eur_per_mwh",
        "intraday_last6_std_d7_eur_per_mwh",
        "intraday_last6_mean_d14_eur_per_mwh",
        "intraday_last6_std_d14_eur_per_mwh",
        "intraday_last6_mean_minus_d14_eur_per_mwh",
        "intraday_last6_mean_ratio_d7",
        "intraday_last6_mean_ratio_d14",
        "intraday_last6_mean_ratio_d7_valid",
        "intraday_last6_mean_ratio_d14_valid",
        "intraday_ar1_forecast_eur_per_mwh",
        "spread_ar1_forecast_eur_per_mwh",
        "day_ahead_target_minus_decision_eur_per_mwh",
    ]


def _validate_mpc_rows(rows: pd.DataFrame, feature_names: tuple[str, ...]) -> None:
    if rows.loc[:, list(feature_names)].isna().any().any():
        missing = rows.loc[:, list(feature_names)].columns[
            rows.loc[:, list(feature_names)].isna().any()
        ].tolist()
        raise ValueError(f"Intraday model features contain missing values: {missing}")
    if not (rows["intraday_history_available_at_utc"] <= rows["as_of_utc"]).all():
        raise AssertionError("Intraday history is not available at its decision time")
    if not (rows["day_ahead_price_available_at_utc"] <= rows["as_of_utc"]).all():
        raise AssertionError("Day-ahead price is not available at its decision time")
    if not (rows["target_available_at_utc"] > rows["as_of_utc"]).all():
        raise AssertionError("Intraday price target is available at or before its decision time")
    if not (rows["target_valid_time_utc"] > rows["as_of_utc"]).all():
        raise AssertionError("MPC rows must forecast strictly future delivery hours")
    if not rows["lead_hours"].between(1, 23).all():
        raise AssertionError("MPC lead hours must be between 1 and 23")
