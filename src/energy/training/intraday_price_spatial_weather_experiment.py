"""Leakage-safe 2024 experiment for spatial IFS intraday-price features.

This is deliberately an experiment rather than a replacement for the deployed
intraday-price model.  It compares the existing price-only CatBoost correction
with an otherwise identical challenger that receives ten-location ECMWF IFS
weather *levels* and weather-forecast *revisions*.  The two forecasts are
selected from archived single runs according to their publication timestamps.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from energy.data.intraday_price import ENRICHED_FEATURES_RELATIVE_PATH, _feature_names
from energy.data.intraday_price_weather import (
    PROCESSED_RELATIVE_PATH as SPATIAL_IFS_WEATHER_RELATIVE_PATH,
    SPATIAL_WEATHER_VARIABLES,
)
from energy.data.splitting import SeasonalWeekKFold, add_week_index


TARGET_PRICE_COLUMN = "target_intraday_price_eur_per_mwh"
TARGET_SPREAD_COLUMN = "target_intraday_spread_eur_per_mwh"
DAY_AHEAD_COLUMN = "day_ahead_price_eur_per_mwh"
DA_BIDDING_HOUR_LOCAL = 11


@dataclass(frozen=True)
class IntradayPriceSpatialWeatherExperimentConfig:
    """Inputs for the first historical spatial-weather challenger."""

    project_root: Path
    data_path: Path | None = None
    weather_path: Path | None = None
    train_start: str = "2024-03-15"
    validation_start: str = "2024-10-01"
    end: str = "2024-12-31"
    correction_weight: float = 0.70
    iterations: int = 500
    random_state: int = 42
    n_splits: int = 3
    artifact_name: str = "intraday_price_spatial_ifs_2024"


@dataclass(frozen=True)
class IntradayPriceSpatialWeatherExperimentResult:
    """Persisted artifacts and the Q4 chronological validation metrics."""

    artifact_dir: Path
    metrics: pd.DataFrame
    rows: int
    weather_feature_names: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_dir": str(self.artifact_dir),
            "rows": self.rows,
            "weather_feature_names": list(self.weather_feature_names),
            "metrics": self.metrics.to_dict(orient="records"),
        }


@dataclass(frozen=True)
class IntradayPriceSpatialWeatherWeekCVResult:
    """Out-of-fold results from season-balanced, whole-week cross-validation."""

    artifact_dir: Path
    fold_metrics: pd.DataFrame
    summary_metrics: pd.DataFrame
    rows: int

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_dir": str(self.artifact_dir),
            "rows": self.rows,
            "fold_metrics": self.fold_metrics.to_dict(orient="records"),
            "summary_metrics": self.summary_metrics.to_dict(orient="records"),
        }


def run_intraday_price_spatial_weather_experiment(
    config: IntradayPriceSpatialWeatherExperimentConfig,
) -> IntradayPriceSpatialWeatherExperimentResult:
    """Fit price-only and spatial-IFS models, then evaluate 2024-Q4.

    The scope is intentionally only ``lead_hours == 1``.  That is the horizon
    at which the MPC policy currently uses an intraday price correction; it
    avoids letting rows for later horizons dominate a model whose operational
    purpose is the next delivery hour.
    """

    _validate_config(config)
    root = config.project_root.expanduser().resolve()
    base_path = _resolve_path(config.data_path, root / ENRICHED_FEATURES_RELATIVE_PATH)
    weather_path = _resolve_path(
        config.weather_path, root / SPATIAL_IFS_WEATHER_RELATIVE_PATH
    )
    frame = _prepare_base_frame(pd.read_parquet(base_path), config)
    enriched, weather_features, audit = attach_spatial_ifs_features(
        frame, pd.read_parquet(weather_path)
    )

    validation_start = pd.Timestamp(config.validation_start)
    development = enriched.loc[
        enriched["delivery_date_local"] < validation_start
    ].copy()
    validation = enriched.loc[
        enriched["delivery_date_local"] >= validation_start
    ].copy()
    if development.empty or validation.empty:
        raise ValueError("Development and validation partitions must both be non-empty")
    if development["target_valid_time_utc"].max() >= validation["as_of_utc"].min():
        raise AssertionError("Development and validation periods overlap")

    base_features = _feature_names("v2")
    weather_features = list(weather_features)
    revision_features = _revision_feature_names(weather_features)
    _assert_no_missing(enriched, [*base_features, *weather_features])

    price_only_model = _fit_model(development, base_features, config)
    revision_model = _fit_model(development, [*base_features, *revision_features], config)
    weather_model = _fit_model(development, [*base_features, *weather_features], config)

    actual = validation[TARGET_PRICE_COLUMN].to_numpy(dtype=float)
    day_ahead = validation[DAY_AHEAD_COLUMN].to_numpy(dtype=float)
    price_only_spread = price_only_model.predict(validation.loc[:, base_features])
    revision_spread = revision_model.predict(
        validation.loc[:, [*base_features, *revision_features]]
    )
    weather_spread = weather_model.predict(
        validation.loc[:, [*base_features, *weather_features]]
    )
    predictions = {
        "day_ahead_baseline": day_ahead,
        "price_only_correction": day_ahead
        + config.correction_weight * price_only_spread,
        "spatial_ifs_revision_only_correction": day_ahead
        + config.correction_weight * revision_spread,
        "spatial_ifs_weather_correction": day_ahead
        + config.correction_weight * weather_spread,
    }
    metrics = pd.DataFrame(
        [
            _metric_row(name, actual, prediction, config.correction_weight)
            for name, prediction in predictions.items()
        ]
    ).sort_values("mae_eur_per_mwh").reset_index(drop=True)

    artifact_dir = root / "artifacts" / "experiments" / config.artifact_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(artifact_dir / "metrics.csv", index=False)
    _prediction_frame(validation, actual, predictions).to_parquet(
        artifact_dir / "predictions.parquet", index=False
    )
    _feature_importance(price_only_model, base_features, "price_only").to_csv(
        artifact_dir / "feature_importance_price_only.csv", index=False
    )
    _feature_importance(
        revision_model, [*base_features, *revision_features], "spatial_ifs_revision_only"
    ).to_csv(artifact_dir / "feature_importance_spatial_ifs_revision_only.csv", index=False)
    _feature_importance(weather_model, [*base_features, *weather_features], "spatial_ifs").to_csv(
        artifact_dir / "feature_importance_spatial_ifs.csv", index=False
    )
    (artifact_dir / "experiment_config.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "project_root": str(root),
                "data_path": str(base_path),
                "weather_path": str(weather_path),
                "base_feature_names": base_features,
                "weather_feature_names": weather_features,
                "revision_feature_names": revision_features,
                "operational_scope": "lead_hours == 1 only",
                "selection": "2024 Q4 chronological validation; no untouched future holdout yet",
                "weather_contract": {
                    "day_ahead": (
                        "latest archived IFS run available by 11:00 Europe/Berlin on "
                        "the day before delivery"
                    ),
                    "intraday": "latest archived IFS run available at as_of_utc",
                    "revision": "intraday IFS aggregate minus day-ahead IFS aggregate",
                    "publication_delay_hours": 6,
                },
                "audit": audit,
            },
            indent=2,
            default=str,
        )
    )

    return IntradayPriceSpatialWeatherExperimentResult(
        artifact_dir=artifact_dir,
        metrics=metrics,
        rows=len(enriched),
        weather_feature_names=tuple(weather_features),
    )


def run_intraday_price_spatial_weather_week_cv(
    config: IntradayPriceSpatialWeatherExperimentConfig,
) -> IntradayPriceSpatialWeatherWeekCVResult:
    """Compare feature sets using randomized season-balanced *whole weeks*.

    This is a complementary feature-ablation protocol.  It preserves every
    row's original point-in-time input contract and keeps a calendar week
    intact, but it is deliberately not a simulation of a live deployment:
    each fitted fold can learn parameters from weeks later than its test weeks.
    The chronological experiment remains the gate for a production candidate.
    """

    _validate_config(config)
    root = config.project_root.expanduser().resolve()
    base_path = _resolve_path(config.data_path, root / ENRICHED_FEATURES_RELATIVE_PATH)
    weather_path = _resolve_path(
        config.weather_path, root / SPATIAL_IFS_WEATHER_RELATIVE_PATH
    )
    frame = _prepare_base_frame(pd.read_parquet(base_path), config)
    enriched, weather_features, audit = attach_spatial_ifs_features(
        frame, pd.read_parquet(weather_path)
    )
    enriched = add_week_index(enriched, time_column="target_valid_time_utc")
    base_features = _feature_names("v2")
    weather_features = list(weather_features)
    revision_features = _revision_feature_names(weather_features)
    _assert_no_missing(enriched, [*base_features, *weather_features])

    splitter = SeasonalWeekKFold(
        n_splits=config.n_splits,
        random_state=config.random_state,
        time_column="target_valid_time_utc",
    )
    fold_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    fold_assignments: list[dict[str, object]] = []
    for fold in splitter.split_with_metadata(enriched):
        development = enriched.iloc[fold.train_indices]
        test = enriched.iloc[fold.test_indices]
        price_only_model = _fit_model(development, base_features, config)
        revision_model = _fit_model(development, [*base_features, *revision_features], config)
        weather_model = _fit_model(development, [*base_features, *weather_features], config)
        actual = test[TARGET_PRICE_COLUMN].to_numpy(dtype=float)
        day_ahead = test[DAY_AHEAD_COLUMN].to_numpy(dtype=float)
        predictions = {
            "day_ahead_baseline": day_ahead,
            "price_only_correction": day_ahead
            + config.correction_weight * price_only_model.predict(test.loc[:, base_features]),
            "spatial_ifs_revision_only_correction": day_ahead
            + config.correction_weight
            * revision_model.predict(test.loc[:, [*base_features, *revision_features]]),
            "spatial_ifs_weather_correction": day_ahead
            + config.correction_weight
            * weather_model.predict(test.loc[:, [*base_features, *weather_features]]),
        }
        fold_rows.extend(
            {
                **_metric_row(name, actual, prediction, config.correction_weight),
                "fold": fold.fold_index,
                "test_weeks": len(fold.test_weeks),
                "train_weeks": len(fold.train_weeks),
            }
            for name, prediction in predictions.items()
        )
        fold_predictions = _prediction_frame(test, actual, predictions)
        fold_predictions.insert(1, "fold", fold.fold_index)
        fold_predictions.insert(
            2,
            "week_index",
            np.tile(test["week_index"].to_numpy(), len(predictions)),
        )
        prediction_frames.append(fold_predictions)
        fold_assignments.append(
            {
                "fold": fold.fold_index,
                "train_weeks": list(fold.train_weeks),
                "test_weeks": list(fold.test_weeks),
            }
        )

    fold_metrics = pd.DataFrame(fold_rows).sort_values(["variant", "fold"]).reset_index(drop=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = _week_cv_summary(predictions, fold_metrics)
    artifact_dir = root / "artifacts" / "experiments" / config.artifact_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(artifact_dir / "metrics_by_fold.csv", index=False)
    summary.to_csv(artifact_dir / "metrics_summary.csv", index=False)
    predictions.to_parquet(artifact_dir / "out_of_fold_predictions.parquet", index=False)
    (artifact_dir / "fold_assignments.json").write_text(json.dumps(fold_assignments, indent=2))
    (artifact_dir / "experiment_config.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "project_root": str(root),
                "data_path": str(base_path),
                "weather_path": str(weather_path),
                "base_feature_names": base_features,
                "weather_feature_names": weather_features,
                "revision_feature_names": revision_features,
                "evaluation_protocol": (
                    "season-balanced randomized whole-week cross-validation; "
                    "complementary feature-ablation evidence, not a deployment backtest"
                ),
                "operational_scope": "lead_hours == 1 only",
                "weather_contract": {
                    "day_ahead": (
                        "latest archived IFS run available by 11:00 Europe/Berlin on "
                        "the day before delivery"
                    ),
                    "intraday": "latest archived IFS run available at as_of_utc",
                    "publication_delay_hours": 6,
                },
                "audit": audit,
            },
            indent=2,
            default=str,
        )
    )
    return IntradayPriceSpatialWeatherWeekCVResult(
        artifact_dir=artifact_dir,
        fold_metrics=fold_metrics,
        summary_metrics=summary,
        rows=len(enriched),
    )


def attach_spatial_ifs_features(
    frame: pd.DataFrame, weather: pd.DataFrame
) -> tuple[pd.DataFrame, tuple[str, ...], dict[str, object]]:
    """Join two point-in-time IFS views and derive compact spatial aggregates.

    ``frame`` must have one row per decision/target pair.  The function is
    public so its availability checks can be tested independently of model
    training.
    """

    required_base = {"as_of_utc", "target_valid_time_utc", "delivery_date_local"}
    required_weather = {
        "location_id",
        "run_init_utc",
        "available_at_utc",
        "valid_time_utc",
        *[f"weather_forecast_{name}" for name in SPATIAL_WEATHER_VARIABLES],
    }
    missing_base = sorted(required_base.difference(frame.columns))
    missing_weather = sorted(required_weather.difference(weather.columns))
    if missing_base:
        raise KeyError(f"Base frame is missing required columns: {missing_base}")
    if missing_weather:
        raise KeyError(f"Spatial weather table is missing required columns: {missing_weather}")

    result = frame.copy().reset_index(drop=True)
    for column in ("as_of_utc", "target_valid_time_utc"):
        result[column] = pd.to_datetime(result[column], utc=True)
    result["delivery_date_local"] = pd.to_datetime(result["delivery_date_local"])
    source = weather.copy()
    for column in ("run_init_utc", "available_at_utc", "valid_time_utc"):
        source[column] = pd.to_datetime(source[column], utc=True)

    result["_row_id"] = np.arange(len(result), dtype=int)
    result["_day_ahead_as_of_utc"] = _day_ahead_as_of_utc(result["delivery_date_local"])
    locations = tuple(sorted(source["location_id"].dropna().unique()))
    if not locations:
        raise ValueError("Spatial weather table has no locations")

    da = _select_latest_weather(
        result, source, cutoff_column="_day_ahead_as_of_utc", expected_locations=locations
    )
    latest = _select_latest_weather(
        result, source, cutoff_column="as_of_utc", expected_locations=locations
    )
    _assert_availability(da, "_day_ahead_as_of_utc", "day-ahead IFS")
    _assert_availability(latest, "as_of_utc", "latest IFS")

    da_features = _aggregate_weather(da, prefix="ifs_day_ahead")
    latest_features = _aggregate_weather(latest, prefix="ifs_latest")
    weather_features = tuple([*da_features.columns, *latest_features.columns])
    enriched = (
        result.merge(da_features, left_on="_row_id", right_index=True, how="left")
        .merge(latest_features, left_on="_row_id", right_index=True, how="left")
    )
    revision_features: list[str] = []
    for variable in SPATIAL_WEATHER_VARIABLES:
        for statistic in ("mean", "std"):
            da_name = f"ifs_day_ahead_{variable}_{statistic}"
            latest_name = f"ifs_latest_{variable}_{statistic}"
            revision_name = f"ifs_revision_{variable}_{statistic}"
            enriched[revision_name] = enriched[latest_name] - enriched[da_name]
            revision_features.append(revision_name)
    weather_features = (*weather_features, *revision_features)

    audit = {
        "rows": len(enriched),
        "locations": list(locations),
        "locations_per_row": len(locations),
        "day_ahead_runs": int(da["run_init_utc"].nunique()),
        "latest_runs": int(latest["run_init_utc"].nunique()),
        "all_day_ahead_available_by_cutoff": True,
        "all_latest_available_by_decision": True,
        "weather_feature_count": len(weather_features),
    }
    return enriched.drop(columns=["_row_id", "_day_ahead_as_of_utc"]), weather_features, audit


def _prepare_base_frame(
    frame: pd.DataFrame, config: IntradayPriceSpatialWeatherExperimentConfig
) -> pd.DataFrame:
    required = {
        *_feature_names("v2"),
        TARGET_PRICE_COLUMN,
        TARGET_SPREAD_COLUMN,
        DAY_AHEAD_COLUMN,
        "delivery_date_local",
        "as_of_utc",
        "target_valid_time_utc",
        "target_available_at_utc",
        "intraday_history_available_at_utc",
        "day_ahead_price_available_at_utc",
        "lead_hours",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Intraday V2 table is missing required columns: {missing}")
    result = frame.copy()
    result["delivery_date_local"] = pd.to_datetime(result["delivery_date_local"])
    for column in (
        "as_of_utc",
        "target_valid_time_utc",
        "target_available_at_utc",
        "intraday_history_available_at_utc",
        "day_ahead_price_available_at_utc",
    ):
        result[column] = pd.to_datetime(result[column], utc=True)
    result = result.loc[
        (result["lead_hours"] == 1)
        & (result["delivery_date_local"] >= pd.Timestamp(config.train_start))
        & (result["delivery_date_local"] <= pd.Timestamp(config.end))
    ].copy()
    if result.empty:
        raise ValueError("No lead-one rows remain in the configured date range")
    if not (result["intraday_history_available_at_utc"] <= result["as_of_utc"]).all():
        raise AssertionError("Intraday history is unavailable at a decision time")
    if not (result["day_ahead_price_available_at_utc"] <= result["as_of_utc"]).all():
        raise AssertionError("Day-ahead price is unavailable at a decision time")
    if not (result["target_available_at_utc"] > result["as_of_utc"]).all():
        raise AssertionError("Intraday target is available at a decision time")
    _assert_no_missing(result, _feature_names("v2"))
    return result.sort_values(["delivery_date_local", "as_of_utc"]).reset_index(drop=True)


def _day_ahead_as_of_utc(delivery_dates: pd.Series) -> pd.Series:
    cutoff = (pd.to_datetime(delivery_dates) - pd.Timedelta(days=1)).dt.normalize()
    cutoff = cutoff + pd.Timedelta(hours=DA_BIDDING_HOUR_LOCAL)
    return cutoff.dt.tz_localize("Europe/Berlin").dt.tz_convert("UTC")


def _select_latest_weather(
    requests: pd.DataFrame,
    weather: pd.DataFrame,
    *,
    cutoff_column: str,
    expected_locations: tuple[str, ...],
) -> pd.DataFrame:
    request_columns = ["_row_id", "target_valid_time_utc", cutoff_column]
    candidates = requests.loc[:, request_columns].merge(
        weather,
        left_on="target_valid_time_utc",
        right_on="valid_time_utc",
        how="left",
        validate="many_to_many",
    )
    candidates = candidates.loc[
        candidates["available_at_utc"] <= candidates[cutoff_column]
    ].copy()
    if candidates.empty:
        raise ValueError(f"No weather runs are available for cutoff {cutoff_column}")
    selected = (
        candidates.sort_values(["_row_id", "location_id", "available_at_utc", "run_init_utc"])
        .drop_duplicates(["_row_id", "location_id"], keep="last")
        .reset_index(drop=True)
    )
    counts = selected.groupby("_row_id")["location_id"].nunique()
    incomplete = counts.loc[counts != len(expected_locations)]
    if not incomplete.empty or len(counts) != len(requests):
        missing_rows = sorted(set(requests["_row_id"]).difference(counts.index))
        raise ValueError(
            "Spatial IFS coverage is incomplete for one or more decision rows; "
            f"missing_rows={missing_rows[:10]}, incomplete_rows={incomplete.index.tolist()[:10]}"
        )
    if set(selected["location_id"].unique()) != set(expected_locations):
        raise AssertionError("Selected weather locations differ from archive locations")
    return selected


def _assert_availability(selected: pd.DataFrame, cutoff_column: str, label: str) -> None:
    if not (selected["available_at_utc"] <= selected[cutoff_column]).all():
        raise AssertionError(f"{label} uses a forecast published after its decision cutoff")
    if not (selected["valid_time_utc"] > selected[cutoff_column]).all():
        raise AssertionError(f"{label} selects weather for a time already in the past")


def _aggregate_weather(selected: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    source_columns = [f"weather_forecast_{variable}" for variable in SPATIAL_WEATHER_VARIABLES]
    aggregated = selected.groupby("_row_id", sort=True)[source_columns].agg(["mean", "std"])
    aggregated.columns = [
        f"{prefix}_{column.removeprefix('weather_forecast_')}_{statistic}"
        for column, statistic in aggregated.columns
    ]
    return aggregated


def _fit_model(
    frame: pd.DataFrame,
    features: list[str],
    config: IntradayPriceSpatialWeatherExperimentConfig,
) -> CatBoostRegressor:
    model = CatBoostRegressor(
        loss_function="Huber:delta=10",
        iterations=config.iterations,
        learning_rate=0.05,
        depth=4,
        l2_leaf_reg=50.0,
        random_seed=config.random_state,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(frame.loc[:, features], frame[TARGET_SPREAD_COLUMN])
    return model


def _metric_row(
    variant: str, actual: np.ndarray, prediction: np.ndarray, correction_weight: float
) -> dict[str, object]:
    return {
        "variant": variant,
        "rows": len(actual),
        "correction_weight": correction_weight if variant != "day_ahead_baseline" else 0.0,
        "mae_eur_per_mwh": float(mean_absolute_error(actual, prediction)),
        "rmse_eur_per_mwh": float(mean_squared_error(actual, prediction) ** 0.5),
    }


def _prediction_frame(
    validation: pd.DataFrame, actual: np.ndarray, predictions: dict[str, np.ndarray]
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for name, prediction in predictions.items():
        frames.append(
            pd.DataFrame(
                {
                    "variant": name,
                    "as_of_utc": validation["as_of_utc"].to_numpy(),
                    "target_valid_time_utc": validation["target_valid_time_utc"].to_numpy(),
                    "delivery_date_local": validation["delivery_date_local"].dt.strftime("%Y-%m-%d").to_numpy(),
                    "actual_intraday_price_eur_per_mwh": actual,
                    "prediction_eur_per_mwh": prediction,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _week_cv_summary(
    predictions: pd.DataFrame, fold_metrics: pd.DataFrame
) -> pd.DataFrame:
    """Report both fold dispersion and pooled out-of-fold error."""

    rows: list[dict[str, object]] = []
    for variant, group in predictions.groupby("variant", sort=True):
        actual = group["actual_intraday_price_eur_per_mwh"].to_numpy(dtype=float)
        prediction = group["prediction_eur_per_mwh"].to_numpy(dtype=float)
        per_fold = fold_metrics.loc[fold_metrics["variant"] == variant]
        rows.append(
            {
                "variant": variant,
                "oof_rows": len(group),
                "oof_mae_eur_per_mwh": float(mean_absolute_error(actual, prediction)),
                "oof_rmse_eur_per_mwh": float(mean_squared_error(actual, prediction) ** 0.5),
                "fold_mae_mean_eur_per_mwh": float(per_fold["mae_eur_per_mwh"].mean()),
                "fold_mae_std_eur_per_mwh": float(per_fold["mae_eur_per_mwh"].std(ddof=1)),
                "fold_rmse_mean_eur_per_mwh": float(per_fold["rmse_eur_per_mwh"].mean()),
                "fold_rmse_std_eur_per_mwh": float(per_fold["rmse_eur_per_mwh"].std(ddof=1)),
            }
        )
    return pd.DataFrame(rows).sort_values("oof_mae_eur_per_mwh").reset_index(drop=True)


def _feature_importance(
    model: CatBoostRegressor, features: list[str], model_scope: str
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_scope": model_scope,
            "feature": features,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)


def _revision_feature_names(weather_features: list[str]) -> list[str]:
    revision_features = [
        feature for feature in weather_features if feature.startswith("ifs_revision_")
    ]
    if len(revision_features) != 12:
        raise AssertionError(
            f"Expected twelve spatial IFS revision features, found {len(revision_features)}"
        )
    return revision_features


def _assert_no_missing(frame: pd.DataFrame, features: list[str] | tuple[str, ...]) -> None:
    missing = frame.loc[:, list(features)].isna().any()
    if missing.any():
        raise ValueError(f"Model features contain missing values: {missing[missing].index.tolist()}")


def _resolve_path(configured: Path | None, default: Path) -> Path:
    path = (configured or default).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Required input file was not found: {path}")
    return path


def _validate_config(config: IntradayPriceSpatialWeatherExperimentConfig) -> None:
    if pd.Timestamp(config.train_start) >= pd.Timestamp(config.validation_start):
        raise ValueError("train_start must be earlier than validation_start")
    if pd.Timestamp(config.validation_start) > pd.Timestamp(config.end):
        raise ValueError("validation_start must be on or before end")
    if not 0.0 <= config.correction_weight <= 1.0:
        raise ValueError("correction_weight must be between zero and one")
    if config.iterations < 1:
        raise ValueError("iterations must be positive")
    if config.n_splits < 2:
        raise ValueError("n_splits must be at least two")
