"""Reproducible training application for the day-ahead PV forecast model.

The registered model deliberately uses only data available when a day-ahead
forecast is made: calendar fields, deterministic solar geometry and forecast
weather. Actual weather and the Oracle proxy are excluded from this package.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from energy.data import SeasonalWeekSplitter


TARGET_COLUMN = "pv_power_mean_w"
MODEL_NAME = "day_ahead_pv_catboost"
WEATHER_FEATURES = (
    "weather_forecast_temperature_2m",
    "weather_forecast_shortwave_radiation",
    "weather_forecast_direct_radiation",
    "weather_forecast_diffuse_radiation",
    "weather_forecast_wind_speed_10m",
    "weather_forecast_cloud_cover",
)


@dataclass(frozen=True)
class PVDayAheadTrainingConfig:
    """Inputs and reproducibility settings for one PV training run."""

    project_root: Path
    data_path: Path | None = None
    year: int = 2024
    tracking_uri: str | None = None
    experiment_name: str | None = None
    hour_start: int = 6
    hour_end: int = 21
    min_clear_sky_ghi: float = 25.0
    random_state: int = 42
    registered_model_name: str | None = None


@dataclass(frozen=True)
class PVDayAheadTrainingResult:
    """Stable, serializable identifiers and held-out evaluation values."""

    run_id: str
    tracking_uri: str
    data_path: str
    model_uri: str
    registered_model_name: str | None
    registered_model_version: str | None
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def train_day_ahead_pv(config: PVDayAheadTrainingConfig) -> PVDayAheadTrainingResult:
    """Fit, evaluate, log and optionally register the deployable PV model.

    A season-balanced whole-week split leaves the test partition untouched by
    early stopping. The final model is refit on the train and validation rows
    with the iteration count selected on validation data only.
    """

    mlflow, CatBoostRegressor, infer_signature, pvlib = _training_dependencies()
    project_root = config.project_root.expanduser().resolve()
    data_path = _resolve_data_path(config, project_root)
    frame = pd.read_parquet(data_path)
    prepared, feature_names = _prepare_day_ahead_frame(frame, config, pvlib)

    split = SeasonalWeekSplitter(
        train_size=0.70,
        validation_size=0.15,
        test_size=0.15,
        random_state=config.random_state,
    ).split(prepared)
    _check_split(split)

    candidate = _make_model(CatBoostRegressor, config, use_best_model=True)
    candidate.fit(
        split.train.loc[:, feature_names],
        split.train[TARGET_COLUMN],
        eval_set=(split.validation.loc[:, feature_names], split.validation[TARGET_COLUMN]),
        verbose=False,
    )
    selected_iterations = max(1, candidate.get_best_iteration() + 1)
    validation_prediction = _predict_nonnegative(candidate, split.validation.loc[:, feature_names])

    fit_frame = pd.concat([split.train, split.validation], ignore_index=True)
    final_model = _make_model(
        CatBoostRegressor,
        config,
        iterations=selected_iterations,
        use_best_model=False,
    )
    final_model.fit(fit_frame.loc[:, feature_names], fit_frame[TARGET_COLUMN], verbose=False)
    test_prediction = _predict_nonnegative(final_model, split.test.loc[:, feature_names])

    reference_peak_w = float(prepared[TARGET_COLUMN].max())
    validation_metrics = _metrics(
        split.validation[TARGET_COLUMN], validation_prediction, reference_peak_w
    )
    test_metrics = _metrics(split.test[TARGET_COLUMN], test_prediction, reference_peak_w)
    metrics = {
        **{f"validation_{key}": value for key, value in validation_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
        "selected_iterations": float(selected_iterations),
        "reference_peak_power_w": reference_peak_w,
    }

    tracking_uri = _resolve_tracking_uri(config, project_root)
    experiment_name = config.experiment_name or os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "pv-day-ahead"
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    input_example = fit_frame.loc[:, feature_names].head(5)
    raw_example_prediction = final_model.predict(input_example)
    signature = infer_signature(input_example, raw_example_prediction)
    with mlflow.start_run(run_name=f"pv-day-ahead-{config.year}") as run:
        mlflow.log_params(
            {
                "year": config.year,
                "hour_start": config.hour_start,
                "hour_end": config.hour_end,
                "min_clear_sky_ghi_w_m2": config.min_clear_sky_ghi,
                "random_state": config.random_state,
                "iterations_selected_on_validation": selected_iterations,
                "catboost_loss": "RMSE",
                "catboost_learning_rate": 0.05,
                "catboost_depth": 6,
                "catboost_l2_leaf_reg": 10,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "model_scope": "deployable_day_ahead",
                "forecast_inputs_only": "true",
                "oracle_proxy": "excluded_after_oof_ablation",
                "split_strategy": "seasonal_whole_week",
                "postprocessing": "clip_predictions_at_zero",
            }
        )
        mlflow.log_dict(_feature_contract(config, feature_names), "feature_contract.json")
        mlflow.log_dict(_split_contract(split), "split_contract.json")
        mlflow.log_dict(metrics, "metrics.json")
        model_info = mlflow.catboost.log_model(
            final_model,
            name=MODEL_NAME,
            signature=signature,
            input_example=input_example,
            registered_model_name=config.registered_model_name,
        )

    return PVDayAheadTrainingResult(
        run_id=run.info.run_id,
        tracking_uri=tracking_uri,
        data_path=str(data_path),
        model_uri=f"runs:/{run.info.run_id}/{MODEL_NAME}",
        registered_model_name=config.registered_model_name,
        registered_model_version=_registered_version(model_info),
        metrics=metrics,
    )


def _training_dependencies():
    """Import the optional ML stack only when this application is invoked."""

    try:
        import mlflow
        import mlflow.catboost
        import pvlib
        from catboost import CatBoostRegressor
        from mlflow.models import infer_signature
    except ImportError as error:  # pragma: no cover - depends on caller environment
        raise RuntimeError(
            "Install the training extra first: python -m pip install -e '.[train]'"
        ) from error
    return mlflow, CatBoostRegressor, infer_signature, pvlib


def _resolve_data_path(config: PVDayAheadTrainingConfig, project_root: Path) -> Path:
    data_path = config.data_path or (
        project_root
        / "data"
        / "features"
        / "day_ahead_pv"
        / f"day_ahead_pv_{config.year}.parquet"
    )
    data_path = data_path.expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Day-ahead PV dataset was not found: {data_path}")
    return data_path


def _resolve_tracking_uri(config: PVDayAheadTrainingConfig, project_root: Path) -> str:
    if config.tracking_uri:
        return config.tracking_uri
    if environment_uri := os.getenv("MLFLOW_TRACKING_URI"):
        return environment_uri
    database_path = project_root / "mlflow" / "mlflow.db"
    database_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{database_path}"


def _prepare_day_ahead_frame(
    frame: pd.DataFrame,
    config: PVDayAheadTrainingConfig,
    pvlib: Any,
) -> tuple[pd.DataFrame, list[str]]:
    """Create deterministic geometry and forecast-only sky features."""

    required = {
        TARGET_COLUMN,
        "valid_time_utc",
        "week_index",
        "hour_local",
        "hour_sin",
        "day_of_year_sin",
        "horizon_hours",
        *WEATHER_FEATURES,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Day-ahead dataset is missing required columns: {missing}")
    if not 0 <= config.hour_start <= config.hour_end <= 23:
        raise ValueError("hour_start and hour_end must be local hours from 0 through 23")

    result = (
        frame.loc[frame["hour_local"].between(config.hour_start, config.hour_end)]
        .copy()
        .sort_values("valid_time_utc")
        .reset_index(drop=True)
    )
    if result.empty:
        raise ValueError("No rows remain after applying the local-hour filter")

    location = pvlib.location.Location(
        latitude=48.89,
        longitude=8.70,
        tz="Europe/Berlin",
    )
    times = pd.DatetimeIndex(pd.to_datetime(result["valid_time_utc"], utc=True))
    solar_position = location.get_solarposition(times)
    clear_sky = location.get_clearsky(times, model="ineichen")
    result["solar_elevation_deg"] = solar_position["apparent_elevation"].to_numpy()
    result["solar_azimuth_deg"] = solar_position["azimuth"].to_numpy()
    result["clear_sky_ghi_w_m2"] = clear_sky["ghi"].to_numpy()

    clear_sky_index = np.full(len(result), -999.0)
    clear_sky_ghi = result["clear_sky_ghi_w_m2"].to_numpy()
    valid_clear_sky = clear_sky_ghi >= config.min_clear_sky_ghi
    np.divide(
        result["weather_forecast_shortwave_radiation"].to_numpy(),
        clear_sky_ghi,
        out=clear_sky_index,
        where=valid_clear_sky,
    )
    result["forecast_clear_sky_index"] = clear_sky_index
    result["forecast_clear_sky_index_valid"] = valid_clear_sky.astype("int8")

    feature_names = [
        "hour_local",
        "hour_sin",
        "day_of_year_sin",
        "solar_elevation_deg",
        "solar_azimuth_deg",
        "clear_sky_ghi_w_m2",
        *WEATHER_FEATURES,
        "forecast_clear_sky_index",
        "forecast_clear_sky_index_valid",
        "horizon_hours",
    ]
    if result.loc[:, feature_names].isna().any().any():
        bad_columns = result.loc[:, feature_names].columns[
            result.loc[:, feature_names].isna().any()
        ].tolist()
        raise ValueError(f"Model features contain missing values: {bad_columns}")
    return result, feature_names


def _make_model(
    CatBoostRegressor: Any,
    config: PVDayAheadTrainingConfig,
    *,
    iterations: int = 500,
    use_best_model: bool,
) -> Any:
    return CatBoostRegressor(
        loss_function="RMSE",
        iterations=iterations,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=10,
        random_seed=config.random_state,
        use_best_model=use_best_model,
        early_stopping_rounds=100 if use_best_model else None,
        allow_writing_files=False,
        verbose=False,
    )


def _predict_nonnegative(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Apply the physical non-negativity rule used by the evaluation."""

    return np.maximum(model.predict(features), 0.0)


def _metrics(
    target: pd.Series,
    prediction: np.ndarray,
    reference_peak_w: float,
) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    mae_w = float(mean_absolute_error(target, prediction))
    rmse_w = float(mean_squared_error(target, prediction) ** 0.5)
    return {
        "mae_w": mae_w,
        "rmse_w": rmse_w,
        "nmae_observed_peak": mae_w / reference_peak_w,
        "nrmse_observed_peak": rmse_w / reference_peak_w,
    }


def _check_split(split: Any) -> None:
    if split.train.empty or split.validation.empty or split.test.empty:
        raise ValueError("The seasonal whole-week split produced an empty partition")
    partitions = [set(split.train_weeks), set(split.validation_weeks), set(split.test_weeks)]
    if any(
        left.intersection(right)
        for index, left in enumerate(partitions)
        for right in partitions[index + 1 :]
    ):
        raise AssertionError("Whole weeks must not appear in more than one partition")


def _feature_contract(config: PVDayAheadTrainingConfig, feature_names: list[str]) -> dict[str, Any]:
    return {
        "model_scope": "day_ahead_pv_forecast",
        "target": TARGET_COLUMN,
        "features": feature_names,
        "availability": "All features are available at day-ahead decision time.",
        "excluded_inputs": [
            "weather_actual_*",
            "actual PV output",
            "Oracle-model proxy",
        ],
        "solar_geometry": {
            "proxy_location": {
                "latitude": 48.89,
                "longitude": 8.70,
                "timezone": "Europe/Berlin",
            },
            "clear_sky_model": "pvlib Ineichen GHI",
        },
        "forecast_clear_sky_index": {
            "formula": "forecast shortwave radiation / clear-sky GHI",
            "minimum_clear_sky_ghi_w_m2": config.min_clear_sky_ghi,
            "below_threshold_value": -999.0,
            "validity_flag": "forecast_clear_sky_index_valid",
        },
        "prediction_postprocessing": "Apply max(prediction_w, 0) before use.",
    }


def _split_contract(split: Any) -> dict[str, Any]:
    return {
        "strategy": "season-balanced random allocation of complete local calendar weeks",
        "development_use": "validation selects the iteration count; test is held out",
        "train_weeks": list(split.train_weeks),
        "validation_weeks": list(split.validation_weeks),
        "test_weeks": list(split.test_weeks),
        "rows": {
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
        },
    }


def _registered_version(model_info: Any) -> str | None:
    version = getattr(model_info, "registered_model_version", None)
    return str(version) if version is not None else None
