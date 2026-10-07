"""Training application for direct residual correction of the PV day-ahead forecast.

The registered model predicts a point residual for one future solar-active hour.
At a live MPC decision, it is called once per remaining hour. Its input includes
the frozen day-ahead prediction and factual PV / residual history that is known
strictly before the decision hour.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from energy.data import SeasonalWeekKFold
from energy.data.builder import WEATHER_VARIABLES


TARGET_COLUMN = "pv_power_mean_w"
DAY_AHEAD_PREDICTION = "day_ahead_prediction_w"
RESIDUAL_TARGET = "target_residual_w"
MPC_MODEL_NAME = "pv_mpc_direct_residual_catboost"
MINIMUM_ACTIVE_GHI = 25.0
UPDATED_WEATHER_FEATURE_MODES = ("replace", "revision", "legacy")

FORECAST_FEATURES = (
    "hour_local",
    "hour_sin",
    "day_of_year_sin",
    "solar_elevation_deg",
    "solar_azimuth_deg",
    "clear_sky_ghi_w_m2",
    "weather_forecast_temperature_2m",
    "weather_forecast_shortwave_radiation",
    "weather_forecast_direct_radiation",
    "weather_forecast_diffuse_radiation",
    "weather_forecast_wind_speed_10m",
    "weather_forecast_cloud_cover",
    "forecast_clear_sky_index",
    "forecast_clear_sky_index_valid",
    "horizon_hours",
)


@dataclass(frozen=True)
class PVMpcResidualTrainingConfig:
    """Inputs and reproducibility settings for direct residual-model training."""

    project_root: Path
    data_path: Path | None = None
    weather_snapshot_path: Path | None = None
    year: int = 2024
    train_start_date: str | None = None
    tracking_uri: str | None = None
    experiment_name: str | None = None
    registered_model_name: str | None = None
    n_lags: int = 4
    min_clear_sky_ghi: float = MINIMUM_ACTIVE_GHI
    cv_splits: int = 3
    iterations: int = 500
    random_state: int = 42
    updated_weather_feature_mode: str = "replace"


@dataclass(frozen=True)
class PVMpcResidualTrainingResult:
    """MLflow identifiers and cross-validated MPC trajectory metrics."""

    run_id: str
    tracking_uri: str
    data_path: str
    model_uri: str
    registered_model_name: str | None
    registered_model_version: str | None
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def train_pv_mpc_residual(
    config: PVMpcResidualTrainingConfig,
) -> PVMpcResidualTrainingResult:
    """Evaluate and fit a deployable direct residual-correction CatBoost model.

    The outer folds measure the full two-stage pipeline. The day-ahead baseline
    is fitted without each outer test fold; inside outer training, its
    predictions are out-of-fold before they become residual-model inputs.
    Afterwards, a final residual model is fitted from global OOF baseline
    predictions and logged to MLflow.
    """

    mlflow, CatBoostRegressor, infer_signature, pvlib = _training_dependencies()
    _validate_config(config)
    project_root = config.project_root.expanduser().resolve()
    data_path = _resolve_data_path(config, project_root)
    full_frame = pd.read_parquet(data_path)
    active_frame = _prepare_active_frame(full_frame, config, pvlib)
    weather_snapshots = _load_weather_snapshots(config, project_root)
    uses_updated_weather = weather_snapshots is not None
    residual_features = _residual_features(
        config.n_lags,
        uses_updated_weather=uses_updated_weather,
        updated_weather_feature_mode=config.updated_weather_feature_mode,
    )

    fold_metrics, fold_contracts = _evaluate_outer_folds(
        active_frame,
        residual_features,
        config,
        CatBoostRegressor,
        weather_snapshots,
    )
    metrics = _aggregate_metrics(fold_metrics)

    # Global OOF predictions retain realistic day-ahead errors for final
    # residual training; fitting the baseline in-sample would shrink them.
    global_oof = _add_oof_day_ahead_prediction(
        active_frame,
        config,
        CatBoostRegressor,
        random_state=config.random_state,
    )
    final_training_rows = _make_direct_residual_rows(
        global_oof,
        config,
        weather_snapshots=weather_snapshots,
    )
    final_model = _make_model(CatBoostRegressor, config)
    final_model.fit(
        final_training_rows.loc[:, residual_features],
        final_training_rows[RESIDUAL_TARGET],
        verbose=False,
    )

    tracking_uri = _resolve_tracking_uri(config, project_root)
    experiment_name = config.experiment_name or os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "pv-mpc-residual"
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    input_example = final_training_rows.loc[:, residual_features].head(5)
    signature = infer_signature(input_example, final_model.predict(input_example))
    with mlflow.start_run(run_name=f"pv-mpc-residual-{config.year}") as run:
        mlflow.log_params(
            {
                "year": config.year,
                "n_lags": config.n_lags,
                "min_clear_sky_ghi_w_m2": config.min_clear_sky_ghi,
                "cv_splits": config.cv_splits,
                "random_state": config.random_state,
                "catboost_iterations": config.iterations,
                "catboost_learning_rate": 0.05,
                "catboost_depth": 6,
                "catboost_l2_leaf_reg": 10,
                "weather_snapshot_path": str(config.weather_snapshot_path) if config.weather_snapshot_path else "",
                "updated_weather_feature_mode": config.updated_weather_feature_mode,
                "train_start_date": config.train_start_date or "",
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "model_scope": "mpc_direct_residual_correction",
                "forecast_inputs_only": "true",
                "actual_pv_cutoff": "strictly_before_decision_hour",
                "weather_snapshot": "updated_ecmwf_ifs" if uses_updated_weather else "day_ahead_v1",
                "prediction_postprocessing": "clip_day_ahead_plus_residual_at_zero",
                "base_forecast_contract": "frozen_day_ahead_prediction_required",
            }
        )
        mlflow.log_dict(
            _feature_contract(config, residual_features, uses_updated_weather), "feature_contract.json"
        )
        mlflow.log_dict(
            {"outer_folds": fold_contracts}, "split_contract.json"
        )
        mlflow.log_dict({"fold_metrics": fold_metrics}, "fold_metrics.json")
        mlflow.log_dict(metrics, "metrics.json")
        mlflow.log_dict(
            _feature_importance(final_model, residual_features),
            "feature_importance.json",
        )
        model_info = mlflow.catboost.log_model(
            final_model,
            name=MPC_MODEL_NAME,
            signature=signature,
            input_example=input_example,
            registered_model_name=config.registered_model_name,
        )

    return PVMpcResidualTrainingResult(
        run_id=run.info.run_id,
        tracking_uri=tracking_uri,
        data_path=str(data_path),
        model_uri=f"runs:/{run.info.run_id}/{MPC_MODEL_NAME}",
        registered_model_name=config.registered_model_name,
        registered_model_version=_registered_version(model_info),
        metrics=metrics,
    )


def _training_dependencies():
    try:
        import mlflow
        import mlflow.catboost
        import pvlib
        from catboost import CatBoostRegressor
        from mlflow.models import infer_signature
    except ImportError as error:  # pragma: no cover - caller environment dependent
        raise RuntimeError(
            "Install the training extra first: python -m pip install -e '.[train]'"
        ) from error
    return mlflow, CatBoostRegressor, infer_signature, pvlib


def _validate_config(config: PVMpcResidualTrainingConfig) -> None:
    if config.n_lags < 1:
        raise ValueError("n_lags must be at least 1")
    if config.cv_splits < 2:
        raise ValueError("cv_splits must be at least 2")
    if config.iterations < 1:
        raise ValueError("iterations must be at least 1")
    if config.min_clear_sky_ghi <= 0:
        raise ValueError("min_clear_sky_ghi must be positive")
    if config.updated_weather_feature_mode not in UPDATED_WEATHER_FEATURE_MODES:
        raise ValueError(
            "updated_weather_feature_mode must be one of "
            f"{UPDATED_WEATHER_FEATURE_MODES}"
        )
    if config.train_start_date is not None:
        try:
            pd.Timestamp(config.train_start_date)
        except ValueError as error:
            raise ValueError("train_start_date must be an ISO-8601 local date") from error


def _resolve_data_path(config: PVMpcResidualTrainingConfig, project_root: Path) -> Path:
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


def _load_weather_snapshots(
    config: PVMpcResidualTrainingConfig,
    project_root: Path,
) -> pd.DataFrame | None:
    """Load a separately materialised weather snapshot without changing D-1 inputs."""

    if config.weather_snapshot_path is None:
        return None
    path = config.weather_snapshot_path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"MPC weather snapshot dataset was not found: {path}")
    frame = pd.read_parquet(path)
    weather_columns = [f"weather_forecast_{variable}" for variable in WEATHER_VARIABLES]
    required = {
        "delivery_date_local",
        "valid_time_utc",
        "as_of_utc",
        "decision_hour_local",
        "weather_snapshot_available_at_utc",
        "weather_snapshot_lead_hours",
        "weather_snapshot_age_hours",
        *weather_columns,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"MPC weather snapshot dataset is missing columns: {missing}")
    result = frame.copy()
    for column in (
        "valid_time_utc",
        "as_of_utc",
        "weather_snapshot_available_at_utc",
    ):
        result[column] = pd.to_datetime(result[column], utc=True)
    result["delivery_date_local"] = result["delivery_date_local"].astype(str)
    if config.train_start_date is not None:
        result = result.loc[result["delivery_date_local"] >= config.train_start_date].copy()
    if result.empty:
        raise ValueError("No weather snapshot rows remain after the configured training cutoff")
    if result.duplicated(
        ["delivery_date_local", "decision_hour_local", "valid_time_utc"]
    ).any():
        raise ValueError("MPC weather snapshots contain duplicate decision / target rows")
    if not (result["weather_snapshot_available_at_utc"] <= result["as_of_utc"]).all():
        raise AssertionError("A selected weather update is newer than its MPC decision")
    if not (result["valid_time_utc"] > result["as_of_utc"]).all():
        raise AssertionError("A selected weather update contains a non-future PV target")
    if result.loc[:, weather_columns].isna().any().any():
        raise ValueError("MPC weather snapshots contain missing forecast variables")
    return result


def _resolve_tracking_uri(config: PVMpcResidualTrainingConfig, project_root: Path) -> str:
    if config.tracking_uri:
        return config.tracking_uri
    if environment_uri := os.getenv("MLFLOW_TRACKING_URI"):
        return environment_uri
    database_path = project_root / "mlflow" / "mlflow.db"
    database_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{database_path}"


def _prepare_active_frame(
    frame: pd.DataFrame,
    config: PVMpcResidualTrainingConfig,
    pvlib: Any,
) -> pd.DataFrame:
    required = {
        TARGET_COLUMN,
        "valid_time_utc",
        "delivery_date_local",
        "week_index",
        *FORECAST_FEATURES[0:3],
        "horizon_hours",
        "weather_forecast_temperature_2m",
        "weather_forecast_shortwave_radiation",
        "weather_forecast_direct_radiation",
        "weather_forecast_diffuse_radiation",
        "weather_forecast_wind_speed_10m",
        "weather_forecast_cloud_cover",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Day-ahead dataset is missing required columns: {missing}")

    result = frame.copy().sort_values("valid_time_utc").reset_index(drop=True)
    if config.train_start_date is not None:
        result = result.loc[
            result["delivery_date_local"].astype(str) >= config.train_start_date
        ].copy()
    location = pvlib.location.Location(48.89, 8.70, tz="Europe/Berlin")
    times = pd.DatetimeIndex(pd.to_datetime(result["valid_time_utc"], utc=True))
    solar_position = location.get_solarposition(times)
    clear_sky = location.get_clearsky(times, model="ineichen")
    result["solar_elevation_deg"] = solar_position["apparent_elevation"].to_numpy()
    result["solar_azimuth_deg"] = solar_position["azimuth"].to_numpy()
    result["clear_sky_ghi_w_m2"] = clear_sky["ghi"].to_numpy()

    clear_sky_ghi = result["clear_sky_ghi_w_m2"].to_numpy()
    is_solar_active = clear_sky_ghi >= config.min_clear_sky_ghi
    clear_sky_index = np.full(len(result), -999.0)
    np.divide(
        result["weather_forecast_shortwave_radiation"].to_numpy(),
        clear_sky_ghi,
        out=clear_sky_index,
        where=is_solar_active,
    )
    result["forecast_clear_sky_index"] = clear_sky_index
    result["forecast_clear_sky_index_valid"] = is_solar_active.astype("int8")
    result["is_solar_active"] = is_solar_active

    # A missing midday sample is not a short solar day. Remove such delivery
    # days before constructing history vectors.
    contiguous_by_day = (
        result.loc[result["is_solar_active"]]
        .groupby("delivery_date_local")["hour_local"]
        .agg(_is_consecutive_hour_sequence)
    )
    result = result.loc[
        result["is_solar_active"]
        & result["delivery_date_local"].map(contiguous_by_day).fillna(False)
    ].copy()
    if result.empty:
        raise ValueError("No complete solar-active PV rows remain")
    if result.loc[:, list(FORECAST_FEATURES)].isna().any().any():
        missing_features = result.loc[:, list(FORECAST_FEATURES)].columns[
            result.loc[:, list(FORECAST_FEATURES)].isna().any()
        ].tolist()
        raise ValueError(f"Forecast features contain missing values: {missing_features}")
    return result.reset_index(drop=True)


def _is_consecutive_hour_sequence(hours: pd.Series) -> bool:
    ordered = hours.sort_values().to_numpy()
    return len(ordered) > 0 and bool(np.all(np.diff(ordered) == 1))


def _make_model(CatBoostRegressor: Any, config: PVMpcResidualTrainingConfig) -> Any:
    return CatBoostRegressor(
        loss_function="RMSE",
        iterations=config.iterations,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=10,
        random_seed=config.random_state,
        allow_writing_files=False,
        verbose=False,
    )


def _add_oof_day_ahead_prediction(
    train_frame: pd.DataFrame,
    config: PVMpcResidualTrainingConfig,
    CatBoostRegressor: Any,
    *,
    random_state: int,
) -> pd.DataFrame:
    result = train_frame.copy()
    predictions = np.full(len(result), np.nan)
    cv = SeasonalWeekKFold(n_splits=config.cv_splits, random_state=random_state)
    for train_indices, validation_indices in cv.split(result):
        model = _make_model(CatBoostRegressor, config)
        model.fit(
            result.iloc[train_indices].loc[:, list(FORECAST_FEATURES)],
            result.iloc[train_indices][TARGET_COLUMN],
        )
        predictions[validation_indices] = np.maximum(
            model.predict(result.iloc[validation_indices].loc[:, list(FORECAST_FEATURES)]),
            0.0,
        )
    if np.isnan(predictions).any():
        raise AssertionError("OOF day-ahead prediction is missing for at least one row")
    result[DAY_AHEAD_PREDICTION] = predictions
    return result


def _add_outer_day_ahead_prediction(
    outer_train: pd.DataFrame,
    outer_test: pd.DataFrame,
    config: PVMpcResidualTrainingConfig,
    CatBoostRegressor: Any,
    outer_fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = _add_oof_day_ahead_prediction(
        outer_train,
        config,
        CatBoostRegressor,
        random_state=100 + outer_fold,
    )
    test = outer_test.copy()
    model = _make_model(CatBoostRegressor, config)
    model.fit(train.loc[:, list(FORECAST_FEATURES)], train[TARGET_COLUMN])
    test[DAY_AHEAD_PREDICTION] = np.maximum(
        model.predict(test.loc[:, list(FORECAST_FEATURES)]),
        0.0,
    )
    return train, test


def _make_direct_residual_rows(
    day_ahead_frame: pd.DataFrame,
    config: PVMpcResidualTrainingConfig,
    *,
    weather_snapshots: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Convert active hourly rows into MPC decision / target point rows."""

    rows: list[dict[str, Any]] = []
    weather_lookup = _weather_snapshot_lookup(weather_snapshots)
    for delivery_date, day in day_ahead_frame.groupby("delivery_date_local", sort=False):
        day = day.sort_values("hour_local").copy()
        day["residual_w"] = day[TARGET_COLUMN] - day[DAY_AHEAD_PREDICTION]

        for decision_hour in range(24):
            history = day.loc[day["hour_local"] < decision_hour].tail(config.n_lags)
            future = day.loc[day["hour_local"] > decision_hour]
            if history.empty or future.empty:
                continue

            history_residuals = history["residual_w"].to_numpy()
            recent_history = history.iloc[::-1]
            recent_residuals = history_residuals[::-1]
            lag_values = np.full(config.n_lags, np.nan)
            lag_values[: len(recent_residuals)] = recent_residuals
            residual_trend = (
                float(np.polyfit(np.arange(len(history_residuals)), history_residuals, deg=1)[0])
                if len(history_residuals) >= 2
                else 0.0
            )

            for _, target in future.iterrows():
                row = target.loc[list(FORECAST_FEATURES)].to_dict()
                if weather_lookup is not None:
                    snapshot = weather_lookup.get(
                        (
                            str(delivery_date),
                            decision_hour,
                            _as_utc_timestamp(target["valid_time_utc"]),
                        )
                    )
                    if snapshot is None:
                        continue
                    _add_weather_snapshot_features(row, snapshot)
                row.update(
                    {
                        "delivery_date_local": delivery_date,
                        "week_index": int(target["week_index"]),
                        "decision_hour_local": decision_hour,
                        "remaining_horizon_hours": int(target["hour_local"] - decision_hour),
                        DAY_AHEAD_PREDICTION: float(target[DAY_AHEAD_PREDICTION]),
                        "n_active_residual_lags": len(recent_residuals),
                        "residual_lag_mean_w": float(recent_residuals.mean()),
                        "residual_lag_std_w": float(recent_residuals.std(ddof=0)),
                        "residual_lag_trend_w_per_hour": residual_trend,
                        RESIDUAL_TARGET: float(target["residual_w"]),
                        "target_pv_power_w": float(target[TARGET_COLUMN]),
                    }
                )
                for lag in range(config.n_lags):
                    available = lag < len(recent_residuals)
                    row[f"residual_lag_{lag + 1}_w"] = lag_values[lag]
                    row[f"residual_lag_{lag + 1}_available"] = int(available)
                    if available:
                        previous = recent_history.iloc[lag]
                        row[f"pv_lag_{lag + 1}_w"] = float(previous[TARGET_COLUMN])
                        row[f"day_ahead_lag_{lag + 1}_w"] = float(
                            previous[DAY_AHEAD_PREDICTION]
                        )
                    else:
                        row[f"pv_lag_{lag + 1}_w"] = np.nan
                        row[f"day_ahead_lag_{lag + 1}_w"] = np.nan
                rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No MPC decision / target rows were created")
    return result


def _residual_features(
    n_lags: int,
    *,
    uses_updated_weather: bool = False,
    updated_weather_feature_mode: str = "replace",
) -> list[str]:
    features = [
        *FORECAST_FEATURES,
        "decision_hour_local",
        "remaining_horizon_hours",
        DAY_AHEAD_PREDICTION,
        "n_active_residual_lags",
        "residual_lag_mean_w",
        "residual_lag_std_w",
        "residual_lag_trend_w_per_hour",
        *[f"residual_lag_{lag}_w" for lag in range(1, n_lags + 1)],
        *[f"pv_lag_{lag}_w" for lag in range(1, n_lags + 1)],
        *[f"day_ahead_lag_{lag}_w" for lag in range(1, n_lags + 1)],
        *[f"residual_lag_{lag}_available" for lag in range(1, n_lags + 1)],
    ]
    if not uses_updated_weather:
        return features
    if updated_weather_feature_mode not in UPDATED_WEATHER_FEATURE_MODES:
        raise ValueError(
            "updated_weather_feature_mode must be one of "
            f"{UPDATED_WEATHER_FEATURE_MODES}"
        )
    if updated_weather_feature_mode == "replace":
        replacements = {
            **{
                f"weather_forecast_{variable}": f"weather_update_{variable}"
                for variable in WEATHER_VARIABLES
            },
            "forecast_clear_sky_index": "weather_update_clear_sky_index",
            "forecast_clear_sky_index_valid": (
                "weather_update_clear_sky_index_valid"
            ),
        }
        features = [replacements.get(feature, feature) for feature in features]
        features.extend(
            ["weather_snapshot_lead_hours", "weather_snapshot_age_hours"]
        )
    elif updated_weather_feature_mode == "revision":
        features.extend(
            [
                *[
                    f"weather_update_delta_{variable}"
                    for variable in WEATHER_VARIABLES
                ],
                "weather_snapshot_lead_hours",
                "weather_snapshot_age_hours",
            ]
        )
    else:
        features.extend(
            [
                *[f"weather_update_{variable}" for variable in WEATHER_VARIABLES],
                *[f"weather_update_delta_{variable}" for variable in WEATHER_VARIABLES],
                "weather_update_clear_sky_index",
                "weather_update_clear_sky_index_valid",
                "weather_snapshot_lead_hours",
                "weather_snapshot_age_hours",
            ]
        )
    if len(features) != len(set(features)):
        raise AssertionError("Residual feature list contains duplicate columns")
    return features


def _as_utc_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _weather_snapshot_lookup(
    weather_snapshots: pd.DataFrame | None,
) -> dict[tuple[str, int, pd.Timestamp], dict[str, Any]] | None:
    if weather_snapshots is None:
        return None
    return {
        (
            str(row.delivery_date_local),
            int(row.decision_hour_local),
            _as_utc_timestamp(row.valid_time_utc),
        ): row._asdict()
        for row in weather_snapshots.itertuples(index=False)
    }


def _add_weather_snapshot_features(row: dict[str, Any], snapshot: dict[str, Any]) -> None:
    """Keep D-1 ICON inputs and add the fresh IFS level plus its source delta."""

    for variable in WEATHER_VARIABLES:
        original = float(row[f"weather_forecast_{variable}"])
        updated = float(snapshot[f"weather_forecast_{variable}"])
        row[f"weather_update_{variable}"] = updated
        row[f"weather_update_delta_{variable}"] = updated - original
    clear_sky_ghi = float(row["clear_sky_ghi_w_m2"])
    if clear_sky_ghi > 0:
        row["weather_update_clear_sky_index"] = (
            row["weather_update_shortwave_radiation"] / clear_sky_ghi
        )
        row["weather_update_clear_sky_index_valid"] = 1
    else:
        row["weather_update_clear_sky_index"] = -999.0
        row["weather_update_clear_sky_index_valid"] = 0
    row["weather_snapshot_lead_hours"] = int(snapshot["weather_snapshot_lead_hours"])
    row["weather_snapshot_age_hours"] = int(snapshot["weather_snapshot_age_hours"])


def _evaluate_outer_folds(
    active_frame: pd.DataFrame,
    residual_features: list[str],
    config: PVMpcResidualTrainingConfig,
    CatBoostRegressor: Any,
    weather_snapshots: pd.DataFrame | None,
) -> tuple[list[dict[str, float]], list[dict[str, Any]]]:
    metrics: list[dict[str, float]] = []
    contracts: list[dict[str, Any]] = []
    outer_cv = SeasonalWeekKFold(
        n_splits=config.cv_splits,
        random_state=config.random_state,
    )
    for fold in outer_cv.split_with_metadata(active_frame):
        outer_train, outer_test = _add_outer_day_ahead_prediction(
            active_frame.iloc[fold.train_indices],
            active_frame.iloc[fold.test_indices],
            config,
            CatBoostRegressor,
            fold.fold_index,
        )
        residual_train = _make_direct_residual_rows(
            outer_train,
            config,
            weather_snapshots=weather_snapshots,
        )
        residual_test = _make_direct_residual_rows(
            outer_test,
            config,
            weather_snapshots=weather_snapshots,
        )
        residual_model = _make_model(CatBoostRegressor, config)
        residual_model.fit(
            residual_train.loc[:, residual_features],
            residual_train[RESIDUAL_TARGET],
            verbose=False,
        )
        base = residual_test[DAY_AHEAD_PREDICTION].to_numpy()
        correction = residual_model.predict(residual_test.loc[:, residual_features])
        persistence_correction = residual_test["residual_lag_1_w"].to_numpy()
        predictions = {
            "baseline": base,
            "persistence": np.maximum(base + persistence_correction, 0.0),
            "direct_residual": np.maximum(base + correction, 0.0),
        }
        for variant, prediction in predictions.items():
            metrics.append(
                {
                    "fold": float(fold.fold_index),
                    "variant": variant,
                    "mae_w": float(
                        _mean_absolute_error(residual_test["target_pv_power_w"], prediction)
                    ),
                    "rmse_w": float(
                        _mean_squared_error(residual_test["target_pv_power_w"], prediction)
                        ** 0.5
                    ),
                    "rows": float(len(residual_test)),
                }
            )
        contracts.append(
            {
                "fold": fold.fold_index,
                "train_weeks": list(fold.train_weeks),
                "test_weeks": list(fold.test_weeks),
                "residual_train_rows": len(residual_train),
                "residual_test_rows": len(residual_test),
            }
        )
    return metrics, contracts


def _mean_absolute_error(target: pd.Series, prediction: np.ndarray) -> float:
    from sklearn.metrics import mean_absolute_error

    return float(mean_absolute_error(target, prediction))


def _mean_squared_error(target: pd.Series, prediction: np.ndarray) -> float:
    from sklearn.metrics import mean_squared_error

    return float(mean_squared_error(target, prediction))


def _aggregate_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    frame = pd.DataFrame(fold_metrics)
    result: dict[str, float] = {}
    for variant, part in frame.groupby("variant"):
        result[f"cv_{variant}_mae_w_mean"] = float(part["mae_w"].mean())
        result[f"cv_{variant}_mae_w_std"] = float(part["mae_w"].std(ddof=0))
        result[f"cv_{variant}_rmse_w_mean"] = float(part["rmse_w"].mean())
        result[f"cv_{variant}_rmse_w_std"] = float(part["rmse_w"].std(ddof=0))
    baseline = frame.loc[frame["variant"] == "baseline"].set_index("fold")
    direct = frame.loc[frame["variant"] == "direct_residual"].set_index("fold")
    persistence = frame.loc[frame["variant"] == "persistence"].set_index("fold")
    result["cv_direct_residual_mae_improvement_w_mean"] = float(
        (baseline["mae_w"] - direct["mae_w"]).mean()
    )
    result["cv_direct_residual_vs_persistence_mae_improvement_w_mean"] = float(
        (persistence["mae_w"] - direct["mae_w"]).mean()
    )
    return result


def _feature_contract(
    config: PVMpcResidualTrainingConfig,
    residual_features: list[str],
    uses_updated_weather: bool,
) -> dict[str, Any]:
    return {
        "model_scope": "direct_pointwise_mpc_pv_residual_correction",
        "target": "actual_pv_power_w - frozen_day_ahead_prediction_w",
        "features": residual_features,
        "decision_boundary": "At decision hour tau, only factual PV values before tau are used.",
        "prediction_boundary": "The model predicts future solar-active hours strictly after tau.",
        "base_forecast_requirement": (
            "day_ahead_prediction_w must be the frozen forecast made before the delivery day."
        ),
        "historical_lags": {
            "count": config.n_lags,
            "values": ["actual PV", "day-ahead PV", "residual"],
            "missing_early_day_values": "NaN plus residual_lag_*_available flag",
        },
        "solar_active_rule": {
            "formula": "clear-sky GHI >= threshold",
            "threshold_w_m2": config.min_clear_sky_ghi,
            "proxy_location": {"latitude": 48.89, "longitude": 8.70, "timezone": "Europe/Berlin"},
        },
        "weather_inputs": (
            "The frozen day-ahead base forecast remains unchanged. The residual model "
            f"uses the latest published ECMWF IFS snapshot in "
            f"'{config.updated_weather_feature_mode}' feature mode."
            if uses_updated_weather
            else "Original day-ahead weather forecast in V1; no intraday update yet."
        ),
        "prediction_postprocessing": "Apply max(day_ahead_prediction_w + residual_prediction_w, 0).",
    }


def _feature_importance(model: Any, feature_names: list[str]) -> dict[str, float]:
    values = model.get_feature_importance()
    return {
        feature: float(value)
        for feature, value in sorted(
            zip(feature_names, values, strict=True), key=lambda item: item[1], reverse=True
        )
    }


def _registered_version(model_info: Any) -> str | None:
    version = getattr(model_info, "registered_model_version", None)
    return str(version) if version is not None else None
