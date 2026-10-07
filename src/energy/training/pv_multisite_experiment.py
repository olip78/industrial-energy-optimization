"""Parallel multi-site PV backtest for direct, MIMO MLP and MIMO LSTM models.

This experiment is intentionally isolated from the production-like single-site
training applications.  It pools eligible MPVBench profiles only for training;
week splits keep all profiles from one delivery week together, so no factual
PV from a second installation leaks into a test-week forecast.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from energy.data import SeasonalWeekKFold
from energy.modeling.pv_mimo import (
    MIMOLSTMTrainingConfig,
    MIMOMLPTrainingConfig,
    build_lstm_sequence_samples,
    build_mimo_residual_samples,
    fit_mimo_residual_lstm,
    fit_mimo_residual_mlp,
    refit_config_for_selected_epoch,
    refit_lstm_config_for_selected_epoch,
    trajectory_prediction_frame,
)
from energy.training.pv_mpc_residual import FORECAST_FEATURES


TARGET_COLUMN = "pv_power_mean_w"
DAY_AHEAD_PREDICTION = "day_ahead_prediction_w"
RESIDUAL_TARGET = "target_residual_w"
MINIMUM_ACTIVE_GHI = 25.0
DERIVED_SOLAR_FEATURES = {
    "solar_elevation_deg",
    "solar_azimuth_deg",
    "clear_sky_ghi_w_m2",
    "forecast_clear_sky_index",
    "forecast_clear_sky_index_valid",
}


@dataclass(frozen=True)
class MultiSitePVExperimentConfig:
    """Settings for the isolated three-model multi-site backtest."""

    project_root: Path
    data_path: Path | None = None
    year: int = 2024
    n_lags: int = 4
    cv_splits: int = 3
    catboost_iterations: int = 500
    min_clear_sky_ghi: float = MINIMUM_ACTIVE_GHI
    random_state: int = 42
    mlp: MIMOMLPTrainingConfig = MIMOMLPTrainingConfig()
    lstm: MIMOLSTMTrainingConfig = MIMOLSTMTrainingConfig(
        batch_size=256,
        max_epochs=150,
        patience=20,
    )


@dataclass(frozen=True)
class MultiSitePVExperimentResult:
    """Backtest data and the dedicated parallel artefact directory."""

    metrics: pd.DataFrame
    lead_metrics: pd.DataFrame
    selection: pd.DataFrame
    artifact_dir: Path

    def write(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.metrics.to_csv(self.artifact_dir / "fold_metrics.csv", index=False)
        self.lead_metrics.to_csv(self.artifact_dir / "lead_metrics.csv", index=False)
        self.selection.to_csv(self.artifact_dir / "model_selection.csv", index=False)
        summary = _summarise_metrics(self.metrics)
        (self.artifact_dir / "summary.json").write_text(
            json.dumps(summary, indent=2)
        )


def run_multisite_pv_experiment(
    config: MultiSitePVExperimentConfig,
) -> MultiSitePVExperimentResult:
    """Run direct CatBoost, MIMO MLP and MIMO LSTM under one shared protocol."""

    CatBoostRegressor, pvlib = _dependencies()
    _validate_config(config)
    root = config.project_root.expanduser().resolve()
    data_path = config.data_path or (
        root
        / "data"
        / "features"
        / "pv_multisite_v1"
        / f"day_ahead_pv_multisite_{config.year}.parquet"
    )
    data_path = data_path.expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Multi-site training table was not found: {data_path}")

    full_frame = pd.read_parquet(data_path)
    site_columns = _site_one_hot_columns(full_frame)
    source_forecast_features = [
        feature for feature in FORECAST_FEATURES if feature not in DERIVED_SOLAR_FEATURES
    ] + site_columns
    forecast_features = [*FORECAST_FEATURES, *site_columns]
    active_frame = _prepare_active_frame(
        full_frame,
        config,
        pvlib,
        source_forecast_features,
        forecast_features,
    )
    max_horizon = _maximum_horizon(active_frame)

    metric_rows: list[dict[str, object]] = []
    lead_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    outer_cv = SeasonalWeekKFold(config.cv_splits, random_state=config.random_state)

    for fold in outer_cv.split_with_metadata(active_frame):
        outer_train, outer_test = _add_outer_day_ahead_prediction(
            active_frame.iloc[fold.train_indices],
            active_frame.iloc[fold.test_indices],
            config,
            CatBoostRegressor,
            forecast_features,
            fold.fold_index,
        )

        direct_train = _make_direct_residual_rows(outer_train, config, forecast_features)
        direct_test = _make_direct_residual_rows(outer_test, config, forecast_features)
        direct_features = _direct_features(config.n_lags, forecast_features)
        direct_model = _make_catboost(CatBoostRegressor, config)
        direct_model.fit(
            direct_train.loc[:, direct_features],
            direct_train[RESIDUAL_TARGET],
            verbose=False,
        )
        direct_test["direct_prediction_w"] = np.maximum(
            direct_test[DAY_AHEAD_PREDICTION].to_numpy()
            + direct_model.predict(direct_test.loc[:, direct_features]),
            0.0,
        )

        mimo_train = build_mimo_residual_samples(
            outer_train,
            future_features=forecast_features,
            n_lags=config.n_lags,
            max_horizon=max_horizon,
            site_column="site_id",
        )
        mimo_test = build_mimo_residual_samples(
            outer_test,
            future_features=forecast_features,
            n_lags=config.n_lags,
            max_horizon=max_horizon,
            site_column="site_id",
        )
        selection_fold = next(
            SeasonalWeekKFold(
                config.cv_splits,
                random_state=10_000 + fold.fold_index,
            ).split_with_metadata(outer_train)
        )

        selected_mlp = fit_mimo_residual_mlp(
            mimo_train.select_weeks(selection_fold.train_weeks),
            mimo_train.select_weeks(selection_fold.test_weeks),
            config=config.mlp,
        )
        final_mlp = fit_mimo_residual_mlp(
            mimo_train,
            validation=None,
            config=refit_config_for_selected_epoch(config.mlp, selected_mlp.best_epoch),
        )
        mlp_rows = trajectory_prediction_frame(
            mimo_test,
            final_mlp.predict_residual_w(mimo_test),
        ).rename(columns={"mimo_prediction_w": "mimo_mlp_prediction_w"})

        lstm_train = build_lstm_sequence_samples(
            mimo_train,
            future_features=forecast_features,
            n_lags=config.n_lags,
        )
        lstm_test = build_lstm_sequence_samples(
            mimo_test,
            future_features=forecast_features,
            n_lags=config.n_lags,
        )
        selected_lstm = fit_mimo_residual_lstm(
            lstm_train.select_weeks(selection_fold.train_weeks),
            lstm_train.select_weeks(selection_fold.test_weeks),
            config=config.lstm,
        )
        final_lstm = fit_mimo_residual_lstm(
            lstm_train,
            validation=None,
            config=refit_lstm_config_for_selected_epoch(config.lstm, selected_lstm.best_epoch),
        )
        lstm_rows = trajectory_prediction_frame(
            lstm_test.trajectory,
            final_lstm.predict_residual_w(lstm_test),
        )[[
            "site_id",
            "delivery_date_local",
            "decision_hour_local",
            "lead_hours",
            "mimo_prediction_w",
        ]].rename(columns={"mimo_prediction_w": "mimo_lstm_prediction_w"})

        rows = _join_predictions(mlp_rows, lstm_rows, direct_test)
        _append_metrics(metric_rows, lead_rows, rows, fold.fold_index)
        selection_rows.extend(
            [
                {
                    "fold": fold.fold_index,
                    "model": "mimo_mlp",
                    "selected_epoch": selected_mlp.best_epoch,
                    "selection_validation_mae_w": selected_mlp.best_validation_mae_w,
                    "train_weeks": json.dumps(list(selection_fold.train_weeks)),
                    "validation_weeks": json.dumps(list(selection_fold.test_weeks)),
                },
                {
                    "fold": fold.fold_index,
                    "model": "mimo_lstm",
                    "selected_epoch": selected_lstm.best_epoch,
                    "selection_validation_mae_w": selected_lstm.best_validation_mae_w,
                    "train_weeks": json.dumps(list(selection_fold.train_weeks)),
                    "validation_weeks": json.dumps(list(selection_fold.test_weeks)),
                },
            ]
        )

    artifact_dir = root / "artifacts" / "experiments" / "pv_multisite_v1" / str(config.year)
    result = MultiSitePVExperimentResult(
        metrics=pd.DataFrame(metric_rows),
        lead_metrics=pd.DataFrame(lead_rows),
        selection=pd.DataFrame(selection_rows),
        artifact_dir=artifact_dir,
    )
    result.write()
    return result


def _dependencies() -> tuple[Any, Any]:
    try:
        import pvlib
        from catboost import CatBoostRegressor
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("Install the experiment stack: python -m pip install -e '.[train,neural]'") from error
    return CatBoostRegressor, pvlib


def _validate_config(config: MultiSitePVExperimentConfig) -> None:
    if config.n_lags < 1:
        raise ValueError("n_lags must be at least 1")
    if config.cv_splits < 2:
        raise ValueError("cv_splits must be at least 2")
    if config.catboost_iterations < 1:
        raise ValueError("catboost_iterations must be positive")
    if config.min_clear_sky_ghi <= 0:
        raise ValueError("min_clear_sky_ghi must be positive")


def _site_one_hot_columns(frame: pd.DataFrame) -> list[str]:
    columns = sorted(column for column in frame if column.startswith("site_id_"))
    if not columns:
        raise KeyError("Multi-site data must contain one-hot site_id_<profile> columns")
    return columns


def _prepare_active_frame(
    frame: pd.DataFrame,
    config: MultiSitePVExperimentConfig,
    pvlib: Any,
    source_forecast_features: list[str],
    forecast_features: list[str],
) -> pd.DataFrame:
    required = {
        TARGET_COLUMN,
        "site_id",
        "valid_time_utc",
        "delivery_date_local",
        "week_index",
        "hour_local",
        *source_forecast_features,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Multi-site frame is missing required columns: {missing}")
    result = frame.copy().sort_values(["site_id", "valid_time_utc"]).reset_index(drop=True)
    location = pvlib.location.Location(48.89, 8.70, tz="Europe/Berlin")
    unique_times = pd.DatetimeIndex(pd.to_datetime(result["valid_time_utc"], utc=True).unique())
    solar_position = location.get_solarposition(unique_times)
    clear_sky = location.get_clearsky(unique_times, model="ineichen")
    solar_features = pd.DataFrame(
        {
            "valid_time_utc": unique_times,
            "solar_elevation_deg": solar_position["apparent_elevation"].to_numpy(),
            "solar_azimuth_deg": solar_position["azimuth"].to_numpy(),
            "clear_sky_ghi_w_m2": clear_sky["ghi"].to_numpy(),
        }
    )
    result = result.merge(solar_features, on="valid_time_utc", how="left", validate="many_to_one")
    active = result["clear_sky_ghi_w_m2"].to_numpy() >= config.min_clear_sky_ghi
    result["is_solar_active"] = active
    clear_sky_index = np.full(len(result), -999.0)
    np.divide(
        result["weather_forecast_shortwave_radiation"].to_numpy(),
        result["clear_sky_ghi_w_m2"].to_numpy(),
        out=clear_sky_index,
        where=active,
    )
    result["forecast_clear_sky_index"] = clear_sky_index
    result["forecast_clear_sky_index_valid"] = active.astype("int8")

    active_frame = result.loc[result["is_solar_active"]].copy()
    active_frame["_complete_active_interval"] = active_frame.groupby(
        ["site_id", "delivery_date_local"]
    )["hour_local"].transform(_is_consecutive_hour_sequence)
    active_frame = active_frame.loc[active_frame["_complete_active_interval"]].drop(
        columns="_complete_active_interval"
    )
    if active_frame.empty:
        raise ValueError("No complete multi-site solar-active rows remain")
    all_features = [*forecast_features, "solar_elevation_deg", "solar_azimuth_deg", "clear_sky_ghi_w_m2", "forecast_clear_sky_index", "forecast_clear_sky_index_valid"]
    if active_frame[all_features].isna().any().any():
        missing_features = active_frame[all_features].columns[active_frame[all_features].isna().any()].tolist()
        raise ValueError(f"Multi-site forecast features contain missing values: {missing_features}")
    return active_frame.reset_index(drop=True)


def _is_consecutive_hour_sequence(hours: pd.Series) -> bool:
    ordered = hours.sort_values().to_numpy()
    return len(ordered) > 0 and bool(np.all(np.diff(ordered) == 1))


def _model_features(forecast_features: list[str]) -> list[str]:
    return list(dict.fromkeys(forecast_features))


def _make_catboost(CatBoostRegressor: Any, config: MultiSitePVExperimentConfig) -> Any:
    return CatBoostRegressor(
        loss_function="RMSE",
        iterations=config.catboost_iterations,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=10,
        random_seed=config.random_state,
        allow_writing_files=False,
        verbose=False,
    )


def _add_oof_day_ahead_prediction(
    frame: pd.DataFrame,
    config: MultiSitePVExperimentConfig,
    CatBoostRegressor: Any,
    forecast_features: list[str],
    *,
    random_state: int,
) -> pd.DataFrame:
    result = frame.copy()
    features = _model_features(forecast_features)
    predictions = np.full(len(result), np.nan)
    cv = SeasonalWeekKFold(config.cv_splits, random_state=random_state)
    for train_indices, validation_indices in cv.split(result):
        model = _make_catboost(CatBoostRegressor, config)
        model.fit(result.iloc[train_indices][features], result.iloc[train_indices][TARGET_COLUMN])
        predictions[validation_indices] = np.maximum(
            model.predict(result.iloc[validation_indices][features]), 0.0
        )
    if np.isnan(predictions).any():
        raise AssertionError("Multi-site OOF day-ahead prediction is incomplete")
    result[DAY_AHEAD_PREDICTION] = predictions
    return result


def _add_outer_day_ahead_prediction(
    outer_train: pd.DataFrame,
    outer_test: pd.DataFrame,
    config: MultiSitePVExperimentConfig,
    CatBoostRegressor: Any,
    forecast_features: list[str],
    outer_fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = _add_oof_day_ahead_prediction(
        outer_train,
        config,
        CatBoostRegressor,
        forecast_features,
        random_state=100 + outer_fold,
    )
    features = _model_features(forecast_features)
    model = _make_catboost(CatBoostRegressor, config)
    model.fit(train[features], train[TARGET_COLUMN])
    test = outer_test.copy()
    test[DAY_AHEAD_PREDICTION] = np.maximum(model.predict(test[features]), 0.0)
    return train, test


def _make_direct_residual_rows(
    frame: pd.DataFrame,
    config: MultiSitePVExperimentConfig,
    forecast_features: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    model_features = _model_features(forecast_features)
    for (site_id, delivery_date), day in frame.groupby(["site_id", "delivery_date_local"], sort=False):
        day = day.sort_values("hour_local").copy()
        day["_residual_w"] = day[TARGET_COLUMN] - day[DAY_AHEAD_PREDICTION]
        for decision_hour in range(24):
            history = day.loc[day["hour_local"] < decision_hour].tail(config.n_lags)
            future = day.loc[day["hour_local"] > decision_hour]
            if history.empty or future.empty:
                continue
            recent = history.iloc[::-1]
            for _, target in future.iterrows():
                row = target[model_features].to_dict()
                row.update(
                    {
                        "site_id": site_id,
                        "delivery_date_local": delivery_date,
                        "week_index": int(target["week_index"]),
                        "decision_hour_local": decision_hour,
                        "lead_hours": int(target["hour_local"] - decision_hour),
                        DAY_AHEAD_PREDICTION: float(target[DAY_AHEAD_PREDICTION]),
                        RESIDUAL_TARGET: float(target["_residual_w"]),
                        "target_pv_power_w": float(target[TARGET_COLUMN]),
                    }
                )
                for lag in range(config.n_lags):
                    if lag < len(recent):
                        previous = recent.iloc[lag]
                        row[f"residual_lag_{lag + 1}_w"] = float(previous["_residual_w"])
                        row[f"pv_lag_{lag + 1}_w"] = float(previous[TARGET_COLUMN])
                        row[f"day_ahead_lag_{lag + 1}_w"] = float(previous[DAY_AHEAD_PREDICTION])
                        row[f"residual_lag_{lag + 1}_available"] = 1
                    else:
                        row[f"residual_lag_{lag + 1}_w"] = np.nan
                        row[f"pv_lag_{lag + 1}_w"] = np.nan
                        row[f"day_ahead_lag_{lag + 1}_w"] = np.nan
                        row[f"residual_lag_{lag + 1}_available"] = 0
                rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No multi-site direct residual rows were created")
    return result


def _direct_features(n_lags: int, forecast_features: list[str]) -> list[str]:
    return [
        *_model_features(forecast_features),
        DAY_AHEAD_PREDICTION,
        *[f"residual_lag_{lag}_w" for lag in range(1, n_lags + 1)],
        *[f"pv_lag_{lag}_w" for lag in range(1, n_lags + 1)],
        *[f"day_ahead_lag_{lag}_w" for lag in range(1, n_lags + 1)],
        *[f"residual_lag_{lag}_available" for lag in range(1, n_lags + 1)],
    ]


def _maximum_horizon(frame: pd.DataFrame) -> int:
    maximum = 0
    for _, day in frame.groupby(["site_id", "delivery_date_local"], sort=False):
        for decision_hour in range(24):
            if (day["hour_local"] < decision_hour).any():
                maximum = max(maximum, int((day["hour_local"] > decision_hour).sum()))
    if maximum < 1:
        raise ValueError("No future solar-active horizon is available")
    return maximum


def _join_predictions(
    mlp_rows: pd.DataFrame,
    lstm_rows: pd.DataFrame,
    direct_rows: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["site_id", "delivery_date_local", "decision_hour_local", "lead_hours"]
    direct = direct_rows[
        [*keys, "direct_prediction_w"]
    ]
    result = mlp_rows.merge(lstm_rows, on=keys, how="inner", validate="one_to_one")
    result = result.merge(direct, on=keys, how="inner", validate="one_to_one")
    if result.empty:
        raise ValueError("No common decision / target rows across multi-site models")
    return result


def _append_metrics(
    metric_rows: list[dict[str, object]],
    lead_rows: list[dict[str, object]],
    rows: pd.DataFrame,
    fold: int,
) -> None:
    variants = {
        "day_ahead_baseline": "day_ahead_prediction_w",
        "direct_residual_catboost": "direct_prediction_w",
        "mimo_mlp": "mimo_mlp_prediction_w",
        "mimo_lstm": "mimo_lstm_prediction_w",
    }
    for site_id, part in [("pooled", rows), *list(rows.groupby("site_id", sort=True))]:
        for variant, prediction_column in variants.items():
            error = part["target_pv_power_w"] - part[prediction_column]
            metric_rows.append(
                {
                    "fold": fold,
                    "site_id": site_id,
                    "variant": variant,
                    "mae_w": float(error.abs().mean()),
                    "rmse_w": float((error.pow(2).mean()) ** 0.5),
                    "rows": len(part),
                }
            )
        for lead, lead_part in part.groupby("lead_hours"):
            row: dict[str, object] = {
                "fold": fold,
                "site_id": site_id,
                "lead_hours": int(lead),
                "rows": len(lead_part),
            }
            for variant, prediction_column in variants.items():
                row[f"{variant}_mae_w"] = float(
                    (lead_part["target_pv_power_w"] - lead_part[prediction_column]).abs().mean()
                )
            lead_rows.append(row)


def _summarise_metrics(metrics: pd.DataFrame) -> dict[str, object]:
    summary = (
        metrics.groupby(["site_id", "variant"])[["mae_w", "rmse_w"]]
        .agg(["mean", "std"])
        .round(6)
    )
    return {
        "metric_definition": "mean / sample standard deviation across outer week folds",
        "metrics": {
            f"{site_id}/{variant}": {
                f"{metric}_{statistic}": float(value)
                for (metric, statistic), value in values.dropna().items()
            }
            for (site_id, variant), values in summary.iterrows()
        },
    }
