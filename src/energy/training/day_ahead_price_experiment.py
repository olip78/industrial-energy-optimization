"""Chronological comparison of compact day-ahead price forecasting models."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass(frozen=True)
class DayAheadPriceExperimentConfig:
    project_root: Path
    data_path: Path | None = None
    train_end: str = "2024-12-31"
    test_start: str = "2025-01-01"
    test_end: str = "2025-09-30"
    iterations: int = 500
    random_state: int = 42


@dataclass(frozen=True)
class DayAheadPriceExperimentResult:
    artifact_dir: Path
    metrics: pd.DataFrame

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_dir": str(self.artifact_dir),
            "metrics": self.metrics.to_dict(orient="records"),
        }


def _daily_ranking_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    correlations: list[float] = []
    top_overlap: list[float] = []
    bottom_overlap: list[float] = []
    for _, day in predictions.groupby("delivery_date_local", sort=False):
        correlations.append(day["actual_price_eur_per_mwh"].corr(day["prediction_eur_per_mwh"], method="spearman"))
        top_actual = set(day.nlargest(6, "actual_price_eur_per_mwh").index)
        top_predicted = set(day.nlargest(6, "prediction_eur_per_mwh").index)
        bottom_actual = set(day.nsmallest(6, "actual_price_eur_per_mwh").index)
        bottom_predicted = set(day.nsmallest(6, "prediction_eur_per_mwh").index)
        top_overlap.append(len(top_actual & top_predicted) / 6.0)
        bottom_overlap.append(len(bottom_actual & bottom_predicted) / 6.0)
    return {
        "daily_spearman_mean": float(np.nanmean(correlations)),
        "top_quartile_overlap": float(np.mean(top_overlap)),
        "bottom_quartile_overlap": float(np.mean(bottom_overlap)),
    }


def _metrics(predictions: pd.DataFrame) -> dict[str, float]:
    actual = predictions["actual_price_eur_per_mwh"]
    predicted = predictions["prediction_eur_per_mwh"]
    return {
        "mae_eur_per_mwh": float(mean_absolute_error(actual, predicted)),
        "rmse_eur_per_mwh": float(mean_squared_error(actual, predicted) ** 0.5),
        **_daily_ranking_metrics(predictions),
    }


def _read_inputs(config: DayAheadPriceExperimentConfig) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    root = config.project_root.resolve()
    data_path = config.data_path or (
        root
        / "data"
        / "features"
        / "day_ahead_price_spatial"
        / "day_ahead_price_spatial_2024-02-18_2025-09-30.parquet"
    )
    manifest_path = root / "data" / "metadata" / "day_ahead_price_spatial_v1_manifest.json"
    frame = pd.read_parquet(data_path)
    manifest = json.loads(manifest_path.read_text())
    return frame, manifest["feature_groups"]


def run_day_ahead_price_experiment(
    config: DayAheadPriceExperimentConfig,
) -> DayAheadPriceExperimentResult:
    """Evaluate models on a future chronological holdout without model selection on it."""

    frame, groups = _read_inputs(config)
    frame["delivery_date_local"] = pd.to_datetime(frame["delivery_date_local"])
    train_end = pd.Timestamp(config.train_end)
    test_start = pd.Timestamp(config.test_start)
    test_end = pd.Timestamp(config.test_end)
    train = frame.loc[frame["delivery_date_local"] <= train_end].copy()
    test = frame.loc[
        frame["delivery_date_local"].between(test_start, test_end, inclusive="both")
    ].copy()
    if train.empty or test.empty or train["delivery_date_local"].max() >= test["delivery_date_local"].min():
        raise ValueError("Train/test windows must be non-empty and strictly chronological")

    variants: dict[str, list[str] | None] = {
        "persistence_d1": None,
        "price_only_catboost": groups["price_only"],
        "local_weather_catboost": [*groups["price_only"], *groups["local_weather"]],
        "spatial_weather_catboost": [*groups["price_only"], *groups["spatial_weather"]],
    }
    target = "day_ahead_price_eur_per_mwh"
    artifact_dir = (
        config.project_root.resolve()
        / "artifacts"
        / "experiments"
        / "day_ahead_price_spatial_v1"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []

    for variant, features in variants.items():
        if features is None:
            prediction = test["price_lag_d1_same_hour"].to_numpy(dtype=float)
            feature_count = 1
        else:
            model = CatBoostRegressor(
                loss_function="RMSE",
                iterations=config.iterations,
                depth=6,
                learning_rate=0.05,
                l2_leaf_reg=10.0,
                random_seed=config.random_state,
                verbose=False,
                allow_writing_files=False,
            )
            model.fit(train[features], train[target])
            prediction = model.predict(test[features])
            feature_count = len(features)
            importance_frames.append(
                pd.DataFrame(
                    {
                        "variant": variant,
                        "feature": features,
                        "importance": model.feature_importances_,
                    }
                ).sort_values("importance", ascending=False)
            )
        predictions = pd.DataFrame(
            {
                "variant": variant,
                "valid_time_utc": test["valid_time_utc"],
                "delivery_date_local": test["delivery_date_local"].dt.strftime("%Y-%m-%d"),
                "hour_local": test["hour_local"],
                "actual_price_eur_per_mwh": test[target],
                "prediction_eur_per_mwh": prediction,
            }
        )
        prediction_frames.append(predictions)
        metric_rows.append(
            {
                "variant": variant,
                "feature_count": feature_count,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                **_metrics(predictions),
            }
        )

    metrics = pd.DataFrame(metric_rows).sort_values("mae_eur_per_mwh").reset_index(drop=True)
    metrics.to_csv(artifact_dir / "metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(
        artifact_dir / "hourly_predictions.parquet", index=False
    )
    pd.concat(importance_frames, ignore_index=True).to_csv(
        artifact_dir / "feature_importance.csv", index=False
    )
    (artifact_dir / "experiment_config.json").write_text(
        json.dumps(
            {
                "train_end": config.train_end,
                "test_start": config.test_start,
                "test_end": config.test_end,
                "iterations": config.iterations,
                "random_state": config.random_state,
                "selection_policy": "Hyperparameters were fixed before evaluating the 2025 holdout; no early stopping or tuning uses test rows.",
            },
            indent=2,
        )
    )
    return DayAheadPriceExperimentResult(artifact_dir, metrics)
