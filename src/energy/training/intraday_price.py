"""Training application for the calibrated one-hour intraday price correction.

The CatBoost model predicts the intraday-minus-day-ahead spread.  MPC applies
that correction only to the next delivery hour, shrunk by a calibration weight
selected on an earlier chronological validation period.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from energy.data.intraday_price import ENRICHED_FEATURES_RELATIVE_PATH, _feature_names


TARGET_PRICE_COLUMN = "target_intraday_price_eur_per_mwh"
TARGET_SPREAD_COLUMN = "target_intraday_spread_eur_per_mwh"
DAY_AHEAD_COLUMN = "day_ahead_price_eur_per_mwh"
MODEL_NAME = "intraday_price_spread_catboost"
FEATURE_VERSION = "v2"


@dataclass(frozen=True)
class IntradayPriceTrainingConfig:
    """Inputs and reproducibility settings for the intraday spread model."""

    project_root: Path
    data_path: Path | None = None
    year: int = 2024
    validation_start: str = "2024-10-01"
    correction_weight: float = 0.70
    iterations: int = 500
    random_state: int = 42
    tracking_uri: str | None = None
    experiment_name: str | None = None
    registered_model_name: str | None = None


@dataclass(frozen=True)
class IntradayPriceTrainingResult:
    """MLflow identifiers and chronological validation metrics."""

    run_id: str
    tracking_uri: str
    data_path: str
    model_uri: str
    registered_model_name: str | None
    registered_model_version: str | None
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def train_intraday_price(
    config: IntradayPriceTrainingConfig,
) -> IntradayPriceTrainingResult:
    """Fit, evaluate, log and optionally register the intraday spread model.

    Q4 of ``year`` remains a chronological validation period. The final CatBoost
    model is refit on all eligible rows of the year after the fixed Huber/depth-4
    configuration and correction weight have been evaluated there.
    """

    mlflow, CatBoostRegressor, infer_signature = _training_dependencies()
    _validate_config(config)
    root = config.project_root.expanduser().resolve()
    data_path = _resolve_data_path(config, root)
    frame = _prepare_frame(pd.read_parquet(data_path), config)
    feature_names = _feature_names(FEATURE_VERSION)

    validation_start = pd.Timestamp(config.validation_start)
    development = frame.loc[frame["delivery_date_local"] < validation_start].copy()
    validation = frame.loc[frame["delivery_date_local"] >= validation_start].copy()
    if development.empty or validation.empty:
        raise ValueError("Development and validation periods must both be non-empty")
    if development["target_valid_time_utc"].max() >= validation["as_of_utc"].min():
        raise AssertionError("Development and validation periods overlap")

    validation_model = _make_model(CatBoostRegressor, config)
    validation_model.fit(development[feature_names], development[TARGET_SPREAD_COLUMN])
    validation_spread = validation_model.predict(validation[feature_names])
    validation_prediction = calibrated_intraday_price(
        validation[DAY_AHEAD_COLUMN].to_numpy(dtype=float),
        validation_spread,
        validation["lead_hours"].to_numpy(dtype=int),
        correction_weight=config.correction_weight,
    )
    metrics = _validation_metrics(validation, validation_prediction)

    final_model = _make_model(CatBoostRegressor, config)
    final_model.fit(frame[feature_names], frame[TARGET_SPREAD_COLUMN])

    tracking_uri = _resolve_tracking_uri(config, root)
    experiment_name = config.experiment_name or os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "intraday-price"
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    input_example = frame.loc[:, feature_names].head(5)
    signature = infer_signature(input_example, final_model.predict(input_example))
    with mlflow.start_run(run_name=f"intraday-price-{config.year}-v2") as run:
        mlflow.log_params(
            {
                "year": config.year,
                "feature_version": FEATURE_VERSION,
                "validation_start": config.validation_start,
                "correction_weight": config.correction_weight,
                "random_state": config.random_state,
                "catboost_iterations": config.iterations,
                "catboost_loss": "Huber:delta=10",
                "catboost_learning_rate": 0.05,
                "catboost_depth": 4,
                "catboost_l2_leaf_reg": 50.0,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "model_scope": "mpc_intraday_spread_correction",
                "forecast_inputs_only": "true",
                "feature_version": FEATURE_VERSION,
                "target": "intraday_minus_day_ahead_spread",
                "prediction_contract": "CatBoost outputs spread; decision layer adds 0.70 * spread only at lead_hours=1.",
                "market_data_limit": "public realised hourly index; no trade timestamps, volume, bid or ask.",
            }
        )
        mlflow.log_dict(_feature_contract(feature_names, config), "feature_contract.json")
        mlflow.log_dict(_split_contract(development, validation, config), "split_contract.json")
        mlflow.log_dict(metrics, "metrics.json")
        mlflow.log_dict(
            _feature_importance(final_model, feature_names), "feature_importance.json"
        )
        model_info = mlflow.catboost.log_model(
            final_model,
            name=MODEL_NAME,
            signature=signature,
            input_example=input_example,
            registered_model_name=config.registered_model_name,
        )

    return IntradayPriceTrainingResult(
        run_id=run.info.run_id,
        tracking_uri=tracking_uri,
        data_path=str(data_path),
        model_uri=f"runs:/{run.info.run_id}/{MODEL_NAME}",
        registered_model_name=config.registered_model_name,
        registered_model_version=_registered_version(model_info),
        metrics=metrics,
    )


def calibrated_intraday_price(
    day_ahead_price: np.ndarray,
    predicted_spread: np.ndarray,
    lead_hours: np.ndarray,
    *,
    correction_weight: float = 0.70,
) -> np.ndarray:
    """Apply the calibrated spread correction solely at the next MPC horizon."""

    if not 0.0 <= correction_weight <= 1.0:
        raise ValueError("correction_weight must be between zero and one")
    baseline = np.asarray(day_ahead_price, dtype=float)
    spread = np.asarray(predicted_spread, dtype=float)
    lead = np.asarray(lead_hours, dtype=int)
    if not (len(baseline) == len(spread) == len(lead)):
        raise ValueError("day_ahead_price, predicted_spread and lead_hours must align")
    result = baseline.copy()
    result[lead == 1] += correction_weight * spread[lead == 1]
    return result


def _training_dependencies():
    try:
        import mlflow
        import mlflow.catboost
        from catboost import CatBoostRegressor
        from mlflow.models import infer_signature
    except ImportError as error:  # pragma: no cover - caller environment dependent
        raise RuntimeError(
            "Install the training extra first: python -m pip install -e '.[train]'"
        ) from error
    return mlflow, CatBoostRegressor, infer_signature


def _validate_config(config: IntradayPriceTrainingConfig) -> None:
    if not 0.0 <= config.correction_weight <= 1.0:
        raise ValueError("correction_weight must be between zero and one")
    if config.iterations < 1:
        raise ValueError("iterations must be positive")


def _resolve_data_path(config: IntradayPriceTrainingConfig, root: Path) -> Path:
    data_path = config.data_path or root / ENRICHED_FEATURES_RELATIVE_PATH
    data_path = data_path.expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(
            f"Enriched intraday feature table was not found: {data_path}. "
            "Build it with --feature-version v2 first."
        )
    return data_path


def _prepare_frame(frame: pd.DataFrame, config: IntradayPriceTrainingConfig) -> pd.DataFrame:
    features = _feature_names(FEATURE_VERSION)
    required = {
        *features,
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
        raise KeyError(f"Intraday V2 dataset is missing required columns: {missing}")
    result = frame.copy()
    result["delivery_date_local"] = pd.to_datetime(result["delivery_date_local"])
    result = result.loc[result["delivery_date_local"].dt.year == config.year].copy()
    if result.empty:
        raise ValueError(f"No intraday rows remain for year {config.year}")
    if result[features].isna().any().any():
        raise ValueError("Intraday model features contain missing values")
    if not (result["intraday_history_available_at_utc"] <= result["as_of_utc"]).all():
        raise AssertionError("Intraday history is unavailable at a decision time")
    if not (result["day_ahead_price_available_at_utc"] <= result["as_of_utc"]).all():
        raise AssertionError("Day-ahead price is unavailable at a decision time")
    if not (result["target_available_at_utc"] > result["as_of_utc"]).all():
        raise AssertionError("Intraday target is available at a decision time")
    return result.sort_values(["delivery_date_local", "as_of_utc", "lead_hours"]).reset_index(drop=True)


def _make_model(CatBoostRegressor: Any, config: IntradayPriceTrainingConfig) -> Any:
    return CatBoostRegressor(
        loss_function="Huber:delta=10",
        iterations=config.iterations,
        learning_rate=0.05,
        depth=4,
        l2_leaf_reg=50.0,
        random_seed=config.random_state,
        verbose=False,
        allow_writing_files=False,
    )


def _validation_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    actual = frame[TARGET_PRICE_COLUMN].to_numpy(dtype=float)
    baseline = frame[DAY_AHEAD_COLUMN].to_numpy(dtype=float)
    next_hour = frame["lead_hours"].to_numpy(dtype=int) == 1
    metrics: dict[str, float] = {}
    for prefix, mask in (("next_hour", next_hour), ("all_remaining_hours", np.ones(len(frame), dtype=bool))):
        metrics[f"validation_{prefix}_baseline_mae_eur_per_mwh"] = float(
            mean_absolute_error(actual[mask], baseline[mask])
        )
        metrics[f"validation_{prefix}_hybrid_mae_eur_per_mwh"] = float(
            mean_absolute_error(actual[mask], prediction[mask])
        )
        metrics[f"validation_{prefix}_baseline_rmse_eur_per_mwh"] = float(
            mean_squared_error(actual[mask], baseline[mask]) ** 0.5
        )
        metrics[f"validation_{prefix}_hybrid_rmse_eur_per_mwh"] = float(
            mean_squared_error(actual[mask], prediction[mask]) ** 0.5
        )
    return metrics


def _feature_contract(
    feature_names: list[str], config: IntradayPriceTrainingConfig
) -> dict[str, Any]:
    return {
        "model_scope": "intraday_price_spread_correction",
        "feature_version": FEATURE_VERSION,
        "raw_target": TARGET_SPREAD_COLUMN,
        "raw_model_output": "predicted intraday-minus-day-ahead spread in EUR/MWh",
        "features": feature_names,
        "availability": {
            "intraday_history": "six completed delivery hours before as_of",
            "day_ahead_curve": "auction-cleared curve available before as_of",
            "target": "strictly later than as_of",
        },
        "decision_layer_formula": (
            "For lead_hours = 1: day_ahead_price + correction_weight * predicted_spread. "
            "For lead_hours >= 2: day_ahead_price."
        ),
        "correction_weight": config.correction_weight,
        "excluded_inputs": [
            "future intraday prices",
            "same-contract trade timestamps and volumes",
            "bid/ask order-book data",
        ],
    }


def _split_contract(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    config: IntradayPriceTrainingConfig,
) -> dict[str, Any]:
    return {
        "strategy": "chronological development/validation split",
        "development": {
            "delivery_dates_before": config.validation_start,
            "rows": len(development),
        },
        "validation": {
            "delivery_dates_from": config.validation_start,
            "rows": len(validation),
            "selection_scope": "next delivery hour only",
        },
        "final_fit": "all eligible rows in the configured year after validation",
    }


def _feature_importance(model: Any, feature_names: list[str]) -> dict[str, float]:
    return {
        feature: float(importance)
        for feature, importance in sorted(
            zip(feature_names, model.feature_importances_, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
    }


def _resolve_tracking_uri(config: IntradayPriceTrainingConfig, root: Path) -> str:
    if config.tracking_uri:
        return config.tracking_uri
    if uri := os.getenv("MLFLOW_TRACKING_URI"):
        return uri
    database_path = root / "mlflow" / "mlflow.db"
    database_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{database_path}"


def _registered_version(model_info: Any) -> str | None:
    version = getattr(model_info, "registered_model_version", None)
    return str(version) if version is not None else None
