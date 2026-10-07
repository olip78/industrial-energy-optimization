"""Chronological evaluation for the compact intraday-price forecaster.

The experiment deliberately separates model selection from the untouched 2025
holdout. The day-ahead price is known at every intraday decision time and is
the baseline. CatBoost may correct only the next delivery hour if a 2024-Q4
validation window demonstrates an improvement at that horizon.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from energy.data.intraday_price import DAY_AHEAD_COLUMN, FEATURES_RELATIVE_PATH, _feature_names


TARGET_PRICE_COLUMN = "target_intraday_price_eur_per_mwh"
TARGET_SPREAD_COLUMN = "target_intraday_spread_eur_per_mwh"
BASELINE_VARIANT = "day_ahead_baseline"
CHALLENGER_VARIANT = "spread_catboost_l1_regularised"
PERSISTENCE_VARIANT = "last_intraday_persistence"
HYBRID_VARIANT = "day_ahead_plus_lead_1_spread"


@dataclass(frozen=True)
class IntradayPriceExperimentConfig:
    project_root: Path
    data_path: Path | None = None
    train_year: int = 2024
    test_year: int = 2025
    test_end: str = "2025-09-30"
    validation_start: str = "2024-10-01"
    feature_version: str = "v1"
    artifact_name: str = "intraday_price_v1"
    iterations: int = 500
    random_state: int = 42


@dataclass(frozen=True)
class IntradayPriceExperimentResult:
    artifact_dir: Path
    metrics: pd.DataFrame
    selected_variant: str

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_dir": str(self.artifact_dir),
            "selected_variant": self.selected_variant,
            "metrics": self.metrics.to_dict(orient="records"),
        }


def run_intraday_price_experiment(
    config: IntradayPriceExperimentConfig,
) -> IntradayPriceExperimentResult:
    """Select on a 2024 validation window, then report an untouched 2025 holdout.

    The experiment always saves the CatBoost spread model as a challenger. The
    V1 path uses its correction for lead one only when it beats the known
    day-ahead curve for lead one on the earlier validation period; this prevents
    an attractive 2025 result from silently becoming a tuning decision.
    """

    root = config.project_root.expanduser().resolve()
    path = (config.data_path or root / FEATURES_RELATIVE_PATH).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Intraday MPC feature table is missing: {path}")
    frame = pd.read_parquet(path)
    frame["delivery_date_local"] = pd.to_datetime(frame["delivery_date_local"])
    train = frame.loc[frame["delivery_date_local"].dt.year == config.train_year].copy()
    test = frame.loc[
        (frame["delivery_date_local"].dt.year == config.test_year)
        & (frame["delivery_date_local"] <= pd.Timestamp(config.test_end))
    ].copy()
    if train.empty or test.empty:
        raise ValueError("Chronological train and test partitions must both be non-empty")
    if train["target_valid_time_utc"].max() >= test["as_of_utc"].min():
        raise AssertionError("Temporal partitions overlap")

    validation_start = pd.Timestamp(config.validation_start)
    development = train.loc[train["delivery_date_local"] < validation_start].copy()
    validation = train.loc[train["delivery_date_local"] >= validation_start].copy()
    if development.empty or validation.empty:
        raise ValueError("Development and validation partitions must both be non-empty")
    if development["target_valid_time_utc"].max() >= validation["as_of_utc"].min():
        raise AssertionError("Development and validation periods overlap")

    feature_names = _feature_names(config.feature_version)
    candidate_specs = _candidate_specs(config)
    lead_one_validation = validation.loc[validation["lead_hours"] == 1].copy()
    validation_target = lead_one_validation[TARGET_PRICE_COLUMN].to_numpy(dtype=float)
    validation_baseline = lead_one_validation[DAY_AHEAD_COLUMN].to_numpy(dtype=float)
    validation_metrics: list[dict[str, object]] = [
        _metrics_row(
            BASELINE_VARIANT,
            validation_target,
            validation_baseline,
            config,
            stage="selection_validation_2024_q4",
            lead_hours=None,
            selected_for_v1=False,
        )
    ]
    candidate_scores: list[tuple[str, dict[str, Any], float]] = []
    for name, parameters in candidate_specs:
        model = _fit_spread_model(development, feature_names, parameters)
        prediction = validation_baseline + model.predict(lead_one_validation.loc[:, feature_names])
        score = float(mean_absolute_error(validation_target, prediction))
        candidate_scores.append((name, parameters, score))
        validation_metrics.append(
            _metrics_row(
                name,
                validation_target,
                prediction,
                config,
                stage="selection_validation_2024_q4",
                lead_hours=None,
                selected_for_v1=False,
            )
        )

    best_name, best_parameters, best_mae = min(candidate_scores, key=lambda item: item[2])
    baseline_validation_mae = float(validation_metrics[0]["mae_eur_per_mwh"])
    selected_variant = (
        BASELINE_VARIANT if baseline_validation_mae <= best_mae else HYBRID_VARIANT
    )
    for row in validation_metrics:
        row["selected_for_v1"] = (
            row["variant"] == BASELINE_VARIANT
            if selected_variant == BASELINE_VARIANT
            else row["variant"] == best_name
        )

    validation_full_target = validation[TARGET_PRICE_COLUMN].to_numpy(dtype=float)
    validation_full_baseline = validation[DAY_AHEAD_COLUMN].to_numpy(dtype=float)
    best_validation_prediction = validation_full_baseline + _fit_spread_model(
        development, feature_names, best_parameters
    ).predict(validation.loc[:, feature_names])
    validation_hybrid_prediction = validation_full_baseline.copy()
    validation_lead_one = validation["lead_hours"].to_numpy() == 1
    validation_hybrid_prediction[validation_lead_one] = best_validation_prediction[
        validation_lead_one
    ]
    hybrid_validation_metrics = pd.DataFrame(
        [
            _metrics_row(
                BASELINE_VARIANT,
                validation_full_target,
                validation_full_baseline,
                config,
                stage="selection_validation_2024_q4_all_horizons",
                lead_hours=None,
                selected_for_v1=selected_variant == BASELINE_VARIANT,
            ),
            _metrics_row(
                HYBRID_VARIANT,
                validation_full_target,
                validation_hybrid_prediction,
                config,
                stage="selection_validation_2024_q4_all_horizons",
                lead_hours=None,
                selected_for_v1=selected_variant == HYBRID_VARIANT,
            ),
        ]
    )

    # Refit the selected lead-one challenger on all 2024 data. It is used only
    # at the immediate MPC horizon; the remainder stays on the day-ahead curve.
    challenger = _fit_spread_model(train, feature_names, best_parameters)
    target = test[TARGET_PRICE_COLUMN].to_numpy(dtype=float)
    day_ahead_prediction = test[DAY_AHEAD_COLUMN].to_numpy(dtype=float)
    challenger_prediction = day_ahead_prediction + challenger.predict(test.loc[:, feature_names])
    hybrid_prediction = day_ahead_prediction.copy()
    lead_one_test = test["lead_hours"].to_numpy() == 1
    hybrid_prediction[lead_one_test] = challenger_prediction[lead_one_test]
    variants = {
        BASELINE_VARIANT: day_ahead_prediction,
        PERSISTENCE_VARIANT: test["intraday_price_lag_1_eur_per_mwh"].to_numpy(dtype=float),
        best_name: challenger_prediction,
        HYBRID_VARIANT: hybrid_prediction,
    }

    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for variant, prediction in variants.items():
        is_selected = variant == selected_variant
        prediction_frames.append(
            pd.DataFrame(
                {
                    "variant": variant,
                    "as_of_utc": test["as_of_utc"],
                    "target_valid_time_utc": test["target_valid_time_utc"],
                    "delivery_date_local": test["delivery_date_local"].dt.strftime("%Y-%m-%d"),
                    "decision_hour_local": test["decision_hour_local"],
                    "lead_hours": test["lead_hours"],
                    "actual_intraday_price_eur_per_mwh": target,
                    "prediction_eur_per_mwh": prediction,
                }
            )
        )
        metric_rows.append(
            _metrics_row(
                variant,
                target,
                prediction,
                config,
                stage="future_holdout_2025",
                lead_hours=None,
                selected_for_v1=is_selected,
            )
        )
        for lead_hours, group_indices in test.groupby("lead_hours").groups.items():
            positions = test.index.get_indexer(group_indices)
            metric_rows.append(
                _metrics_row(
                    variant,
                    target[positions],
                    prediction[positions],
                    config,
                    stage="future_holdout_2025",
                    lead_hours=int(lead_hours),
                    selected_for_v1=is_selected,
                )
            )

    metrics = pd.DataFrame(metric_rows)
    overall = metrics.loc[metrics["lead_hours"].isna()].sort_values("mae_eur_per_mwh")
    artifact_dir = root / "artifacts" / "experiments" / config.artifact_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    overall.to_csv(artifact_dir / "metrics.csv", index=False)
    metrics.loc[metrics["lead_hours"].notna()].to_csv(
        artifact_dir / "metrics_by_lead_hour.csv", index=False
    )
    pd.DataFrame(validation_metrics).sort_values("mae_eur_per_mwh").to_csv(
        artifact_dir / "selection_validation_metrics.csv", index=False
    )
    hybrid_validation_metrics.to_csv(
        artifact_dir / "hybrid_validation_metrics.csv", index=False
    )
    pd.concat(prediction_frames, ignore_index=True).to_parquet(
        artifact_dir / "predictions.parquet", index=False
    )
    pd.DataFrame(
        {"feature": feature_names, "importance": challenger.feature_importances_}
    ).sort_values("importance", ascending=False).to_csv(
        artifact_dir / "feature_importance.csv", index=False
    )
    (artifact_dir / "experiment_config.json").write_text(
        json.dumps(
            {
                "train_year": config.train_year,
                "test_year": config.test_year,
                "test_end": config.test_end,
                "validation_start": config.validation_start,
                "feature_version": config.feature_version,
                "artifact_name": config.artifact_name,
                "iterations": config.iterations,
                "random_state": config.random_state,
                "target": (
                    "Intraday Continuous Average Price (DE-LU), represented during "
                    "training as its realised spread to the known day-ahead price."
                ),
                "selection_policy": {
                    "development": "2024-01-01 through 2024-09-30",
                    "validation": "2024-10-01 through 2024-12-31",
                    "criterion": "MAE in EUR/MWh at lead_hours = 1",
                    "baseline": BASELINE_VARIANT,
                    "best_catboost_candidate": best_name,
                    "baseline_lead_1_validation_mae_eur_per_mwh": baseline_validation_mae,
                    "best_catboost_lead_1_validation_mae_eur_per_mwh": best_mae,
                    "selected_for_v1": selected_variant,
                    "rule": (
                        "Use the CatBoost correction only for the next delivery hour "
                        "when it beats the known day-ahead curve at lead one on the "
                        "2024 validation period."
                    ),
                },
                "final_challenger": {
                    "variant": best_name,
                    "configuration_selected_with_2024_only": best_name,
                    "parameters": best_parameters,
                    "fit_data": "all eligible 2024 rows",
                },
            },
            indent=2,
        )
    )
    return IntradayPriceExperimentResult(
        artifact_dir=artifact_dir,
        metrics=overall,
        selected_variant=selected_variant,
    )


def _candidate_specs(config: IntradayPriceExperimentConfig) -> list[tuple[str, dict[str, Any]]]:
    """Small predeclared set; no 2025 result can affect selection."""

    common = {
        "iterations": config.iterations,
        "learning_rate": 0.05,
        "random_seed": config.random_state,
        "verbose": False,
        "allow_writing_files": False,
    }
    return [
        (
            "spread_catboost_l1_depth6",
            {**common, "loss_function": "MAE", "depth": 6, "l2_leaf_reg": 10.0},
        ),
        (
            CHALLENGER_VARIANT,
            {**common, "loss_function": "MAE", "depth": 4, "l2_leaf_reg": 50.0},
        ),
        (
            "spread_catboost_huber_depth4",
            {**common, "loss_function": "Huber:delta=10", "depth": 4, "l2_leaf_reg": 50.0},
        ),
    ]


def _fit_spread_model(
    train: pd.DataFrame,
    feature_names: list[str],
    parameters: dict[str, Any],
) -> CatBoostRegressor:
    model = CatBoostRegressor(**parameters)
    model.fit(train.loc[:, feature_names], train[TARGET_SPREAD_COLUMN])
    return model


def _metrics_row(
    variant: str,
    target: np.ndarray,
    prediction: np.ndarray,
    config: IntradayPriceExperimentConfig,
    *,
    stage: str,
    lead_hours: int | None,
    selected_for_v1: bool,
) -> dict[str, object]:
    return {
        "stage": stage,
        "variant": variant,
        "selected_for_v1": selected_for_v1,
        "train_year": config.train_year,
        "test_year": config.test_year,
        "test_end": config.test_end,
        "rows": int(len(target)),
        "mae_eur_per_mwh": float(mean_absolute_error(target, prediction)),
        "rmse_eur_per_mwh": float(mean_squared_error(target, prediction) ** 0.5),
        "mean_error_eur_per_mwh": float(np.mean(prediction - target)),
        "lead_hours": lead_hours,
    }
