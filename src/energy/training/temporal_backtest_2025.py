"""One frozen, chronological 2024 -> 2025 evaluation for the project models.

The module intentionally evaluates finished V1 models rather than selecting
models or hyperparameters on 2025.  It covers three forecasting contracts:

* the deployable pointwise day-ahead PV forecast;
* the frozen day-ahead PV forecast plus intraday MPC residual corrections;
* the compact day-ahead electricity-price comparison.

The price source remains hourly only through September 2025, so that component
has a shorter 2025 evaluation window.  PV evaluation uses all available 2025
rows.  This is an offline backtest, not a live MPC or market simulator.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from energy.modeling.pv_mimo import (
    MIMOLSTMTrainingConfig,
    MIMOMLPTrainingConfig,
    _maximum_remaining_active_horizon,
    build_lstm_sequence_samples,
    build_mimo_residual_samples,
    fit_mimo_residual_lstm,
    fit_mimo_residual_mlp,
    trajectory_prediction_frame,
)
from energy.training import pv_day_ahead as day_ahead
from energy.training import pv_mpc_residual as mpc
from energy.training.day_ahead_price_experiment import (
    DayAheadPriceExperimentConfig,
    run_day_ahead_price_experiment,
)


@dataclass(frozen=True)
class TemporalBacktest2025Config:
    """Fixed settings for the chronological 2024 -> 2025 backtest."""

    project_root: Path
    train_year: int = 2024
    test_year: int = 2025
    n_lags: int = 4
    catboost_iterations: int = 500
    neural_epochs: int = 300
    random_state: int = 42


@dataclass(frozen=True)
class TemporalBacktest2025Result:
    """Artifact location and compact metrics table for the completed run."""

    artifact_dir: Path
    metrics: pd.DataFrame

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_dir": str(self.artifact_dir),
            "metrics": self.metrics.to_dict(orient="records"),
        }


def run_temporal_backtest_2025(
    config: TemporalBacktest2025Config,
) -> TemporalBacktest2025Result:
    """Fit on all of 2024 and evaluate only on the future 2025 period.

    2025 rows are never used for model selection, early stopping, scaling or
    construction of the residual-model training target.  The residual models
    use out-of-fold 2024 head predictions when constructing their training
    rows, then use one head model fitted on all 2024 rows in 2025.
    """

    _validate_config(config)
    project_root = config.project_root.expanduser().resolve()
    train_raw = _read_pv_data(project_root, config.train_year)
    test_raw = _read_pv_data(project_root, config.test_year)
    CatBoostRegressor, pvlib = _pv_dependencies()

    artifact_dir = (
        project_root
        / "artifacts"
        / "experiments"
        / "temporal_backtest_v1"
        / f"{config.train_year}_to_{config.test_year}"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)

    day_ahead_predictions, day_ahead_metrics = _evaluate_day_ahead_pv(
        train_raw,
        test_raw,
        config,
        CatBoostRegressor,
        pvlib,
    )
    mpc_predictions, mpc_metrics, mpc_lead_metrics = _evaluate_mpc_pv(
        train_raw,
        test_raw,
        config,
        CatBoostRegressor,
        pvlib,
    )
    price_metrics = _evaluate_day_ahead_price(project_root, config)

    metrics = pd.concat(
        [day_ahead_metrics, mpc_metrics, price_metrics], ignore_index=True
    )
    metrics.to_csv(artifact_dir / "metrics.csv", index=False)
    mpc_lead_metrics.to_csv(artifact_dir / "pv_mpc_metrics_by_lead_hour.csv", index=False)
    day_ahead_predictions.to_parquet(
        artifact_dir / "pv_day_ahead_hourly_predictions.parquet", index=False
    )
    mpc_predictions.to_parquet(
        artifact_dir / "pv_mpc_trajectory_predictions.parquet", index=False
    )

    summary = {
        "config": _json_config(config),
        "selection_policy": (
            "All model settings were fixed before inspection of 2025 metrics. "
            "The test year is used only for the final forward evaluation."
        ),
        "pv_day_ahead": {
            "training_rows": int(len(train_raw)),
            "test_rows": int(len(test_raw)),
            "target": day_ahead.TARGET_COLUMN,
        },
        "pv_mpc": {
            "n_lags": config.n_lags,
            "base_forecast_training": (
                "CatBoost head model fitted on all active 2024 rows; OOF 2024 "
                "head forecasts are used only to form honest residual-model training rows."
            ),
            "test_contract": (
                "At every decision hour, only frozen day-ahead values and actual PV "
                "from earlier solar-active hours are available to correction models."
            ),
        },
        "day_ahead_price": {
            "test_window": "2025-01-01 through 2025-09-30",
            "reason_for_shorter_window": (
                "The source becomes 15-minute after September 2025, while this V1 "
                "price experiment deliberately uses hourly auction data."
            ),
        },
    }
    (artifact_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return TemporalBacktest2025Result(artifact_dir=artifact_dir, metrics=metrics)


def _read_pv_data(project_root: Path, year: int) -> pd.DataFrame:
    path = project_root / "data" / "features" / "day_ahead_pv" / f"day_ahead_pv_{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"The day-ahead PV data for {year} is missing: {path}. "
            "Run `energy build-training-data --project-root . --year <year>` first."
        )
    return pd.read_parquet(path)


def _pv_dependencies() -> tuple[Any, Any]:
    try:
        import pvlib
        from catboost import CatBoostRegressor
    except ImportError as error:  # pragma: no cover - depends on caller environment
        raise RuntimeError(
            "Install the optional training and neural dependencies before running "
            "the temporal backtest."
        ) from error
    return CatBoostRegressor, pvlib


def _evaluate_day_ahead_pv(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    config: TemporalBacktest2025Config,
    CatBoostRegressor: Any,
    pvlib: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    day_ahead_config = day_ahead.PVDayAheadTrainingConfig(
        project_root=config.project_root,
        year=config.train_year,
        random_state=config.random_state,
    )
    train, features = day_ahead._prepare_day_ahead_frame(train_raw, day_ahead_config, pvlib)
    test, test_features = day_ahead._prepare_day_ahead_frame(test_raw, day_ahead_config, pvlib)
    if features != test_features:
        raise AssertionError("The day-ahead PV feature contract differs between years")

    model = day_ahead._make_model(
        CatBoostRegressor,
        day_ahead_config,
        iterations=config.catboost_iterations,
        use_best_model=False,
    )
    model.fit(train.loc[:, features], train[day_ahead.TARGET_COLUMN], verbose=False)
    prediction = day_ahead._predict_nonnegative(model, test.loc[:, features])
    output = test.loc[:, [
        "valid_time_utc",
        "delivery_date_local",
        "hour_local",
        day_ahead.TARGET_COLUMN,
        "clear_sky_ghi_w_m2",
    ]].copy()
    output = output.rename(columns={day_ahead.TARGET_COLUMN: "target_pv_power_w"})
    output["variant"] = "day_ahead_catboost"
    output["prediction_w"] = prediction
    output["is_solar_active"] = (
        output["clear_sky_ghi_w_m2"] >= day_ahead_config.min_clear_sky_ghi
    )

    train_peak_w = float(train[day_ahead.TARGET_COLUMN].max())
    metric_rows = [
        _pv_metric_row(
            component="pv_day_ahead",
            variant="day_ahead_catboost",
            evaluation_slice="all_local_06_21_hours",
            frame=output,
            prediction_column="prediction_w",
            train_peak_w=train_peak_w,
            train_year=config.train_year,
            test_year=config.test_year,
        ),
        _pv_metric_row(
            component="pv_day_ahead",
            variant="day_ahead_catboost",
            evaluation_slice="solar_active_hours",
            frame=output.loc[output["is_solar_active"]].copy(),
            prediction_column="prediction_w",
            train_peak_w=train_peak_w,
            train_year=config.train_year,
            test_year=config.test_year,
        ),
    ]
    return output, pd.DataFrame(metric_rows)


def _evaluate_mpc_pv(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    config: TemporalBacktest2025Config,
    CatBoostRegressor: Any,
    pvlib: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mpc_config = mpc.PVMpcResidualTrainingConfig(
        project_root=config.project_root,
        year=config.train_year,
        n_lags=config.n_lags,
        iterations=config.catboost_iterations,
        random_state=config.random_state,
    )
    train_active = mpc._prepare_active_frame(train_raw, mpc_config, pvlib)
    test_active = mpc._prepare_active_frame(test_raw, mpc_config, pvlib)

    # Cross-fitted 2024 head predictions let the correction models learn errors
    # with realistic magnitude.  The final 2025 head model is then fit once on
    # every eligible 2024 row.
    train_with_oof_head = mpc._add_oof_day_ahead_prediction(
        train_active,
        mpc_config,
        CatBoostRegressor,
        random_state=config.random_state,
    )
    frozen_head = mpc._make_model(CatBoostRegressor, mpc_config)
    frozen_head.fit(
        train_active.loc[:, list(mpc.FORECAST_FEATURES)],
        train_active[mpc.TARGET_COLUMN],
        verbose=False,
    )
    test_with_head = test_active.copy()
    test_with_head[mpc.DAY_AHEAD_PREDICTION] = np.maximum(
        frozen_head.predict(test_active.loc[:, list(mpc.FORECAST_FEATURES)]), 0.0
    )

    direct_rows_train = mpc._make_direct_residual_rows(train_with_oof_head, mpc_config)
    direct_rows_test = mpc._make_direct_residual_rows(test_with_head, mpc_config)
    residual_features = mpc._residual_features(config.n_lags)
    direct_model = mpc._make_model(CatBoostRegressor, mpc_config)
    direct_model.fit(
        direct_rows_train.loc[:, residual_features],
        direct_rows_train[mpc.RESIDUAL_TARGET],
        verbose=False,
    )
    direct_rows_test["direct_residual_prediction_w"] = direct_model.predict(
        direct_rows_test.loc[:, residual_features]
    )
    direct_rows_test["direct_prediction_w"] = np.maximum(
        direct_rows_test[mpc.DAY_AHEAD_PREDICTION]
        + direct_rows_test["direct_residual_prediction_w"],
        0.0,
    )
    direct_rows_test["persistence_prediction_w"] = np.maximum(
        direct_rows_test[mpc.DAY_AHEAD_PREDICTION]
        + direct_rows_test["residual_lag_1_w"],
        0.0,
    )

    mimo_predictions = _mimo_predictions(
        train_with_oof_head,
        test_with_head,
        config,
    )
    direct_output = direct_rows_test.loc[:, [
        "delivery_date_local",
        "week_index",
        "decision_hour_local",
        "remaining_horizon_hours",
        "target_pv_power_w",
        mpc.DAY_AHEAD_PREDICTION,
        "persistence_prediction_w",
        "direct_prediction_w",
    ]].copy()
    direct_output = direct_output.rename(
        columns={
            "remaining_horizon_hours": "lead_hours",
            mpc.DAY_AHEAD_PREDICTION: "day_ahead_prediction_w",
        }
    )
    long_direct = direct_output.melt(
        id_vars=[
            "delivery_date_local",
            "week_index",
            "decision_hour_local",
            "lead_hours",
            "target_pv_power_w",
            "day_ahead_prediction_w",
        ],
        value_vars=["persistence_prediction_w", "direct_prediction_w"],
        var_name="variant",
        value_name="prediction_w",
    )
    long_direct["variant"] = long_direct["variant"].map(
        {
            "persistence_prediction_w": "persistence_residual",
            "direct_prediction_w": "direct_residual_catboost",
        }
    )
    baseline = direct_output.loc[
        :,
        [
            "delivery_date_local",
            "week_index",
            "decision_hour_local",
            "lead_hours",
            "target_pv_power_w",
            "day_ahead_prediction_w",
        ],
    ].copy()
    baseline["variant"] = "frozen_day_ahead_baseline"
    baseline["prediction_w"] = baseline["day_ahead_prediction_w"]

    combined = pd.concat(
        [
            baseline,
            long_direct,
            mimo_predictions,
        ],
        ignore_index=True,
        sort=False,
    )
    expected_pairs = len(baseline)
    counts = combined.groupby("variant", sort=False).size()
    if not (counts == expected_pairs).all():
        raise AssertionError(
            "MPC variants must be scored on identical decision/target pairs; "
            f"got {counts.to_dict()} versus {expected_pairs} baseline pairs"
        )

    train_peak_w = float(train_active[mpc.TARGET_COLUMN].max())
    metric_rows = [
        _pv_metric_row(
            component="pv_mpc",
            variant=variant,
            evaluation_slice="solar_active_mpc_trajectory_pairs",
            frame=group,
            prediction_column="prediction_w",
            train_peak_w=train_peak_w,
            train_year=config.train_year,
            test_year=config.test_year,
        )
        for variant, group in combined.groupby("variant", sort=False)
    ]
    lead_rows = [
        _pv_metric_row(
            component="pv_mpc",
            variant=variant,
            evaluation_slice="solar_active_mpc_trajectory_pairs",
            frame=group,
            prediction_column="prediction_w",
            train_peak_w=train_peak_w,
            train_year=config.train_year,
            test_year=config.test_year,
            lead_hours=int(lead_hours),
        )
        for (variant, lead_hours), group in combined.groupby(["variant", "lead_hours"], sort=False)
    ]
    return combined, pd.DataFrame(metric_rows), pd.DataFrame(lead_rows)


def _mimo_predictions(
    train_with_oof_head: pd.DataFrame,
    test_with_head: pd.DataFrame,
    config: TemporalBacktest2025Config,
) -> pd.DataFrame:
    maximum_horizon = max(
        _maximum_remaining_active_horizon(train_with_oof_head, ["delivery_date_local"]),
        _maximum_remaining_active_horizon(test_with_head, ["delivery_date_local"]),
    )
    train_samples = build_mimo_residual_samples(
        train_with_oof_head,
        future_features=mpc.FORECAST_FEATURES,
        n_lags=config.n_lags,
        max_horizon=maximum_horizon,
    )
    test_samples = build_mimo_residual_samples(
        test_with_head,
        future_features=mpc.FORECAST_FEATURES,
        n_lags=config.n_lags,
        max_horizon=maximum_horizon,
    )

    mlp_fit = fit_mimo_residual_mlp(
        train_samples,
        validation=None,
        config=MIMOMLPTrainingConfig(
            max_epochs=config.neural_epochs,
            patience=None,
            random_state=config.random_state,
        ),
    )
    mlp = trajectory_prediction_frame(
        test_samples, mlp_fit.predict_residual_w(test_samples)
    )
    mlp = mlp.rename(columns={"mimo_prediction_w": "prediction_w"})
    mlp["variant"] = "mimo_mlp_residual"

    train_sequences = build_lstm_sequence_samples(
        train_samples,
        future_features=mpc.FORECAST_FEATURES,
        n_lags=config.n_lags,
    )
    test_sequences = build_lstm_sequence_samples(
        test_samples,
        future_features=mpc.FORECAST_FEATURES,
        n_lags=config.n_lags,
    )
    lstm_fit = fit_mimo_residual_lstm(
        train_sequences,
        validation=None,
        config=MIMOLSTMTrainingConfig(
            max_epochs=config.neural_epochs,
            patience=None,
            random_state=config.random_state,
        ),
    )
    lstm = trajectory_prediction_frame(
        test_samples, lstm_fit.predict_residual_w(test_sequences)
    )
    lstm = lstm.rename(columns={"mimo_prediction_w": "prediction_w"})
    lstm["variant"] = "mimo_lstm_residual"

    columns = [
        "delivery_date_local",
        "week_index",
        "decision_hour_local",
        "lead_hours",
        "target_pv_power_w",
        "day_ahead_prediction_w",
        "variant",
        "prediction_w",
    ]
    return pd.concat([mlp.loc[:, columns], lstm.loc[:, columns]], ignore_index=True)


def _evaluate_day_ahead_price(
    project_root: Path,
    config: TemporalBacktest2025Config,
) -> pd.DataFrame:
    result = run_day_ahead_price_experiment(
        DayAheadPriceExperimentConfig(
            project_root=project_root,
            train_end=f"{config.train_year}-12-31",
            test_start=f"{config.test_year}-01-01",
            test_end=f"{config.test_year}-09-30",
            iterations=config.catboost_iterations,
            random_state=config.random_state,
        )
    )
    price = result.metrics.copy()
    price.insert(0, "component", "day_ahead_price")
    price.insert(2, "evaluation_slice", "hourly_2025_jan_sep")
    price.insert(3, "train_year", config.train_year)
    price.insert(4, "test_year", config.test_year)
    price.insert(5, "unit", "EUR/MWh")
    return price


def _pv_metric_row(
    *,
    component: str,
    variant: str,
    evaluation_slice: str,
    frame: pd.DataFrame,
    prediction_column: str,
    train_peak_w: float,
    train_year: int,
    test_year: int,
    lead_hours: int | None = None,
) -> dict[str, Any]:
    if frame.empty:
        raise ValueError(f"Cannot compute {component}/{variant} metrics on zero rows")
    target = frame["target_pv_power_w"].to_numpy(dtype=float)
    prediction = frame[prediction_column].to_numpy(dtype=float)
    row: dict[str, Any] = {
        "component": component,
        "variant": variant,
        "evaluation_slice": evaluation_slice,
        "train_year": train_year,
        "test_year": test_year,
        "unit": "W",
        "rows": int(len(frame)),
        "mae_w": float(mean_absolute_error(target, prediction)),
        "rmse_w": float(mean_squared_error(target, prediction) ** 0.5),
        "nmae_train_observed_peak": float(mean_absolute_error(target, prediction) / train_peak_w),
        "nrmse_train_observed_peak": float(
            (mean_squared_error(target, prediction) ** 0.5) / train_peak_w
        ),
    }
    if lead_hours is not None:
        row["lead_hours"] = lead_hours
    return row


def _json_config(config: TemporalBacktest2025Config) -> dict[str, Any]:
    result = asdict(config)
    result["project_root"] = str(config.project_root.expanduser().resolve())
    return result


def _validate_config(config: TemporalBacktest2025Config) -> None:
    if config.train_year >= config.test_year:
        raise ValueError("train_year must precede test_year")
    if config.n_lags < 1:
        raise ValueError("n_lags must be at least one")
    if config.catboost_iterations < 1 or config.neural_epochs < 1:
        raise ValueError("catboost_iterations and neural_epochs must be positive")
