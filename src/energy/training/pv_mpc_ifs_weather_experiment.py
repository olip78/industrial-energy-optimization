"""Chronological ablation of fresh IFS weather in direct MPC PV correction."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from energy.training import pv_mpc_residual as mpc


@dataclass(frozen=True)
class PVMpcIfsWeatherExperimentConfig:
    project_root: Path
    train_start_date: str = "2024-03-14"
    test_start_date: str = "2025-01-01"
    test_end_date: str = "2025-09-30"
    n_lags: int = 4
    iterations: int = 500
    random_state: int = 42
    artifact_name: str = "pv_mpc_ifs_weather_update_2024_to_2025"


@dataclass(frozen=True)
class PVMpcIfsWeatherExperimentResult:
    artifact_dir: Path
    metrics: pd.DataFrame

    def to_dict(self) -> dict[str, object]:
        return {"artifact_dir": str(self.artifact_dir), "metrics": self.metrics.to_dict(orient="records")}


def run_pv_mpc_ifs_weather_experiment(
    config: PVMpcIfsWeatherExperimentConfig,
) -> PVMpcIfsWeatherExperimentResult:
    """Evaluate same-sample residual models with D-1 and fresh IFS weather.

    The two correction models share the frozen day-ahead head prediction and
    actual-PV residual lags.  Only future weather inputs differ.  Both are
    restricted to decision/target pairs with an on-time published IFS run.
    """

    root = config.project_root.expanduser().resolve()
    CatBoostRegressor, pvlib = _dependencies()
    train_config = _model_config(root, 2024, config.train_start_date, config)
    test_config = _model_config(root, 2025, config.test_start_date, config)
    train_raw = pd.read_parquet(_day_ahead_path(root, 2024))
    test_raw = pd.read_parquet(_day_ahead_path(root, 2025))
    train_active = mpc._prepare_active_frame(train_raw, train_config, pvlib)
    test_active = mpc._prepare_active_frame(test_raw, test_config, pvlib)
    test_active = test_active.loc[
        test_active["delivery_date_local"].astype(str) <= config.test_end_date
    ].copy()
    train_snapshots = mpc._load_weather_snapshots(train_config, root)
    test_snapshots = mpc._load_weather_snapshots(test_config, root)
    assert train_snapshots is not None and test_snapshots is not None

    train_head_oof = mpc._add_oof_day_ahead_prediction(
        train_active, train_config, CatBoostRegressor, random_state=config.random_state
    )
    head_model = mpc._make_model(CatBoostRegressor, train_config)
    head_model.fit(train_active.loc[:, list(mpc.FORECAST_FEATURES)], train_active[mpc.TARGET_COLUMN], verbose=False)
    test_head = test_active.copy()
    test_head[mpc.DAY_AHEAD_PREDICTION] = np.maximum(
        head_model.predict(test_active.loc[:, list(mpc.FORECAST_FEATURES)]), 0.0
    )

    base_train = mpc._make_direct_residual_rows(train_head_oof, train_config)
    base_test = mpc._make_direct_residual_rows(test_head, test_config)
    fresh_train = mpc._make_direct_residual_rows(
        train_head_oof, train_config, weather_snapshots=train_snapshots
    )
    fresh_test = mpc._make_direct_residual_rows(
        test_head, test_config, weather_snapshots=test_snapshots
    )
    keys = ["delivery_date_local", "decision_hour_local", "remaining_horizon_hours"]
    base_train = _align_to_fresh_pairs(base_train, fresh_train, keys)
    base_test = _align_to_fresh_pairs(base_test, fresh_test, keys)
    _assert_same_pairs(base_train, fresh_train, keys, "train")
    _assert_same_pairs(base_test, fresh_test, keys, "test")

    base_features = mpc._residual_features(config.n_lags)
    fresh_features = mpc._residual_features(config.n_lags, uses_updated_weather=True)
    base_model = mpc._make_model(CatBoostRegressor, train_config)
    base_model.fit(base_train.loc[:, base_features], base_train[mpc.RESIDUAL_TARGET], verbose=False)
    fresh_model = mpc._make_model(CatBoostRegressor, train_config)
    fresh_model.fit(fresh_train.loc[:, fresh_features], fresh_train[mpc.RESIDUAL_TARGET], verbose=False)

    output = fresh_test.loc[:, [*keys, "target_pv_power_w", mpc.DAY_AHEAD_PREDICTION, "weather_snapshot_lead_hours", "weather_snapshot_age_hours"]].copy()
    output = output.rename(columns={mpc.DAY_AHEAD_PREDICTION: "day_ahead_prediction_w"})
    output["direct_residual_original_weather_w"] = np.maximum(
        base_test[mpc.DAY_AHEAD_PREDICTION].to_numpy()
        + base_model.predict(base_test.loc[:, base_features]),
        0.0,
    )
    output["direct_residual_updated_ifs_weather_w"] = np.maximum(
        fresh_test[mpc.DAY_AHEAD_PREDICTION].to_numpy()
        + fresh_model.predict(fresh_test.loc[:, fresh_features]),
        0.0,
    )
    metrics = _metrics(output)
    artifact_dir = root / "artifacts" / "experiments" / config.artifact_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output.to_parquet(artifact_dir / "trajectory_predictions.parquet", index=False)
    metrics.to_csv(artifact_dir / "metrics.csv", index=False)
    (artifact_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": _config_dict(config),
                "rows": len(output),
                "sample_contract": (
                    "All variants share identical MPC decision/target pairs, frozen day-ahead "
                    "predictions and factual residual lags. Only future weather inputs differ."
                ),
                "metrics": metrics.to_dict(orient="records"),
            },
            indent=2,
        )
    )
    return PVMpcIfsWeatherExperimentResult(artifact_dir=artifact_dir, metrics=metrics)


def _model_config(
    root: Path,
    year: int,
    start_date: str,
    config: PVMpcIfsWeatherExperimentConfig,
) -> mpc.PVMpcResidualTrainingConfig:
    return mpc.PVMpcResidualTrainingConfig(
        project_root=root,
        year=year,
        weather_snapshot_path=(
            root / "data" / "features" / "mpc_pv_weather_ifs_snapshots" / f"mpc_pv_weather_ifs_snapshots_{year}.parquet"
        ),
        train_start_date=start_date,
        n_lags=config.n_lags,
        iterations=config.iterations,
        random_state=config.random_state,
    )


def _day_ahead_path(root: Path, year: int) -> Path:
    path = root / "data" / "features" / "day_ahead_pv" / f"day_ahead_pv_{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Day-ahead PV data is missing: {path}")
    return path


def _align_to_fresh_pairs(base: pd.DataFrame, fresh: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    order = fresh.loc[:, keys].set_index(keys).index
    indexed = base.set_index(keys)
    if not order.isin(indexed.index).all():
        raise AssertionError("Fresh-weather pairs are absent from the original-weather rows")
    return indexed.loc[order].reset_index()


def _assert_same_pairs(base: pd.DataFrame, fresh: pd.DataFrame, keys: list[str], split: str) -> None:
    if not base.loc[:, keys].reset_index(drop=True).equals(fresh.loc[:, keys].reset_index(drop=True)):
        raise AssertionError(f"Original and fresh-weather {split} samples differ")


def _metrics(frame: pd.DataFrame) -> pd.DataFrame:
    target = frame["target_pv_power_w"]
    rows = []
    for variant, prediction in (
        ("frozen_day_ahead", frame["day_ahead_prediction_w"]),
        ("direct_residual_original_weather", frame["direct_residual_original_weather_w"]),
        ("direct_residual_updated_ifs_weather", frame["direct_residual_updated_ifs_weather_w"]),
    ):
        rows.append(
            {
                "variant": variant,
                "mae_w": float(mean_absolute_error(target, prediction)),
                "rmse_w": float(mean_squared_error(target, prediction) ** 0.5),
                "rows": int(len(frame)),
            }
        )
    return pd.DataFrame(rows)


def _dependencies() -> tuple[Any, Any]:
    try:
        import pvlib
        from catboost import CatBoostRegressor
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise RuntimeError("Install the project's training dependencies first") from error
    return CatBoostRegressor, pvlib


def _config_dict(config: PVMpcIfsWeatherExperimentConfig) -> dict[str, object]:
    result = asdict(config)
    result["project_root"] = str(config.project_root)
    return result
