"""Rolling evaluation of intraday weather refreshes for MPC PV forecasts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from energy.data.builder import WEATHER_VARIABLES
from energy.training import pv_mpc_residual as mpc


@dataclass(frozen=True)
class PVMpcWeatherRefreshEvaluationConfig:
    project_root: Path
    train_start_date: str = "2024-03-14"
    rolling_start_date: str = "2025-04-01"
    selection_end_date: str = "2025-06-30"
    holdout_start_date: str = "2025-07-01"
    test_end_date: str = "2025-09-30"
    n_lags: int = 4
    iterations: int = 500
    random_state: int = 42
    artifact_name: str = "pv_mpc_weather_refresh_rolling_v1"


@dataclass(frozen=True)
class PVMpcWeatherRefreshEvaluationResult:
    artifact_dir: Path
    selected_variant: str
    metrics: pd.DataFrame

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_dir": str(self.artifact_dir),
            "selected_variant": self.selected_variant,
            "metrics": self.metrics.to_dict(orient="records"),
        }


def run_pv_mpc_weather_refresh_evaluation(
    config: PVMpcWeatherRefreshEvaluationConfig,
) -> PVMpcWeatherRefreshEvaluationResult:
    """Select fresh-weather features in Q2 and evaluate once on Q3 2025.

    Each calendar-month forecast is fitted from 2024 cross-fitted residuals and
    only those 2025 delivery days that precede the month. The day-ahead model
    itself remains frozen after its 2024 fit.
    """

    root = config.project_root.expanduser().resolve()
    CatBoostRegressor, pvlib = _dependencies()
    base24, fresh24, base25, fresh25 = _prepare_rows(
        config, root, CatBoostRegressor, pvlib
    )
    feature_sets = _feature_sets(config.n_lags)
    predictions: list[pd.DataFrame] = []
    importance_rows: list[dict[str, object]] = []

    frozen_original = _make_model(CatBoostRegressor, config)
    frozen_original.fit(
        base24.loc[:, feature_sets["rolling_original_weather"]],
        base24[mpc.RESIDUAL_TARGET],
        verbose=False,
    )

    starts = pd.date_range(
        config.rolling_start_date,
        pd.Timestamp(config.test_end_date).replace(day=1),
        freq="MS",
    )
    for start in starts:
        end = min(start + pd.offsets.MonthEnd(0), pd.Timestamp(config.test_end_date))
        start_text, end_text = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        prior = fresh25["delivery_date_local"] < start_text
        test_mask = fresh25["delivery_date_local"].between(start_text, end_text)
        train_base = pd.concat([base24, base25.loc[prior]], ignore_index=True)
        train_fresh = pd.concat([fresh24, fresh25.loc[prior]], ignore_index=True)
        test_base = base25.loc[test_mask].reset_index(drop=True)
        test_fresh = fresh25.loc[test_mask].reset_index(drop=True)
        if test_fresh.empty:
            continue
        block = start.strftime("%Y-%m")
        predictions.append(
            _prediction_frame(
                test_fresh,
                test_fresh[mpc.DAY_AHEAD_PREDICTION].to_numpy(),
                "frozen_day_ahead",
                block,
            )
        )
        frozen_prediction = (
            test_base[mpc.DAY_AHEAD_PREDICTION].to_numpy()
            + frozen_original.predict(
                test_base.loc[:, feature_sets["rolling_original_weather"]]
            )
        )
        predictions.append(
            _prediction_frame(
                test_fresh,
                frozen_prediction,
                "frozen_2024_original_weather",
                block,
            )
        )
        for variant, features in feature_sets.items():
            train = train_base if variant == "rolling_original_weather" else train_fresh
            test = test_base if variant == "rolling_original_weather" else test_fresh
            fitted = _make_model(CatBoostRegressor, config)
            fitted.fit(
                train.loc[:, features], train[mpc.RESIDUAL_TARGET], verbose=False
            )
            prediction = test[mpc.DAY_AHEAD_PREDICTION].to_numpy() + fitted.predict(
                test.loc[:, features]
            )
            predictions.append(
                _prediction_frame(test_fresh, prediction, variant, block)
            )
            if end_text == config.test_end_date:
                importance_rows.extend(
                    {
                        "variant": variant,
                        "feature": feature,
                        "importance": float(value),
                    }
                    for feature, value in zip(
                        features, fitted.get_feature_importance(), strict=True
                    )
                )

    output = pd.concat(predictions, ignore_index=True)
    metrics = _metrics(output, config)
    selection = metrics.loc[metrics["period"] == "selection"].copy()
    candidates = selection.loc[
        selection["variant"].str.startswith("rolling_")
        & (selection["variant"] != "rolling_original_weather")
    ]
    selected = str(candidates.sort_values(["mae_w", "rmse_w"]).iloc[0]["variant"])
    holdout = output.loc[
        output["delivery_date_local"].between(
            config.holdout_start_date, config.test_end_date
        )
    ].copy()
    lead_metrics = _lead_metrics(holdout)
    importance = pd.DataFrame(importance_rows).sort_values(
        ["variant", "importance"], ascending=[True, False]
    )

    artifact_dir = root / "artifacts" / "experiments" / config.artifact_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output.to_parquet(artifact_dir / "rolling_predictions.parquet", index=False)
    metrics.to_csv(artifact_dir / "metrics.csv", index=False)
    lead_metrics.to_csv(artifact_dir / "holdout_lead_metrics.csv", index=False)
    importance.to_csv(artifact_dir / "feature_importance.csv", index=False)
    (artifact_dir / "feature_sets.json").write_text(
        json.dumps(feature_sets, indent=2)
    )
    for variant, filename in (
        ("rolling_original_weather", "pv_mpc_predictions_original_weather.parquet"),
        (selected, "pv_mpc_predictions_selected_updated_weather.parquet"),
    ):
        exported = holdout.loc[
            holdout["variant"] == variant,
            [
                "delivery_date_local",
                "decision_hour_local",
                "lead_hours",
                "prediction_w",
            ],
        ].copy()
        exported["variant"] = "direct_residual_catboost"
        exported.to_parquet(artifact_dir / filename, index=False)
    summary = {
        "config": _json_config(config),
        "selected_variant": selected,
        "selection_rule": "lowest Q2 MAE, with RMSE as tie-break",
        "locked_holdout": (
            f"{config.holdout_start_date} through {config.test_end_date}"
        ),
        "training_contract": (
            "At every monthly boundary: 2024 cross-fitted residual rows plus "
            "strictly earlier 2025 days; frozen 2024 day-ahead head."
        ),
        "rows_2024": len(fresh24),
        "rows_2025": len(fresh25),
    }
    (artifact_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return PVMpcWeatherRefreshEvaluationResult(artifact_dir, selected, metrics)


def _prepare_rows(
    config: PVMpcWeatherRefreshEvaluationConfig,
    root: Path,
    CatBoostRegressor: Any,
    pvlib: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_config = _training_config(config, root, 2024, config.train_start_date)
    test_config = _training_config(config, root, 2025, "2025-01-01")
    raw24 = pd.read_parquet(_day_ahead_path(root, 2024))
    raw25 = pd.read_parquet(_day_ahead_path(root, 2025))
    active24 = mpc._prepare_active_frame(raw24, train_config, pvlib)
    active25 = mpc._prepare_active_frame(raw25, test_config, pvlib)
    active25 = active25.loc[
        active25["delivery_date_local"].astype(str) <= config.test_end_date
    ].copy()
    oof24 = mpc._add_oof_day_ahead_prediction(
        active24,
        train_config,
        CatBoostRegressor,
        random_state=config.random_state,
    )
    head = mpc._make_model(CatBoostRegressor, train_config)
    head.fit(
        active24.loc[:, list(mpc.FORECAST_FEATURES)],
        active24[mpc.TARGET_COLUMN],
        verbose=False,
    )
    head25 = active25.copy()
    head25[mpc.DAY_AHEAD_PREDICTION] = np.maximum(
        head.predict(active25.loc[:, list(mpc.FORECAST_FEATURES)]), 0.0
    )
    snapshots24 = mpc._load_weather_snapshots(train_config, root)
    snapshots25 = mpc._load_weather_snapshots(test_config, root)
    assert snapshots24 is not None and snapshots25 is not None
    base24 = mpc._make_direct_residual_rows(oof24, train_config)
    fresh24 = mpc._make_direct_residual_rows(
        oof24, train_config, weather_snapshots=snapshots24
    )
    base25 = mpc._make_direct_residual_rows(head25, test_config)
    fresh25 = mpc._make_direct_residual_rows(
        head25, test_config, weather_snapshots=snapshots25
    )
    base24, fresh24 = _align(base24, fresh24)
    base25, fresh25 = _align(base25, fresh25)
    for frame in (base24, fresh24, base25, fresh25):
        frame["delivery_date_local"] = frame["delivery_date_local"].astype(str)
    return base24, fresh24, base25, fresh25


def _feature_sets(n_lags: int) -> dict[str, list[str]]:
    return {
        "rolling_original_weather": mpc._residual_features(n_lags),
        "rolling_updated_weather_replace": mpc._residual_features(
            n_lags,
            uses_updated_weather=True,
            updated_weather_feature_mode="replace",
        ),
        "rolling_original_plus_revision": mpc._residual_features(
            n_lags,
            uses_updated_weather=True,
            updated_weather_feature_mode="revision",
        ),
        "rolling_legacy_all_weather_features": mpc._residual_features(
            n_lags,
            uses_updated_weather=True,
            updated_weather_feature_mode="legacy",
        ),
    }


def _training_config(
    config: PVMpcWeatherRefreshEvaluationConfig,
    root: Path,
    year: int,
    start: str,
) -> mpc.PVMpcResidualTrainingConfig:
    return mpc.PVMpcResidualTrainingConfig(
        project_root=root,
        year=year,
        train_start_date=start,
        weather_snapshot_path=(
            root
            / "data/features/mpc_pv_weather_ifs_snapshots"
            / f"mpc_pv_weather_ifs_snapshots_{year}.parquet"
        ),
        n_lags=config.n_lags,
        iterations=config.iterations,
        random_state=config.random_state,
    )


def _make_model(CatBoostRegressor: Any, config: PVMpcWeatherRefreshEvaluationConfig) -> Any:
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


def _prediction_frame(
    test: pd.DataFrame,
    prediction: np.ndarray,
    variant: str,
    block: str,
) -> pd.DataFrame:
    result = test.loc[
        :,
        [
            "delivery_date_local",
            "decision_hour_local",
            "remaining_horizon_hours",
            "target_pv_power_w",
            mpc.DAY_AHEAD_PREDICTION,
            "weather_snapshot_lead_hours",
            "weather_snapshot_age_hours",
        ],
    ].copy()
    result = result.rename(
        columns={
            "remaining_horizon_hours": "lead_hours",
            mpc.DAY_AHEAD_PREDICTION: "day_ahead_prediction_w",
        }
    )
    result["variant"] = variant
    result["prediction_w"] = np.maximum(prediction, 0.0)
    result["rolling_block"] = block
    return result


def _metrics(
    predictions: pd.DataFrame,
    config: PVMpcWeatherRefreshEvaluationConfig,
) -> pd.DataFrame:
    periods = (
        ("selection", config.rolling_start_date, config.selection_end_date),
        ("locked_holdout", config.holdout_start_date, config.test_end_date),
        ("all_rolling_oos", config.rolling_start_date, config.test_end_date),
    )
    rows = []
    for period, start, end in periods:
        part = predictions.loc[predictions["delivery_date_local"].between(start, end)]
        for variant, group in part.groupby("variant", sort=False):
            error = group["prediction_w"] - group["target_pv_power_w"]
            rows.append(
                {
                    "period": period,
                    "variant": variant,
                    "rows": len(group),
                    "mae_w": float(
                        mean_absolute_error(
                            group["target_pv_power_w"], group["prediction_w"]
                        )
                    ),
                    "rmse_w": float(
                        mean_squared_error(
                            group["target_pv_power_w"], group["prediction_w"]
                        )
                        ** 0.5
                    ),
                    "p95_absolute_error_w": float(np.quantile(np.abs(error), 0.95)),
                    "bias_w": float(error.mean()),
                }
            )
    return pd.DataFrame(rows)


def _lead_metrics(holdout: pd.DataFrame) -> pd.DataFrame:
    return (
        holdout.assign(
            absolute_error_w=lambda frame: np.abs(
                frame["prediction_w"] - frame["target_pv_power_w"]
            )
        )
        .groupby(["variant", "lead_hours"], as_index=False)
        .agg(mae_w=("absolute_error_w", "mean"), rows=("absolute_error_w", "size"))
    )


def _align(base: pd.DataFrame, fresh: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = [
        "delivery_date_local",
        "decision_hour_local",
        "remaining_horizon_hours",
    ]
    order = fresh.loc[:, keys].set_index(keys).index
    aligned = base.set_index(keys).loc[order].reset_index()
    fresh = fresh.reset_index(drop=True)
    if not aligned.loc[:, keys].equals(fresh.loc[:, keys]):
        raise AssertionError("Original and refreshed weather rows are not aligned")
    return aligned, fresh


def _day_ahead_path(root: Path, year: int) -> Path:
    path = root / "data/features/day_ahead_pv" / f"day_ahead_pv_{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Day-ahead PV data is missing: {path}")
    return path


def _dependencies() -> tuple[Any, Any]:
    try:
        import pvlib
        from catboost import CatBoostRegressor
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Install the project's training dependencies first") from error
    return CatBoostRegressor, pvlib


def _json_config(config: PVMpcWeatherRefreshEvaluationConfig) -> dict[str, object]:
    result = asdict(config)
    result["project_root"] = str(config.project_root)
    return result
