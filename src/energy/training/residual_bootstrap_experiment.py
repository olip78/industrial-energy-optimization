"""Rolling-origin residual library and joint bootstrap evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from energy.optimization.scenario import ReferenceScenario
from energy.training import pv_day_ahead
from energy.training.day_ahead_price_experiment import (
    DayAheadPriceExperimentConfig,
    _read_inputs as read_price_inputs,
)
from energy.uncertainty import JointResidualBootstrap


PRICE_TARGET = "day_ahead_price_eur_per_mwh"
HOURS_LOCAL = tuple(range(6, 22))


@dataclass(frozen=True)
class ResidualBootstrapExperimentConfig:
    project_root: Path
    residual_year: int = 2024
    test_start: str = "2025-01-01"
    test_end: str = "2025-09-30"
    initial_train_days: int = 84
    block_days: int = 28
    iterations: int = 500
    n_scenarios: int = 500
    random_state: int = 42
    center_residuals: bool = False
    pv_capacity_kwh_per_hour: float = 10.0
    artifact_name: str = "joint_residual_bootstrap_v1"


@dataclass(frozen=True)
class ResidualBootstrapExperimentResult:
    artifact_dir: Path
    residual_days: int
    test_days: int
    metrics: pd.DataFrame

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_dir": str(self.artifact_dir),
            "residual_days": self.residual_days,
            "test_days": self.test_days,
            "metrics": self.metrics.to_dict(orient="records"),
        }


def run_residual_bootstrap_experiment(
    config: ResidualBootstrapExperimentConfig,
) -> ResidualBootstrapExperimentResult:
    """Build 2024 rolling-origin errors and evaluate bootstrap bands in 2025."""

    _validate_config(config)
    root = config.project_root.expanduser().resolve()
    scenario = ReferenceScenario()
    pv, pv_features = _read_pv(root, config)
    price, price_features = _read_price(root, config)

    residual_library = _rolling_origin_residuals(
        pv,
        pv_features,
        price,
        price_features,
        config,
        scenario,
    )
    generator = JointResidualBootstrap(
        residual_library,
        hours_local=HOURS_LOCAL,
        center_residuals=config.center_residuals,
    )
    point_test = _future_point_forecasts(
        pv,
        pv_features,
        price,
        price_features,
        config,
        scenario,
    )
    bands, example_scenarios = _evaluate_scenarios(point_test, generator, config)
    metrics = _metric_table(bands)
    hourly_metrics = _hourly_metric_table(bands)
    residual_diagnostics = _residual_diagnostics(residual_library)
    dependence_summary, training_correlation, sampled_correlation = _dependence_audit(
        generator, config
    )

    artifact_dir = root / "artifacts" / "experiments" / config.artifact_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    residual_library.to_parquet(artifact_dir / "rolling_origin_residual_library.parquet", index=False)
    bands.to_parquet(artifact_dir / "out_of_sample_prediction_bands.parquet", index=False)
    example_scenarios.to_parquet(artifact_dir / "example_scenario_batch.parquet", index=False)
    metrics.to_csv(artifact_dir / "metrics.csv", index=False)
    hourly_metrics.to_csv(artifact_dir / "metrics_by_hour.csv", index=False)
    residual_diagnostics.to_csv(artifact_dir / "residual_diagnostics.csv", index=False)
    dependence_summary.to_csv(artifact_dir / "dependence_summary.csv", index=False)
    training_correlation.to_csv(artifact_dir / "correlation_training_library.csv")
    sampled_correlation.to_csv(artifact_dir / "correlation_bootstrap_sample.csv")
    (artifact_dir / "experiment_config.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "project_root": str(root),
                "working_hours_local": list(HOURS_LOCAL),
                "pv_point_model": "existing day-ahead CatBoost feature/model contract",
                "price_point_model": "existing spatial-weather CatBoost feature/model contract",
                "residual_contract": (
                    "paired complete 16-hour PV and price forecast errors from expanding-window "
                    "out-of-sample predictions; source days are sampled jointly"
                ),
                "evaluation_contract": (
                    "residual library ends in 2024; all interval and CRPS metrics use the "
                    "untouched 2025-01-01 through 2025-09-30 common period"
                ),
            },
            indent=2,
            default=str,
        )
    )
    return ResidualBootstrapExperimentResult(
        artifact_dir=artifact_dir,
        residual_days=len(generator.library_days),
        test_days=int(bands["delivery_date_local"].nunique()),
        metrics=metrics,
    )


def _read_pv(
    root: Path, config: ResidualBootstrapExperimentConfig
) -> tuple[pd.DataFrame, list[str]]:
    try:
        import pvlib
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Install the training dependencies before running this experiment") from error
    frames: list[pd.DataFrame] = []
    features: list[str] | None = None
    pv_config = pv_day_ahead.PVDayAheadTrainingConfig(
        project_root=root,
        year=config.residual_year,
        hour_start=HOURS_LOCAL[0],
        hour_end=HOURS_LOCAL[-1],
        random_state=config.random_state,
    )
    for year in sorted({config.residual_year, pd.Timestamp(config.test_start).year, pd.Timestamp(config.test_end).year}):
        path = root / "data" / "features" / "day_ahead_pv" / f"day_ahead_pv_{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Day-ahead PV data is missing: {path}")
        prepared, year_features = pv_day_ahead._prepare_day_ahead_frame(
            pd.read_parquet(path), pv_config, pvlib
        )
        if features is not None and features != year_features:
            raise AssertionError("PV feature contract differs between years")
        features = year_features
        frames.append(prepared)
    result = pd.concat(frames, ignore_index=True)
    result["delivery_date_local"] = pd.to_datetime(result["delivery_date_local"])
    return result, list(features or [])


def _read_price(
    root: Path, config: ResidualBootstrapExperimentConfig
) -> tuple[pd.DataFrame, list[str]]:
    frame, groups = read_price_inputs(DayAheadPriceExperimentConfig(project_root=root))
    frame["delivery_date_local"] = pd.to_datetime(frame["delivery_date_local"])
    features = [*groups["price_only"], *groups["spatial_weather"]]
    if frame[features].isna().any().any():
        raise ValueError("Day-ahead price feature table contains missing values")
    return frame, features


def _rolling_origin_residuals(
    pv: pd.DataFrame,
    pv_features: list[str],
    price: pd.DataFrame,
    price_features: list[str],
    config: ResidualBootstrapExperimentConfig,
    scenario: ReferenceScenario,
) -> pd.DataFrame:
    year = config.residual_year
    pv_year = pv.loc[pv["delivery_date_local"].dt.year == year].copy()
    price_year = price.loc[price["delivery_date_local"].dt.year == year].copy()
    price_working = price_year.loc[price_year["hour_local"].isin(HOURS_LOCAL)]
    common_days = _common_complete_days(pv_year, price_working)
    if len(common_days) <= config.initial_train_days:
        raise ValueError("Not enough common days for the configured initial training window")

    outputs: list[pd.DataFrame] = []
    forecast_days = common_days[config.initial_train_days :]
    for fold, start in enumerate(range(0, len(forecast_days), config.block_days)):
        test_days = forecast_days[start : start + config.block_days]
        cutoff = pd.Timestamp(test_days[0])
        pv_train = pv_year.loc[pv_year["delivery_date_local"] < cutoff]
        pv_test = pv_year.loc[pv_year["delivery_date_local"].dt.date.isin(test_days)]
        price_train = price_year.loc[price_year["delivery_date_local"] < cutoff]
        price_test = price_year.loc[
            price_year["delivery_date_local"].dt.date.isin(test_days)
            & price_year["hour_local"].isin(HOURS_LOCAL)
        ]
        pv_prediction = _fit_predict_pv(pv_train, pv_test, pv_features, config)
        price_prediction = _fit_predict_price(
            price_train, price_test, price_features, config
        )
        pv_output = pv_test.loc[
            :, ["delivery_date_local", "hour_local", pv_day_ahead.TARGET_COLUMN]
        ].copy()
        pv_output["pv_prediction_w"] = pv_prediction
        price_output = price_test.loc[
            :, ["delivery_date_local", "hour_local", PRICE_TARGET]
        ].copy()
        price_output["price_prediction_eur_per_mwh"] = price_prediction
        merged = pv_output.merge(
            price_output,
            on=["delivery_date_local", "hour_local"],
            validate="one_to_one",
        )
        merged["rolling_fold"] = fold
        outputs.append(merged)

    result = pd.concat(outputs, ignore_index=True).sort_values(
        ["delivery_date_local", "hour_local"]
    )
    result["pv_residual_w"] = (
        result[pv_day_ahead.TARGET_COLUMN] - result["pv_prediction_w"]
    )
    result["pv_residual_kwh"] = (
        result["pv_residual_w"] / 1_000.0 * scenario.pv_profile_scale
    )
    result["price_residual_eur_per_mwh"] = (
        result[PRICE_TARGET] - result["price_prediction_eur_per_mwh"]
    )
    _assert_complete(result)
    result["delivery_date_local"] = result["delivery_date_local"].dt.strftime("%Y-%m-%d")
    return result.reset_index(drop=True)


def _future_point_forecasts(
    pv: pd.DataFrame,
    pv_features: list[str],
    price: pd.DataFrame,
    price_features: list[str],
    config: ResidualBootstrapExperimentConfig,
    scenario: ReferenceScenario,
) -> pd.DataFrame:
    train_end = pd.Timestamp(f"{config.residual_year}-12-31")
    start = pd.Timestamp(config.test_start)
    end = pd.Timestamp(config.test_end)
    pv_train = pv.loc[pv["delivery_date_local"] <= train_end]
    pv_test = pv.loc[pv["delivery_date_local"].between(start, end, inclusive="both")]
    price_train = price.loc[price["delivery_date_local"] <= train_end]
    price_test = price.loc[
        price["delivery_date_local"].between(start, end, inclusive="both")
        & price["hour_local"].isin(HOURS_LOCAL)
    ]
    common_days = _common_complete_days(pv_test, price_test)
    pv_test = pv_test.loc[pv_test["delivery_date_local"].dt.date.isin(common_days)].copy()
    price_test = price_test.loc[
        price_test["delivery_date_local"].dt.date.isin(common_days)
    ].copy()
    pv_prediction = _fit_predict_pv(pv_train, pv_test, pv_features, config)
    price_prediction = _fit_predict_price(price_train, price_test, price_features, config)
    pv_output = pv_test.loc[
        :, ["delivery_date_local", "hour_local", pv_day_ahead.TARGET_COLUMN]
    ].copy()
    pv_output["actual_pv_kwh"] = (
        pv_output[pv_day_ahead.TARGET_COLUMN] / 1_000.0 * scenario.pv_profile_scale
    )
    pv_output["point_pv_kwh"] = pv_prediction / 1_000.0 * scenario.pv_profile_scale
    price_output = price_test.loc[
        :, ["delivery_date_local", "hour_local", PRICE_TARGET]
    ].copy()
    price_output = price_output.rename(columns={PRICE_TARGET: "actual_price_eur_per_mwh"})
    price_output["point_price_eur_per_mwh"] = price_prediction
    result = pv_output.merge(
        price_output,
        on=["delivery_date_local", "hour_local"],
        validate="one_to_one",
    )
    _assert_complete(result)
    return result.sort_values(["delivery_date_local", "hour_local"]).reset_index(drop=True)


def _fit_predict_pv(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    config: ResidualBootstrapExperimentConfig,
) -> np.ndarray:
    model_config = pv_day_ahead.PVDayAheadTrainingConfig(
        project_root=config.project_root,
        random_state=config.random_state,
    )
    model = pv_day_ahead._make_model(
        CatBoostRegressor,
        model_config,
        iterations=config.iterations,
        use_best_model=False,
    )
    model.fit(train.loc[:, features], train[pv_day_ahead.TARGET_COLUMN], verbose=False)
    return pv_day_ahead._predict_nonnegative(model, test.loc[:, features])


def _fit_predict_price(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    config: ResidualBootstrapExperimentConfig,
) -> np.ndarray:
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
    model.fit(train.loc[:, features], train[PRICE_TARGET])
    return model.predict(test.loc[:, features])


def _evaluate_scenarios(
    point_test: pd.DataFrame,
    generator: JointResidualBootstrap,
    config: ResidualBootstrapExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    example: pd.DataFrame | None = None
    for delivery_day, day in point_test.groupby("delivery_date_local", sort=True):
        day = day.sort_values("hour_local")
        day_date = pd.Timestamp(delivery_day).date()
        batch = generator.sample(
            point_pv_kwh=day["point_pv_kwh"].to_numpy(dtype=float),
            point_price_eur_per_mwh=day["point_price_eur_per_mwh"].to_numpy(dtype=float),
            n_scenarios=config.n_scenarios,
            random_state=config.random_state + int(pd.Timestamp(delivery_day).strftime("%Y%m%d")),
            as_of_date=day_date,
            pv_capacity_kwh_per_hour=config.pv_capacity_kwh_per_hour,
        )
        output = day.copy()
        for prefix, scenarios, actual_column in (
            ("pv", batch.pv_kwh, "actual_pv_kwh"),
            ("price", batch.day_ahead_price_eur_per_mwh, "actual_price_eur_per_mwh"),
        ):
            output[f"{prefix}_scenario_mean"] = scenarios.mean(axis=0)
            output[f"{prefix}_p10"] = np.quantile(scenarios, 0.10, axis=0)
            output[f"{prefix}_p50"] = np.quantile(scenarios, 0.50, axis=0)
            output[f"{prefix}_p90"] = np.quantile(scenarios, 0.90, axis=0)
            output[f"{prefix}_crps"] = _ensemble_crps(
                scenarios, output[actual_column].to_numpy(dtype=float)
            )
        rows.append(output)
        if example is None:
            scenario_id = np.repeat(np.arange(config.n_scenarios), len(HOURS_LOCAL))
            example = pd.DataFrame(
                {
                    "delivery_date_local": str(day_date),
                    "scenario_id": scenario_id,
                    "source_residual_day": np.repeat(
                        [str(value) for value in batch.source_residual_days],
                        len(HOURS_LOCAL),
                    ),
                    "hour_local": np.tile(HOURS_LOCAL, config.n_scenarios),
                    "pv_kwh": batch.pv_kwh.reshape(-1),
                    "day_ahead_price_eur_per_mwh": batch.day_ahead_price_eur_per_mwh.reshape(-1),
                }
            )
    if example is None:
        raise ValueError("No future test days were available")
    return pd.concat(rows, ignore_index=True), example


def _metric_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specs = (
        (
            "pv_generation",
            "kWh",
            "actual_pv_kwh",
            "point_pv_kwh",
            "pv_scenario_mean",
            "pv_p10",
            "pv_p90",
            "pv_crps",
        ),
        (
            "day_ahead_price",
            "EUR/MWh",
            "actual_price_eur_per_mwh",
            "point_price_eur_per_mwh",
            "price_scenario_mean",
            "price_p10",
            "price_p90",
            "price_crps",
        ),
    )
    for target, unit, actual_col, point_col, mean_col, low_col, high_col, crps_col in specs:
        actual = frame[actual_col].to_numpy(dtype=float)
        point = frame[point_col].to_numpy(dtype=float)
        scenario_mean = frame[mean_col].to_numpy(dtype=float)
        rows.append(
            {
                "target": target,
                "unit": unit,
                "rows": len(frame),
                "point_mae": float(mean_absolute_error(actual, point)),
                "point_rmse": float(mean_squared_error(actual, point) ** 0.5),
                "scenario_mean_mae": float(mean_absolute_error(actual, scenario_mean)),
                "scenario_mean_rmse": float(mean_squared_error(actual, scenario_mean) ** 0.5),
                "p10_p90_coverage": float(
                    ((actual >= frame[low_col]) & (actual <= frame[high_col])).mean()
                ),
                "p10_p90_mean_width": float((frame[high_col] - frame[low_col]).mean()),
                "ensemble_crps": float(frame[crps_col].mean()),
            }
        )
    return pd.DataFrame(rows)


def _hourly_metric_table(frame: pd.DataFrame) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for target, actual, low, high in (
        ("pv_generation", "actual_pv_kwh", "pv_p10", "pv_p90"),
        ("day_ahead_price", "actual_price_eur_per_mwh", "price_p10", "price_p90"),
    ):
        prepared = frame.assign(
            _inside=(frame[actual] >= frame[low]) & (frame[actual] <= frame[high]),
            _width=frame[high] - frame[low],
        )
        table = (
            prepared.groupby("hour_local")
            .agg(
                rows=(actual, "size"),
                p10_p90_coverage=("_inside", "mean"),
                p10_p90_mean_width=("_width", "mean"),
            )
            .reset_index()
        )
        table.insert(0, "target", target)
        outputs.append(table)
    return pd.concat(outputs, ignore_index=True)


def _residual_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for target, column, unit in (
        ("pv_generation", "pv_residual_kwh", "kWh"),
        ("day_ahead_price", "price_residual_eur_per_mwh", "EUR/MWh"),
    ):
        table = frame.groupby("hour_local")[column].agg(
            rows="size", mean="mean", std="std", p10=lambda x: x.quantile(0.1), p90=lambda x: x.quantile(0.9)
        ).reset_index()
        table.insert(0, "unit", unit)
        table.insert(0, "target", target)
        outputs.append(table)
    return pd.concat(outputs, ignore_index=True)


def _dependence_audit(
    generator: JointResidualBootstrap,
    config: ResidualBootstrapExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    blocks = generator.residual_blocks
    training = blocks.corr()
    rng = np.random.default_rng(config.random_state)
    sampled = blocks.iloc[rng.choice(len(blocks), size=10_000, replace=True)].reset_index(drop=True)
    sampled_correlation = sampled.corr()
    difference = sampled_correlation.to_numpy() - training.to_numpy()
    summary = pd.DataFrame(
        [
            {
                "training_days": len(blocks),
                "sampled_blocks": len(sampled),
                "correlation_rmse": float(np.sqrt(np.mean(difference**2))),
                "correlation_max_abs_error": float(np.max(np.abs(difference))),
            }
        ]
    )
    return summary, training, sampled_correlation


def _ensemble_crps(scenarios: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Empirical ensemble CRPS for every forecast horizon."""

    values = np.asarray(scenarios, dtype=float)
    observations = np.asarray(actual, dtype=float)
    first = np.mean(np.abs(values - observations[None, :]), axis=0)
    ordered = np.sort(values, axis=0)
    count = len(ordered)
    coefficients = (2 * np.arange(count) - count + 1)[:, None]
    second = np.sum(coefficients * ordered, axis=0) / (count**2)
    return first - second


def _common_complete_days(left: pd.DataFrame, right: pd.DataFrame) -> list[Any]:
    expected = set(HOURS_LOCAL)
    left_days = {
        pd.Timestamp(day).date()
        for day, group in left.groupby("delivery_date_local")
        if set(group["hour_local"]) == expected
    }
    right_days = {
        pd.Timestamp(day).date()
        for day, group in right.groupby("delivery_date_local")
        if set(group["hour_local"]) == expected
    }
    return sorted(left_days & right_days)


def _assert_complete(frame: pd.DataFrame) -> None:
    counts = frame.groupby("delivery_date_local")["hour_local"].agg(["count", "nunique"])
    if not ((counts["count"] == len(HOURS_LOCAL)) & (counts["nunique"] == len(HOURS_LOCAL))).all():
        raise AssertionError("At least one daily trajectory is incomplete or duplicated")


def _validate_config(config: ResidualBootstrapExperimentConfig) -> None:
    if config.initial_train_days < 30:
        raise ValueError("initial_train_days must be at least 30")
    if config.block_days < 1 or config.iterations < 1 or config.n_scenarios < 10:
        raise ValueError("block_days and iterations must be positive; n_scenarios must be at least 10")
    if pd.Timestamp(config.test_start) > pd.Timestamp(config.test_end):
        raise ValueError("test_start must not be after test_end")
    if config.pv_capacity_kwh_per_hour <= 0:
        raise ValueError("pv_capacity_kwh_per_hour must be positive")
