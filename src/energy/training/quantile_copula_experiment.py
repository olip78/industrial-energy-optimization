"""Conditional MultiQuantile forecasts coupled by a historical empirical copula."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from energy.optimization import ReferenceScenario
from energy.training import pv_day_ahead
from energy.training.residual_bootstrap_experiment import (
    HOURS_LOCAL,
    PRICE_TARGET,
    _assert_complete,
    _common_complete_days,
    _ensemble_crps,
    _hourly_metric_table,
    _metric_table,
    _read_price,
    _read_pv,
)
from energy.uncertainty import QuantileEmpiricalCopula


DEFAULT_QUANTILES = (0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95)


@dataclass(frozen=True)
class QuantileCopulaExperimentConfig:
    project_root: Path
    residual_year: int = 2024
    test_start: str = "2025-01-01"
    test_end: str = "2025-09-30"
    initial_train_days: int = 84
    block_days: int = 28
    iterations: int = 500
    quantile_levels: tuple[float, ...] = DEFAULT_QUANTILES
    n_scenarios: int = 500
    random_state: int = 42
    pv_capacity_kwh_per_hour: float = 10.0
    artifact_name: str = "quantile_copula_v1"


@dataclass(frozen=True)
class QuantileCopulaExperimentResult:
    artifact_dir: Path
    copula_days: int
    test_days: int
    metrics: pd.DataFrame

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_dir": str(self.artifact_dir),
            "copula_days": self.copula_days,
            "test_days": self.test_days,
            "metrics": self.metrics.to_dict(orient="records"),
        }


def run_quantile_copula_experiment(
    config: QuantileCopulaExperimentConfig,
) -> QuantileCopulaExperimentResult:
    """Fit leakage-safe quantile marginals and evaluate copula scenarios."""

    _validate_config(config)
    root = config.project_root.expanduser().resolve()
    scenario = ReferenceScenario()
    pv, pv_features = _read_pv(root, config)
    price, price_features = _read_price(root, config)
    rolling_quantiles, crossing_2024 = _rolling_origin_quantiles(
        pv, pv_features, price, price_features, config, scenario
    )
    copula_library = _build_empirical_copula(rolling_quantiles, config)
    generator = QuantileEmpiricalCopula(
        copula_library,
        quantile_levels=config.quantile_levels,
        hours_local=HOURS_LOCAL,
    )
    test_quantiles, crossing_2025 = _future_quantile_forecasts(
        pv, pv_features, price, price_features, config, scenario
    )
    bands, example_scenarios = _evaluate_scenarios(
        test_quantiles, generator, config
    )
    metrics = _metric_table(bands)
    pinball = _pinball_summary(test_quantiles, config)
    metrics = metrics.merge(pinball, on=["target", "unit"], validate="one_to_one")
    calibration = _quantile_calibration(test_quantiles, config)
    hourly_metrics = _hourly_metric_table(bands)
    dependence, training_correlation, sampled_correlation = _dependence_audit(
        copula_library, config
    )
    crossing = pd.concat((crossing_2024, crossing_2025), ignore_index=True)

    artifact_dir = root / "artifacts" / "experiments" / config.artifact_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    rolling_quantiles.to_parquet(
        artifact_dir / "rolling_origin_quantile_predictions.parquet", index=False
    )
    copula_library.to_parquet(
        artifact_dir / "empirical_copula_library.parquet", index=False
    )
    test_quantiles.to_parquet(
        artifact_dir / "test_quantile_forecasts.parquet", index=False
    )
    bands.to_parquet(
        artifact_dir / "out_of_sample_prediction_bands.parquet", index=False
    )
    example_scenarios.to_parquet(
        artifact_dir / "example_scenario_batch.parquet", index=False
    )
    metrics.to_csv(artifact_dir / "metrics.csv", index=False)
    calibration.to_csv(artifact_dir / "quantile_calibration.csv", index=False)
    hourly_metrics.to_csv(artifact_dir / "metrics_by_hour.csv", index=False)
    crossing.to_csv(artifact_dir / "quantile_crossing.csv", index=False)
    dependence.to_csv(artifact_dir / "dependence_summary.csv", index=False)
    training_correlation.to_csv(artifact_dir / "correlation_copula_library.csv")
    sampled_correlation.to_csv(artifact_dir / "correlation_sampled_copula.csv")
    (artifact_dir / "experiment_config.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "project_root": str(root),
                "quantile_levels": list(config.quantile_levels),
                "working_hours_local": list(HOURS_LOCAL),
                "marginal_model": "CatBoost MultiQuantile with monotone rearrangement",
                "copula_contract": (
                    "Complete daily vectors of within-hour ranks of rolling-origin "
                    "P50 residuals; one 32-dimensional historical rank row is sampled jointly"
                ),
                "tail_contract": (
                    "Copula ranks outside the trained P05/P95 grid are clipped to "
                    "the supported marginal endpoints"
                ),
            },
            indent=2,
            default=str,
        )
    )
    return QuantileCopulaExperimentResult(
        artifact_dir=artifact_dir,
        copula_days=len(generator.library_days),
        test_days=int(bands["delivery_date_local"].nunique()),
        metrics=metrics,
    )


def _rolling_origin_quantiles(
    pv: pd.DataFrame,
    pv_features: list[str],
    price: pd.DataFrame,
    price_features: list[str],
    config: QuantileCopulaExperimentConfig,
    scenario: ReferenceScenario,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pv_year = pv.loc[pv["delivery_date_local"].dt.year == config.residual_year].copy()
    price_year = price.loc[
        price["delivery_date_local"].dt.year == config.residual_year
    ].copy()
    common_days = _common_complete_days(pv_year, price_year)
    if len(common_days) <= config.initial_train_days:
        raise ValueError("Not enough complete days for the rolling-origin experiment")
    outputs: list[pd.DataFrame] = []
    crossings: list[dict[str, object]] = []
    forecast_days = common_days[config.initial_train_days :]
    for fold, start in enumerate(range(0, len(forecast_days), config.block_days)):
        test_days = forecast_days[start : start + config.block_days]
        cutoff = pd.Timestamp(test_days[0])
        pv_train = pv_year.loc[pv_year["delivery_date_local"] < cutoff]
        pv_test = pv_year.loc[pv_year["delivery_date_local"].dt.date.isin(test_days)]
        price_train = price_year.loc[price_year["delivery_date_local"] < cutoff]
        price_test = price_year.loc[
            price_year["delivery_date_local"].dt.date.isin(test_days)
        ]
        pv_prediction, pv_crossing = _fit_predict_quantiles(
            pv_train,
            pv_test,
            pv_features,
            pv_day_ahead.TARGET_COLUMN,
            config,
            clip_nonnegative=True,
        )
        price_prediction, price_crossing = _fit_predict_quantiles(
            price_train,
            price_test,
            price_features,
            PRICE_TARGET,
            config,
            clip_nonnegative=False,
        )
        output = _merge_quantile_outputs(
            pv_test,
            price_test,
            pv_prediction / 1_000.0 * scenario.pv_profile_scale,
            price_prediction,
            config,
            scenario,
        )
        output["rolling_fold"] = fold
        outputs.append(output)
        crossings.extend(
            (
                _crossing_row("rolling_2024", "pv_generation", fold, pv_crossing),
                _crossing_row("rolling_2024", "day_ahead_price", fold, price_crossing),
            )
        )
    result = pd.concat(outputs, ignore_index=True).sort_values(
        ["delivery_date_local", "hour_local"]
    )
    _assert_complete(result)
    return result.reset_index(drop=True), pd.DataFrame(crossings)


def _future_quantile_forecasts(
    pv: pd.DataFrame,
    pv_features: list[str],
    price: pd.DataFrame,
    price_features: list[str],
    config: QuantileCopulaExperimentConfig,
    scenario: ReferenceScenario,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_end = pd.Timestamp(f"{config.residual_year}-12-31")
    start = pd.Timestamp(config.test_start)
    end = pd.Timestamp(config.test_end)
    pv_train = pv.loc[pv["delivery_date_local"] <= train_end]
    price_train = price.loc[price["delivery_date_local"] <= train_end]
    pv_test = pv.loc[pv["delivery_date_local"].between(start, end, inclusive="both")]
    price_test = price.loc[
        price["delivery_date_local"].between(start, end, inclusive="both")
    ]
    common_days = _common_complete_days(pv_test, price_test)
    pv_test = pv_test.loc[pv_test["delivery_date_local"].dt.date.isin(common_days)].copy()
    price_test = price_test.loc[
        price_test["delivery_date_local"].dt.date.isin(common_days)
    ].copy()
    pv_prediction, pv_crossing = _fit_predict_quantiles(
        pv_train,
        pv_test,
        pv_features,
        pv_day_ahead.TARGET_COLUMN,
        config,
        clip_nonnegative=True,
    )
    price_prediction, price_crossing = _fit_predict_quantiles(
        price_train,
        price_test,
        price_features,
        PRICE_TARGET,
        config,
        clip_nonnegative=False,
    )
    output = _merge_quantile_outputs(
        pv_test,
        price_test,
        pv_prediction / 1_000.0 * scenario.pv_profile_scale,
        price_prediction,
        config,
        scenario,
    ).sort_values(["delivery_date_local", "hour_local"])
    _assert_complete(output)
    crossing = pd.DataFrame(
        [
            _crossing_row("future_2025", "pv_generation", None, pv_crossing),
            _crossing_row("future_2025", "day_ahead_price", None, price_crossing),
        ]
    )
    return output.reset_index(drop=True), crossing


def _fit_predict_quantiles(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target: str,
    config: QuantileCopulaExperimentConfig,
    *,
    clip_nonnegative: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    alpha = ",".join(f"{value:g}" for value in config.quantile_levels)
    model = CatBoostRegressor(
        loss_function=f"MultiQuantile:alpha={alpha}",
        iterations=config.iterations,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=10.0,
        random_seed=config.random_state,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(train.loc[:, features], train[target], verbose=False)
    raw = np.asarray(model.predict(test.loc[:, features]), dtype=float)
    if raw.ndim == 1:
        raw = raw[:, None]
    crossing_rows = np.any(np.diff(raw, axis=1) < 0, axis=1)
    rearranged = np.sort(raw, axis=1)
    if clip_nonnegative:
        rearranged = np.maximum(rearranged, 0.0)
    return rearranged, {
        "rows": float(len(raw)),
        "crossing_rows_before_rearrangement": float(crossing_rows.sum()),
        "crossing_rate_before_rearrangement": float(crossing_rows.mean()),
    }


def _merge_quantile_outputs(
    pv_frame: pd.DataFrame,
    price_frame: pd.DataFrame,
    pv_prediction_kwh: np.ndarray,
    price_prediction: np.ndarray,
    config: QuantileCopulaExperimentConfig,
    scenario: ReferenceScenario,
) -> pd.DataFrame:
    keys = ["delivery_date_local", "hour_local"]
    pv_output = pv_frame.loc[:, [*keys, pv_day_ahead.TARGET_COLUMN]].copy()
    pv_output["actual_pv_kwh"] = (
        pv_output[pv_day_ahead.TARGET_COLUMN] / 1_000.0 * scenario.pv_profile_scale
    )
    pv_output = pv_output.drop(columns=[pv_day_ahead.TARGET_COLUMN])
    price_output = price_frame.loc[:, [*keys, PRICE_TARGET]].rename(
        columns={PRICE_TARGET: "actual_price_eur_per_mwh"}
    )
    for index, level in enumerate(config.quantile_levels):
        pv_output[_quantile_column("pv", level)] = pv_prediction_kwh[:, index]
        price_output[_quantile_column("price", level)] = price_prediction[:, index]
    return pv_output.merge(price_output, on=keys, validate="one_to_one")


def _build_empirical_copula(
    rolling: pd.DataFrame, config: QuantileCopulaExperimentConfig
) -> pd.DataFrame:
    median = min(config.quantile_levels, key=lambda value: abs(value - 0.5))
    if not np.isclose(median, 0.5):
        raise ValueError("quantile_levels must contain 0.5 for the copula residuals")
    frame = rolling.copy()
    frame["pv_residual"] = frame["actual_pv_kwh"] - frame[_quantile_column("pv", median)]
    frame["price_residual"] = (
        frame["actual_price_eur_per_mwh"]
        - frame[_quantile_column("price", median)]
    )
    days = sorted(pd.to_datetime(frame["delivery_date_local"]).dt.date.unique())
    output = pd.DataFrame({"delivery_date_local": days})
    output = output.set_index("delivery_date_local")
    for target, residual_column in (
        ("pv", "pv_residual"),
        ("price", "price_residual"),
    ):
        pivot = frame.pivot(
            index="delivery_date_local", columns="hour_local", values=residual_column
        ).loc[:, list(HOURS_LOCAL)]
        pivot.index = pd.to_datetime(pivot.index).date
        for hour in HOURS_LOCAL:
            ranks = pivot[hour].rank(method="average")
            output[f"{target}_u_h{hour:02d}"] = (ranks - 0.5) / len(ranks)
    return output.reset_index()


def _evaluate_scenarios(
    forecasts: pd.DataFrame,
    generator: QuantileEmpiricalCopula,
    config: QuantileCopulaExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    example: pd.DataFrame | None = None
    pv_columns = [_quantile_column("pv", value) for value in config.quantile_levels]
    price_columns = [
        _quantile_column("price", value) for value in config.quantile_levels
    ]
    for delivery_day, day in forecasts.groupby("delivery_date_local", sort=True):
        day = day.sort_values("hour_local")
        day_date = pd.Timestamp(delivery_day).date()
        batch = generator.sample(
            pv_quantiles_kwh=day[pv_columns].to_numpy(dtype=float),
            price_quantiles_eur_per_mwh=day[price_columns].to_numpy(dtype=float),
            n_scenarios=config.n_scenarios,
            random_state=config.random_state
            + int(pd.Timestamp(delivery_day).strftime("%Y%m%d")),
            as_of_date=day_date,
            pv_capacity_kwh_per_hour=config.pv_capacity_kwh_per_hour,
        )
        output = day.loc[
            :, ["delivery_date_local", "hour_local", "actual_pv_kwh", "actual_price_eur_per_mwh"]
        ].copy()
        output["point_pv_kwh"] = day[_quantile_column("pv", 0.5)].to_numpy()
        output["point_price_eur_per_mwh"] = day[
            _quantile_column("price", 0.5)
        ].to_numpy()
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
            example = pd.DataFrame(
                {
                    "delivery_date_local": str(day_date),
                    "scenario_id": np.repeat(
                        np.arange(config.n_scenarios), len(HOURS_LOCAL)
                    ),
                    "source_copula_day": np.repeat(
                        [str(value) for value in batch.source_copula_days],
                        len(HOURS_LOCAL),
                    ),
                    "hour_local": np.tile(HOURS_LOCAL, config.n_scenarios),
                    "pv_uniform_rank": batch.pv_uniform_ranks.reshape(-1),
                    "price_uniform_rank": batch.price_uniform_ranks.reshape(-1),
                    "pv_kwh": batch.pv_kwh.reshape(-1),
                    "day_ahead_price_eur_per_mwh": batch.day_ahead_price_eur_per_mwh.reshape(-1),
                }
            )
    if example is None:
        raise ValueError("No future days were available for scenario evaluation")
    return pd.concat(rows, ignore_index=True), example


def _pinball_summary(
    forecasts: pd.DataFrame, config: QuantileCopulaExperimentConfig
) -> pd.DataFrame:
    rows = []
    for target, unit, actual_column, prefix in (
        ("pv_generation", "kWh", "actual_pv_kwh", "pv"),
        ("day_ahead_price", "EUR/MWh", "actual_price_eur_per_mwh", "price"),
    ):
        actual = forecasts[actual_column].to_numpy(dtype=float)
        losses = []
        for level in config.quantile_levels:
            prediction = forecasts[_quantile_column(prefix, level)].to_numpy(dtype=float)
            error = actual - prediction
            losses.append(np.mean(np.maximum(level * error, (level - 1.0) * error)))
        rows.append(
            {
                "target": target,
                "unit": unit,
                "mean_pinball_loss": float(np.mean(losses)),
            }
        )
    return pd.DataFrame(rows)


def _quantile_calibration(
    forecasts: pd.DataFrame, config: QuantileCopulaExperimentConfig
) -> pd.DataFrame:
    rows = []
    for target, unit, actual_column, prefix in (
        ("pv_generation", "kWh", "actual_pv_kwh", "pv"),
        ("day_ahead_price", "EUR/MWh", "actual_price_eur_per_mwh", "price"),
    ):
        actual = forecasts[actual_column].to_numpy(dtype=float)
        for level in config.quantile_levels:
            prediction = forecasts[_quantile_column(prefix, level)].to_numpy(dtype=float)
            error = actual - prediction
            rows.append(
                {
                    "target": target,
                    "unit": unit,
                    "quantile": level,
                    "empirical_cdf": float(np.mean(actual <= prediction)),
                    "calibration_error": float(np.mean(actual <= prediction) - level),
                    "pinball_loss": float(
                        np.mean(np.maximum(level * error, (level - 1.0) * error))
                    ),
                }
            )
    return pd.DataFrame(rows)


def _dependence_audit(
    library: pd.DataFrame, config: QuantileCopulaExperimentConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rank_columns = [column for column in library if column != "delivery_date_local"]
    training = library[rank_columns].corr()
    rng = np.random.default_rng(config.random_state)
    sampled = library.iloc[
        rng.choice(len(library), size=10_000, replace=True)
    ][rank_columns].reset_index(drop=True)
    sampled_correlation = sampled.corr()
    difference = sampled_correlation.to_numpy() - training.to_numpy()
    summary = pd.DataFrame(
        [
            {
                "training_days": len(library),
                "sampled_rows": len(sampled),
                "rank_correlation_rmse": float(np.sqrt(np.mean(difference**2))),
                "rank_correlation_max_abs_error": float(np.max(np.abs(difference))),
            }
        ]
    )
    return summary, training, sampled_correlation


def _quantile_column(prefix: str, level: float) -> str:
    return f"{prefix}_q{int(round(level * 100)):02d}"


def _crossing_row(scope: str, target: str, fold, values: dict[str, float]):
    return {"scope": scope, "target": target, "fold": fold, **values}


def _validate_config(config: QuantileCopulaExperimentConfig) -> None:
    levels = np.asarray(config.quantile_levels, dtype=float)
    if len(levels) < 3 or not np.all(np.diff(levels) > 0):
        raise ValueError("quantile_levels must contain at least three increasing values")
    if levels[0] <= 0 or levels[-1] >= 1 or not np.isclose(levels, 0.5).any():
        raise ValueError("quantile_levels must lie inside (0, 1) and contain 0.5")
    if config.initial_train_days < 30:
        raise ValueError("initial_train_days must be at least 30")
    if config.block_days < 1 or config.iterations < 1 or config.n_scenarios < 10:
        raise ValueError("block_days/iterations must be positive and scenarios at least 10")
    if pd.Timestamp(config.test_start) > pd.Timestamp(config.test_end):
        raise ValueError("test_start must not be after test_end")
    if config.pv_capacity_kwh_per_hour <= 0:
        raise ValueError("pv_capacity_kwh_per_hour must be positive")
