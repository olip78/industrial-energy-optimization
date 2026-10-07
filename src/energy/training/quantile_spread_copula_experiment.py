"""Day-ahead quantile scenarios for PV, auction price and intraday spread.

This V2 experiment extends the existing leakage-safe PV/price MultiQuantile
forecasts with a third marginal for the realised intraday-minus-day-ahead
spread.  Hourly conformal corrections are fitted on rolling-origin 2024
predictions, and one empirical copula row supplies a coherent 48-dimensional
daily rank path (three targets by sixteen operating hours).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from energy.training.economic_backtest import _read_intraday_actual
from energy.training.quantile_copula_experiment import (
    DEFAULT_QUANTILES,
    _fit_predict_quantiles,
    _quantile_column,
)
from energy.training.residual_bootstrap_experiment import (
    HOURS_LOCAL,
    _ensemble_crps,
    _read_price,
)
from energy.uncertainty import CentralIntervalCQR


SPREAD_TARGET = "actual_spread_eur_per_mwh"
TARGETS = (
    ("pv_generation", "pv", "actual_pv_kwh", "kWh", 0.0, 10.0),
    (
        "day_ahead_price",
        "price",
        "actual_price_eur_per_mwh",
        "EUR/MWh",
        None,
        None,
    ),
    (
        "intraday_spread",
        "spread",
        SPREAD_TARGET,
        "EUR/MWh",
        None,
        None,
    ),
)


@dataclass(frozen=True)
class QuantileSpreadCopulaExperimentConfig:
    project_root: Path
    source_artifact_name: str = "quantile_copula_v1"
    residual_year: int = 2024
    test_start: str = "2025-01-01"
    test_end: str = "2025-09-30"
    iterations: int = 500
    quantile_levels: tuple[float, ...] = DEFAULT_QUANTILES
    n_scenarios: int = 500
    random_state: int = 42
    pv_capacity_kwh_per_hour: float = 10.0
    artifact_name: str = "quantile_spread_copula_v2"


@dataclass(frozen=True)
class QuantileSpreadCopulaExperimentResult:
    artifact_dir: Path
    calibration_days: int
    test_days: int
    metrics: pd.DataFrame

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_dir": str(self.artifact_dir),
            "calibration_days": self.calibration_days,
            "test_days": self.test_days,
            "metrics": self.metrics.to_dict(orient="records"),
        }


@dataclass(frozen=True)
class ThreeTargetScenarioBatch:
    pv_kwh: np.ndarray
    day_ahead_price_eur_per_mwh: np.ndarray
    intraday_spread_eur_per_mwh: np.ndarray
    source_copula_days: tuple[object, ...]
    uniform_ranks: dict[str, np.ndarray]


def run_quantile_spread_copula_experiment(
    config: QuantileSpreadCopulaExperimentConfig,
) -> QuantileSpreadCopulaExperimentResult:
    """Fit the spread marginal, conformalize all marginals and audit scenarios."""

    _validate_config(config)
    root = config.project_root.expanduser().resolve()
    source = root / "artifacts" / "experiments" / config.source_artifact_name
    rolling_base = pd.read_parquet(
        source / "rolling_origin_quantile_predictions.parquet"
    )
    future_base = pd.read_parquet(source / "test_quantile_forecasts.parquet")
    rolling_base = _normalise_dates(rolling_base)
    future_base = _normalise_dates(future_base)

    spread, spread_features = _read_day_ahead_spread_frame(root, config)
    rolling_raw, rolling_crossing = _add_rolling_spread_quantiles(
        rolling_base, spread, spread_features, config
    )
    future_raw, future_crossing = _add_future_spread_quantiles(
        future_base, spread, spread_features, config
    )
    copula_library = _build_empirical_copula(rolling_raw, config)
    future_calibrated, corrections = _conformalize(
        rolling_raw, future_raw, config
    )

    raw_bands, _ = _evaluate_scenarios(
        future_raw, copula_library, config, generator="raw"
    )
    calibrated_bands, example = _evaluate_scenarios(
        future_calibrated, copula_library, config, generator="cqr"
    )
    metrics = pd.concat(
        (
            _metric_table(raw_bands, future_raw, config, "raw"),
            _metric_table(
                calibrated_bands, future_calibrated, config, "cqr"
            ),
        ),
        ignore_index=True,
    )
    calibration = pd.concat(
        (
            _quantile_calibration(future_raw, config).assign(generator="raw"),
            _quantile_calibration(future_calibrated, config).assign(
                generator="cqr"
            ),
        ),
        ignore_index=True,
    )
    dependence = _dependence_summary(copula_library)

    artifact_dir = root / "artifacts" / "experiments" / config.artifact_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    rolling_raw.to_parquet(
        artifact_dir / "rolling_origin_quantile_predictions.parquet", index=False
    )
    future_raw.to_parquet(
        artifact_dir / "test_raw_quantile_forecasts.parquet", index=False
    )
    future_calibrated.to_parquet(
        artifact_dir / "test_calibrated_quantile_forecasts.parquet", index=False
    )
    copula_library.to_parquet(
        artifact_dir / "empirical_copula_library.parquet", index=False
    )
    raw_bands.to_parquet(
        artifact_dir / "raw_prediction_bands.parquet", index=False
    )
    calibrated_bands.to_parquet(
        artifact_dir / "cqr_prediction_bands.parquet", index=False
    )
    example.to_parquet(artifact_dir / "example_scenario_batch.parquet", index=False)
    corrections.to_csv(artifact_dir / "conformal_corrections.csv", index=False)
    calibration.to_csv(artifact_dir / "quantile_calibration.csv", index=False)
    metrics.to_csv(artifact_dir / "metrics.csv", index=False)
    dependence.to_csv(artifact_dir / "dependence_summary.csv", index=False)
    pd.DataFrame([rolling_crossing, future_crossing]).to_csv(
        artifact_dir / "spread_quantile_crossing.csv", index=False
    )
    (artifact_dir / "experiment_config.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "project_root": str(root),
                "quantile_levels": list(config.quantile_levels),
                "working_hours_local": list(HOURS_LOCAL),
                "information_cutoff": (
                    "Every spread feature is inherited from the day-ahead price "
                    "table and is available before the auction; realised day-ahead "
                    "and intraday prices are targets only."
                ),
                "dependence_contract": (
                    "One historical rolling-origin rank row jointly supplies PV, "
                    "day-ahead-price and intraday-spread ranks for all 16 hours."
                ),
                "calibration_contract": (
                    "Hourly symmetric CQR corrections use only rolling-origin "
                    "2024 errors; the 2025 period is untouched."
                ),
            },
            indent=2,
            default=str,
        )
    )
    return QuantileSpreadCopulaExperimentResult(
        artifact_dir=artifact_dir,
        calibration_days=int(rolling_raw["delivery_date_local"].nunique()),
        test_days=int(future_calibrated["delivery_date_local"].nunique()),
        metrics=metrics,
    )


def sample_three_target_scenarios(
    *,
    day_forecasts: pd.DataFrame,
    copula_library: pd.DataFrame,
    quantile_levels: tuple[float, ...],
    n_scenarios: int,
    random_state: int,
    as_of_date,
    pv_capacity_kwh_per_hour: float,
) -> ThreeTargetScenarioBatch:
    """Invert three conditional marginal grids with one sampled copula row."""

    day = day_forecasts.sort_values("hour_local")
    if tuple(day["hour_local"].astype(int)) != HOURS_LOCAL:
        raise ValueError("day_forecasts must contain the complete operating window")
    library = copula_library.copy()
    library["delivery_date_local"] = pd.to_datetime(
        library["delivery_date_local"]
    ).dt.date
    eligible = library.loc[library["delivery_date_local"] < as_of_date]
    if eligible.empty:
        raise ValueError("No copula rows precede as_of_date")
    rng = np.random.default_rng(random_state)
    positions = rng.choice(len(eligible), size=n_scenarios, replace=True)
    sampled = eligible.iloc[positions]
    outputs: dict[str, np.ndarray] = {}
    ranks: dict[str, np.ndarray] = {}
    for _, prefix, _, _, _, _ in TARGETS:
        rank_columns = [f"{prefix}_u_h{hour:02d}" for hour in HOURS_LOCAL]
        uniform = sampled[rank_columns].to_numpy(dtype=float)
        quantile_columns = [
            _quantile_column(prefix, level) for level in quantile_levels
        ]
        marginal = day[quantile_columns].to_numpy(dtype=float)
        outputs[prefix] = _inverse_marginals(uniform, marginal, quantile_levels)
        ranks[prefix] = uniform
    outputs["pv"] = np.clip(
        outputs["pv"], 0.0, pv_capacity_kwh_per_hour
    )
    return ThreeTargetScenarioBatch(
        pv_kwh=outputs["pv"],
        day_ahead_price_eur_per_mwh=outputs["price"],
        intraday_spread_eur_per_mwh=outputs["spread"],
        source_copula_days=tuple(sampled["delivery_date_local"]),
        uniform_ranks=ranks,
    )


def _read_day_ahead_spread_frame(
    root: Path, config: QuantileSpreadCopulaExperimentConfig
) -> tuple[pd.DataFrame, list[str]]:
    price, features = _read_price(root, config)
    intraday = _read_intraday_actual(root)
    price["valid_time_utc"] = pd.to_datetime(price["valid_time_utc"], utc=True)
    merged = price.merge(
        intraday, on="valid_time_utc", how="inner", validate="one_to_one"
    )
    merged[SPREAD_TARGET] = (
        merged["actual_intraday_price_eur_per_mwh"]
        - merged["day_ahead_price_eur_per_mwh"]
    )
    if merged[[SPREAD_TARGET, *features]].isna().any().any():
        raise ValueError("Day-ahead spread training frame contains missing values")
    return merged, features


def _add_rolling_spread_quantiles(
    base: pd.DataFrame,
    spread: pd.DataFrame,
    features: list[str],
    config: QuantileSpreadCopulaExperimentConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    outputs: list[pd.DataFrame] = []
    crossing_rows = 0.0
    total_rows = 0.0
    year = spread.loc[
        spread["delivery_date_local"].dt.year == config.residual_year
    ]
    keys = ["delivery_date_local", "hour_local"]
    for fold, fold_base in base.groupby("rolling_fold", sort=True):
        test_days = sorted(fold_base["delivery_date_local"].dt.date.unique())
        cutoff = pd.Timestamp(test_days[0])
        train = year.loc[year["delivery_date_local"] < cutoff]
        test = year.merge(
            fold_base[keys].drop_duplicates(), on=keys, how="inner", validate="one_to_one"
        )
        prediction, crossing = _fit_predict_quantiles(
            train,
            test,
            features,
            SPREAD_TARGET,
            config,
            clip_nonnegative=False,
        )
        addition = test[keys + [SPREAD_TARGET]].copy()
        for index, level in enumerate(config.quantile_levels):
            addition[_quantile_column("spread", level)] = prediction[:, index]
        output = fold_base.merge(addition, on=keys, validate="one_to_one")
        outputs.append(output)
        crossing_rows += crossing["crossing_rows_before_rearrangement"]
        total_rows += crossing["rows"]
    result = pd.concat(outputs, ignore_index=True).sort_values(keys)
    _assert_complete(result)
    return result, {
        "scope": "rolling_2024",
        "rows": int(total_rows),
        "crossing_rows_before_rearrangement": int(crossing_rows),
        "crossing_rate_before_rearrangement": float(crossing_rows / total_rows),
    }


def _add_future_spread_quantiles(
    base: pd.DataFrame,
    spread: pd.DataFrame,
    features: list[str],
    config: QuantileSpreadCopulaExperimentConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    keys = ["delivery_date_local", "hour_local"]
    train_end = pd.Timestamp(f"{config.residual_year}-12-31")
    train = spread.loc[spread["delivery_date_local"] <= train_end]
    test = spread.merge(
        base[keys].drop_duplicates(), on=keys, how="inner", validate="one_to_one"
    )
    prediction, crossing = _fit_predict_quantiles(
        train,
        test,
        features,
        SPREAD_TARGET,
        config,
        clip_nonnegative=False,
    )
    addition = test[keys + [SPREAD_TARGET]].copy()
    for index, level in enumerate(config.quantile_levels):
        addition[_quantile_column("spread", level)] = prediction[:, index]
    result = base.merge(addition, on=keys, validate="one_to_one").sort_values(keys)
    _assert_complete(result)
    return result, {"scope": "future_2025", **crossing}


def _build_empirical_copula(
    rolling: pd.DataFrame, config: QuantileSpreadCopulaExperimentConfig
) -> pd.DataFrame:
    days = sorted(rolling["delivery_date_local"].dt.date.unique())
    output = pd.DataFrame({"delivery_date_local": days}).set_index(
        "delivery_date_local"
    )
    for _, prefix, actual, _, _, _ in TARGETS:
        residual = rolling[actual] - rolling[_quantile_column(prefix, 0.5)]
        frame = rolling.assign(_residual=residual)
        pivot = frame.pivot(
            index="delivery_date_local", columns="hour_local", values="_residual"
        ).loc[:, list(HOURS_LOCAL)]
        pivot.index = pd.to_datetime(pivot.index).date
        for hour in HOURS_LOCAL:
            ranks = pivot[hour].rank(method="average")
            output[f"{prefix}_u_h{hour:02d}"] = (ranks - 0.5) / len(ranks)
    return output.reset_index()


def _conformalize(
    rolling: pd.DataFrame,
    future: pd.DataFrame,
    config: QuantileSpreadCopulaExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    calibrated = future.copy()
    correction_frames: list[pd.DataFrame] = []
    for target, prefix, actual, unit, lower, upper in TARGETS:
        columns = [
            _quantile_column(prefix, level) for level in config.quantile_levels
        ]
        calibrator = CentralIntervalCQR.fit(
            actual=rolling[actual].to_numpy(dtype=float),
            quantile_predictions=rolling[columns].to_numpy(dtype=float),
            groups=rolling["hour_local"].to_numpy(dtype=int),
            quantile_levels=config.quantile_levels,
        )
        calibrated.loc[:, columns] = calibrator.transform(
            future[columns].to_numpy(dtype=float),
            future["hour_local"].to_numpy(dtype=int),
            lower_bound=lower,
            upper_bound=(
                config.pv_capacity_kwh_per_hour if target == "pv_generation" else upper
            ),
        )
        correction = calibrator.corrections.rename(
            columns={"group": "hour_local"}
        ).copy()
        correction.insert(0, "unit", unit)
        correction.insert(0, "target", target)
        correction_frames.append(correction)
    return calibrated, pd.concat(correction_frames, ignore_index=True)


def _evaluate_scenarios(
    forecasts: pd.DataFrame,
    copula_library: pd.DataFrame,
    config: QuantileSpreadCopulaExperimentConfig,
    *,
    generator: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    example: pd.DataFrame | None = None
    for delivery_day, day in forecasts.groupby("delivery_date_local", sort=True):
        day_date = pd.Timestamp(delivery_day).date()
        batch = sample_three_target_scenarios(
            day_forecasts=day,
            copula_library=copula_library,
            quantile_levels=config.quantile_levels,
            n_scenarios=config.n_scenarios,
            random_state=(
                config.random_state + int(pd.Timestamp(delivery_day).strftime("%Y%m%d"))
            ),
            as_of_date=day_date,
            pv_capacity_kwh_per_hour=config.pv_capacity_kwh_per_hour,
        )
        output = day[
            [
                "delivery_date_local",
                "hour_local",
                "actual_pv_kwh",
                "actual_price_eur_per_mwh",
                SPREAD_TARGET,
            ]
        ].copy()
        output["generator"] = generator
        for target, prefix, actual, _, _, _ in TARGETS:
            scenarios = {
                "pv": batch.pv_kwh,
                "price": batch.day_ahead_price_eur_per_mwh,
                "spread": batch.intraday_spread_eur_per_mwh,
            }[prefix]
            output[f"{prefix}_scenario_mean"] = scenarios.mean(axis=0)
            output[f"{prefix}_p10"] = np.quantile(scenarios, 0.10, axis=0)
            output[f"{prefix}_p50"] = np.quantile(scenarios, 0.50, axis=0)
            output[f"{prefix}_p90"] = np.quantile(scenarios, 0.90, axis=0)
            output[f"{prefix}_crps"] = _ensemble_crps(
                scenarios, output[actual].to_numpy(dtype=float)
            )
        rows.append(output)
        if example is None:
            count = config.n_scenarios
            hours = len(HOURS_LOCAL)
            example = pd.DataFrame(
                {
                    "delivery_date_local": str(day_date),
                    "scenario_id": np.repeat(np.arange(count), hours),
                    "source_copula_day": np.repeat(
                        [str(value) for value in batch.source_copula_days], hours
                    ),
                    "hour_local": np.tile(HOURS_LOCAL, count),
                    "pv_kwh": batch.pv_kwh.reshape(-1),
                    "day_ahead_price_eur_per_mwh": (
                        batch.day_ahead_price_eur_per_mwh.reshape(-1)
                    ),
                    "intraday_spread_eur_per_mwh": (
                        batch.intraday_spread_eur_per_mwh.reshape(-1)
                    ),
                    "intraday_price_eur_per_mwh": (
                        batch.day_ahead_price_eur_per_mwh
                        + batch.intraday_spread_eur_per_mwh
                    ).reshape(-1),
                }
            )
    if example is None:
        raise ValueError("No complete future days available")
    return pd.concat(rows, ignore_index=True), example


def _metric_table(
    bands: pd.DataFrame,
    forecasts: pd.DataFrame,
    config: QuantileSpreadCopulaExperimentConfig,
    generator: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target, prefix, actual, unit, _, _ in TARGETS:
        y = bands[actual].to_numpy(dtype=float)
        point = forecasts[_quantile_column(prefix, 0.5)].to_numpy(dtype=float)
        scenario_mean = bands[f"{prefix}_scenario_mean"].to_numpy(dtype=float)
        lower = bands[f"{prefix}_p10"].to_numpy(dtype=float)
        upper = bands[f"{prefix}_p90"].to_numpy(dtype=float)
        losses = []
        for level in config.quantile_levels:
            prediction = forecasts[_quantile_column(prefix, level)].to_numpy(
                dtype=float
            )
            error = y - prediction
            losses.append(
                np.mean(np.maximum(level * error, (level - 1.0) * error))
            )
        rows.append(
            {
                "generator": generator,
                "target": target,
                "unit": unit,
                "rows": len(y),
                "point_mae": float(np.mean(np.abs(y - point))),
                "point_rmse": float(np.sqrt(np.mean((y - point) ** 2))),
                "scenario_mean_mae": float(np.mean(np.abs(y - scenario_mean))),
                "scenario_mean_rmse": float(
                    np.sqrt(np.mean((y - scenario_mean) ** 2))
                ),
                "p10_p90_coverage": float(((y >= lower) & (y <= upper)).mean()),
                "p10_p90_mean_width": float(np.mean(upper - lower)),
                "ensemble_crps": float(bands[f"{prefix}_crps"].mean()),
                "mean_pinball_loss": float(np.mean(losses)),
            }
        )
    return pd.DataFrame(rows)


def _quantile_calibration(
    forecasts: pd.DataFrame, config: QuantileSpreadCopulaExperimentConfig
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target, prefix, actual, unit, _, _ in TARGETS:
        y = forecasts[actual].to_numpy(dtype=float)
        for level in config.quantile_levels:
            prediction = forecasts[_quantile_column(prefix, level)].to_numpy(
                dtype=float
            )
            error = y - prediction
            rows.append(
                {
                    "target": target,
                    "unit": unit,
                    "quantile": level,
                    "empirical_cdf": float(np.mean(y <= prediction)),
                    "calibration_error": float(np.mean(y <= prediction) - level),
                    "pinball_loss": float(
                        np.mean(np.maximum(level * error, (level - 1.0) * error))
                    ),
                }
            )
    return pd.DataFrame(rows)


def _dependence_summary(library: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in library if column != "delivery_date_local"]
    correlation = library[columns].corr()
    cross_target = []
    for left, right in (("pv", "price"), ("pv", "spread"), ("price", "spread")):
        left_columns = [column for column in columns if column.startswith(f"{left}_")]
        right_columns = [column for column in columns if column.startswith(f"{right}_")]
        values = correlation.loc[left_columns, right_columns].to_numpy(dtype=float)
        cross_target.append(
            {
                "target_pair": f"{left}__{right}",
                "mean_rank_correlation": float(np.mean(values)),
                "mean_abs_rank_correlation": float(np.mean(np.abs(values))),
                "max_abs_rank_correlation": float(np.max(np.abs(values))),
            }
        )
    return pd.DataFrame(cross_target)


def _inverse_marginals(
    ranks: np.ndarray,
    quantiles: np.ndarray,
    levels: tuple[float, ...],
) -> np.ndarray:
    sorted_quantiles = np.sort(np.asarray(quantiles, dtype=float), axis=1)
    clipped = np.clip(ranks, levels[0], levels[-1])
    result = np.empty_like(clipped, dtype=float)
    for hour_index in range(len(HOURS_LOCAL)):
        result[:, hour_index] = np.interp(
            clipped[:, hour_index], levels, sorted_quantiles[hour_index]
        )
    return result


def _normalise_dates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["delivery_date_local"] = pd.to_datetime(
        result["delivery_date_local"]
    )
    return result


def _assert_complete(frame: pd.DataFrame) -> None:
    counts = frame.groupby("delivery_date_local")["hour_local"].agg(
        ["count", "nunique"]
    )
    expected = len(HOURS_LOCAL)
    if not ((counts["count"] == expected) & (counts["nunique"] == expected)).all():
        raise ValueError("Forecast table must contain one complete operating window per day")


def _validate_config(config: QuantileSpreadCopulaExperimentConfig) -> None:
    if tuple(config.quantile_levels) != tuple(DEFAULT_QUANTILES):
        raise ValueError("V2 uses the existing nine-quantile grid")
    if config.iterations < 1:
        raise ValueError("iterations must be positive")
    if config.n_scenarios < 10:
        raise ValueError("n_scenarios must be at least ten")
    if config.pv_capacity_kwh_per_hour <= 0:
        raise ValueError("pv_capacity_kwh_per_hour must be positive")
