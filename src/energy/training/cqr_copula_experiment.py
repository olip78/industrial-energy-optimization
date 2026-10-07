"""Conformal calibration of quantile marginals before empirical-copula sampling."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from energy.training.quantile_copula_experiment import (
    DEFAULT_QUANTILES,
    _evaluate_scenarios,
    _hourly_metric_table,
    _metric_table,
    _pinball_summary,
    _quantile_calibration,
    _quantile_column,
)
from energy.uncertainty import CentralIntervalCQR, QuantileEmpiricalCopula


@dataclass(frozen=True)
class CqrCopulaExperimentConfig:
    project_root: Path
    source_artifact_name: str = "quantile_copula_v1"
    quantile_levels: tuple[float, ...] = DEFAULT_QUANTILES
    n_scenarios: int = 500
    random_state: int = 42
    pv_capacity_kwh_per_hour: float = 10.0
    artifact_name: str = "cqr_copula_v1"


@dataclass(frozen=True)
class CqrCopulaExperimentResult:
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


def run_cqr_copula_experiment(
    config: CqrCopulaExperimentConfig,
) -> CqrCopulaExperimentResult:
    """Fit hourly CQR corrections on 2024 OOF scores and evaluate 2025."""

    _validate_config(config)
    root = config.project_root.expanduser().resolve()
    source_dir = root / "artifacts" / "experiments" / config.source_artifact_name
    rolling = pd.read_parquet(
        source_dir / "rolling_origin_quantile_predictions.parquet"
    )
    future = pd.read_parquet(source_dir / "test_quantile_forecasts.parquet")
    copula_library = pd.read_parquet(source_dir / "empirical_copula_library.parquet")
    calibrated, corrections = _calibrate(rolling, future, config)
    generator = QuantileEmpiricalCopula(
        copula_library,
        quantile_levels=config.quantile_levels,
        hours_local=tuple(range(6, 22)),
    )
    bands, example_scenarios = _evaluate_scenarios(calibrated, generator, config)
    metrics = _metric_table(bands)
    metrics = metrics.merge(
        _pinball_summary(calibrated, config),
        on=["target", "unit"],
        validate="one_to_one",
    )
    calibration = _quantile_calibration(calibrated, config)
    hourly_metrics = _hourly_metric_table(bands)
    raw_metrics = pd.read_csv(source_dir / "metrics.csv")
    comparison = pd.concat(
        (
            raw_metrics.assign(generator="raw_quantile_copula"),
            metrics.assign(generator="cqr_quantile_copula"),
        ),
        ignore_index=True,
    )

    artifact_dir = root / "artifacts" / "experiments" / config.artifact_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    calibrated.to_parquet(
        artifact_dir / "test_calibrated_quantile_forecasts.parquet", index=False
    )
    copula_library.to_parquet(
        artifact_dir / "empirical_copula_library.parquet", index=False
    )
    bands.to_parquet(
        artifact_dir / "out_of_sample_prediction_bands.parquet", index=False
    )
    example_scenarios.to_parquet(
        artifact_dir / "example_scenario_batch.parquet", index=False
    )
    corrections.to_csv(artifact_dir / "conformal_corrections.csv", index=False)
    calibration.to_csv(artifact_dir / "quantile_calibration.csv", index=False)
    metrics.to_csv(artifact_dir / "metrics.csv", index=False)
    comparison.to_csv(artifact_dir / "raw_vs_cqr_metrics.csv", index=False)
    hourly_metrics.to_csv(artifact_dir / "metrics_by_hour.csv", index=False)
    (artifact_dir / "experiment_config.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "project_root": str(root),
                "quantile_levels": list(config.quantile_levels),
                "calibration_contract": (
                    "Non-negative symmetric CQR scores from 2024 rolling-origin "
                    "predictions, fitted separately by target, local hour and "
                    "central interval; all calibration dates precede 2025"
                ),
                "dependence_contract": (
                    "The empirical copula library is unchanged from raw quantile-copula V1"
                ),
            },
            indent=2,
            default=str,
        )
    )
    return CqrCopulaExperimentResult(
        artifact_dir=artifact_dir,
        calibration_days=int(pd.to_datetime(rolling["delivery_date_local"]).dt.date.nunique()),
        test_days=int(pd.to_datetime(calibrated["delivery_date_local"]).dt.date.nunique()),
        metrics=metrics,
    )


def _calibrate(
    rolling: pd.DataFrame,
    future: pd.DataFrame,
    config: CqrCopulaExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    calibrated = future.copy()
    correction_frames: list[pd.DataFrame] = []
    for target, unit, actual_column, prefix, lower, upper in (
        (
            "pv_generation",
            "kWh",
            "actual_pv_kwh",
            "pv",
            0.0,
            config.pv_capacity_kwh_per_hour,
        ),
        (
            "day_ahead_price",
            "EUR/MWh",
            "actual_price_eur_per_mwh",
            "price",
            None,
            None,
        ),
    ):
        columns = [_quantile_column(prefix, value) for value in config.quantile_levels]
        calibrator = CentralIntervalCQR.fit(
            actual=rolling[actual_column].to_numpy(dtype=float),
            quantile_predictions=rolling[columns].to_numpy(dtype=float),
            groups=rolling["hour_local"].to_numpy(dtype=int),
            quantile_levels=config.quantile_levels,
        )
        transformed = calibrator.transform(
            future[columns].to_numpy(dtype=float),
            future["hour_local"].to_numpy(dtype=int),
            lower_bound=lower,
            upper_bound=upper,
        )
        calibrated.loc[:, columns] = transformed
        correction = calibrator.corrections.rename(columns={"group": "hour_local"}).copy()
        correction.insert(0, "unit", unit)
        correction.insert(0, "target", target)
        correction_frames.append(correction)
    return calibrated, pd.concat(correction_frames, ignore_index=True)


def _validate_config(config: CqrCopulaExperimentConfig) -> None:
    levels = np.asarray(config.quantile_levels, dtype=float)
    if tuple(config.quantile_levels) != tuple(DEFAULT_QUANTILES):
        raise ValueError("CQR V1 must use the quantile grid fitted by quantile_copula_v1")
    if config.n_scenarios < 10:
        raise ValueError("n_scenarios must be at least 10")
    if config.pv_capacity_kwh_per_hour <= 0:
        raise ValueError("pv_capacity_kwh_per_hour must be positive")
