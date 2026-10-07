"""Three-target whole-day residual bootstrap and 2025 forecast audit."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from energy.training.residual_bootstrap_experiment import (
    HOURS_LOCAL,
    _ensemble_crps,
)
from energy.uncertainty.residual_bootstrap_spread import (
    JointThreeTargetResidualBootstrap,
)


TARGETS = (
    ("pv_generation", "pv", "actual_pv_kwh", "point_pv_kwh", "kWh"),
    (
        "day_ahead_price",
        "price",
        "actual_price_eur_per_mwh",
        "point_price_eur_per_mwh",
        "EUR/MWh",
    ),
    (
        "intraday_spread",
        "spread",
        "actual_spread_eur_per_mwh",
        "point_spread_eur_per_mwh",
        "EUR/MWh",
    ),
)


@dataclass(frozen=True)
class ResidualBootstrapSpreadExperimentConfig:
    project_root: Path
    residual_artifact_name: str = "joint_residual_bootstrap_current_v3"
    quantile_artifact_name: str = "quantile_spread_copula_v2"
    n_scenarios: int = 500
    random_state: int = 42
    pv_capacity_kwh_per_hour: float = 10.0
    seasonal_bandwidth_days: float = 25.0
    global_mixture_weight: float = 0.15
    artifact_name: str = "joint_residual_bootstrap_spread_seasonal_v4"


@dataclass(frozen=True)
class ResidualBootstrapSpreadExperimentResult:
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


def run_residual_bootstrap_spread_experiment(
    config: ResidualBootstrapSpreadExperimentConfig,
) -> ResidualBootstrapSpreadExperimentResult:
    """Join the spread residual and compare raw versus centred bootstrap."""

    _validate_config(config)
    root = config.project_root.expanduser().resolve()
    residual_source = (
        root / "artifacts" / "experiments" / config.residual_artifact_name
    )
    quantile_source = (
        root / "artifacts" / "experiments" / config.quantile_artifact_name
    )
    residual_library = _build_residual_library(residual_source, quantile_source)
    future = _build_future_point_forecasts(residual_source, quantile_source)

    bands: list[pd.DataFrame] = []
    examples: list[pd.DataFrame] = []
    generator_specs = (
        ("raw", False, "uniform", False),
        ("centered", True, "uniform", False),
        ("seasonal", True, "seasonal", True),
    )
    for generator_name, centered, sampling_scheme, standardize_pv in generator_specs:
        generator = JointThreeTargetResidualBootstrap(
            residual_library,
            hours_local=HOURS_LOCAL,
            center_residuals=centered,
            sampling_scheme=sampling_scheme,
            seasonal_bandwidth_days=config.seasonal_bandwidth_days,
            global_mixture_weight=config.global_mixture_weight,
            standardize_pv_residuals=standardize_pv,
        )
        evaluated, example = _evaluate(
            future, generator, config, generator_name
        )
        bands.append(evaluated)
        examples.append(example)
    prediction_bands = pd.concat(bands, ignore_index=True)
    metrics = _metric_table(prediction_bands)
    diagnostics = _residual_diagnostics(residual_library)
    dependence = _dependence_summary(residual_library)

    artifact_dir = root / "artifacts" / "experiments" / config.artifact_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    residual_library.to_parquet(
        artifact_dir / "rolling_origin_residual_library.parquet", index=False
    )
    future.to_parquet(artifact_dir / "test_point_forecasts.parquet", index=False)
    prediction_bands.to_parquet(
        artifact_dir / "out_of_sample_prediction_bands.parquet", index=False
    )
    pd.concat(examples, ignore_index=True).to_parquet(
        artifact_dir / "example_scenario_batches.parquet", index=False
    )
    metrics.to_csv(artifact_dir / "metrics.csv", index=False)
    diagnostics.to_csv(artifact_dir / "residual_diagnostics.csv", index=False)
    dependence.to_csv(artifact_dir / "dependence_summary.csv", index=False)
    (artifact_dir / "experiment_config.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "project_root": str(root),
                "working_hours_local": list(HOURS_LOCAL),
                "residual_contract": (
                    "one complete rolling-origin 2024 source day jointly supplies "
                    "PV, day-ahead-price and intraday-spread residuals for all hours"
                ),
                "centering_contract": (
                    "the centered generator subtracts the eligible historical "
                    "hour-specific residual mean independently for each target"
                ),
                "seasonal_contract": (
                    "the seasonal generator samples paired whole-day residual "
                    "vectors with a circular day-of-year Gaussian kernel, mixes "
                    "in a global component, and rescales standardized PV residuals"
                ),
                "point_forecast_contract": (
                    "PV and day-ahead price reuse the frozen RMSE point forecasts; "
                    "spread uses the rolling/future quantile-regression median"
                ),
            },
            indent=2,
            default=str,
        )
    )
    return ResidualBootstrapSpreadExperimentResult(
        artifact_dir=artifact_dir,
        residual_days=int(residual_library["delivery_date_local"].nunique()),
        test_days=int(future["delivery_date_local"].nunique()),
        metrics=metrics,
    )


def _build_residual_library(
    residual_source: Path,
    quantile_source: Path,
) -> pd.DataFrame:
    base = pd.read_parquet(
        residual_source / "rolling_origin_residual_library.parquet"
    )
    spread = pd.read_parquet(
        quantile_source / "rolling_origin_quantile_predictions.parquet"
    )
    keys = ["delivery_date_local", "hour_local"]
    base["delivery_date_local"] = pd.to_datetime(base["delivery_date_local"])
    spread["delivery_date_local"] = pd.to_datetime(spread["delivery_date_local"])
    addition = spread[
        keys + ["actual_spread_eur_per_mwh", "spread_q50"]
    ].copy()
    addition["spread_residual_eur_per_mwh"] = (
        addition["actual_spread_eur_per_mwh"] - addition["spread_q50"]
    )
    result = base.merge(addition, on=keys, how="inner", validate="one_to_one")
    _assert_complete(result)
    if len(result) != len(base):
        raise ValueError("Spread residuals do not cover the existing residual library")
    return result.sort_values(keys).reset_index(drop=True)


def _build_future_point_forecasts(
    residual_source: Path,
    quantile_source: Path,
) -> pd.DataFrame:
    base = pd.read_parquet(
        residual_source / "out_of_sample_prediction_bands.parquet"
    )
    spread = pd.read_parquet(
        quantile_source / "test_raw_quantile_forecasts.parquet"
    )
    keys = ["delivery_date_local", "hour_local"]
    base["delivery_date_local"] = pd.to_datetime(base["delivery_date_local"])
    spread["delivery_date_local"] = pd.to_datetime(spread["delivery_date_local"])
    addition = spread[
        keys + ["actual_spread_eur_per_mwh", "spread_q50"]
    ].rename(columns={"spread_q50": "point_spread_eur_per_mwh"})
    keep = keys + [
        "actual_pv_kwh",
        "point_pv_kwh",
        "actual_price_eur_per_mwh",
        "point_price_eur_per_mwh",
    ]
    result = base[keep].merge(
        addition, on=keys, how="inner", validate="one_to_one"
    )
    _assert_complete(result)
    if len(result) != len(base):
        raise ValueError("Spread point forecasts do not cover the 2025 test frame")
    return result.sort_values(keys).reset_index(drop=True)


def _evaluate(
    future: pd.DataFrame,
    generator: JointThreeTargetResidualBootstrap,
    config: ResidualBootstrapSpreadExperimentConfig,
    generator_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    example: pd.DataFrame | None = None
    for delivery_day, day in future.groupby("delivery_date_local", sort=True):
        day = day.sort_values("hour_local")
        day_date = pd.Timestamp(delivery_day).date()
        batch = generator.sample(
            point_pv_kwh=day["point_pv_kwh"].to_numpy(dtype=float),
            point_price_eur_per_mwh=day[
                "point_price_eur_per_mwh"
            ].to_numpy(dtype=float),
            point_spread_eur_per_mwh=day[
                "point_spread_eur_per_mwh"
            ].to_numpy(dtype=float),
            n_scenarios=config.n_scenarios,
            random_state=(
                config.random_state
                + int(pd.Timestamp(delivery_day).strftime("%Y%m%d"))
            ),
            as_of_date=day_date,
            pv_capacity_kwh_per_hour=config.pv_capacity_kwh_per_hour,
        )
        output = day.copy()
        output["generator"] = generator_name
        output["sampling_effective_days"] = batch.sampling_effective_days
        for _, prefix, actual, _, _ in TARGETS:
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
                    "generator": generator_name,
                    "delivery_date_local": str(day_date),
                    "scenario_id": np.repeat(np.arange(count), hours),
                    "source_residual_day": np.repeat(
                        [str(value) for value in batch.source_residual_days],
                        hours,
                    ),
                    "sampling_effective_days": batch.sampling_effective_days,
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
        raise ValueError("No future days available")
    return pd.concat(rows, ignore_index=True), example


def _metric_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for generator, group in frame.groupby("generator", sort=True):
        for target, prefix, actual, point, unit in TARGETS:
            y = group[actual].to_numpy(dtype=float)
            point_values = group[point].to_numpy(dtype=float)
            scenario_mean = group[f"{prefix}_scenario_mean"].to_numpy(dtype=float)
            lower = group[f"{prefix}_p10"].to_numpy(dtype=float)
            upper = group[f"{prefix}_p90"].to_numpy(dtype=float)
            rows.append(
                {
                    "generator": generator,
                    "target": target,
                    "unit": unit,
                    "rows": len(group),
                    "point_mae": float(np.mean(np.abs(y - point_values))),
                    "point_rmse": float(np.sqrt(np.mean((y - point_values) ** 2))),
                    "scenario_mean_mae": float(np.mean(np.abs(y - scenario_mean))),
                    "scenario_mean_rmse": float(
                        np.sqrt(np.mean((y - scenario_mean) ** 2))
                    ),
                    "p10_p90_coverage": float(
                        ((y >= lower) & (y <= upper)).mean()
                    ),
                    "p10_p90_mean_width": float(np.mean(upper - lower)),
                    "ensemble_crps": float(group[f"{prefix}_crps"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _residual_diagnostics(library: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, column in (
        ("pv_generation", "pv_residual_kwh"),
        ("day_ahead_price", "price_residual_eur_per_mwh"),
        ("intraday_spread", "spread_residual_eur_per_mwh"),
    ):
        summary = library.groupby("hour_local")[column].agg(
            rows="size", mean="mean", std="std", median="median"
        )
        summary.insert(0, "target", target)
        rows.append(summary.reset_index())
    return pd.concat(rows, ignore_index=True)


def _dependence_summary(library: pd.DataFrame) -> pd.DataFrame:
    pivots = {}
    for prefix, column in (
        ("pv", "pv_residual_kwh"),
        ("price", "price_residual_eur_per_mwh"),
        ("spread", "spread_residual_eur_per_mwh"),
    ):
        pivot = library.pivot(
            index="delivery_date_local", columns="hour_local", values=column
        ).loc[:, list(HOURS_LOCAL)]
        pivot.columns = [f"{prefix}_h{hour:02d}" for hour in HOURS_LOCAL]
        pivots[prefix] = pivot
    correlation = pd.concat(pivots.values(), axis=1).corr(method="spearman")
    rows = []
    for left, right in (("pv", "price"), ("pv", "spread"), ("price", "spread")):
        left_columns = [column for column in correlation if column.startswith(left)]
        right_columns = [column for column in correlation if column.startswith(right)]
        values = correlation.loc[left_columns, right_columns].to_numpy(dtype=float)
        rows.append(
            {
                "target_pair": f"{left}__{right}",
                "mean_rank_correlation": float(np.mean(values)),
                "mean_abs_rank_correlation": float(np.mean(np.abs(values))),
                "max_abs_rank_correlation": float(np.max(np.abs(values))),
            }
        )
    return pd.DataFrame(rows)


def _assert_complete(frame: pd.DataFrame) -> None:
    counts = frame.groupby("delivery_date_local")["hour_local"].agg(
        ["count", "nunique"]
    )
    expected = len(HOURS_LOCAL)
    if not ((counts["count"] == expected) & (counts["nunique"] == expected)).all():
        raise ValueError("Frame does not contain one complete operating window per day")


def _validate_config(config: ResidualBootstrapSpreadExperimentConfig) -> None:
    if config.n_scenarios < 10:
        raise ValueError("n_scenarios must be at least ten")
    if config.pv_capacity_kwh_per_hour <= 0:
        raise ValueError("pv_capacity_kwh_per_hour must be positive")
    if config.seasonal_bandwidth_days <= 0:
        raise ValueError("seasonal_bandwidth_days must be positive")
    if not 0 <= config.global_mixture_weight <= 1:
        raise ValueError("global_mixture_weight must lie in [0, 1]")
