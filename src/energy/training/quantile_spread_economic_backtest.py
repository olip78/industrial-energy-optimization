"""Day-ahead-only economic replay for the three-target quantile scenarios."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from energy.optimization.dynamic_battery import evaluate_dynamic_actual_day
from energy.optimization.dynamic_replay import (
    replay_dynamic_day_ahead_only_day,
    replay_dynamic_oracle_day,
    replay_dynamic_rule_based_day,
)
from energy.optimization.dynamic_stochastic_spread import (
    solve_dynamic_stochastic_spread_day_ahead,
)
from energy.optimization.grid_tariff import (
    PFORZHEIM_SLP_2025_GRID_TARIFF,
    GridTariffParameters,
)
from energy.optimization.rule_based import fit_rule_based_price_shape
from energy.optimization.scenario import V1_REFERENCE_SCENARIO
from energy.training.dynamic_charge_economic_backtest import (
    _daily_row,
    _dynamic_battery,
    _hourly_rows,
)
from energy.training.economic_backtest import (
    INTRADAY_BLEND_WEIGHT,
    EconomicBacktestConfig,
    _assemble_replay_inputs,
    _night_reference_prices,
    _read_canonical,
    _read_intraday_actual,
    _read_intraday_predictions,
    _read_price_predictions,
    _read_pv_day_ahead_predictions,
    _read_pv_mpc_predictions,
    _validate_date_window,
)
from energy.training.quantile_spread_copula_experiment import (
    DEFAULT_QUANTILES,
    sample_three_target_scenarios,
)
from energy.training.stochastic_economic_backtest import _paired_block_bootstrap


@dataclass(frozen=True)
class QuantileSpreadEconomicBacktestConfig:
    project_root: Path
    start: str = "2025-01-01"
    end: str = "2025-09-30"
    quantile_artifact_name: str = "quantile_spread_copula_v2"
    generators: tuple[str, ...] = ("raw", "cqr")
    n_scenarios: int = 500
    cvar_alpha: float = 0.95
    risk_weights: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50)
    random_state: int = 42
    correction_weight: float = INTRADAY_BLEND_WEIGHT
    economic_bootstrap_resamples: int = 5_000
    economic_bootstrap_block_days: int = 7
    grid_tariff: GridTariffParameters = PFORZHEIM_SLP_2025_GRID_TARIFF
    artifact_name: str = "quantile_spread_economic_backtest_v2"


@dataclass(frozen=True)
class QuantileSpreadEconomicBacktestResult:
    artifact_dir: Path
    daily_results: pd.DataFrame
    summary: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {"artifact_dir": str(self.artifact_dir), "summary": self.summary}


def run_quantile_spread_economic_backtest(
    config: QuantileSpreadEconomicBacktestConfig,
) -> QuantileSpreadEconomicBacktestResult:
    """Replay raw/CQR SAA and CVaR plans without MPC or intraday recourse."""

    _validate_config(config)
    root = config.project_root.expanduser().resolve()
    start, end = _validate_date_window(config.start, config.end)
    scenario = V1_REFERENCE_SCENARIO
    history = _read_canonical(root, 2024)
    test = _read_canonical(root, 2025)
    test = test.merge(
        _read_intraday_actual(root),
        on="valid_time_utc",
        how="left",
        validate="one_to_one",
    )
    base_config = EconomicBacktestConfig(
        project_root=root,
        start=config.start,
        end=config.end,
        correction_weight=config.correction_weight,
    )
    inputs, coverage = _assemble_replay_inputs(
        start=start,
        end=end,
        scenario=scenario,
        test=test,
        price_predictions=_read_price_predictions(base_config, root),
        pv_day_ahead_predictions=_read_pv_day_ahead_predictions(base_config, root),
        pv_mpc_predictions=_read_pv_mpc_predictions(base_config, root),
        intraday_predictions=_read_intraday_predictions(base_config, root),
        night_references=_night_reference_prices(root),
        correction_weight=config.correction_weight,
    )
    if not inputs:
        raise ValueError("No complete delivery days remain for the replay")

    source = root / "artifacts" / "experiments" / config.quantile_artifact_name
    copula_library = pd.read_parquet(source / "empirical_copula_library.parquet")
    forecast_files = {
        "raw": "test_raw_quantile_forecasts.parquet",
        "cqr": "test_calibrated_quantile_forecasts.parquet",
    }
    forecasts_by_generator: dict[str, dict[object, pd.DataFrame]] = {}
    for generator in config.generators:
        forecast = pd.read_parquet(source / forecast_files[generator])
        forecast["delivery_date_local"] = pd.to_datetime(
            forecast["delivery_date_local"]
        ).dt.date
        forecasts_by_generator[generator] = {
            day: frame.sort_values("hour_local")
            for day, frame in forecast.groupby("delivery_date_local", sort=False)
        }

    price_shape = fit_rule_based_price_shape(
        history,
        price_column="day_ahead_price_eur_per_mwh",
        date_column="delivery_date_local",
        hour_column="hour_local",
    )
    hours = np.asarray(scenario.working_hours, dtype=int)
    load_lower, load_upper = scenario.load_bounds()
    daily_rows: list[dict[str, object]] = []
    hourly_rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []

    for day_index, day_inputs in enumerate(inputs):
        battery = _dynamic_battery(
            scenario, day_inputs.night_reference_price_eur_per_mwh
        )
        rule = replay_dynamic_rule_based_day(
            inputs=day_inputs,
            scenario=scenario,
            price_shape=price_shape,
            grid_tariff=config.grid_tariff,
        )
        deterministic = replay_dynamic_day_ahead_only_day(
            inputs=day_inputs,
            scenario=scenario,
            grid_tariff=config.grid_tariff,
        )
        oracle = replay_dynamic_oracle_day(
            inputs=day_inputs,
            scenario=scenario,
            grid_tariff=config.grid_tariff,
        )
        schedules = {
            "rule_based": (
                rule.rule_plan.load_kwh,
                rule.rule_plan.charge_kwh,
                rule.rule_plan.discharge_kwh,
                rule.rule_plan.curtailment_kwh,
                rule.rule_plan.soc_kwh,
                rule.rule_plan.day_ahead_position_kwh,
                np.full(len(hours), np.nan),
                rule.ledger,
            ),
            "day_ahead_only": (
                deterministic.day_ahead_plan.load_kwh,
                deterministic.day_ahead_plan.charge_kwh,
                deterministic.day_ahead_plan.discharge_kwh,
                deterministic.ledger.actual_curtailment_kwh,
                deterministic.day_ahead_plan.soc_kwh,
                deterministic.day_ahead_plan.net_position_kwh,
                day_inputs.day_ahead_price_forecast_eur_per_mwh[hours],
                deterministic.ledger,
            ),
            "oracle": (
                oracle.oracle_plan.actual.load_kwh,
                oracle.oracle_plan.actual.charge_kwh,
                oracle.oracle_plan.actual.discharge_kwh,
                oracle.oracle_plan.actual.curtailment_kwh,
                oracle.oracle_plan.actual.soc_kwh,
                oracle.oracle_plan.day_ahead_position_kwh,
                day_inputs.actual_day_ahead_price_eur_per_mwh[hours],
                oracle.ledger,
            ),
        }

        for generator_index, generator in enumerate(config.generators):
            day_forecast = forecasts_by_generator[generator].get(
                day_inputs.delivery_day
            )
            if day_forecast is None:
                raise ValueError(
                    f"Missing {generator} quantiles for {day_inputs.delivery_day}"
                )
            batch = sample_three_target_scenarios(
                day_forecasts=day_forecast,
                copula_library=copula_library,
                quantile_levels=DEFAULT_QUANTILES,
                n_scenarios=config.n_scenarios,
                random_state=config.random_state + day_index,
                as_of_date=day_inputs.delivery_day,
                pv_capacity_kwh_per_hour=10.0,
            )
            for risk_weight in (0.0, *config.risk_weights):
                plan = solve_dynamic_stochastic_spread_day_ahead(
                    pv_scenarios_kwh=batch.pv_kwh,
                    day_ahead_price_scenarios_eur_per_mwh=(
                        batch.day_ahead_price_eur_per_mwh
                    ),
                    intraday_spread_scenarios_eur_per_mwh=(
                        batch.intraday_spread_eur_per_mwh
                    ),
                    point_pv_forecast_kwh=(
                        day_inputs.day_ahead_pv_forecast_kwh[hours]
                    ),
                    load_min_kwh=load_lower[hours],
                    load_max_kwh=load_upper[hours],
                    required_load_kwh=scenario.daily_load_kwh,
                    battery=battery,
                    discharge_allowed=np.ones(len(hours), dtype=bool),
                    grid_tariff=config.grid_tariff,
                    cvar_alpha=config.cvar_alpha,
                    risk_weight=risk_weight,
                )
                factual_curtailment = (
                    plan.curtailment_fraction
                    * day_inputs.actual_pv_kwh[hours]
                )
                ledger = evaluate_dynamic_actual_day(
                    day_ahead_position_kwh=plan.day_ahead_position_kwh,
                    day_ahead_price_eur_per_mwh=(
                        day_inputs.actual_day_ahead_price_eur_per_mwh[hours]
                    ),
                    actual_load_kwh=plan.load_kwh,
                    actual_charge_kwh=plan.charge_kwh,
                    actual_pv_kwh=day_inputs.actual_pv_kwh[hours],
                    actual_discharge_kwh=plan.discharge_kwh,
                    actual_intraday_price_eur_per_mwh=(
                        day_inputs.actual_intraday_price_eur_per_mwh[hours]
                    ),
                    battery=battery,
                    actual_curtailment_kwh=factual_curtailment,
                    grid_tariff=config.grid_tariff,
                )
                strategy = _strategy_name(generator, risk_weight)
                schedules[strategy] = (
                    plan.load_kwh,
                    plan.charge_kwh,
                    plan.discharge_kwh,
                    factual_curtailment,
                    plan.soc_kwh,
                    plan.day_ahead_position_kwh,
                    day_inputs.day_ahead_price_forecast_eur_per_mwh[hours],
                    ledger,
                )
                diagnostics.append(
                    {
                        "delivery_date_local": day_inputs.delivery_day.isoformat(),
                        "generator": generator,
                        "strategy": strategy,
                        "risk_weight": risk_weight,
                        "predicted_expected_cost_eur": plan.expected_cost_eur,
                        "predicted_cvar_cost_eur": plan.cvar_cost_eur,
                        "predicted_objective_eur": plan.objective_eur,
                        "unique_source_copula_days": len(
                            set(batch.source_copula_days)
                        ),
                        "scenario_seed": config.random_state + day_index,
                        "generator_index": generator_index,
                    }
                )

        for strategy, values in schedules.items():
            (
                load,
                charge,
                discharge,
                curtailment,
                soc,
                position,
                decision_price,
                ledger,
            ) = values
            delivery_day = day_inputs.delivery_day.isoformat()
            daily_rows.append(
                _daily_row(
                    strategy=strategy,
                    delivery_day=delivery_day,
                    ledger=ledger,
                    load=load,
                    charge=charge,
                    discharge=discharge,
                    curtailment=curtailment,
                    soc=soc,
                    position=position,
                    actual_pv=day_inputs.actual_pv_kwh[hours],
                    battery_capacity=scenario.battery_available_energy_kwh,
                    raw_night_reference=(
                        day_inputs.night_reference_price_eur_per_mwh
                    ),
                )
            )
            hourly_rows.extend(
                _hourly_rows(
                    strategy=strategy,
                    delivery_day=delivery_day,
                    hours=hours,
                    load=load,
                    charge=charge,
                    discharge=discharge,
                    curtailment=curtailment,
                    soc=soc,
                    position=position,
                    actual_pv=day_inputs.actual_pv_kwh[hours],
                    actual_da_price=(
                        day_inputs.actual_day_ahead_price_eur_per_mwh[hours]
                    ),
                    actual_id_price=(
                        day_inputs.actual_intraday_price_eur_per_mwh[hours]
                    ),
                    decision_charge_price=decision_price,
                    night_reference_price=(
                        day_inputs.night_reference_price_eur_per_mwh
                    ),
                    grid_tariff=config.grid_tariff,
                )
            )

    daily = pd.DataFrame(daily_rows).sort_values(
        ["delivery_date_local", "strategy"]
    )
    summary = _economic_summary(daily, config)
    artifact_dir = root / "artifacts" / "experiments" / config.artifact_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(artifact_dir / "daily_results.csv", index=False)
    pd.DataFrame(hourly_rows).to_parquet(
        artifact_dir / "hourly_decisions.parquet", index=False
    )
    pd.DataFrame(diagnostics).to_csv(
        artifact_dir / "scenario_objectives.csv", index=False
    )
    coverage.to_csv(artifact_dir / "coverage.csv", index=False)
    (artifact_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (artifact_dir / "experiment_config.json").write_text(
        json.dumps(
            {
                "start": config.start,
                "end": config.end,
                "operating_hours": list(scenario.working_hours),
                "quantile_artifact_name": config.quantile_artifact_name,
                "generators": list(config.generators),
                "n_scenarios": config.n_scenarios,
                "cvar_alpha": config.cvar_alpha,
                "risk_weights": list(config.risk_weights),
                "random_state": config.random_state,
                "grid_tariff": config.grid_tariff.to_dict(),
                "mpc_included": False,
                "recourse_controls": False,
                "common_decisions": [
                    "production_load",
                    "battery_charge",
                    "battery_discharge",
                    "curtailment_fraction",
                ],
                "position_contract": (
                    "q_DA is the common schedule evaluated at the frozen point "
                    "PV forecast; deliberate DA/ID speculation is excluded"
                ),
                "intraday_price_contract": "P_ID = P_DA + spread",
                "factual_settlement": (
                    "the selected plan is held fixed and settled once against "
                    "actual 2025 PV, day-ahead and intraday prices"
                ),
            },
            indent=2,
        )
    )
    return QuantileSpreadEconomicBacktestResult(
        artifact_dir=artifact_dir,
        daily_results=daily,
        summary=summary,
    )


def _strategy_name(generator: str, risk_weight: float) -> str:
    if risk_weight == 0.0:
        return f"quantile_{generator}_saa"
    suffix = f"{risk_weight:g}".replace(".", "_")
    return f"quantile_{generator}_cvar_lambda_{suffix}"


def _economic_summary(
    daily: pd.DataFrame,
    config: QuantileSpreadEconomicBacktestConfig,
) -> dict[str, object]:
    metrics: dict[str, dict[str, float]] = {}
    for strategy, group in daily.groupby("strategy", sort=True):
        costs = group["total_cost_eur"].to_numpy(dtype=float)
        metrics[strategy] = {
            "total_cost_eur": float(costs.sum()),
            "mean_daily_cost_eur": float(costs.mean()),
            "std_daily_cost_eur": float(costs.std(ddof=1)),
            "p95_daily_cost_eur": float(np.quantile(costs, config.cvar_alpha)),
            "realized_cvar95_daily_cost_eur": _upper_tail_mean(
                costs, config.cvar_alpha
            ),
            "maximum_daily_cost_eur": float(costs.max()),
            "day_ahead_cost_eur": float(group["day_ahead_cost_eur"].sum()),
            "intraday_deviation_cost_eur": float(
                group["intraday_deviation_cost_eur"].sum()
            ),
            "regulated_grid_cost_eur": float(
                group["regulated_grid_cost_eur"].sum()
            ),
            "battery_reference_cost_eur": float(
                group["battery_reference_cost_eur"].sum()
            ),
            "battery_charge_kwh": float(group["battery_charge_kwh"].sum()),
            "battery_discharge_kwh": float(
                group["battery_discharge_kwh"].sum()
            ),
            "equivalent_full_cycles": float(
                group["equivalent_full_cycles"].sum()
            ),
            "pv_curtailment_kwh": float(group["pv_curtailment_kwh"].sum()),
            "mean_final_soc_kwh": float(group["final_soc_kwh"].mean()),
            "physical_grid_import_kwh": float(
                group["physical_grid_import_kwh"].sum()
            ),
            "physical_grid_export_kwh": float(
                group["physical_grid_export_kwh"].sum()
            ),
        }

    baseline = _cost_series(daily, "day_ahead_only")
    oracle_total = metrics["oracle"]["total_cost_eur"]
    comparisons: dict[str, dict[str, float]] = {}
    candidates = sorted(name for name in metrics if name.startswith("quantile_"))
    tail_count = max(1, math.ceil((1.0 - config.cvar_alpha) * len(baseline)))
    baseline_tail_days = baseline.nlargest(tail_count).index
    for index, strategy in enumerate(candidates):
        candidate = _cost_series(daily, strategy)
        if not baseline.index.equals(candidate.index):
            raise ValueError(f"Date mismatch between baseline and {strategy}")
        saving = baseline - candidate
        comparisons[strategy] = {
            "total_saving_vs_day_ahead_eur": float(saving.sum()),
            "mean_daily_saving_eur": float(saving.mean()),
            "median_daily_saving_eur": float(saving.median()),
            "share_days_cheaper": float((saving > 1e-9).mean()),
            "share_days_more_expensive": float((saving < -1e-9).mean()),
            "realized_cvar95_reduction_eur_per_day": float(
                metrics["day_ahead_only"]["realized_cvar95_daily_cost_eur"]
                - metrics[strategy]["realized_cvar95_daily_cost_eur"]
            ),
            "mean_saving_on_baseline_worst_5pct_days_eur": float(
                saving.loc[baseline_tail_days].mean()
            ),
            "gap_to_oracle_eur": float(
                metrics[strategy]["total_cost_eur"] - oracle_total
            ),
            **_paired_block_bootstrap(
                baseline.to_numpy(dtype=float),
                candidate.to_numpy(dtype=float),
                alpha=config.cvar_alpha,
                block_days=min(
                    config.economic_bootstrap_block_days,
                    len(baseline),
                ),
                n_resamples=config.economic_bootstrap_resamples,
                random_state=config.random_state + 10_000 + index,
            ),
        }
    return {
        "delivery_days": int(daily["delivery_date_local"].nunique()),
        "operating_window": "06:00-22:00 (hours 6..21)",
        "scenario_count_per_day": config.n_scenarios,
        "cvar_alpha": config.cvar_alpha,
        "strategy_metrics": metrics,
        "comparisons_vs_day_ahead": comparisons,
        "rule_to_oracle_opportunity_eur": float(
            metrics["rule_based"]["total_cost_eur"] - oracle_total
        ),
        "day_ahead_gap_to_oracle_eur": float(
            metrics["day_ahead_only"]["total_cost_eur"] - oracle_total
        ),
    }


def _cost_series(daily: pd.DataFrame, strategy: str) -> pd.Series:
    return (
        daily.loc[
            daily["strategy"].eq(strategy),
            ["delivery_date_local", "total_cost_eur"],
        ]
        .sort_values("delivery_date_local")
        .set_index("delivery_date_local")["total_cost_eur"]
    )


def _upper_tail_mean(values: np.ndarray, alpha: float) -> float:
    count = max(1, math.ceil((1.0 - alpha) * len(values)))
    return float(np.sort(np.asarray(values, dtype=float))[-count:].mean())


def _validate_config(config: QuantileSpreadEconomicBacktestConfig) -> None:
    if not config.generators or any(
        generator not in {"raw", "cqr"} for generator in config.generators
    ):
        raise ValueError("generators must contain raw and/or cqr")
    if len(set(config.generators)) != len(config.generators):
        raise ValueError("generators must be unique")
    if config.n_scenarios < 2:
        raise ValueError("n_scenarios must be at least two")
    if not 0 < config.cvar_alpha < 1:
        raise ValueError("cvar_alpha must lie strictly inside (0, 1)")
    if any(weight <= 0 or not np.isfinite(weight) for weight in config.risk_weights):
        raise ValueError("risk_weights must be finite and positive")
    if len(set(config.risk_weights)) != len(config.risk_weights):
        raise ValueError("risk_weights must be unique")
    if config.economic_bootstrap_resamples < 100:
        raise ValueError("economic_bootstrap_resamples must be at least 100")
    if config.economic_bootstrap_block_days < 1:
        raise ValueError("economic_bootstrap_block_days must be positive")
