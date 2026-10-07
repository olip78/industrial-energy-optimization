"""Economic replay with dynamic battery charging and physical grid tariffs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from energy.optimization.dynamic_battery import (
    DynamicBatteryParameters,
    DynamicEconomicLedger,
    evaluate_dynamic_actual_day,
)
from energy.optimization.dynamic_replay import (
    replay_dynamic_day_ahead_only_day,
    replay_dynamic_deterministic_day,
    replay_dynamic_oracle_day,
    replay_dynamic_rule_based_day,
)
from energy.optimization.dynamic_stochastic import (
    solve_dynamic_stochastic_day_ahead,
)
from energy.optimization.grid_tariff import (
    PFORZHEIM_SLP_2025_GRID_TARIFF,
    GridTariffParameters,
)
from energy.optimization.rule_based import fit_rule_based_price_shape
from energy.optimization.scenario import V1_REFERENCE_SCENARIO
from energy.training.economic_backtest import (
    DA_PRICE_VARIANT,
    DA_PV_VARIANT,
    INTRADAY_BASELINE_VARIANT,
    INTRADAY_BLEND_WEIGHT,
    INTRADAY_CHALLENGER_VARIANT,
    MPC_PV_VARIANT,
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
from energy.uncertainty import QuantileEmpiricalCopula


QUANTILE_LEVELS = (0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95)
STOCHASTIC_STRATEGY = "cqr_copula_cvar_lambda_0_1"


@dataclass(frozen=True)
class DynamicChargeEconomicBacktestConfig:
    project_root: Path
    start: str = "2025-01-01"
    end: str = "2025-09-30"
    artifact_name: str = "economic_backtest_v4_final_formulation"
    correction_weight: float = INTRADAY_BLEND_WEIGHT
    pv_mpc_prediction_path: Path | None = None
    include_stochastic: bool = False
    quantile_artifact_name: str = "cqr_copula_v1"
    quantile_forecast_filename: str = "test_calibrated_quantile_forecasts.parquet"
    n_scenarios: int = 500
    cvar_alpha: float = 0.95
    stochastic_risk_weight: float = 0.10
    random_state: int = 42
    grid_tariff: GridTariffParameters = PFORZHEIM_SLP_2025_GRID_TARIFF


@dataclass(frozen=True)
class DynamicChargeEconomicBacktestResult:
    artifact_dir: Path
    daily_results: pd.DataFrame
    coverage: pd.DataFrame
    summary: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {"artifact_dir": str(self.artifact_dir), "summary": self.summary}


def run_dynamic_charge_economic_backtest(
    config: DynamicChargeEconomicBacktestConfig,
) -> DynamicChargeEconomicBacktestResult:
    """Replay rule, day-ahead, MPC and Oracle policies under one tariff ledger."""

    if not 0.0 <= config.correction_weight <= 1.0:
        raise ValueError("correction_weight must lie in [0, 1]")
    root = config.project_root.expanduser().resolve()
    start, end = _validate_date_window(config.start, config.end)
    scenario = V1_REFERENCE_SCENARIO
    history = _read_canonical(root, 2024)
    test = _read_canonical(root, 2025)
    intraday_actual = _read_intraday_actual(root)
    test = test.merge(
        intraday_actual,
        on="valid_time_utc",
        how="left",
        validate="one_to_one",
    )
    base_config = EconomicBacktestConfig(
        project_root=root,
        start=config.start,
        end=config.end,
        correction_weight=config.correction_weight,
        pv_mpc_prediction_path=config.pv_mpc_prediction_path,
    )
    price_predictions = _read_price_predictions(base_config, root)
    pv_day_ahead_predictions = _read_pv_day_ahead_predictions(base_config, root)
    pv_mpc_predictions = _read_pv_mpc_predictions(base_config, root)
    intraday_predictions = _read_intraday_predictions(base_config, root)
    night_references = _night_reference_prices(root)
    copula: QuantileEmpiricalCopula | None = None
    quantiles_by_day: dict[object, pd.DataFrame] = {}
    if config.include_stochastic:
        quantile_dir = (
            root / "artifacts" / "experiments" / config.quantile_artifact_name
        )
        copula_library = pd.read_parquet(
            quantile_dir / "empirical_copula_library.parquet"
        )
        quantile_forecasts = pd.read_parquet(
            quantile_dir / config.quantile_forecast_filename
        )
        quantile_forecasts["delivery_date_local"] = pd.to_datetime(
            quantile_forecasts["delivery_date_local"]
        ).dt.date
        quantiles_by_day = {
            day: frame.sort_values("hour_local")
            for day, frame in quantile_forecasts.groupby(
                "delivery_date_local", sort=False
            )
        }
        copula = QuantileEmpiricalCopula(
            copula_library,
            quantile_levels=QUANTILE_LEVELS,
            hours_local=scenario.working_hours,
        )
    price_shape = fit_rule_based_price_shape(
        history,
        price_column="day_ahead_price_eur_per_mwh",
        date_column="delivery_date_local",
        hour_column="hour_local",
    )
    inputs, coverage = _assemble_replay_inputs(
        start=start,
        end=end,
        scenario=scenario,
        test=test,
        price_predictions=price_predictions,
        pv_day_ahead_predictions=pv_day_ahead_predictions,
        pv_mpc_predictions=pv_mpc_predictions,
        intraday_predictions=intraday_predictions,
        night_references=night_references,
        correction_weight=config.correction_weight,
    )
    if not inputs:
        raise ValueError("No complete delivery days remain for the replay")

    hours = np.asarray(scenario.working_hours, dtype=int)
    daily_rows: list[dict[str, object]] = []
    hourly_rows: list[dict[str, object]] = []
    stochastic_rows: list[dict[str, object]] = []
    pv_quantile_columns = [_quantile_column("pv", value) for value in QUANTILE_LEVELS]
    price_quantile_columns = [
        _quantile_column("price", value) for value in QUANTILE_LEVELS
    ]
    for day_index, day_inputs in enumerate(inputs):
        rule = replay_dynamic_rule_based_day(
            inputs=day_inputs,
            scenario=scenario,
            price_shape=price_shape,
            grid_tariff=config.grid_tariff,
        )
        day_ahead = replay_dynamic_day_ahead_only_day(
            inputs=day_inputs,
            scenario=scenario,
            grid_tariff=config.grid_tariff,
        )
        deterministic = replay_dynamic_deterministic_day(
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
                np.full(len(hours), np.nan, dtype=float),
                rule.ledger,
            ),
            "day_ahead_only": (
                day_ahead.day_ahead_plan.load_kwh,
                day_ahead.day_ahead_plan.charge_kwh,
                day_ahead.day_ahead_plan.discharge_kwh,
                day_ahead.ledger.actual_curtailment_kwh,
                day_ahead.day_ahead_plan.soc_kwh,
                day_ahead.day_ahead_plan.net_position_kwh,
                day_inputs.day_ahead_price_forecast_eur_per_mwh[hours],
                day_ahead.ledger,
            ),
            "deterministic": (
                deterministic.executed_load_kwh,
                deterministic.executed_charge_kwh,
                deterministic.executed_discharge_kwh,
                deterministic.executed_curtailment_kwh,
                deterministic.executed_soc_kwh,
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
        if config.include_stochastic:
            assert copula is not None
            day_quantiles = quantiles_by_day.get(day_inputs.delivery_day)
            if day_quantiles is None:
                raise ValueError(
                    f"Missing CQR quantile forecasts for {day_inputs.delivery_day}"
                )
            batch = copula.sample(
                pv_quantiles_kwh=day_quantiles[pv_quantile_columns].to_numpy(
                    dtype=float
                ),
                price_quantiles_eur_per_mwh=day_quantiles[
                    price_quantile_columns
                ].to_numpy(dtype=float),
                n_scenarios=config.n_scenarios,
                random_state=config.random_state + day_index,
                as_of_date=day_inputs.delivery_day,
                pv_capacity_kwh_per_hour=10.0,
            )
            battery = _dynamic_battery(
                scenario,
                day_inputs.night_reference_price_eur_per_mwh,
            )
            stochastic = solve_dynamic_stochastic_day_ahead(
                pv_scenarios_kwh=batch.pv_kwh,
                price_scenarios_eur_per_mwh=batch.day_ahead_price_eur_per_mwh,
                point_pv_forecast_kwh=day_inputs.day_ahead_pv_forecast_kwh[hours],
                point_price_forecast_eur_per_mwh=(
                    day_inputs.day_ahead_price_forecast_eur_per_mwh[hours]
                ),
                load_min_kwh=scenario.load_bounds()[0][hours],
                load_max_kwh=scenario.load_bounds()[1][hours],
                required_load_kwh=scenario.daily_load_kwh,
                battery=battery,
                discharge_allowed=np.ones(len(hours), dtype=bool),
                grid_tariff=config.grid_tariff,
                cvar_alpha=config.cvar_alpha,
                risk_weight=config.stochastic_risk_weight,
            )
            stochastic_ledger = evaluate_dynamic_actual_day(
                day_ahead_position_kwh=stochastic.day_ahead_position_kwh,
                day_ahead_price_eur_per_mwh=(
                    day_inputs.actual_day_ahead_price_eur_per_mwh[hours]
                ),
                actual_load_kwh=stochastic.load_kwh,
                actual_charge_kwh=stochastic.charge_kwh,
                actual_pv_kwh=day_inputs.actual_pv_kwh[hours],
                actual_discharge_kwh=stochastic.discharge_kwh,
                actual_intraday_price_eur_per_mwh=(
                    day_inputs.actual_intraday_price_eur_per_mwh[hours]
                ),
                battery=battery,
                actual_curtailment_kwh=np.zeros(len(hours), dtype=float),
                grid_tariff=config.grid_tariff,
            )
            stochastic_rows.append(
                {
                    "delivery_date_local": day_inputs.delivery_day.isoformat(),
                    "strategy": STOCHASTIC_STRATEGY,
                    "risk_weight": config.stochastic_risk_weight,
                    "predicted_expected_cost_eur": stochastic.expected_cost_eur,
                    "predicted_cvar_cost_eur": stochastic.cvar_cost_eur,
                    "predicted_objective_eur": stochastic.objective_eur,
                    "unique_source_copula_days": len(
                        set(batch.source_copula_days)
                    ),
                }
            )
            schedules[STOCHASTIC_STRATEGY] = (
                stochastic.load_kwh,
                stochastic.charge_kwh,
                stochastic.discharge_kwh,
                np.zeros(len(hours), dtype=float),
                stochastic.soc_kwh,
                stochastic.day_ahead_position_kwh,
                day_inputs.day_ahead_price_forecast_eur_per_mwh[hours],
                stochastic_ledger,
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
            daily_rows.append(
                _daily_row(
                    strategy=strategy,
                    delivery_day=day_inputs.delivery_day.isoformat(),
                    ledger=ledger,
                    load=load,
                    charge=charge,
                    discharge=discharge,
                    curtailment=curtailment,
                    soc=soc,
                    position=position,
                    actual_pv=day_inputs.actual_pv_kwh[hours],
                    battery_capacity=scenario.battery_available_energy_kwh,
                    raw_night_reference=day_inputs.night_reference_price_eur_per_mwh,
                )
            )
            hourly_rows.extend(
                _hourly_rows(
                    strategy=strategy,
                    delivery_day=day_inputs.delivery_day.isoformat(),
                    hours=hours,
                    load=load,
                    charge=charge,
                    discharge=discharge,
                    curtailment=curtailment,
                    soc=soc,
                    position=position,
                    actual_pv=day_inputs.actual_pv_kwh[hours],
                    actual_da_price=day_inputs.actual_day_ahead_price_eur_per_mwh[
                        hours
                    ],
                    actual_id_price=day_inputs.actual_intraday_price_eur_per_mwh[
                        hours
                    ],
                    decision_charge_price=decision_price,
                    night_reference_price=(
                        day_inputs.night_reference_price_eur_per_mwh
                    ),
                    grid_tariff=config.grid_tariff,
                )
            )

    daily_results = pd.DataFrame(daily_rows).sort_values(
        ["delivery_date_local", "strategy"]
    )
    hourly_results = pd.DataFrame(hourly_rows).sort_values(
        ["delivery_date_local", "strategy", "hour_local"]
    )
    summary = _summary(daily_results)
    artifact_dir = root / "artifacts" / "experiments" / config.artifact_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    daily_results.to_csv(artifact_dir / "daily_results.csv", index=False)
    hourly_results.to_parquet(artifact_dir / "hourly_decisions.parquet", index=False)
    if config.include_stochastic:
        pd.DataFrame(stochastic_rows).to_csv(
            artifact_dir / "stochastic_scenario_objectives.csv", index=False
        )
    coverage.to_csv(artifact_dir / "coverage.csv", index=False)
    (artifact_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    experiment_config = {
        "start": config.start,
        "end": config.end,
        "operating_hours": list(scenario.working_hours),
        "daily_load_kwh": scenario.daily_load_kwh,
        "pv_profile_scale": scenario.pv_profile_scale,
        "battery_capacity_kwh": scenario.battery_available_energy_kwh,
        "initial_soc_kwh": scenario.battery_available_energy_kwh,
        "max_charge_kwh_per_hour": scenario.battery_max_discharge_kwh_per_hour,
        "max_discharge_kwh_per_hour": scenario.battery_max_discharge_kwh_per_hour,
        "charge_efficiency": scenario.charge_efficiency,
        "discharge_efficiency": scenario.discharge_efficiency,
        "degradation_eur_per_kwh": scenario.degradation_eur_per_kwh,
        "grid_tariff": config.grid_tariff.to_dict(),
        "fixed_daily_grid_charge_applied": False,
        "charge_rule": (
            "charging is feasible in every operating hour and selected by the "
            "day-ahead optimizer from wholesale price, physical import tariff, "
            "PV export opportunity cost and terminal battery value"
        ),
        "mode_rule": "binary mutually exclusive charge/discharge mode",
        "night_reference_rule": (
            "rolling mean of the latest 14 completed nights; each night is the "
            "mean of day-ahead labels 22:00, 23:00 and 00:00-06:00 and is "
            "floored at zero before rolling aggregation"
        ),
        "terminal_value_rule": (
            "cost liability for restoring final SoC to capacity at the causal "
            "14-night wholesale benchmark plus the night import tariff"
        ),
        "terminal_soc_constraint": None,
        "mpc_battery_policy": (
            "each step reoptimizes remaining load, charge, discharge, state of "
            "charge and PV curtailment; only the day-ahead position is fixed"
        ),
        "rule_based_charge_policy": (
            "actual PV refills headroom created by earlier rule discharge; the "
            "rule never charges from the grid"
        ),
        "curtailment_policy": (
            "optimizer chooses curtailment; the rule curtails residual PV only "
            "when the known day-ahead clearing price is negative"
        ),
        "oracle_policy": {
            "information": "factual PV and factual day-ahead prices only",
            "intraday_price_used": False,
            "nomination_equals_physical_schedule": True,
            "deliberate_imbalance_allowed": False,
        },
        "stochastic_policy": {
            "included": config.include_stochastic,
            "strategy": STOCHASTIC_STRATEGY,
            "quantile_artifact_name": config.quantile_artifact_name,
            "quantile_forecast_filename": config.quantile_forecast_filename,
            "quantile_levels": list(QUANTILE_LEVELS),
            "n_scenarios": config.n_scenarios,
            "cvar_alpha": config.cvar_alpha,
            "risk_weight": config.stochastic_risk_weight,
            "copula_library_days": (
                len(copula.library_days) if copula is not None else None
            ),
            "charge_mask": "all operating hours",
            "position_centre": "frozen point PV forecast",
        },
        "frozen_forecasts": {
            "day_ahead_price_variant": DA_PRICE_VARIANT,
            "day_ahead_pv_variant": DA_PV_VARIANT,
            "mpc_pv_variant": MPC_PV_VARIANT,
            "intraday_baseline_variant": INTRADAY_BASELINE_VARIANT,
            "intraday_challenger_variant": INTRADAY_CHALLENGER_VARIANT,
            "intraday_lead_1_correction_weight": config.correction_weight,
            "pv_mpc_prediction_path": (
                str(config.pv_mpc_prediction_path)
                if config.pv_mpc_prediction_path is not None
                else None
            ),
        },
        "excluded_days": int((coverage["status"] != "used").sum()),
    }
    (artifact_dir / "experiment_config.json").write_text(
        json.dumps(experiment_config, indent=2)
    )
    return DynamicChargeEconomicBacktestResult(
        artifact_dir=artifact_dir,
        daily_results=daily_results,
        coverage=coverage,
        summary=summary,
    )


def _daily_row(
    *,
    strategy: str,
    delivery_day: str,
    ledger: DynamicEconomicLedger,
    load: np.ndarray,
    charge: np.ndarray,
    discharge: np.ndarray,
    curtailment: np.ndarray,
    soc: np.ndarray,
    position: np.ndarray,
    actual_pv: np.ndarray,
    battery_capacity: float,
    raw_night_reference: float,
) -> dict[str, object]:
    return {
        "delivery_date_local": delivery_day,
        "strategy": strategy,
        "total_cost_eur": float(ledger.total_cost_eur),
        "day_ahead_cost_eur": float(ledger.day_ahead_cost_eur),
        "intraday_deviation_cost_eur": float(
            ledger.intraday_deviation_cost_eur
        ),
        "battery_reference_cost_eur": float(ledger.battery_reference_cost_eur),
        "battery_night_energy_cost_eur": float(
            ledger.battery_night_energy_cost_eur
        ),
        "battery_night_grid_cost_eur": float(
            ledger.battery_night_grid_cost_eur
        ),
        "battery_degradation_cost_eur": float(
            ledger.battery_degradation_cost_eur
        ),
        "terminal_recharge_cost_eur": float(
            ledger.terminal_recharge_cost_eur
        ),
        "terminal_recharge_energy_kwh": float(
            ledger.terminal_recharge_energy_kwh
        ),
        "terminal_soc_value_eur_per_kwh": float(
            ledger.terminal_soc_value_eur_per_kwh
        ),
        "daytime_network_cost_eur": float(ledger.daytime_network_cost_eur),
        "levies_cost_eur": float(ledger.levies_cost_eur),
        "concession_fee_cost_eur": float(ledger.concession_fee_cost_eur),
        "electricity_tax_cost_eur": float(ledger.electricity_tax_cost_eur),
        "export_fee_cost_eur": float(ledger.export_fee_cost_eur),
        "regulated_grid_cost_eur": float(
            ledger.battery_night_grid_cost_eur
            + ledger.daytime_network_cost_eur
            + ledger.levies_cost_eur
            + ledger.concession_fee_cost_eur
            + ledger.electricity_tax_cost_eur
            + ledger.export_fee_cost_eur
        ),
        "day_ahead_position_kwh": float(position.sum()),
        "intraday_deviation_kwh": float(ledger.actual_deviation_kwh.sum()),
        "physical_grid_import_kwh": float(
            ledger.physical_grid_import_kwh.sum()
        ),
        "physical_grid_export_kwh": float(
            ledger.physical_grid_export_kwh.sum()
        ),
        "actual_pv_kwh": float(actual_pv.sum()),
        "pv_curtailment_kwh": float(curtailment.sum()),
        "used_pv_kwh": float(ledger.actual_used_pv_kwh.sum()),
        "battery_charge_kwh": float(charge.sum()),
        "battery_discharge_kwh": float(discharge.sum()),
        "equivalent_full_cycles": float(discharge.sum() / battery_capacity),
        "initial_soc_kwh": float(soc[0]),
        "minimum_soc_kwh": float(soc.min()),
        "final_soc_kwh": float(soc[-1]),
        "production_load_kwh": float(load.sum()),
        "raw_night_reference_price_eur_per_mwh": float(raw_night_reference),
        "floored_night_reference_price_eur_per_mwh": float(
            max(raw_night_reference, 0.0)
        ),
    }


def _hourly_rows(
    *,
    strategy: str,
    delivery_day: str,
    hours: np.ndarray,
    load: np.ndarray,
    charge: np.ndarray,
    discharge: np.ndarray,
    curtailment: np.ndarray,
    soc: np.ndarray,
    position: np.ndarray,
    actual_pv: np.ndarray,
    actual_da_price: np.ndarray,
    actual_id_price: np.ndarray,
    decision_charge_price: np.ndarray,
    night_reference_price: float,
    grid_tariff: GridTariffParameters,
) -> list[dict[str, object]]:
    return [
        {
            "delivery_date_local": delivery_day,
            "strategy": strategy,
            "hour_local": int(hour),
            "load_kwh": float(load[index]),
            "battery_charge_kwh": float(charge[index]),
            "battery_discharge_kwh": float(discharge[index]),
            "pv_curtailment_kwh": float(curtailment[index]),
            "used_pv_kwh": float(actual_pv[index] - curtailment[index]),
            "soc_start_kwh": float(soc[index]),
            "soc_end_kwh": float(soc[index + 1]),
            "day_ahead_position_kwh": float(position[index]),
            "actual_pv_kwh": float(actual_pv[index]),
            "actual_day_ahead_price_eur_per_mwh": float(actual_da_price[index]),
            "actual_intraday_price_eur_per_mwh": float(actual_id_price[index]),
            "decision_charge_price_eur_per_mwh": float(
                decision_charge_price[index]
            ),
            "physical_grid_exchange_kwh": float(
                load[index]
                + charge[index]
                - actual_pv[index]
                + curtailment[index]
                - discharge[index]
            ),
            "physical_grid_import_kwh": float(
                max(
                    load[index]
                    + charge[index]
                    - actual_pv[index]
                    + curtailment[index]
                    - discharge[index],
                    0.0,
                )
            ),
            "physical_grid_export_kwh": float(
                max(
                    actual_pv[index]
                    - curtailment[index]
                    + discharge[index]
                    - load[index]
                    - charge[index],
                    0.0,
                )
            ),
            "daytime_variable_grid_rate_eur_per_kwh": float(
                grid_tariff.daytime_variable_import_eur_per_kwh
            ),
            "grid_charge_input_cost_eur_per_kwh": float(
                decision_charge_price[index] / 1_000.0
                + grid_tariff.daytime_variable_import_eur_per_kwh
            ),
            "expected_overnight_input_cost_eur_per_kwh": float(
                max(night_reference_price, 0.0) / 1_000.0
                + grid_tariff.nighttime_variable_import_eur_per_kwh
            ),
            "grid_charge_cheaper_than_overnight": bool(
                np.isfinite(decision_charge_price[index])
                and decision_charge_price[index] / 1_000.0
                + grid_tariff.daytime_variable_import_eur_per_kwh
                < max(night_reference_price, 0.0) / 1_000.0
                + grid_tariff.nighttime_variable_import_eur_per_kwh
            ),
        }
        for index, hour in enumerate(hours)
    ]


def _summary(daily: pd.DataFrame) -> dict[str, object]:
    metrics: dict[str, dict[str, float]] = {}
    for strategy, group in daily.groupby("strategy", sort=True):
        costs = group["total_cost_eur"].to_numpy(dtype=float)
        metrics[strategy] = {
            "total_cost_eur": float(costs.sum()),
            "mean_daily_cost_eur": float(costs.mean()),
            "p95_daily_cost_eur": float(np.quantile(costs, 0.95)),
            "maximum_daily_cost_eur": float(costs.max()),
            "realized_cvar95_daily_cost_eur": _upper_tail_mean(costs, 0.95),
            "day_ahead_cost_eur": float(group["day_ahead_cost_eur"].sum()),
            "intraday_deviation_cost_eur": float(
                group["intraday_deviation_cost_eur"].sum()
            ),
            "battery_reference_cost_eur": float(
                group["battery_reference_cost_eur"].sum()
            ),
            "battery_night_energy_cost_eur": float(
                group["battery_night_energy_cost_eur"].sum()
            ),
            "battery_night_grid_cost_eur": float(
                group["battery_night_grid_cost_eur"].sum()
            ),
            "battery_degradation_cost_eur": float(
                group["battery_degradation_cost_eur"].sum()
            ),
            "terminal_recharge_cost_eur": float(
                group["terminal_recharge_cost_eur"].sum()
            ),
            "terminal_recharge_energy_kwh": float(
                group["terminal_recharge_energy_kwh"].sum()
            ),
            "regulated_grid_cost_eur": float(
                group["regulated_grid_cost_eur"].sum()
            ),
            "daytime_network_cost_eur": float(
                group["daytime_network_cost_eur"].sum()
            ),
            "levies_cost_eur": float(group["levies_cost_eur"].sum()),
            "concession_fee_cost_eur": float(
                group["concession_fee_cost_eur"].sum()
            ),
            "electricity_tax_cost_eur": float(
                group["electricity_tax_cost_eur"].sum()
            ),
            "export_fee_cost_eur": float(group["export_fee_cost_eur"].sum()),
            "pv_curtailment_kwh": float(group["pv_curtailment_kwh"].sum()),
            "physical_grid_import_kwh": float(
                group["physical_grid_import_kwh"].sum()
            ),
            "physical_grid_export_kwh": float(
                group["physical_grid_export_kwh"].sum()
            ),
            "battery_charge_kwh": float(group["battery_charge_kwh"].sum()),
            "mean_daily_battery_charge_kwh": float(
                group["battery_charge_kwh"].mean()
            ),
            "battery_discharge_kwh": float(
                group["battery_discharge_kwh"].sum()
            ),
            "mean_daily_battery_discharge_kwh": float(
                group["battery_discharge_kwh"].mean()
            ),
            "equivalent_full_cycles": float(
                group["equivalent_full_cycles"].sum()
            ),
            "mean_daily_equivalent_full_cycles": float(
                group["equivalent_full_cycles"].mean()
            ),
            "mean_final_soc_kwh": float(group["final_soc_kwh"].mean()),
        }
    rule = metrics["rule_based"]["total_cost_eur"]
    day_ahead = metrics["day_ahead_only"]["total_cost_eur"]
    deterministic = metrics["deterministic"]["total_cost_eur"]
    oracle = metrics["oracle"]["total_cost_eur"]
    rule_to_oracle_opportunity = rule - oracle
    day_ahead_gap_to_oracle = day_ahead - oracle
    raw_nights = daily.loc[
        daily["strategy"] == "day_ahead_only",
        "raw_night_reference_price_eur_per_mwh",
    ]
    result: dict[str, object] = {
        "delivery_days": int(daily["delivery_date_local"].nunique()),
        "operating_window": "06:00-22:00 (hours 6..21)",
        "strategy_metrics": metrics,
        "rule_to_oracle_opportunity_eur": float(rule_to_oracle_opportunity),
        "day_ahead_saving_vs_rule_eur": float(rule - day_ahead),
        "day_ahead_gap_to_oracle_eur": float(day_ahead_gap_to_oracle),
        "day_ahead_oracle_opportunity_capture_pct": _safe_percentage(
            rule - day_ahead,
            rule_to_oracle_opportunity,
        ),
        "mpc_incremental_saving_vs_day_ahead_eur": float(
            day_ahead - deterministic
        ),
        "mpc_share_of_remaining_day_ahead_gap_pct": _safe_percentage(
            day_ahead - deterministic,
            day_ahead_gap_to_oracle,
        ),
        "deterministic_saving_vs_rule_eur": float(rule - deterministic),
        "deterministic_oracle_opportunity_capture_pct": _safe_percentage(
            rule - deterministic,
            rule_to_oracle_opportunity,
        ),
        "deterministic_gap_to_oracle_eur": float(deterministic - oracle),
        "negative_previous_night_reference_days": int((raw_nights < 0).sum()),
    }
    if STOCHASTIC_STRATEGY in metrics:
        stochastic = metrics[STOCHASTIC_STRATEGY]["total_cost_eur"]
        result["stochastic_incremental_cost_vs_day_ahead_eur"] = float(
            stochastic - day_ahead
        )
        result["stochastic_cvar95_reduction_vs_day_ahead_eur_per_day"] = float(
            metrics["day_ahead_only"]["realized_cvar95_daily_cost_eur"]
            - metrics[STOCHASTIC_STRATEGY][
                "realized_cvar95_daily_cost_eur"
            ]
        )
    return result


def _dynamic_battery(
    scenario,
    night_reference_price_eur_per_mwh: float,
) -> DynamicBatteryParameters:
    return DynamicBatteryParameters(
        capacity_kwh=scenario.battery_available_energy_kwh,
        initial_soc_kwh=scenario.battery_available_energy_kwh,
        max_charge_kwh_per_hour=scenario.battery_max_discharge_kwh_per_hour,
        max_discharge_kwh_per_hour=scenario.battery_max_discharge_kwh_per_hour,
        night_reference_price_eur_per_mwh=night_reference_price_eur_per_mwh,
        charge_efficiency=scenario.charge_efficiency,
        discharge_efficiency=scenario.discharge_efficiency,
        degradation_eur_per_kwh=scenario.degradation_eur_per_kwh,
    )


def _upper_tail_mean(values: np.ndarray, alpha: float) -> float:
    count = max(1, math.ceil((1.0 - alpha) * len(values)))
    return float(np.sort(np.asarray(values, dtype=float))[-count:].mean())


def _safe_percentage(numerator: float, denominator: float) -> float | None:
    if abs(denominator) <= 1e-12:
        return None
    return float(100.0 * numerator / denominator)


def _quantile_column(prefix: str, level: float) -> str:
    return f"{prefix}_q{int(round(level * 100)):02d}"
