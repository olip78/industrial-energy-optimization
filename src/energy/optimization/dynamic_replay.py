"""Sequential replay with an explicit battery state of charge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from energy.optimization.dynamic_battery import (
    DynamicBatteryParameters,
    DynamicEconomicLedger,
    DynamicOraclePlan,
    DynamicSchedule,
    battery_reference_cost_eur_per_kwh,
    evaluate_dynamic_actual_day,
    solve_dynamic_oracle,
    solve_dynamic_schedule,
)
from energy.optimization.replay import DayReplayInputs
from energy.optimization.grid_tariff import NO_GRID_TARIFF, GridTariffParameters
from energy.optimization.rule_based import RuleBasedPriceShape, build_rule_based_plan
from energy.optimization.scenario import ReferenceScenario


@dataclass(frozen=True)
class DynamicRulePlan:
    load_kwh: np.ndarray
    charge_kwh: np.ndarray
    discharge_kwh: np.ndarray
    curtailment_kwh: np.ndarray
    curtailment_fraction: np.ndarray
    soc_kwh: np.ndarray
    day_ahead_position_kwh: np.ndarray
    price_score_eur_per_mwh: np.ndarray


@dataclass(frozen=True)
class DynamicDeterministicReplayResult:
    day_ahead_plan: DynamicSchedule
    executed_load_kwh: np.ndarray
    executed_charge_kwh: np.ndarray
    executed_discharge_kwh: np.ndarray
    executed_curtailment_kwh: np.ndarray
    executed_soc_kwh: np.ndarray
    mpc_plans: Sequence[DynamicSchedule]
    ledger: DynamicEconomicLedger


@dataclass(frozen=True)
class DynamicDayAheadOnlyReplayResult:
    day_ahead_plan: DynamicSchedule
    ledger: DynamicEconomicLedger


@dataclass(frozen=True)
class DynamicRuleReplayResult:
    rule_plan: DynamicRulePlan
    ledger: DynamicEconomicLedger


@dataclass(frozen=True)
class DynamicOracleReplayResult:
    oracle_plan: DynamicOraclePlan
    ledger: DynamicEconomicLedger


def replay_dynamic_deterministic_day(
    *,
    inputs: DayReplayInputs,
    scenario: ReferenceScenario,
    grid_tariff: GridTariffParameters = NO_GRID_TARIFF,
) -> DynamicDeterministicReplayResult:
    """Run day-ahead dispatch and sequential full-state MPC.

    The day-ahead position is fixed after the auction.  At every decision
    step MPC reallocates all still-flexible load and reoptimizes battery charge,
    battery discharge and PV curtailment over the remaining working window.
    """

    hours = np.asarray(scenario.working_hours, dtype=int)
    lower_full, upper_full = scenario.load_bounds()
    lower = lower_full[hours]
    upper = upper_full[hours]
    allowed = np.ones(len(hours), dtype=bool)
    battery = _battery(scenario, inputs.night_reference_price_eur_per_mwh)
    day_ahead_plan = solve_dynamic_schedule(
        pv_forecast_kwh=inputs.day_ahead_pv_forecast_kwh[hours],
        price_eur_per_mwh=inputs.day_ahead_price_forecast_eur_per_mwh[hours],
        load_min_kwh=lower,
        load_max_kwh=upper,
        required_load_kwh=scenario.daily_load_kwh,
        battery=battery,
        discharge_allowed=allowed,
        grid_tariff=grid_tariff,
    )

    count = len(hours)
    executed_load = np.zeros(count, dtype=float)
    executed_charge = np.zeros(count, dtype=float)
    executed_discharge = np.zeros(count, dtype=float)
    executed_curtailment = np.zeros(count, dtype=float)
    executed_soc = np.zeros(count + 1, dtype=float)
    executed_soc[0] = battery.initial_soc_kwh
    mpc_plans: list[DynamicSchedule] = []
    for index, hour in enumerate(hours):
        tail_hours = hours[index:]
        remaining_load = scenario.daily_load_kwh - executed_load[:index].sum()
        mpc_plan = solve_dynamic_schedule(
            pv_forecast_kwh=inputs.mpc_pv_forecast_kwh[hour, tail_hours],
            price_eur_per_mwh=inputs.mpc_intraday_price_forecast_eur_per_mwh[
                hour, tail_hours
            ],
            fixed_position_kwh=day_ahead_plan.net_position_kwh[index:],
            load_min_kwh=lower[index:],
            load_max_kwh=upper[index:],
            required_load_kwh=remaining_load,
            battery=battery,
            discharge_allowed=allowed[index:],
            initial_soc_kwh=executed_soc[index],
            grid_tariff=grid_tariff,
        )
        executed_load[index] = mpc_plan.load_kwh[0]
        executed_charge[index] = mpc_plan.charge_kwh[0]
        executed_discharge[index] = mpc_plan.discharge_kwh[0]
        executed_curtailment[index] = (
            mpc_plan.curtailment_fraction[0]
            * inputs.actual_pv_kwh[hour]
        )
        executed_soc[index + 1] = mpc_plan.soc_kwh[1]
        mpc_plans.append(mpc_plan)

    if not np.isclose(executed_load.sum(), scenario.daily_load_kwh, atol=1e-7):
        raise AssertionError("dynamic MPC did not complete the daily load")
    ledger = evaluate_dynamic_actual_day(
        day_ahead_position_kwh=day_ahead_plan.net_position_kwh,
        day_ahead_price_eur_per_mwh=inputs.actual_day_ahead_price_eur_per_mwh[
            hours
        ],
        actual_load_kwh=executed_load,
        actual_charge_kwh=executed_charge,
        actual_pv_kwh=inputs.actual_pv_kwh[hours],
        actual_discharge_kwh=executed_discharge,
        actual_intraday_price_eur_per_mwh=inputs.actual_intraday_price_eur_per_mwh[
            hours
        ],
        battery=battery,
        actual_curtailment_kwh=executed_curtailment,
        grid_tariff=grid_tariff,
    )
    return DynamicDeterministicReplayResult(
        day_ahead_plan=day_ahead_plan,
        executed_load_kwh=executed_load,
        executed_charge_kwh=executed_charge,
        executed_discharge_kwh=executed_discharge,
        executed_curtailment_kwh=executed_curtailment,
        executed_soc_kwh=executed_soc,
        mpc_plans=tuple(mpc_plans),
        ledger=ledger,
    )


def replay_dynamic_day_ahead_only_day(
    *,
    inputs: DayReplayInputs,
    scenario: ReferenceScenario,
    grid_tariff: GridTariffParameters = NO_GRID_TARIFF,
) -> DynamicDayAheadOnlyReplayResult:
    """Execute the day-ahead schedule without intraday replanning."""

    hours = np.asarray(scenario.working_hours, dtype=int)
    lower_full, upper_full = scenario.load_bounds()
    battery = _battery(scenario, inputs.night_reference_price_eur_per_mwh)
    plan = solve_dynamic_schedule(
        pv_forecast_kwh=inputs.day_ahead_pv_forecast_kwh[hours],
        price_eur_per_mwh=inputs.day_ahead_price_forecast_eur_per_mwh[hours],
        load_min_kwh=lower_full[hours],
        load_max_kwh=upper_full[hours],
        required_load_kwh=scenario.daily_load_kwh,
        battery=battery,
        discharge_allowed=np.ones(len(hours), dtype=bool),
        grid_tariff=grid_tariff,
    )
    ledger = evaluate_dynamic_actual_day(
        day_ahead_position_kwh=plan.net_position_kwh,
        day_ahead_price_eur_per_mwh=inputs.actual_day_ahead_price_eur_per_mwh[
            hours
        ],
        actual_load_kwh=plan.load_kwh,
        actual_charge_kwh=plan.charge_kwh,
        actual_pv_kwh=inputs.actual_pv_kwh[hours],
        actual_discharge_kwh=plan.discharge_kwh,
        actual_intraday_price_eur_per_mwh=inputs.actual_intraday_price_eur_per_mwh[
            hours
        ],
        battery=battery,
        actual_curtailment_kwh=(
            plan.curtailment_fraction * inputs.actual_pv_kwh[hours]
        ),
        grid_tariff=grid_tariff,
    )
    return DynamicDayAheadOnlyReplayResult(day_ahead_plan=plan, ledger=ledger)


def replay_dynamic_rule_based_day(
    *,
    inputs: DayReplayInputs,
    scenario: ReferenceScenario,
    price_shape: RuleBasedPriceShape,
    grid_tariff: GridTariffParameters = NO_GRID_TARIFF,
) -> DynamicRuleReplayResult:
    """Replay the causal rule policy under the common physical ledger.

    The rule buys its fixed production-load profile day-ahead.  Actual PV first
    refills battery headroom created by an earlier rule discharge.  Residual PV
    is exported unless the already-known day-ahead clearing price is negative,
    in which case it is curtailed.  The rule never charges from the grid.
    """

    hours = np.asarray(scenario.working_hours, dtype=int)
    # Flooring here makes the rule's discharge threshold match the ledger.
    battery = _battery(scenario, inputs.night_reference_price_eur_per_mwh)
    base = build_rule_based_plan(
        delivery_day=inputs.delivery_day,
        price_shape=price_shape,
        scenario=scenario,
        night_reference_price_eur_per_mwh=max(
            inputs.night_reference_price_eur_per_mwh, 0.0
        ),
        battery_discharge_cost_eur_per_kwh=(
            battery_reference_cost_eur_per_kwh(battery, grid_tariff)
        ),
        avoided_import_charge_eur_per_kwh=(
            grid_tariff.daytime_variable_import_eur_per_kwh
        ),
    )
    load = base.load_kwh[hours].copy()
    charge = np.zeros(len(hours), dtype=float)
    discharge = np.zeros(len(hours), dtype=float)
    curtailment = np.zeros(len(hours), dtype=float)
    soc = np.zeros(len(hours) + 1, dtype=float)
    soc[0] = battery.initial_soc_kwh
    for index, hour in enumerate(hours):
        requested_discharge = float(base.discharge_kwh[hour])
        if requested_discharge > 1e-9:
            deliverable = soc[index] * battery.discharge_efficiency
            discharge[index] = min(requested_discharge, deliverable)
        else:
            available_pv = max(float(inputs.actual_pv_kwh[hour]), 0.0)
            storage_headroom_input = (
                battery.capacity_kwh - soc[index]
            ) / battery.charge_efficiency
            charge[index] = min(
                available_pv,
                battery.max_charge_kwh_per_hour,
                max(storage_headroom_input, 0.0),
            )
        soc[index + 1] = (
            soc[index]
            + battery.charge_efficiency * charge[index]
            - discharge[index] / battery.discharge_efficiency
        )
        residual_pv = max(
            float(inputs.actual_pv_kwh[hour]) - charge[index],
            0.0,
        )
        if inputs.actual_day_ahead_price_eur_per_mwh[hour] < 0.0:
            curtailment[index] = residual_pv
    position = load.copy()
    plan = DynamicRulePlan(
        load_kwh=load,
        charge_kwh=charge,
        discharge_kwh=discharge,
        curtailment_kwh=curtailment,
        curtailment_fraction=np.divide(
            curtailment,
            inputs.actual_pv_kwh[hours],
            out=np.zeros_like(curtailment),
            where=inputs.actual_pv_kwh[hours] > 1e-9,
        ),
        soc_kwh=soc,
        day_ahead_position_kwh=position,
        price_score_eur_per_mwh=base.price_score_eur_per_mwh[hours],
    )
    ledger = evaluate_dynamic_actual_day(
        day_ahead_position_kwh=position,
        day_ahead_price_eur_per_mwh=inputs.actual_day_ahead_price_eur_per_mwh[
            hours
        ],
        actual_load_kwh=load,
        actual_charge_kwh=charge,
        actual_pv_kwh=inputs.actual_pv_kwh[hours],
        actual_discharge_kwh=discharge,
        actual_intraday_price_eur_per_mwh=inputs.actual_intraday_price_eur_per_mwh[
            hours
        ],
        battery=battery,
        actual_curtailment_kwh=curtailment,
        grid_tariff=grid_tariff,
    )
    return DynamicRuleReplayResult(rule_plan=plan, ledger=ledger)


def replay_dynamic_oracle_day(
    *,
    inputs: DayReplayInputs,
    scenario: ReferenceScenario,
    grid_tariff: GridTariffParameters = NO_GRID_TARIFF,
) -> DynamicOracleReplayResult:
    """Replay the perfect-information day-ahead lower bound.

    Only factual PV and day-ahead prices enter the Oracle decision.  The same
    physical schedule defines nomination and execution.
    """

    hours = np.asarray(scenario.working_hours, dtype=int)
    lower_full, upper_full = scenario.load_bounds()
    battery = _battery(scenario, inputs.night_reference_price_eur_per_mwh)
    plan = solve_dynamic_oracle(
        actual_pv_kwh=inputs.actual_pv_kwh[hours],
        actual_day_ahead_price_eur_per_mwh=inputs.actual_day_ahead_price_eur_per_mwh[
            hours
        ],
        load_min_kwh=lower_full[hours],
        load_max_kwh=upper_full[hours],
        required_load_kwh=scenario.daily_load_kwh,
        battery=battery,
        discharge_allowed=np.ones(len(hours), dtype=bool),
        grid_tariff=grid_tariff,
    )
    ledger = evaluate_dynamic_actual_day(
        day_ahead_position_kwh=plan.day_ahead_position_kwh,
        day_ahead_price_eur_per_mwh=inputs.actual_day_ahead_price_eur_per_mwh[
            hours
        ],
        actual_load_kwh=plan.actual.load_kwh,
        actual_charge_kwh=plan.actual.charge_kwh,
        actual_pv_kwh=inputs.actual_pv_kwh[hours],
        actual_discharge_kwh=plan.actual.discharge_kwh,
        actual_intraday_price_eur_per_mwh=inputs.actual_intraday_price_eur_per_mwh[
            hours
        ],
        battery=battery,
        actual_curtailment_kwh=plan.actual.curtailment_kwh,
        grid_tariff=grid_tariff,
    )
    if not np.allclose(ledger.actual_deviation_kwh, 0.0, atol=1e-9):
        raise AssertionError("day-ahead Oracle created an intraday deviation")
    return DynamicOracleReplayResult(oracle_plan=plan, ledger=ledger)


def _battery(
    scenario: ReferenceScenario,
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
