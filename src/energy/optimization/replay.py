"""Sequential daily replay for the rule-based and deterministic V1 policies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence

import numpy as np

from energy.optimization.deterministic import (
    DayAheadPlan,
    EconomicLedger,
    MpcPlan,
    OraclePlan,
    evaluate_actual_day,
    solve_day_ahead,
    solve_mpc,
    solve_oracle_perfect_information,
)
from energy.optimization.rule_based import (
    RuleBasedDayPlan,
    RuleBasedPriceShape,
    build_rule_based_plan,
)
from energy.optimization.scenario import ReferenceScenario


HOURS_PER_DAY = 24


@dataclass(frozen=True)
class DayReplayInputs:
    """One delivery day's forecasts and realised settlement values.

    Matrix row ``h`` represents the forecast visible immediately before delivery
    hour ``h`` is executed.  Columns ``h..23`` are the remaining MPC horizon;
    values before the diagonal are ignored.  The same convention lets a stored
    backtest prediction table be used without exposing future facts.
    """

    delivery_day: date
    day_ahead_pv_forecast_kwh: np.ndarray
    day_ahead_price_forecast_eur_per_mwh: np.ndarray
    actual_pv_kwh: np.ndarray
    actual_day_ahead_price_eur_per_mwh: np.ndarray
    actual_intraday_price_eur_per_mwh: np.ndarray
    mpc_pv_forecast_kwh: np.ndarray
    mpc_intraday_price_forecast_eur_per_mwh: np.ndarray
    night_reference_price_eur_per_mwh: float

    def __post_init__(self) -> None:
        vector_fields = (
            "day_ahead_pv_forecast_kwh",
            "day_ahead_price_forecast_eur_per_mwh",
            "actual_pv_kwh",
            "actual_day_ahead_price_eur_per_mwh",
            "actual_intraday_price_eur_per_mwh",
        )
        for name in vector_fields:
            _validate_vector(getattr(self, name), name)
        matrix_fields = (
            "mpc_pv_forecast_kwh",
            "mpc_intraday_price_forecast_eur_per_mwh",
        )
        for name in matrix_fields:
            _validate_matrix(getattr(self, name), name)
        if not np.isfinite(self.night_reference_price_eur_per_mwh):
            raise ValueError("night_reference_price_eur_per_mwh must be finite")


@dataclass(frozen=True)
class DeterministicReplayResult:
    """Fixed day-ahead plan, executed MPC trajectory and realised V1 ledger."""

    day_ahead_plan: DayAheadPlan
    executed_load_kwh: np.ndarray
    executed_discharge_kwh: np.ndarray
    mpc_plans: Sequence[MpcPlan]
    ledger: EconomicLedger


@dataclass(frozen=True)
class DayAheadOnlyReplayResult:
    """Forecast-based day-ahead plan executed without intraday replanning."""

    day_ahead_plan: DayAheadPlan
    ledger: EconomicLedger


@dataclass(frozen=True)
class RuleBasedReplayResult:
    """Fixed rule-policy schedule and its realised V1 ledger."""

    rule_plan: RuleBasedDayPlan
    ledger: EconomicLedger


@dataclass(frozen=True)
class OracleReplayResult:
    """Perfect-information lower bound under the same V1 settlement contract."""

    oracle_plan: OraclePlan
    ledger: EconomicLedger


def replay_deterministic_day(
    *,
    inputs: DayReplayInputs,
    scenario: ReferenceScenario,
) -> DeterministicReplayResult:
    """Run the V1 day-ahead plan followed by one executed MPC action per hour."""

    lower, upper = scenario.load_bounds()
    allowed = scenario.discharge_allowed()
    battery = scenario.battery(inputs.night_reference_price_eur_per_mwh)
    day_ahead_plan = solve_day_ahead(
        pv_forecast_kwh=inputs.day_ahead_pv_forecast_kwh,
        day_ahead_price_eur_per_mwh=inputs.day_ahead_price_forecast_eur_per_mwh,
        load_min_kwh=lower,
        load_max_kwh=upper,
        required_load_kwh=scenario.daily_load_kwh,
        battery=battery,
        discharge_allowed=allowed,
    )

    executed_load = np.zeros(HOURS_PER_DAY, dtype=float)
    executed_discharge = np.zeros(HOURS_PER_DAY, dtype=float)
    mpc_plans: list[MpcPlan] = []
    available_battery = battery.available_energy_kwh
    for hour in range(HOURS_PER_DAY):
        mpc_plan = solve_mpc(
            pv_forecast_kwh=inputs.mpc_pv_forecast_kwh[hour, hour:],
            intraday_price_eur_per_mwh=inputs.mpc_intraday_price_forecast_eur_per_mwh[hour, hour:],
            day_ahead_position_kwh=day_ahead_plan.day_ahead_position_kwh[hour:],
            load_min_kwh=lower[hour:],
            load_max_kwh=upper[hour:],
            required_remaining_load_kwh=scenario.daily_load_kwh - executed_load[:hour].sum(),
            available_battery_energy_kwh=available_battery,
            battery=battery,
            discharge_allowed=allowed[hour:],
        )
        executed_load[hour] = mpc_plan.load_kwh[0]
        executed_discharge[hour] = mpc_plan.discharge_kwh[0]
        available_battery -= executed_discharge[hour]
        mpc_plans.append(mpc_plan)

    if not np.isclose(executed_load.sum(), scenario.daily_load_kwh, atol=1e-7):
        raise AssertionError("MPC replay did not complete the daily production plan")
    ledger = evaluate_actual_day(
        day_ahead_position_kwh=day_ahead_plan.day_ahead_position_kwh,
        day_ahead_price_eur_per_mwh=inputs.actual_day_ahead_price_eur_per_mwh,
        actual_load_kwh=executed_load,
        actual_pv_kwh=inputs.actual_pv_kwh,
        actual_discharge_kwh=executed_discharge,
        actual_intraday_price_eur_per_mwh=inputs.actual_intraday_price_eur_per_mwh,
        battery=battery,
    )
    return DeterministicReplayResult(
        day_ahead_plan=day_ahead_plan,
        executed_load_kwh=executed_load,
        executed_discharge_kwh=executed_discharge,
        mpc_plans=tuple(mpc_plans),
        ledger=ledger,
    )


def replay_day_ahead_only_day(
    *,
    inputs: DayReplayInputs,
    scenario: ReferenceScenario,
) -> DayAheadOnlyReplayResult:
    """Execute the frozen point-forecast day-ahead plan without MPC.

    This is the attribution control: it shares the exact day-ahead forecast and
    nomination with ``replay_deterministic_day`` but forbids all intraday
    changes to load or battery discharge.
    """

    lower, upper = scenario.load_bounds()
    battery = scenario.battery(inputs.night_reference_price_eur_per_mwh)
    day_ahead_plan = solve_day_ahead(
        pv_forecast_kwh=inputs.day_ahead_pv_forecast_kwh,
        day_ahead_price_eur_per_mwh=inputs.day_ahead_price_forecast_eur_per_mwh,
        load_min_kwh=lower,
        load_max_kwh=upper,
        required_load_kwh=scenario.daily_load_kwh,
        battery=battery,
        discharge_allowed=scenario.discharge_allowed(),
    )
    ledger = evaluate_actual_day(
        day_ahead_position_kwh=day_ahead_plan.day_ahead_position_kwh,
        day_ahead_price_eur_per_mwh=inputs.actual_day_ahead_price_eur_per_mwh,
        actual_load_kwh=day_ahead_plan.load_kwh,
        actual_pv_kwh=inputs.actual_pv_kwh,
        actual_discharge_kwh=day_ahead_plan.discharge_kwh,
        actual_intraday_price_eur_per_mwh=inputs.actual_intraday_price_eur_per_mwh,
        battery=battery,
    )
    return DayAheadOnlyReplayResult(day_ahead_plan=day_ahead_plan, ledger=ledger)


def replay_rule_based_day(
    *,
    inputs: DayReplayInputs,
    scenario: ReferenceScenario,
    price_shape: RuleBasedPriceShape,
) -> RuleBasedReplayResult:
    """Settle the fixed historical-clock policy on the realised daily facts."""

    rule_plan = build_rule_based_plan(
        delivery_day=inputs.delivery_day,
        price_shape=price_shape,
        scenario=scenario,
        night_reference_price_eur_per_mwh=inputs.night_reference_price_eur_per_mwh,
    )
    battery = scenario.battery(inputs.night_reference_price_eur_per_mwh)
    ledger = evaluate_actual_day(
        day_ahead_position_kwh=rule_plan.day_ahead_position_kwh,
        day_ahead_price_eur_per_mwh=inputs.actual_day_ahead_price_eur_per_mwh,
        actual_load_kwh=rule_plan.load_kwh,
        actual_pv_kwh=inputs.actual_pv_kwh,
        actual_discharge_kwh=rule_plan.discharge_kwh,
        actual_intraday_price_eur_per_mwh=inputs.actual_intraday_price_eur_per_mwh,
        battery=battery,
    )
    return RuleBasedReplayResult(rule_plan=rule_plan, ledger=ledger)


def replay_oracle_day(
    *,
    inputs: DayReplayInputs,
    scenario: ReferenceScenario,
) -> OracleReplayResult:
    """Settle the perfect-information day-ahead Oracle.

    The Oracle uses factual PV and day-ahead prices only.  Its nomination and
    physical execution are identical, so the intraday ledger must be zero.
    """

    lower, upper = scenario.load_bounds()
    battery = scenario.battery(inputs.night_reference_price_eur_per_mwh)
    oracle_plan = solve_oracle_perfect_information(
        actual_pv_kwh=inputs.actual_pv_kwh,
        actual_day_ahead_price_eur_per_mwh=inputs.actual_day_ahead_price_eur_per_mwh,
        load_min_kwh=lower,
        load_max_kwh=upper,
        required_load_kwh=scenario.daily_load_kwh,
        battery=battery,
        discharge_allowed=scenario.discharge_allowed(),
    )
    ledger = evaluate_actual_day(
        day_ahead_position_kwh=oracle_plan.day_ahead_position_kwh,
        day_ahead_price_eur_per_mwh=inputs.actual_day_ahead_price_eur_per_mwh,
        actual_load_kwh=oracle_plan.actual_load_kwh,
        actual_pv_kwh=inputs.actual_pv_kwh,
        actual_discharge_kwh=oracle_plan.actual_discharge_kwh,
        actual_intraday_price_eur_per_mwh=inputs.actual_intraday_price_eur_per_mwh,
        battery=battery,
    )
    if not np.allclose(ledger.actual_deviation_kwh, 0.0, atol=1e-9):
        raise AssertionError("day-ahead Oracle created an intraday deviation")
    return OracleReplayResult(oracle_plan=oracle_plan, ledger=ledger)


def _validate_vector(values: np.ndarray, name: str) -> None:
    value = np.asarray(values, dtype=float)
    if value.shape != (HOURS_PER_DAY,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite vector with 24 hourly values")


def _validate_matrix(values: np.ndarray, name: str) -> None:
    value = np.asarray(values, dtype=float)
    if value.shape != (HOURS_PER_DAY, HOURS_PER_DAY) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite 24 by 24 matrix")
