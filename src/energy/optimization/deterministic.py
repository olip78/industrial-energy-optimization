"""Linear day-ahead and MPC decisions with a transparent economic ledger.

All energy quantities are kWh for one hourly period.  Market prices arrive in
EUR/MWh and are converted to EUR/kWh internally, preventing a silent 1,000x
unit error in the objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import linprog

MWH_TO_KWH = 1_000.0


@dataclass(frozen=True)
class BatteryParameters:
    """Available daytime discharge resource; overnight charging is exogenous."""

    available_energy_kwh: float
    max_discharge_kwh_per_hour: float
    night_reference_price_eur_per_mwh: float
    charge_efficiency: float
    discharge_efficiency: float
    degradation_eur_per_kwh: float

    def __post_init__(self) -> None:
        if self.available_energy_kwh < 0 or self.max_discharge_kwh_per_hour < 0:
            raise ValueError("Battery energy and hourly discharge limit must be non-negative")
        if not 0 < self.charge_efficiency <= 1 or not 0 < self.discharge_efficiency <= 1:
            raise ValueError("Battery efficiencies must be in (0, 1]")
        if self.degradation_eur_per_kwh < 0:
            raise ValueError("Battery degradation cost must be non-negative")


@dataclass(frozen=True)
class DayAheadPlan:
    """Initial 24-hour physical plan and resulting commercial position."""

    load_kwh: np.ndarray
    discharge_kwh: np.ndarray
    day_ahead_position_kwh: np.ndarray
    objective_eur: float


@dataclass(frozen=True)
class MpcPlan:
    """Replanned tail; only index zero becomes an executed action."""

    load_kwh: np.ndarray
    discharge_kwh: np.ndarray
    predicted_deviation_kwh: np.ndarray
    objective_eur: float


@dataclass(frozen=True)
class OraclePlan:
    """Perfect-information day-ahead plan under the V1 ledger.

    The Oracle knows factual PV and day-ahead prices.  Nomination and physical
    execution are the same schedule, so it cannot create a deliberate
    intraday imbalance.  The duplicated fields are retained for compatibility
    with existing replay artifacts.
    """

    nominated_load_kwh: np.ndarray
    nominated_discharge_kwh: np.ndarray
    day_ahead_position_kwh: np.ndarray
    actual_load_kwh: np.ndarray
    actual_discharge_kwh: np.ndarray
    objective_eur: float


@dataclass(frozen=True)
class EconomicLedger:
    """Ex-post daily cost under the simplified V1 settlement convention."""

    day_ahead_cost_eur: float
    intraday_deviation_cost_eur: float
    battery_cost_eur: float
    total_cost_eur: float
    actual_deviation_kwh: np.ndarray


def battery_full_cost_eur_per_kwh(parameters: BatteryParameters) -> float:
    """Full marginal cost of one delivered kWh from the overnight-charged battery."""

    charged_energy_cost = (
        parameters.night_reference_price_eur_per_mwh
        / MWH_TO_KWH
        / (parameters.charge_efficiency * parameters.discharge_efficiency)
    )
    return charged_energy_cost + parameters.degradation_eur_per_kwh


def solve_day_ahead(
    *,
    pv_forecast_kwh: Iterable[float],
    day_ahead_price_eur_per_mwh: Iterable[float],
    load_min_kwh: Iterable[float],
    load_max_kwh: Iterable[float],
    required_load_kwh: float,
    battery: BatteryParameters,
    discharge_allowed: Iterable[bool],
) -> DayAheadPlan:
    """Minimise the planned day-ahead cost for one delivery day.

    The day-ahead position is derived from the physical balance:
    ``q = load - PV - battery discharge``. Positive position means purchase;
    negative means sale. The optimiser does not choose overnight charging.
    """

    pv = _as_vector(pv_forecast_kwh, "pv_forecast_kwh")
    price = _as_vector(day_ahead_price_eur_per_mwh, "day_ahead_price_eur_per_mwh")
    load_min = _as_vector(load_min_kwh, "load_min_kwh")
    load_max = _as_vector(load_max_kwh, "load_max_kwh")
    allowed = _as_bool_vector(discharge_allowed, "discharge_allowed")
    _validate_shared_length(pv, price, load_min, load_max, allowed)
    _validate_load_bounds(load_min, load_max, required_load_kwh)

    load, discharge, objective = _solve_core(
        price_eur_per_mwh=price,
        pv_kwh=pv,
        load_min_kwh=load_min,
        load_max_kwh=load_max,
        required_load_kwh=required_load_kwh,
        battery=battery,
        discharge_allowed=allowed,
        fixed_position_kwh=None,
    )
    position = load - pv - discharge
    return DayAheadPlan(load, discharge, position, objective)


def solve_mpc(
    *,
    pv_forecast_kwh: Iterable[float],
    intraday_price_eur_per_mwh: Iterable[float],
    day_ahead_position_kwh: Iterable[float],
    load_min_kwh: Iterable[float],
    load_max_kwh: Iterable[float],
    required_remaining_load_kwh: float,
    available_battery_energy_kwh: float,
    battery: BatteryParameters,
    discharge_allowed: Iterable[bool],
) -> MpcPlan:
    """Replan the remaining horizon against a fixed day-ahead position.

    ``day_ahead_position_kwh`` is a bookkeeping constant in the MPC objective,
    but it is retained to calculate the predicted deviation used by the ledger.
    ``available_battery_energy_kwh`` is the factual state after earlier hours.
    """

    pv = _as_vector(pv_forecast_kwh, "pv_forecast_kwh")
    price = _as_vector(intraday_price_eur_per_mwh, "intraday_price_eur_per_mwh")
    position = _as_vector(day_ahead_position_kwh, "day_ahead_position_kwh")
    load_min = _as_vector(load_min_kwh, "load_min_kwh")
    load_max = _as_vector(load_max_kwh, "load_max_kwh")
    allowed = _as_bool_vector(discharge_allowed, "discharge_allowed")
    _validate_shared_length(pv, price, position, load_min, load_max, allowed)
    _validate_load_bounds(load_min, load_max, required_remaining_load_kwh)
    if not 0 <= available_battery_energy_kwh <= battery.available_energy_kwh:
        raise ValueError("available_battery_energy_kwh must be within the battery capacity")

    remaining_battery = BatteryParameters(
        available_energy_kwh=available_battery_energy_kwh,
        max_discharge_kwh_per_hour=battery.max_discharge_kwh_per_hour,
        night_reference_price_eur_per_mwh=battery.night_reference_price_eur_per_mwh,
        charge_efficiency=battery.charge_efficiency,
        discharge_efficiency=battery.discharge_efficiency,
        degradation_eur_per_kwh=battery.degradation_eur_per_kwh,
    )
    load, discharge, objective = _solve_core(
        price_eur_per_mwh=price,
        pv_kwh=pv,
        load_min_kwh=load_min,
        load_max_kwh=load_max,
        required_load_kwh=required_remaining_load_kwh,
        battery=remaining_battery,
        discharge_allowed=allowed,
        fixed_position_kwh=position,
    )
    deviation = load - pv - discharge - position
    return MpcPlan(load, discharge, deviation, objective)


def solve_oracle_perfect_information(
    *,
    actual_pv_kwh: Iterable[float],
    actual_day_ahead_price_eur_per_mwh: Iterable[float],
    load_min_kwh: Iterable[float],
    load_max_kwh: Iterable[float],
    required_load_kwh: float,
    battery: BatteryParameters,
    discharge_allowed: Iterable[bool],
) -> OraclePlan:
    """Solve the perfect-information day-ahead benchmark.

    The Oracle sees factual PV and day-ahead prices, then chooses one physical
    load and battery schedule.  Its nomination is exactly that schedule's net
    position; intraday prices do not enter the optimization and the realised
    deviation is zero.
    """

    plan = solve_day_ahead(
        pv_forecast_kwh=actual_pv_kwh,
        day_ahead_price_eur_per_mwh=actual_day_ahead_price_eur_per_mwh,
        load_min_kwh=load_min_kwh,
        load_max_kwh=load_max_kwh,
        required_load_kwh=required_load_kwh,
        battery=battery,
        discharge_allowed=discharge_allowed,
    )
    return OraclePlan(
        nominated_load_kwh=plan.load_kwh,
        nominated_discharge_kwh=plan.discharge_kwh,
        day_ahead_position_kwh=plan.day_ahead_position_kwh,
        actual_load_kwh=plan.load_kwh.copy(),
        actual_discharge_kwh=plan.discharge_kwh.copy(),
        objective_eur=plan.objective_eur,
    )

def evaluate_actual_day(
    *,
    day_ahead_position_kwh: Iterable[float],
    day_ahead_price_eur_per_mwh: Iterable[float],
    actual_load_kwh: Iterable[float],
    actual_pv_kwh: Iterable[float],
    actual_discharge_kwh: Iterable[float],
    actual_intraday_price_eur_per_mwh: Iterable[float],
    battery: BatteryParameters,
) -> EconomicLedger:
    """Calculate the V1 ex-post ledger from realised actions, PV and prices."""

    position = _as_vector(day_ahead_position_kwh, "day_ahead_position_kwh")
    day_ahead_price = _as_vector(day_ahead_price_eur_per_mwh, "day_ahead_price_eur_per_mwh")
    load = _as_vector(actual_load_kwh, "actual_load_kwh")
    pv = _as_vector(actual_pv_kwh, "actual_pv_kwh")
    discharge = _as_vector(actual_discharge_kwh, "actual_discharge_kwh")
    intraday_price = _as_vector(actual_intraday_price_eur_per_mwh, "actual_intraday_price_eur_per_mwh")
    _validate_shared_length(position, day_ahead_price, load, pv, discharge, intraday_price)
    deviation = load - pv - discharge - position
    day_ahead_cost = float(np.dot(day_ahead_price / MWH_TO_KWH, position))
    intraday_cost = float(np.dot(intraday_price / MWH_TO_KWH, deviation))
    battery_cost = float(discharge.sum() * battery_full_cost_eur_per_kwh(battery))
    return EconomicLedger(
        day_ahead_cost_eur=day_ahead_cost,
        intraday_deviation_cost_eur=intraday_cost,
        battery_cost_eur=battery_cost,
        total_cost_eur=day_ahead_cost + intraday_cost + battery_cost,
        actual_deviation_kwh=deviation,
    )


def _solve_core(
    *,
    price_eur_per_mwh: np.ndarray,
    pv_kwh: np.ndarray,
    load_min_kwh: np.ndarray,
    load_max_kwh: np.ndarray,
    required_load_kwh: float,
    battery: BatteryParameters,
    discharge_allowed: np.ndarray,
    fixed_position_kwh: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve the shared LP; fixed day-ahead position is objective-only constant."""

    hours = len(price_eur_per_mwh)
    price_eur_per_kwh = price_eur_per_mwh / MWH_TO_KWH
    battery_cost = battery_full_cost_eur_per_kwh(battery)
    # The variable vector is [load_0..load_H, discharge_0..discharge_H].
    objective = np.concatenate((price_eur_per_kwh, battery_cost - price_eur_per_kwh))
    # PV and, in MPC, q_DA do not change the decision coefficients. Retain the
    # constants in objective reporting after optimisation for auditability.
    constant = -float(np.dot(price_eur_per_kwh, pv_kwh))
    if fixed_position_kwh is not None:
        constant -= float(np.dot(price_eur_per_kwh, fixed_position_kwh))

    equality_matrix = np.zeros((1, 2 * hours))
    equality_matrix[0, :hours] = 1.0
    inequality_matrix = np.zeros((1, 2 * hours))
    inequality_matrix[0, hours:] = 1.0
    discharge_upper = np.where(
        discharge_allowed, battery.max_discharge_kwh_per_hour, 0.0
    )
    bounds = [*zip(load_min_kwh, load_max_kwh, strict=True), *[(0.0, float(value)) for value in discharge_upper]]
    result = linprog(
        c=objective,
        A_ub=inequality_matrix,
        b_ub=np.array([battery.available_energy_kwh]),
        A_eq=equality_matrix,
        b_eq=np.array([required_load_kwh]),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise ValueError(f"Energy LP did not solve: {result.message}")
    load = result.x[:hours]
    discharge = result.x[hours:]
    return load, discharge, float(result.fun + constant)


def _as_vector(values: Iterable[float], name: str) -> np.ndarray:
    result = np.asarray(list(values), dtype=float)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a non-empty finite one-dimensional vector")
    return result


def _as_bool_vector(values: Iterable[bool], name: str) -> np.ndarray:
    result = np.asarray(list(values), dtype=bool)
    if result.ndim != 1 or not len(result):
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    return result


def _validate_shared_length(*arrays: np.ndarray) -> None:
    if len({len(array) for array in arrays}) != 1:
        raise ValueError("All hourly input vectors must have the same length")


def _validate_load_bounds(
    load_min_kwh: np.ndarray, load_max_kwh: np.ndarray, required_load_kwh: float
) -> None:
    if (load_min_kwh < 0).any() or (load_max_kwh < load_min_kwh).any():
        raise ValueError("Load bounds must be non-negative and ordered")
    if not load_min_kwh.sum() <= required_load_kwh <= load_max_kwh.sum():
        raise ValueError("required load must be feasible within hourly load bounds")
