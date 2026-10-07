"""Dynamic-battery MILP and physical-grid economic ledger.

The optimizer jointly chooses production load, battery actions and PV
curtailment.  Energy missing from the battery at the end of the working
window is valued at its expected all-in overnight replacement cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from energy.optimization.grid_tariff import (
    NO_GRID_TARIFF,
    GridTariffParameters,
)


MWH_TO_KWH = 1_000.0


@dataclass(frozen=True)
class DynamicBatteryParameters:
    """Physical limits and the conservative discharge-cost proxy."""

    capacity_kwh: float
    initial_soc_kwh: float
    max_charge_kwh_per_hour: float
    max_discharge_kwh_per_hour: float
    night_reference_price_eur_per_mwh: float
    charge_efficiency: float
    discharge_efficiency: float
    degradation_eur_per_kwh: float

    def __post_init__(self) -> None:
        if self.capacity_kwh < 0:
            raise ValueError("capacity_kwh must be non-negative")
        if not 0 <= self.initial_soc_kwh <= self.capacity_kwh:
            raise ValueError("initial_soc_kwh must lie within battery capacity")
        if self.max_charge_kwh_per_hour < 0 or self.max_discharge_kwh_per_hour < 0:
            raise ValueError("battery power limits must be non-negative")
        if not 0 < self.charge_efficiency <= 1:
            raise ValueError("charge_efficiency must be in (0, 1]")
        if not 0 < self.discharge_efficiency <= 1:
            raise ValueError("discharge_efficiency must be in (0, 1]")
        if self.degradation_eur_per_kwh < 0:
            raise ValueError("degradation_eur_per_kwh must be non-negative")
        if not np.isfinite(self.night_reference_price_eur_per_mwh):
            raise ValueError("night_reference_price_eur_per_mwh must be finite")


@dataclass(frozen=True)
class DynamicSchedule:
    """One feasible load and battery schedule for a contiguous horizon."""

    load_kwh: np.ndarray
    charge_kwh: np.ndarray
    discharge_kwh: np.ndarray
    curtailment_kwh: np.ndarray
    curtailment_fraction: np.ndarray
    soc_kwh: np.ndarray
    charge_mode: np.ndarray
    net_position_kwh: np.ndarray
    physical_grid_import_kwh: np.ndarray
    physical_grid_export_kwh: np.ndarray
    objective_eur: float


@dataclass(frozen=True)
class DynamicEconomicLedger:
    """Realised ledger for one operating window."""

    day_ahead_cost_eur: float
    intraday_deviation_cost_eur: float
    battery_reference_cost_eur: float
    battery_night_energy_cost_eur: float
    battery_night_grid_cost_eur: float
    battery_degradation_cost_eur: float
    terminal_recharge_cost_eur: float
    terminal_recharge_energy_kwh: float
    terminal_soc_kwh: float
    terminal_soc_value_eur_per_kwh: float
    daytime_network_cost_eur: float
    levies_cost_eur: float
    concession_fee_cost_eur: float
    electricity_tax_cost_eur: float
    export_fee_cost_eur: float
    total_cost_eur: float
    actual_deviation_kwh: np.ndarray
    actual_curtailment_kwh: np.ndarray
    actual_used_pv_kwh: np.ndarray
    physical_grid_import_kwh: np.ndarray
    physical_grid_export_kwh: np.ndarray


@dataclass(frozen=True)
class DynamicOraclePlan:
    """Perfect-information day-ahead plan.

    ``nominated`` and ``actual`` are identical; both names remain to preserve
    the replay artifact interface.
    """

    nominated: DynamicSchedule
    actual: DynamicSchedule
    day_ahead_position_kwh: np.ndarray
    objective_eur: float


def battery_reference_cost_eur_per_kwh(
    battery: DynamicBatteryParameters,
    grid_tariff: GridTariffParameters = NO_GRID_TARIFF,
) -> float:
    """Return night energy, night grid charges and degradation per delivered kWh."""

    components = battery_reference_cost_components_eur_per_kwh(
        battery,
        grid_tariff,
    )
    return float(sum(components))


def terminal_soc_value_eur_per_kwh(
    battery: DynamicBatteryParameters,
    grid_tariff: GridTariffParameters = NO_GRID_TARIFF,
) -> float:
    """Return the overnight replacement value of one stored kWh.

    ``night_reference_price_eur_per_mwh`` is already the causal rolling
    estimate available before the day-ahead decision.  Negative wholesale
    values are floored at zero.  The night import tariff is paid per input
    kWh, while one input kWh stores only ``charge_efficiency`` kWh.
    """

    overnight_input_cost = (
        max(float(battery.night_reference_price_eur_per_mwh), 0.0)
        / MWH_TO_KWH
        + grid_tariff.nighttime_variable_import_eur_per_kwh
    )
    return float(overnight_input_cost / battery.charge_efficiency)


def battery_reference_cost_components_eur_per_kwh(
    battery: DynamicBatteryParameters,
    grid_tariff: GridTariffParameters = NO_GRID_TARIFF,
) -> tuple[float, float, float]:
    """Return night energy, night regulated charges and degradation.

    The first two components are divided by round-trip efficiency because
    ``discharge_kwh`` is energy delivered from the battery to the site bus.
    """

    non_negative_night_price = max(
        float(battery.night_reference_price_eur_per_mwh), 0.0
    )
    round_trip_efficiency = battery.charge_efficiency * battery.discharge_efficiency
    night_energy = (
        non_negative_night_price
        / MWH_TO_KWH
        / round_trip_efficiency
    )
    night_grid = (
        grid_tariff.nighttime_variable_import_eur_per_kwh
        / round_trip_efficiency
    )
    return night_energy, night_grid, battery.degradation_eur_per_kwh


def solve_dynamic_schedule(
    *,
    pv_forecast_kwh: Iterable[float],
    price_eur_per_mwh: Iterable[float],
    load_min_kwh: Iterable[float],
    load_max_kwh: Iterable[float],
    required_load_kwh: float,
    battery: DynamicBatteryParameters,
    discharge_allowed: Iterable[bool],
    initial_soc_kwh: float | None = None,
    fixed_position_kwh: Iterable[float] | None = None,
    charge_allowed: Iterable[bool] | None = None,
    fixed_charge_kwh: Iterable[float] | None = None,
    fixed_discharge_kwh: Iterable[float] | None = None,
    grid_tariff: GridTariffParameters = NO_GRID_TARIFF,
) -> DynamicSchedule:
    """Solve the deterministic charge/discharge MILP.

    Charging, discharging and PV curtailment are economically selected over
    the full horizon.  The objective includes degradation and the overnight
    liability needed to restore the battery to full state of charge after the
    operating window.  A positive ``fixed_position_kwh`` converts the physical
    exchange into an MPC deviation without changing the physical constraints.
    """

    pv = _vector(pv_forecast_kwh, "pv_forecast_kwh")
    price = _vector(price_eur_per_mwh, "price_eur_per_mwh")
    lower = _vector(load_min_kwh, "load_min_kwh")
    upper = _vector(load_max_kwh, "load_max_kwh")
    discharge_mask = _bool_vector(discharge_allowed, "discharge_allowed")
    _validate_same_length(pv, price, lower, upper, discharge_mask)
    hours = len(price)
    if fixed_position_kwh is None:
        fixed_position = None
    else:
        fixed_position = _vector(fixed_position_kwh, "fixed_position_kwh")
        _validate_same_length(price, fixed_position)
    if charge_allowed is None:
        charge_mask = np.ones(hours, dtype=bool)
    else:
        charge_mask = _bool_vector(charge_allowed, "charge_allowed")
        _validate_same_length(price, charge_mask)
    if (lower < 0).any() or (upper < lower).any():
        raise ValueError("load bounds must satisfy 0 <= lower <= upper")
    if (pv < -1e-9).any():
        raise ValueError("pv_forecast_kwh must be non-negative")
    pv = np.maximum(pv, 0.0)
    if not lower.sum() <= required_load_kwh <= upper.sum():
        raise ValueError("required_load_kwh is infeasible under the load bounds")

    initial_soc = (
        battery.initial_soc_kwh
        if initial_soc_kwh is None
        else float(initial_soc_kwh)
    )
    if initial_soc < -1e-7 or initial_soc > battery.capacity_kwh + 1e-7:
        raise ValueError("initial_soc_kwh must lie within battery capacity")
    initial_soc = float(np.clip(initial_soc, 0.0, battery.capacity_kwh))
    fixed_charge = _optional_non_negative_vector(
        fixed_charge_kwh, "fixed_charge_kwh", hours
    )
    fixed_discharge = _optional_non_negative_vector(
        fixed_discharge_kwh, "fixed_discharge_kwh", hours
    )
    if fixed_charge is not None and fixed_discharge is not None:
        if np.any((fixed_charge > 1e-9) & (fixed_discharge > 1e-9)):
            raise ValueError("fixed battery schedule cannot charge and discharge together")

    # Variable blocks: load[H], charge[H], discharge[H], curtailment[H],
    # SoC[H+1], mode[H],
    # physical grid import[H], export[H]. Import/export are constrained to the
    # positive/negative parts of the physical balance; unlike an MPC deviation
    # it never subtracts the already-fixed commercial position.
    load_slice = slice(0, hours)
    charge_slice = slice(hours, 2 * hours)
    discharge_slice = slice(2 * hours, 3 * hours)
    curtailment_slice = slice(3 * hours, 4 * hours)
    soc_slice = slice(4 * hours, 5 * hours + 1)
    mode_slice = slice(5 * hours + 1, 6 * hours + 1)
    import_slice = slice(6 * hours + 1, 7 * hours + 1)
    export_slice = slice(7 * hours + 1, 8 * hours + 1)
    n_variables = 8 * hours + 1

    price_kwh = price / MWH_TO_KWH
    objective = np.zeros(n_variables, dtype=float)
    objective[load_slice] = price_kwh
    objective[charge_slice] = price_kwh
    objective[discharge_slice] = battery.degradation_eur_per_kwh - price_kwh
    objective[curtailment_slice] = price_kwh
    objective[import_slice] = grid_tariff.daytime_variable_import_eur_per_kwh
    objective[export_slice] = grid_tariff.export_fee_eur_per_kwh
    terminal_value = terminal_soc_value_eur_per_kwh(battery, grid_tariff)
    objective[soc_slice.stop - 1] = -terminal_value
    constant = -float(np.dot(price_kwh, pv))
    if fixed_position is not None:
        constant -= float(np.dot(price_kwh, fixed_position))
    constant += terminal_value * battery.capacity_kwh

    lower_bounds = np.zeros(n_variables, dtype=float)
    upper_bounds = np.full(n_variables, np.inf, dtype=float)
    lower_bounds[load_slice] = lower
    upper_bounds[load_slice] = upper
    upper_bounds[charge_slice] = np.where(
        charge_mask, battery.max_charge_kwh_per_hour, 0.0
    )
    upper_bounds[discharge_slice] = np.where(
        discharge_mask, battery.max_discharge_kwh_per_hour, 0.0
    )
    upper_bounds[curtailment_slice] = pv
    if fixed_charge is not None:
        if np.any(fixed_charge > upper_bounds[charge_slice] + 1e-9):
            raise ValueError("fixed_charge_kwh violates charge limits or mask")
        lower_bounds[charge_slice] = fixed_charge
        upper_bounds[charge_slice] = fixed_charge
    if fixed_discharge is not None:
        if np.any(fixed_discharge > upper_bounds[discharge_slice] + 1e-9):
            raise ValueError("fixed_discharge_kwh violates discharge limits or mask")
        lower_bounds[discharge_slice] = fixed_discharge
        upper_bounds[discharge_slice] = fixed_discharge
    upper_bounds[soc_slice] = battery.capacity_kwh
    lower_bounds[4 * hours] = initial_soc
    upper_bounds[4 * hours] = initial_soc
    upper_bounds[mode_slice] = 1.0

    # Required production energy and H state transitions.
    equality = np.zeros((hours + 1, n_variables), dtype=float)
    equality[0, load_slice] = 1.0
    equality_rhs = np.zeros(hours + 1, dtype=float)
    equality_rhs[0] = required_load_kwh
    for index in range(hours):
        row = index + 1
        equality[row, hours + index] = -battery.charge_efficiency
        equality[row, 2 * hours + index] = 1.0 / battery.discharge_efficiency
        equality[row, 4 * hours + index] = -1.0
        equality[row, 4 * hours + index + 1] = 1.0

    # C_h <= C_max z_h, D_h <= D_max (1 - z_h), and
    # load_h + charge_h + curtailment_h - discharge_h - import_h <= PV_h,
    # and the symmetric export inequality.
    inequality = np.zeros((4 * hours, n_variables), dtype=float)
    inequality_upper = np.zeros(4 * hours, dtype=float)
    for index in range(hours):
        mode_column = 5 * hours + 1 + index
        base = 4 * index
        inequality[base, hours + index] = 1.0
        inequality[base, mode_column] = -battery.max_charge_kwh_per_hour
        inequality[base + 1, 2 * hours + index] = 1.0
        inequality[base + 1, mode_column] = battery.max_discharge_kwh_per_hour
        inequality_upper[base + 1] = battery.max_discharge_kwh_per_hour
        inequality[base + 2, index] = 1.0
        inequality[base + 2, hours + index] = 1.0
        inequality[base + 2, 2 * hours + index] = -1.0
        inequality[base + 2, 3 * hours + index] = 1.0
        inequality[base + 2, 6 * hours + 1 + index] = -1.0
        inequality_upper[base + 2] = pv[index]
        inequality[base + 3, index] = -1.0
        inequality[base + 3, hours + index] = -1.0
        inequality[base + 3, 2 * hours + index] = 1.0
        inequality[base + 3, 3 * hours + index] = -1.0
        inequality[base + 3, 7 * hours + 1 + index] = -1.0
        inequality_upper[base + 3] = -pv[index]

    integrality = np.zeros(n_variables, dtype=int)
    integrality[mode_slice] = 1
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=(
            LinearConstraint(equality, equality_rhs, equality_rhs),
            LinearConstraint(
                inequality,
                np.full(4 * hours, -np.inf, dtype=float),
                inequality_upper,
            ),
        ),
        options={"disp": False},
    )
    if not result.success or result.x is None:
        raise ValueError(f"Dynamic battery MILP did not solve: {result.message}")

    load = np.asarray(result.x[load_slice], dtype=float)
    charge = np.asarray(result.x[charge_slice], dtype=float)
    discharge = np.asarray(result.x[discharge_slice], dtype=float)
    curtailment = np.clip(
        np.asarray(result.x[curtailment_slice], dtype=float),
        0.0,
        pv,
    )
    soc = np.clip(
        np.asarray(result.x[soc_slice], dtype=float),
        0.0,
        battery.capacity_kwh,
    )
    mode = np.rint(np.asarray(result.x[mode_slice], dtype=float)).astype(int)
    if np.any((charge > 1e-7) & (discharge > 1e-7)):
        raise AssertionError("MILP returned simultaneous charge and discharge")
    net_position = load + charge - pv + curtailment - discharge
    physical_grid_import = np.maximum(net_position, 0.0)
    physical_grid_export = np.maximum(-net_position, 0.0)
    if fixed_position is not None:
        net_position = net_position - fixed_position
    return DynamicSchedule(
        load_kwh=load,
        charge_kwh=charge,
        discharge_kwh=discharge,
        curtailment_kwh=curtailment,
        curtailment_fraction=np.clip(
            np.divide(
                curtailment,
                pv,
                out=np.zeros_like(curtailment),
                where=pv > 1e-9,
            ),
            0.0,
            1.0,
        ),
        soc_kwh=soc,
        charge_mode=mode,
        net_position_kwh=net_position,
        physical_grid_import_kwh=physical_grid_import,
        physical_grid_export_kwh=physical_grid_export,
        objective_eur=float(result.fun + constant),
    )


def solve_dynamic_oracle(
    *,
    actual_pv_kwh: Iterable[float],
    actual_day_ahead_price_eur_per_mwh: Iterable[float],
    load_min_kwh: Iterable[float],
    load_max_kwh: Iterable[float],
    required_load_kwh: float,
    battery: DynamicBatteryParameters,
    discharge_allowed: Iterable[bool],
    grid_tariff: GridTariffParameters = NO_GRID_TARIFF,
) -> DynamicOraclePlan:
    """Solve the perfect-information day-ahead benchmark.

    Factual PV and day-ahead prices are known.  A single feasible schedule
    defines both the nomination and execution, preventing deliberate intraday
    imbalance and excluding intraday prices from the Oracle decision.
    """

    schedule = solve_dynamic_schedule(
        pv_forecast_kwh=actual_pv_kwh,
        price_eur_per_mwh=actual_day_ahead_price_eur_per_mwh,
        load_min_kwh=load_min_kwh,
        load_max_kwh=load_max_kwh,
        required_load_kwh=required_load_kwh,
        battery=battery,
        discharge_allowed=discharge_allowed,
        grid_tariff=grid_tariff,
    )
    return DynamicOraclePlan(
        nominated=schedule,
        actual=schedule,
        day_ahead_position_kwh=schedule.net_position_kwh,
        objective_eur=schedule.objective_eur,
    )


def evaluate_dynamic_actual_day(
    *,
    day_ahead_position_kwh: Iterable[float],
    day_ahead_price_eur_per_mwh: Iterable[float],
    actual_load_kwh: Iterable[float],
    actual_charge_kwh: Iterable[float],
    actual_pv_kwh: Iterable[float],
    actual_discharge_kwh: Iterable[float],
    actual_intraday_price_eur_per_mwh: Iterable[float],
    battery: DynamicBatteryParameters,
    actual_curtailment_kwh: Iterable[float] | None = None,
    grid_tariff: GridTariffParameters = NO_GRID_TARIFF,
) -> DynamicEconomicLedger:
    """Settle one factual operating window under the agreed proxy ledger."""

    position = _vector(day_ahead_position_kwh, "day_ahead_position_kwh")
    day_ahead_price = _vector(
        day_ahead_price_eur_per_mwh, "day_ahead_price_eur_per_mwh"
    )
    load = _vector(actual_load_kwh, "actual_load_kwh")
    charge = _vector(actual_charge_kwh, "actual_charge_kwh")
    pv = _vector(actual_pv_kwh, "actual_pv_kwh")
    discharge = _vector(actual_discharge_kwh, "actual_discharge_kwh")
    intraday_price = _vector(
        actual_intraday_price_eur_per_mwh,
        "actual_intraday_price_eur_per_mwh",
    )
    curtailment = (
        np.zeros_like(pv)
        if actual_curtailment_kwh is None
        else _vector(actual_curtailment_kwh, "actual_curtailment_kwh")
    )
    _validate_same_length(
        position,
        day_ahead_price,
        load,
        charge,
        pv,
        discharge,
        intraday_price,
        curtailment,
    )
    if (curtailment < -1e-9).any() or (curtailment > pv + 1e-9).any():
        raise ValueError("actual curtailment must lie between zero and actual PV")
    curtailment = np.clip(curtailment, 0.0, pv)
    used_pv = pv - curtailment
    deviation = load + charge - used_pv - discharge - position
    physical_exchange = load + charge - used_pv - discharge
    physical_import = np.maximum(physical_exchange, 0.0)
    physical_export = np.maximum(-physical_exchange, 0.0)
    day_ahead_cost = float(np.dot(day_ahead_price / MWH_TO_KWH, position))
    intraday_cost = float(np.dot(intraday_price / MWH_TO_KWH, deviation))
    discharged = float(discharge.sum())
    terminal_soc = (
        battery.initial_soc_kwh
        + battery.charge_efficiency * float(charge.sum())
        - discharged / battery.discharge_efficiency
    )
    if terminal_soc < -1e-7 or terminal_soc > battery.capacity_kwh + 1e-7:
        raise ValueError("actual battery actions violate state-of-charge limits")
    terminal_soc = float(np.clip(terminal_soc, 0.0, battery.capacity_kwh))
    terminal_recharge_energy = (
        battery.capacity_kwh - terminal_soc
    ) / battery.charge_efficiency
    non_negative_night_price = max(
        float(battery.night_reference_price_eur_per_mwh), 0.0
    )
    battery_night_energy_cost = (
        terminal_recharge_energy * non_negative_night_price / MWH_TO_KWH
    )
    battery_night_grid_cost = (
        terminal_recharge_energy
        * grid_tariff.nighttime_variable_import_eur_per_kwh
    )
    battery_degradation_cost = discharged * battery.degradation_eur_per_kwh
    terminal_recharge_cost = (
        battery_night_energy_cost + battery_night_grid_cost
    )
    battery_cost = (
        terminal_recharge_cost + battery_degradation_cost
    )
    imported = float(physical_import.sum())
    exported = float(physical_export.sum())
    daytime_network_cost = imported * grid_tariff.daytime_network_eur_per_kwh
    levies_cost = imported * grid_tariff.levies_eur_per_kwh
    concession_fee_cost = imported * grid_tariff.concession_fee_eur_per_kwh
    electricity_tax_cost = imported * grid_tariff.electricity_tax_eur_per_kwh
    export_fee_cost = exported * grid_tariff.export_fee_eur_per_kwh
    daytime_grid_cost = (
        daytime_network_cost
        + levies_cost
        + concession_fee_cost
        + electricity_tax_cost
        + export_fee_cost
    )
    return DynamicEconomicLedger(
        day_ahead_cost_eur=day_ahead_cost,
        intraday_deviation_cost_eur=intraday_cost,
        battery_reference_cost_eur=battery_cost,
        battery_night_energy_cost_eur=battery_night_energy_cost,
        battery_night_grid_cost_eur=battery_night_grid_cost,
        battery_degradation_cost_eur=battery_degradation_cost,
        terminal_recharge_cost_eur=terminal_recharge_cost,
        terminal_recharge_energy_kwh=terminal_recharge_energy,
        terminal_soc_kwh=terminal_soc,
        terminal_soc_value_eur_per_kwh=terminal_soc_value_eur_per_kwh(
            battery, grid_tariff
        ),
        daytime_network_cost_eur=daytime_network_cost,
        levies_cost_eur=levies_cost,
        concession_fee_cost_eur=concession_fee_cost,
        electricity_tax_cost_eur=electricity_tax_cost,
        export_fee_cost_eur=export_fee_cost,
        total_cost_eur=(
            day_ahead_cost + intraday_cost + battery_cost + daytime_grid_cost
        ),
        actual_deviation_kwh=deviation,
        actual_curtailment_kwh=curtailment,
        actual_used_pv_kwh=used_pv,
        physical_grid_import_kwh=physical_import,
        physical_grid_export_kwh=physical_export,
    )


def _vector(values: Iterable[float], name: str) -> np.ndarray:
    result = np.asarray(list(values), dtype=float)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a non-empty finite vector")
    return result


def _bool_vector(values: Iterable[bool], name: str) -> np.ndarray:
    result = np.asarray(list(values), dtype=bool)
    if result.ndim != 1 or not len(result):
        raise ValueError(f"{name} must be a non-empty boolean vector")
    return result


def _optional_non_negative_vector(
    values: Iterable[float] | None,
    name: str,
    expected_length: int,
) -> np.ndarray | None:
    if values is None:
        return None
    result = _vector(values, name)
    if len(result) != expected_length:
        raise ValueError(f"{name} must contain {expected_length} values")
    if (result < -1e-7).any():
        raise ValueError(f"{name} must be non-negative")
    return np.maximum(result, 0.0)


def _validate_same_length(*arrays: np.ndarray) -> None:
    if len({len(array) for array in arrays}) != 1:
        raise ValueError("all hourly vectors must have the same length")
