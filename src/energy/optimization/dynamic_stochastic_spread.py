"""Day-ahead stochastic MILP with explicit intraday-spread settlement.

Every physical decision is made before uncertainty is revealed and is common
to all scenarios.  Scenario-specific import/export variables only account for
the positive and negative parts of the resulting physical grid exchange.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix, vstack

from energy.optimization.dynamic_battery import (
    MWH_TO_KWH,
    DynamicBatteryParameters,
    terminal_soc_value_eur_per_kwh,
)
from energy.optimization.grid_tariff import NO_GRID_TARIFF, GridTariffParameters


@dataclass(frozen=True)
class DynamicSpreadStochasticPlan:
    """One common executable day-ahead plan and its scenario costs."""

    load_kwh: np.ndarray
    charge_kwh: np.ndarray
    discharge_kwh: np.ndarray
    curtailment_fraction: np.ndarray
    point_curtailment_kwh: np.ndarray
    soc_kwh: np.ndarray
    charge_mode: np.ndarray
    day_ahead_position_kwh: np.ndarray
    scenario_cost_eur: np.ndarray
    expected_cost_eur: float
    cvar_cost_eur: float
    objective_eur: float
    cvar_alpha: float
    risk_weight: float


def solve_dynamic_stochastic_spread_day_ahead(
    *,
    pv_scenarios_kwh: np.ndarray,
    day_ahead_price_scenarios_eur_per_mwh: np.ndarray,
    intraday_spread_scenarios_eur_per_mwh: np.ndarray,
    point_pv_forecast_kwh: np.ndarray,
    load_min_kwh: np.ndarray,
    load_max_kwh: np.ndarray,
    required_load_kwh: float,
    battery: DynamicBatteryParameters,
    discharge_allowed: np.ndarray,
    grid_tariff: GridTariffParameters = NO_GRID_TARIFF,
    cvar_alpha: float = 0.95,
    risk_weight: float = 0.0,
) -> DynamicSpreadStochasticPlan:
    """Minimize mean scenario cost plus ``risk_weight * CVaR``.

    The submitted position is the common schedule evaluated at the frozen
    point PV forecast.  Consequently, uncertainty creates an imbalance but
    cannot create a deliberate speculative day-ahead/intraday position:

    ``q[h] = L[h] + C[h] - D[h] - (1-u[h]) * point_pv[h]``.

    Here ``u[h]`` is one common curtailment fraction.  In scenario ``s`` the
    same fraction is applied to scenario PV, and the deviation is settled at
    ``day_ahead_price[s,h] + spread[s,h]``.
    """

    pv = _scenario_matrix(pv_scenarios_kwh, "pv_scenarios_kwh")
    da_price = _scenario_matrix(
        day_ahead_price_scenarios_eur_per_mwh,
        "day_ahead_price_scenarios_eur_per_mwh",
    )
    spread = _scenario_matrix(
        intraday_spread_scenarios_eur_per_mwh,
        "intraday_spread_scenarios_eur_per_mwh",
    )
    if pv.shape != da_price.shape or pv.shape != spread.shape:
        raise ValueError("PV, day-ahead price and spread scenarios must share a shape")
    if (pv < -1e-9).any():
        raise ValueError("PV scenarios must be non-negative")
    pv = np.maximum(pv, 0.0)
    scenarios, hours = pv.shape
    point_pv = _vector(point_pv_forecast_kwh, "point_pv_forecast_kwh", hours)
    if (point_pv < -1e-9).any():
        raise ValueError("point_pv_forecast_kwh must be non-negative")
    point_pv = np.maximum(point_pv, 0.0)
    lower = _vector(load_min_kwh, "load_min_kwh", hours)
    upper = _vector(load_max_kwh, "load_max_kwh", hours)
    allowed = np.asarray(discharge_allowed, dtype=bool)
    if allowed.shape != (hours,):
        raise ValueError("discharge_allowed must contain one value per hour")
    if (lower < 0).any() or (upper < lower).any():
        raise ValueError("load bounds must satisfy 0 <= lower <= upper")
    if not lower.sum() <= required_load_kwh <= upper.sum():
        raise ValueError("required_load_kwh is infeasible under the load bounds")
    if not 0 < cvar_alpha < 1:
        raise ValueError("cvar_alpha must lie strictly inside (0, 1)")
    if risk_weight < 0 or not np.isfinite(risk_weight):
        raise ValueError("risk_weight must be finite and non-negative")

    # Common first-stage blocks: L, C, D, curtailment fraction, SoC and mode.
    load_slice = slice(0, hours)
    charge_slice = slice(hours, 2 * hours)
    discharge_slice = slice(2 * hours, 3 * hours)
    curtailment_slice = slice(3 * hours, 4 * hours)
    soc_slice = slice(4 * hours, 5 * hours + 1)
    mode_slice = slice(5 * hours + 1, 6 * hours + 1)
    physical_variables = 6 * hours + 1

    intraday_price = da_price + spread
    da_price_kwh = da_price / MWH_TO_KWH
    intraday_price_kwh = intraday_price / MWH_TO_KWH
    terminal_value = terminal_soc_value_eur_per_kwh(battery, grid_tariff)

    # Exact scenario ledger before grid import/export charges.  Starting from
    # P_DA*q + P_ID*(g_s-q) keeps the financial meaning explicit.
    scenario_coefficients = np.zeros((scenarios, physical_variables), dtype=float)
    scenario_coefficients[:, load_slice] = da_price_kwh
    scenario_coefficients[:, charge_slice] = da_price_kwh
    scenario_coefficients[:, discharge_slice] = (
        battery.degradation_eur_per_kwh - da_price_kwh
    )
    scenario_coefficients[:, curtailment_slice] = (
        da_price_kwh * point_pv[None, :]
        + intraday_price_kwh * (pv - point_pv[None, :])
    )
    scenario_coefficients[:, soc_slice.stop - 1] = -terminal_value
    scenario_constants = (
        -np.sum(da_price_kwh * point_pv[None, :], axis=1)
        + np.sum(
            intraday_price_kwh * (point_pv[None, :] - pv),
            axis=1,
        )
        + terminal_value * battery.capacity_kwh
    )

    # Scenario flows are accounting auxiliaries, not recourse decisions.
    scenario_flow_count = scenarios * hours
    import_slice = slice(
        physical_variables,
        physical_variables + scenario_flow_count,
    )
    export_slice = slice(import_slice.stop, import_slice.stop + scenario_flow_count)
    base_variables = export_slice.stop
    with_cvar = risk_weight > 0.0
    eta_index = base_variables if with_cvar else None
    excess_slice = (
        slice(base_variables + 1, base_variables + 1 + scenarios)
        if with_cvar
        else slice(base_variables, base_variables)
    )
    variable_count = base_variables + (1 + scenarios if with_cvar else 0)

    objective = np.zeros(variable_count, dtype=float)
    objective[:physical_variables] = scenario_coefficients.mean(axis=0)
    objective[import_slice] = (
        grid_tariff.daytime_variable_import_eur_per_kwh / scenarios
    )
    objective[export_slice] = grid_tariff.export_fee_eur_per_kwh / scenarios
    if with_cvar:
        assert eta_index is not None
        objective[eta_index] = risk_weight
        objective[excess_slice] = risk_weight / ((1.0 - cvar_alpha) * scenarios)

    lower_bounds = np.zeros(variable_count, dtype=float)
    upper_bounds = np.full(variable_count, np.inf, dtype=float)
    lower_bounds[load_slice] = lower
    upper_bounds[load_slice] = upper
    upper_bounds[charge_slice] = battery.max_charge_kwh_per_hour
    upper_bounds[discharge_slice] = np.where(
        allowed, battery.max_discharge_kwh_per_hour, 0.0
    )
    upper_bounds[curtailment_slice] = 1.0
    upper_bounds[soc_slice] = battery.capacity_kwh
    lower_bounds[soc_slice.start] = battery.initial_soc_kwh
    upper_bounds[soc_slice.start] = battery.initial_soc_kwh
    upper_bounds[mode_slice] = 1.0
    if with_cvar:
        assert eta_index is not None
        lower_bounds[eta_index] = -np.inf

    equality = lil_matrix((hours + 1, variable_count), dtype=float)
    equality[0, load_slice] = 1.0
    equality_rhs = np.zeros(hours + 1, dtype=float)
    equality_rhs[0] = required_load_kwh
    for hour_index in range(hours):
        row = hour_index + 1
        equality[row, charge_slice.start + hour_index] = -battery.charge_efficiency
        equality[row, discharge_slice.start + hour_index] = (
            1.0 / battery.discharge_efficiency
        )
        equality[row, soc_slice.start + hour_index] = -1.0
        equality[row, soc_slice.start + hour_index + 1] = 1.0

    mode_constraints = lil_matrix((2 * hours, variable_count), dtype=float)
    mode_upper = np.zeros(2 * hours, dtype=float)
    for hour_index in range(hours):
        mode_column = mode_slice.start + hour_index
        mode_constraints[2 * hour_index, charge_slice.start + hour_index] = 1.0
        mode_constraints[2 * hour_index, mode_column] = (
            -battery.max_charge_kwh_per_hour
        )
        mode_constraints[2 * hour_index + 1, discharge_slice.start + hour_index] = 1.0
        mode_constraints[2 * hour_index + 1, mode_column] = (
            battery.max_discharge_kwh_per_hour
        )
        mode_upper[2 * hour_index + 1] = battery.max_discharge_kwh_per_hour

    flow_constraints = lil_matrix((2 * scenario_flow_count, variable_count), dtype=float)
    flow_upper = np.zeros(2 * scenario_flow_count, dtype=float)
    for scenario_index in range(scenarios):
        for hour_index in range(hours):
            flat_index = scenario_index * hours + hour_index
            import_row = 2 * flat_index
            export_row = import_row + 1
            import_column = import_slice.start + flat_index
            export_column = export_slice.start + flat_index
            scenario_pv = pv[scenario_index, hour_index]

            # g_s = L + C - D - PV_s + u*PV_s; import >= g_s.
            flow_constraints[import_row, load_slice.start + hour_index] = 1.0
            flow_constraints[import_row, charge_slice.start + hour_index] = 1.0
            flow_constraints[import_row, discharge_slice.start + hour_index] = -1.0
            flow_constraints[import_row, curtailment_slice.start + hour_index] = scenario_pv
            flow_constraints[import_row, import_column] = -1.0
            flow_upper[import_row] = scenario_pv

            # export >= -g_s.
            flow_constraints[export_row, load_slice.start + hour_index] = -1.0
            flow_constraints[export_row, charge_slice.start + hour_index] = -1.0
            flow_constraints[export_row, discharge_slice.start + hour_index] = 1.0
            flow_constraints[export_row, curtailment_slice.start + hour_index] = -scenario_pv
            flow_constraints[export_row, export_column] = -1.0
            flow_upper[export_row] = -scenario_pv

    inequality = vstack(
        (mode_constraints.tocsr(), flow_constraints.tocsr()),
        format="csr",
    )
    inequality_upper = np.concatenate((mode_upper, flow_upper))
    if with_cvar:
        assert eta_index is not None
        tail = lil_matrix((scenarios, variable_count), dtype=float)
        tail[:, :physical_variables] = scenario_coefficients
        tail[:, eta_index] = -1.0
        for scenario_index in range(scenarios):
            first = scenario_index * hours
            last = first + hours
            tail[
                scenario_index,
                import_slice.start + first : import_slice.start + last,
            ] = grid_tariff.daytime_variable_import_eur_per_kwh
            tail[
                scenario_index,
                export_slice.start + first : export_slice.start + last,
            ] = grid_tariff.export_fee_eur_per_kwh
            tail[scenario_index, excess_slice.start + scenario_index] = -1.0
        inequality = vstack((inequality, tail.tocsr()), format="csr")
        inequality_upper = np.concatenate(
            (inequality_upper, -scenario_constants)
        )

    integrality = np.zeros(variable_count, dtype=int)
    integrality[mode_slice] = 1
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=(
            LinearConstraint(equality.tocsr(), equality_rhs, equality_rhs),
            LinearConstraint(
                inequality,
                np.full(len(inequality_upper), -np.inf, dtype=float),
                inequality_upper,
            ),
        ),
        options={"disp": False},
    )
    if not result.success or result.x is None:
        raise ValueError(f"Spread stochastic MILP did not solve: {result.message}")

    load = np.asarray(result.x[load_slice], dtype=float)
    charge = np.asarray(result.x[charge_slice], dtype=float)
    discharge = np.asarray(result.x[discharge_slice], dtype=float)
    curtailment_fraction = np.clip(
        np.asarray(result.x[curtailment_slice], dtype=float), 0.0, 1.0
    )
    soc = np.clip(
        np.asarray(result.x[soc_slice], dtype=float),
        0.0,
        battery.capacity_kwh,
    )
    mode = np.rint(np.asarray(result.x[mode_slice], dtype=float)).astype(int)
    if np.any((charge > 1e-7) & (discharge > 1e-7)):
        raise AssertionError("MILP returned simultaneous charge and discharge")

    scenario_import = np.asarray(result.x[import_slice], dtype=float).reshape(
        scenarios, hours
    )
    scenario_export = np.asarray(result.x[export_slice], dtype=float).reshape(
        scenarios, hours
    )
    physical = np.asarray(result.x[:physical_variables], dtype=float)
    scenario_cost = (
        scenario_coefficients @ physical
        + scenario_constants
        + grid_tariff.daytime_variable_import_eur_per_kwh
        * scenario_import.sum(axis=1)
        + grid_tariff.export_fee_eur_per_kwh * scenario_export.sum(axis=1)
    )
    expected_cost = float(scenario_cost.mean())
    cvar_cost = _empirical_upper_cvar(scenario_cost, cvar_alpha)
    point_curtailment = curtailment_fraction * point_pv
    position = (
        load
        + charge
        - discharge
        - point_pv
        + point_curtailment
    )
    return DynamicSpreadStochasticPlan(
        load_kwh=load,
        charge_kwh=charge,
        discharge_kwh=discharge,
        curtailment_fraction=curtailment_fraction,
        point_curtailment_kwh=point_curtailment,
        soc_kwh=soc,
        charge_mode=mode,
        day_ahead_position_kwh=position,
        scenario_cost_eur=scenario_cost,
        expected_cost_eur=expected_cost,
        cvar_cost_eur=cvar_cost,
        objective_eur=expected_cost + risk_weight * cvar_cost,
        cvar_alpha=cvar_alpha,
        risk_weight=risk_weight,
    )


def _empirical_upper_cvar(values: np.ndarray, alpha: float) -> float:
    losses = np.sort(np.asarray(values, dtype=float))
    candidates = np.unique(losses)
    objectives = [
        threshold
        + np.maximum(losses - threshold, 0.0).mean() / (1.0 - alpha)
        for threshold in candidates
    ]
    return float(np.min(objectives))


def _scenario_matrix(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or result.shape[0] < 1 or result.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty [scenarios, hours] matrix")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values")
    return result


def _vector(values: np.ndarray, name: str, hours: int) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (hours,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {hours} finite hourly values")
    return result
