"""Scenario-based day-ahead scheduling with an optional CVaR penalty."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack, lil_matrix, vstack

from energy.optimization.deterministic import (
    MWH_TO_KWH,
    BatteryParameters,
    battery_full_cost_eur_per_kwh,
)


@dataclass(frozen=True)
class StochasticDayAheadPlan:
    """One executable schedule optimized against a set of joint scenarios."""

    load_kwh: np.ndarray
    discharge_kwh: np.ndarray
    day_ahead_position_kwh: np.ndarray
    scenario_cost_eur: np.ndarray
    expected_cost_eur: float
    cvar_cost_eur: float
    objective_eur: float
    cvar_alpha: float
    risk_weight: float


def solve_stochastic_day_ahead(
    *,
    pv_scenarios_kwh: np.ndarray,
    price_scenarios_eur_per_mwh: np.ndarray,
    point_pv_forecast_kwh: np.ndarray,
    load_min_kwh: np.ndarray,
    load_max_kwh: np.ndarray,
    required_load_kwh: float,
    battery: BatteryParameters,
    discharge_allowed: np.ndarray,
    cvar_alpha: float = 0.95,
    risk_weight: float = 0.0,
) -> StochasticDayAheadPlan:
    """Minimize scenario mean cost plus ``risk_weight * CVaR``.

    The load and discharge vectors are common to every scenario.  In this
    first stochastic version each scenario's day-ahead price is also the
    point proxy for its intraday settlement price.  This is the same simplifying
    market assumption used by the deterministic day-ahead formulation before
    factual intraday prices become available.

    The submitted position remains physically interpretable and is derived
    from the shared schedule and the point PV forecast.  Scenario PV errors are
    settled inside each scenario objective, while the eventual backtest uses
    the factual intraday price in the common economic ledger.
    """

    pv = _scenario_matrix(pv_scenarios_kwh, "pv_scenarios_kwh")
    price = _scenario_matrix(
        price_scenarios_eur_per_mwh, "price_scenarios_eur_per_mwh"
    )
    if pv.shape != price.shape:
        raise ValueError("PV and price scenario matrices must have the same shape")
    scenarios, hours = pv.shape
    point_pv = _vector(point_pv_forecast_kwh, "point_pv_forecast_kwh", hours)
    lower = _vector(load_min_kwh, "load_min_kwh", hours)
    upper = _vector(load_max_kwh, "load_max_kwh", hours)
    allowed = np.asarray(discharge_allowed, dtype=bool)
    if allowed.shape != (hours,):
        raise ValueError("discharge_allowed must have one value per hour")
    if (lower < 0).any() or (upper < lower).any():
        raise ValueError("load bounds must satisfy 0 <= lower <= upper")
    if not lower.sum() <= required_load_kwh <= upper.sum():
        raise ValueError("required_load_kwh is infeasible under the load bounds")
    if not 0 < cvar_alpha < 1:
        raise ValueError("cvar_alpha must lie strictly between zero and one")
    if risk_weight < 0 or not np.isfinite(risk_weight):
        raise ValueError("risk_weight must be finite and non-negative")

    price_kwh = price / MWH_TO_KWH
    battery_cost = battery_full_cost_eur_per_kwh(battery)
    # Variables shared by all scenarios: [load_1:H, discharge_1:H].
    scenario_coefficients = np.concatenate(
        (price_kwh, battery_cost - price_kwh), axis=1
    )
    scenario_constants = -np.sum(price_kwh * pv, axis=1)
    base_objective = scenario_coefficients.mean(axis=0)

    equality = lil_matrix((1, 2 * hours), dtype=float)
    equality[0, :hours] = 1.0
    discharge_constraint = lil_matrix((1, 2 * hours), dtype=float)
    discharge_constraint[0, hours:] = 1.0
    discharge_upper = np.where(
        allowed, battery.max_discharge_kwh_per_hour, 0.0
    )
    physical_bounds = [
        *zip(lower, upper, strict=True),
        *[(0.0, float(value)) for value in discharge_upper],
    ]

    if risk_weight == 0:
        objective = base_objective
        a_ub = discharge_constraint.tocsr()
        b_ub = np.array([battery.available_energy_kwh], dtype=float)
        a_eq = equality.tocsr()
        bounds = physical_bounds
    else:
        # Extra variables: unrestricted VaR threshold eta and non-negative
        # excess losses u_s.  cost_s(x) - eta - u_s <= 0.
        objective = np.concatenate(
            (
                base_objective,
                np.array([risk_weight]),
                np.full(
                    scenarios,
                    risk_weight / ((1.0 - cvar_alpha) * scenarios),
                ),
            )
        )
        eta_column = csr_matrix(-np.ones((scenarios, 1), dtype=float))
        tail = hstack((eta_column, -eye(scenarios, format="csr")), format="csr")
        cvar_constraints = hstack(
            (csr_matrix(scenario_coefficients), tail), format="csr"
        )
        physical_with_tail = hstack(
            (
                discharge_constraint.tocsr(),
                csr_matrix((1, 1 + scenarios)),
            ),
            format="csr",
        )
        a_ub = vstack((physical_with_tail, cvar_constraints), format="csr")
        b_ub = np.concatenate(
            (
                np.array([battery.available_energy_kwh]),
                -scenario_constants,
            )
        )
        a_eq = hstack(
            (equality.tocsr(), csr_matrix((1, 1 + scenarios))), format="csr"
        )
        bounds = [*physical_bounds, (None, None), *[(0.0, None)] * scenarios]

    result = linprog(
        c=objective,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=np.array([required_load_kwh], dtype=float),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise ValueError(f"Stochastic day-ahead LP did not solve: {result.message}")

    load = np.asarray(result.x[:hours], dtype=float)
    discharge = np.asarray(result.x[hours : 2 * hours], dtype=float)
    position = load - point_pv - discharge
    scenario_cost = scenario_coefficients @ np.concatenate((load, discharge))
    scenario_cost += scenario_constants
    expected_cost = float(scenario_cost.mean())
    cvar_cost = _empirical_upper_cvar(scenario_cost, cvar_alpha)
    objective_value = expected_cost + risk_weight * cvar_cost
    return StochasticDayAheadPlan(
        load_kwh=load,
        discharge_kwh=discharge,
        day_ahead_position_kwh=position,
        scenario_cost_eur=scenario_cost,
        expected_cost_eur=expected_cost,
        cvar_cost_eur=cvar_cost,
        objective_eur=objective_value,
        cvar_alpha=cvar_alpha,
        risk_weight=risk_weight,
    )


def _empirical_upper_cvar(values: np.ndarray, alpha: float) -> float:
    """Return the exact empirical CVaR represented by the LP objective."""

    losses = np.asarray(values, dtype=float)
    eta_candidates = np.unique(losses)
    objective = [
        eta + np.maximum(losses - eta, 0.0).mean() / (1.0 - alpha)
        for eta in eta_candidates
    ]
    return float(np.min(objective))


def _scenario_matrix(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or result.shape[0] < 1 or result.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty [scenarios, hours] matrix")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _vector(values: np.ndarray, name: str, hours: int) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (hours,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {hours} finite hourly values")
    return result
