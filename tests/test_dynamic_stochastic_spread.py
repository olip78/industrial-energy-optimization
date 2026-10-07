from __future__ import annotations

import numpy as np

from energy.optimization.dynamic_battery import (
    DynamicBatteryParameters,
    evaluate_dynamic_actual_day,
)
from energy.optimization.dynamic_stochastic_spread import (
    solve_dynamic_stochastic_spread_day_ahead,
)
from energy.optimization.grid_tariff import GridTariffParameters


def _battery(capacity: float = 2.0) -> DynamicBatteryParameters:
    return DynamicBatteryParameters(
        capacity_kwh=capacity,
        initial_soc_kwh=capacity,
        max_charge_kwh_per_hour=2.0,
        max_discharge_kwh_per_hour=2.0,
        night_reference_price_eur_per_mwh=40.0,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        degradation_eur_per_kwh=0.02,
    )


def test_single_scenario_cost_matches_factual_v4_ledger() -> None:
    tariff = GridTariffParameters(
        daytime_network_eur_per_kwh=0.03,
        nighttime_network_eur_per_kwh=0.01,
        levies_eur_per_kwh=0.02,
    )
    actual_pv = np.asarray([[0.5, 3.0]])
    day_ahead_price = np.asarray([[80.0, -20.0]])
    spread = np.asarray([[10.0, 30.0]])
    point_pv = np.asarray([1.0, 2.0])
    battery = _battery()
    plan = solve_dynamic_stochastic_spread_day_ahead(
        pv_scenarios_kwh=actual_pv,
        day_ahead_price_scenarios_eur_per_mwh=day_ahead_price,
        intraday_spread_scenarios_eur_per_mwh=spread,
        point_pv_forecast_kwh=point_pv,
        load_min_kwh=np.asarray([1.0, 1.0]),
        load_max_kwh=np.asarray([3.0, 3.0]),
        required_load_kwh=4.0,
        battery=battery,
        discharge_allowed=np.ones(2, dtype=bool),
        grid_tariff=tariff,
        risk_weight=0.0,
    )
    ledger = evaluate_dynamic_actual_day(
        day_ahead_position_kwh=plan.day_ahead_position_kwh,
        day_ahead_price_eur_per_mwh=day_ahead_price[0],
        actual_load_kwh=plan.load_kwh,
        actual_charge_kwh=plan.charge_kwh,
        actual_pv_kwh=actual_pv[0],
        actual_discharge_kwh=plan.discharge_kwh,
        actual_intraday_price_eur_per_mwh=day_ahead_price[0] + spread[0],
        battery=battery,
        actual_curtailment_kwh=plan.curtailment_fraction * actual_pv[0],
        grid_tariff=tariff,
    )

    np.testing.assert_allclose(plan.scenario_cost_eur, [ledger.total_cost_eur], atol=1e-7)
    assert not np.any(
        (plan.charge_kwh > 1e-8) & (plan.discharge_kwh > 1e-8)
    )


def test_negative_sale_price_selects_common_pv_curtailment() -> None:
    plan = solve_dynamic_stochastic_spread_day_ahead(
        pv_scenarios_kwh=np.asarray([[5.0]]),
        day_ahead_price_scenarios_eur_per_mwh=np.asarray([[-100.0]]),
        intraday_spread_scenarios_eur_per_mwh=np.asarray([[0.0]]),
        point_pv_forecast_kwh=np.asarray([5.0]),
        load_min_kwh=np.asarray([0.0]),
        load_max_kwh=np.asarray([0.0]),
        required_load_kwh=0.0,
        battery=_battery(capacity=0.0),
        discharge_allowed=np.asarray([False]),
        risk_weight=0.0,
    )

    np.testing.assert_allclose(plan.curtailment_fraction, [1.0], atol=1e-8)
    np.testing.assert_allclose(plan.day_ahead_position_kwh, [0.0], atol=1e-8)
