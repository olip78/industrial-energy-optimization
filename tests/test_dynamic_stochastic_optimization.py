from __future__ import annotations

import unittest

import numpy as np

from energy.optimization.dynamic_battery import DynamicBatteryParameters
from energy.optimization.dynamic_stochastic import (
    solve_dynamic_stochastic_day_ahead,
)
from energy.optimization.grid_tariff import GridTariffParameters


class DynamicStochasticOptimizationTest(unittest.TestCase):
    def test_common_schedule_charges_then_discharges_without_mixing_modes(self) -> None:
        battery = DynamicBatteryParameters(
            capacity_kwh=5.0,
            initial_soc_kwh=0.0,
            max_charge_kwh_per_hour=5.0,
            max_discharge_kwh_per_hour=5.0,
            night_reference_price_eur_per_mwh=0.0,
            charge_efficiency=1.0,
            discharge_efficiency=1.0,
            degradation_eur_per_kwh=0.0,
        )
        plan = solve_dynamic_stochastic_day_ahead(
            pv_scenarios_kwh=np.zeros((3, 2)),
            price_scenarios_eur_per_mwh=np.asarray(
                [[-50.0, 100.0], [-40.0, 120.0], [-60.0, 80.0]]
            ),
            point_pv_forecast_kwh=np.zeros(2),
            point_price_forecast_eur_per_mwh=np.asarray([-50.0, 100.0]),
            load_min_kwh=np.zeros(2),
            load_max_kwh=np.zeros(2),
            required_load_kwh=0.0,
            battery=battery,
            discharge_allowed=np.ones(2, dtype=bool),
            cvar_alpha=0.95,
            risk_weight=0.10,
        )
        np.testing.assert_allclose(plan.charge_kwh, [5.0, 0.0], atol=1e-7)
        np.testing.assert_allclose(plan.discharge_kwh, [0.0, 5.0], atol=1e-7)
        np.testing.assert_allclose(plan.soc_kwh, [0.0, 5.0, 0.0], atol=1e-7)
        self.assertFalse(
            np.any((plan.charge_kwh > 1e-8) & (plan.discharge_kwh > 1e-8))
        )
        np.testing.assert_allclose(plan.day_ahead_position_kwh, [5.0, -5.0])

    def test_scenario_import_tariff_can_reject_grid_charging(self) -> None:
        battery = DynamicBatteryParameters(
            capacity_kwh=5.0,
            initial_soc_kwh=0.0,
            max_charge_kwh_per_hour=5.0,
            max_discharge_kwh_per_hour=5.0,
            night_reference_price_eur_per_mwh=0.0,
            charge_efficiency=1.0,
            discharge_efficiency=1.0,
            degradation_eur_per_kwh=0.01,
        )
        tariff = GridTariffParameters(daytime_network_eur_per_kwh=0.08)
        plan = solve_dynamic_stochastic_day_ahead(
            pv_scenarios_kwh=np.zeros((2, 2)),
            price_scenarios_eur_per_mwh=np.asarray(
                [[-50.0, 20.0], [-40.0, 30.0]]
            ),
            point_pv_forecast_kwh=np.zeros(2),
            point_price_forecast_eur_per_mwh=np.asarray([-50.0, 20.0]),
            load_min_kwh=np.zeros(2),
            load_max_kwh=np.zeros(2),
            required_load_kwh=0.0,
            battery=battery,
            discharge_allowed=np.ones(2, dtype=bool),
            grid_tariff=tariff,
            cvar_alpha=0.95,
            risk_weight=0.10,
        )
        np.testing.assert_allclose(plan.charge_kwh, [0.0, 0.0], atol=1e-8)
        np.testing.assert_allclose(plan.discharge_kwh, [0.0, 0.0], atol=1e-8)

    def test_terminal_value_can_select_positive_price_grid_charge(self) -> None:
        battery = DynamicBatteryParameters(
            capacity_kwh=5.0,
            initial_soc_kwh=0.0,
            max_charge_kwh_per_hour=5.0,
            max_discharge_kwh_per_hour=5.0,
            night_reference_price_eur_per_mwh=100.0,
            charge_efficiency=1.0,
            discharge_efficiency=1.0,
            degradation_eur_per_kwh=0.0,
        )
        plan = solve_dynamic_stochastic_day_ahead(
            pv_scenarios_kwh=np.zeros((2, 1)),
            price_scenarios_eur_per_mwh=np.asarray([[20.0], [30.0]]),
            point_pv_forecast_kwh=np.zeros(1),
            point_price_forecast_eur_per_mwh=np.asarray([25.0]),
            load_min_kwh=np.zeros(1),
            load_max_kwh=np.zeros(1),
            required_load_kwh=0.0,
            battery=battery,
            discharge_allowed=np.ones(1, dtype=bool),
            risk_weight=0.0,
        )
        np.testing.assert_allclose(plan.charge_kwh, [5.0], atol=1e-8)
        np.testing.assert_allclose(plan.soc_kwh, [0.0, 5.0], atol=1e-8)


if __name__ == "__main__":
    unittest.main()
