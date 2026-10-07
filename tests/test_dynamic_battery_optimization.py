from __future__ import annotations

import unittest

import numpy as np

from energy.optimization.dynamic_battery import (
    DynamicBatteryParameters,
    battery_reference_cost_eur_per_kwh,
    evaluate_dynamic_actual_day,
    solve_dynamic_schedule,
)
from energy.optimization.grid_tariff import GridTariffParameters


class DynamicBatteryOptimizationTest(unittest.TestCase):
    def battery(self, *, initial_soc: float, night_price: float = 10.0):
        return DynamicBatteryParameters(
            capacity_kwh=5.0,
            initial_soc_kwh=initial_soc,
            max_charge_kwh_per_hour=5.0,
            max_discharge_kwh_per_hour=5.0,
            night_reference_price_eur_per_mwh=night_price,
            charge_efficiency=1.0,
            discharge_efficiency=1.0,
            degradation_eur_per_kwh=0.01,
        )

    def test_negative_price_charges_then_expensive_hour_discharges(self) -> None:
        plan = solve_dynamic_schedule(
            pv_forecast_kwh=[0.0, 0.0],
            price_eur_per_mwh=[-50.0, 100.0],
            load_min_kwh=[0.0, 0.0],
            load_max_kwh=[0.0, 0.0],
            required_load_kwh=0.0,
            battery=self.battery(initial_soc=0.0),
            discharge_allowed=[True, True],
        )
        np.testing.assert_allclose(plan.charge_kwh, [5.0, 0.0])
        np.testing.assert_allclose(plan.discharge_kwh, [0.0, 5.0])
        np.testing.assert_allclose(plan.soc_kwh, [0.0, 5.0, 0.0])
        self.assertFalse(
            np.any((plan.charge_kwh > 1e-8) & (plan.discharge_kwh > 1e-8))
        )

    def test_full_battery_cannot_charge_until_capacity_is_freed(self) -> None:
        plan = solve_dynamic_schedule(
            pv_forecast_kwh=[0.0],
            price_eur_per_mwh=[-100.0],
            load_min_kwh=[0.0],
            load_max_kwh=[0.0],
            required_load_kwh=0.0,
            battery=self.battery(initial_soc=5.0),
            discharge_allowed=[True],
        )
        np.testing.assert_allclose(plan.charge_kwh, [0.0], atol=1e-8)
        np.testing.assert_allclose(plan.discharge_kwh, [0.0], atol=1e-8)
        np.testing.assert_allclose(plan.soc_kwh, [5.0, 5.0], atol=1e-8)

    def test_positive_price_can_charge_below_overnight_replacement_cost(self) -> None:
        plan = solve_dynamic_schedule(
            pv_forecast_kwh=[0.0],
            price_eur_per_mwh=[1.0],
            load_min_kwh=[0.0],
            load_max_kwh=[0.0],
            required_load_kwh=0.0,
            battery=self.battery(initial_soc=0.0),
            discharge_allowed=[True],
        )
        np.testing.assert_allclose(plan.charge_kwh, [5.0])
        np.testing.assert_allclose(plan.soc_kwh, [0.0, 5.0])

    def test_pv_surplus_can_charge_without_daytime_import_tariff(self) -> None:
        tariff = GridTariffParameters(daytime_network_eur_per_kwh=0.20)
        plan = solve_dynamic_schedule(
            pv_forecast_kwh=[5.0],
            price_eur_per_mwh=[50.0],
            load_min_kwh=[0.0],
            load_max_kwh=[0.0],
            required_load_kwh=0.0,
            battery=self.battery(initial_soc=0.0, night_price=100.0),
            discharge_allowed=[True],
            grid_tariff=tariff,
        )
        np.testing.assert_allclose(plan.charge_kwh, [5.0])
        np.testing.assert_allclose(plan.physical_grid_import_kwh, [0.0])
        np.testing.assert_allclose(plan.physical_grid_export_kwh, [0.0])

    def test_negative_night_reference_is_floored_at_zero(self) -> None:
        battery = self.battery(initial_soc=5.0, night_price=-100.0)
        self.assertAlmostEqual(
            battery_reference_cost_eur_per_kwh(battery),
            battery.degradation_eur_per_kwh,
        )

    def test_ledger_includes_charge_in_the_market_balance(self) -> None:
        battery = self.battery(initial_soc=0.0, night_price=-50.0)
        ledger = evaluate_dynamic_actual_day(
            day_ahead_position_kwh=[0.0, 0.0],
            day_ahead_price_eur_per_mwh=[0.0, 0.0],
            actual_load_kwh=[0.0, 0.0],
            actual_charge_kwh=[5.0, 0.0],
            actual_pv_kwh=[0.0, 0.0],
            actual_discharge_kwh=[0.0, 5.0],
            actual_intraday_price_eur_per_mwh=[-50.0, 100.0],
            battery=battery,
        )
        self.assertAlmostEqual(ledger.intraday_deviation_cost_eur, -0.75)
        self.assertAlmostEqual(ledger.battery_reference_cost_eur, 0.05)
        self.assertAlmostEqual(ledger.total_cost_eur, -0.70)

    def test_grid_charge_applies_to_import_but_not_export(self) -> None:
        tariff = GridTariffParameters(
            daytime_network_eur_per_kwh=0.05,
            levies_eur_per_kwh=0.02,
            concession_fee_eur_per_kwh=0.01,
            electricity_tax_eur_per_kwh=0.005,
            annual_network_base_eur=36.5,
            billing_days_per_year=365,
        )
        ledger = evaluate_dynamic_actual_day(
            day_ahead_position_kwh=[5.0, -5.0],
            day_ahead_price_eur_per_mwh=[0.0, 0.0],
            actual_load_kwh=[5.0, 0.0],
            actual_charge_kwh=[0.0, 0.0],
            actual_pv_kwh=[0.0, 5.0],
            actual_discharge_kwh=[0.0, 0.0],
            actual_intraday_price_eur_per_mwh=[0.0, 0.0],
            battery=self.battery(initial_soc=5.0),
            grid_tariff=tariff,
        )
        np.testing.assert_allclose(ledger.physical_grid_import_kwh, [5.0, 0.0])
        np.testing.assert_allclose(ledger.physical_grid_export_kwh, [0.0, 5.0])
        self.assertAlmostEqual(ledger.daytime_network_cost_eur, 0.25)
        self.assertAlmostEqual(ledger.levies_cost_eur, 0.10)
        self.assertAlmostEqual(ledger.concession_fee_cost_eur, 0.05)
        self.assertAlmostEqual(ledger.electricity_tax_cost_eur, 0.025)
        self.assertAlmostEqual(ledger.total_cost_eur, 0.425)

    def test_negative_price_curtails_pv_instead_of_exporting_it(self) -> None:
        plan = solve_dynamic_schedule(
            pv_forecast_kwh=[5.0],
            price_eur_per_mwh=[-100.0],
            load_min_kwh=[0.0],
            load_max_kwh=[0.0],
            required_load_kwh=0.0,
            battery=self.battery(initial_soc=5.0),
            discharge_allowed=[True],
        )
        np.testing.assert_allclose(plan.curtailment_kwh, [5.0], atol=1e-8)
        np.testing.assert_allclose(plan.net_position_kwh, [0.0], atol=1e-8)

    def test_positive_price_exports_pv_instead_of_curtailing_it(self) -> None:
        plan = solve_dynamic_schedule(
            pv_forecast_kwh=[5.0],
            price_eur_per_mwh=[100.0],
            load_min_kwh=[0.0],
            load_max_kwh=[0.0],
            required_load_kwh=0.0,
            battery=self.battery(initial_soc=5.0),
            discharge_allowed=[False],
        )
        np.testing.assert_allclose(plan.curtailment_kwh, [0.0], atol=1e-8)
        np.testing.assert_allclose(plan.net_position_kwh, [-5.0], atol=1e-8)

    def test_ledger_applies_actual_curtailment_to_physical_balance(self) -> None:
        ledger = evaluate_dynamic_actual_day(
            day_ahead_position_kwh=[0.0],
            day_ahead_price_eur_per_mwh=[0.0],
            actual_load_kwh=[0.0],
            actual_charge_kwh=[0.0],
            actual_pv_kwh=[5.0],
            actual_discharge_kwh=[0.0],
            actual_intraday_price_eur_per_mwh=[-100.0],
            battery=self.battery(initial_soc=5.0),
            actual_curtailment_kwh=[5.0],
        )
        np.testing.assert_allclose(ledger.actual_used_pv_kwh, [0.0])
        np.testing.assert_allclose(ledger.actual_deviation_kwh, [0.0])
        self.assertAlmostEqual(ledger.total_cost_eur, 0.0)

    def test_ledger_values_only_missing_terminal_energy_plus_degradation(self) -> None:
        tariff = GridTariffParameters(nighttime_network_eur_per_kwh=0.02)
        ledger = evaluate_dynamic_actual_day(
            day_ahead_position_kwh=[-5.0],
            day_ahead_price_eur_per_mwh=[0.0],
            actual_load_kwh=[0.0],
            actual_charge_kwh=[0.0],
            actual_pv_kwh=[0.0],
            actual_discharge_kwh=[5.0],
            actual_intraday_price_eur_per_mwh=[0.0],
            battery=self.battery(initial_soc=5.0, night_price=50.0),
            grid_tariff=tariff,
        )
        self.assertAlmostEqual(ledger.terminal_soc_kwh, 0.0)
        self.assertAlmostEqual(ledger.terminal_recharge_energy_kwh, 5.0)
        self.assertAlmostEqual(ledger.battery_night_energy_cost_eur, 0.25)
        self.assertAlmostEqual(ledger.battery_night_grid_cost_eur, 0.10)
        self.assertAlmostEqual(ledger.battery_degradation_cost_eur, 0.05)
        self.assertAlmostEqual(ledger.battery_reference_cost_eur, 0.40)

    def test_fixed_battery_actions_are_preserved(self) -> None:
        plan = solve_dynamic_schedule(
            pv_forecast_kwh=[0.0, 0.0],
            price_eur_per_mwh=[-100.0, 500.0],
            load_min_kwh=[0.0, 0.0],
            load_max_kwh=[0.0, 0.0],
            required_load_kwh=0.0,
            battery=self.battery(initial_soc=5.0, night_price=0.0),
            discharge_allowed=[True, True],
            fixed_charge_kwh=[0.0, 0.0],
            fixed_discharge_kwh=[0.0, 0.0],
        )
        np.testing.assert_allclose(plan.charge_kwh, [0.0, 0.0])
        np.testing.assert_allclose(plan.discharge_kwh, [0.0, 0.0])

    def test_grid_tariff_changes_dispatch_at_a_negative_wholesale_price(self) -> None:
        tariff = GridTariffParameters(
            daytime_network_eur_per_kwh=0.08,
        )
        plan = solve_dynamic_schedule(
            pv_forecast_kwh=[0.0, 0.0],
            price_eur_per_mwh=[-50.0, 20.0],
            load_min_kwh=[0.0, 0.0],
            load_max_kwh=[0.0, 0.0],
            required_load_kwh=0.0,
            battery=self.battery(initial_soc=0.0, night_price=0.0),
            discharge_allowed=[True, True],
            grid_tariff=tariff,
        )
        np.testing.assert_allclose(plan.charge_kwh, [0.0, 0.0], atol=1e-8)
        np.testing.assert_allclose(plan.discharge_kwh, [0.0, 0.0], atol=1e-8)

    def test_solver_tolerance_above_capacity_is_clipped(self) -> None:
        plan = solve_dynamic_schedule(
            pv_forecast_kwh=[0.0],
            price_eur_per_mwh=[0.0],
            load_min_kwh=[0.0],
            load_max_kwh=[0.0],
            required_load_kwh=0.0,
            battery=self.battery(initial_soc=5.0),
            discharge_allowed=[False],
            initial_soc_kwh=5.0 + 1e-10,
        )
        self.assertAlmostEqual(plan.soc_kwh[0], 5.0)
        self.assertLessEqual(plan.soc_kwh.max(), 5.0)


if __name__ == "__main__":
    unittest.main()
