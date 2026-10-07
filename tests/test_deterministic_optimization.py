from __future__ import annotations

import unittest

import numpy as np

from energy.optimization import (
    BatteryParameters,
    battery_full_cost_eur_per_kwh,
    evaluate_actual_day,
    solve_day_ahead,
    solve_mpc,
    solve_oracle_perfect_information,
)


class DeterministicOptimizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.battery = BatteryParameters(
            available_energy_kwh=5.0,
            max_discharge_kwh_per_hour=5.0,
            night_reference_price_eur_per_mwh=10.0,
            charge_efficiency=1.0,
            discharge_efficiency=1.0,
            degradation_eur_per_kwh=0.01,
        )

    def test_day_ahead_moves_flexible_load_and_battery_to_valuable_hours(self) -> None:
        plan = solve_day_ahead(
            pv_forecast_kwh=[0.0, 0.0, 0.0],
            day_ahead_price_eur_per_mwh=[20.0, 100.0, 50.0],
            load_min_kwh=[0.0, 0.0, 0.0],
            load_max_kwh=[10.0, 10.0, 10.0],
            required_load_kwh=10.0,
            battery=self.battery,
            discharge_allowed=[True, True, True],
        )
        np.testing.assert_allclose(plan.load_kwh, [10.0, 0.0, 0.0])
        np.testing.assert_allclose(plan.discharge_kwh, [0.0, 5.0, 0.0])
        np.testing.assert_allclose(plan.day_ahead_position_kwh, [10.0, -5.0, 0.0])

    def test_battery_is_not_discharged_when_market_price_is_below_full_cost(self) -> None:
        plan = solve_day_ahead(
            pv_forecast_kwh=[0.0, 0.0],
            day_ahead_price_eur_per_mwh=[15.0, 15.0],
            load_min_kwh=[0.0, 0.0],
            load_max_kwh=[5.0, 5.0],
            required_load_kwh=5.0,
            battery=self.battery,
            discharge_allowed=[True, True],
        )
        np.testing.assert_allclose(plan.discharge_kwh, [0.0, 0.0])
        self.assertAlmostEqual(battery_full_cost_eur_per_kwh(self.battery), 0.02)

    def test_mpc_uses_fixed_day_ahead_position_only_for_deviation_accounting(self) -> None:
        plan = solve_mpc(
            pv_forecast_kwh=[0.0, 0.0],
            intraday_price_eur_per_mwh=[20.0, 100.0],
            day_ahead_position_kwh=[4.0, 1.0],
            load_min_kwh=[0.0, 0.0],
            load_max_kwh=[5.0, 5.0],
            required_remaining_load_kwh=5.0,
            available_battery_energy_kwh=2.0,
            battery=self.battery,
            discharge_allowed=[True, True],
        )
        np.testing.assert_allclose(plan.load_kwh, [5.0, 0.0])
        np.testing.assert_allclose(plan.discharge_kwh, [0.0, 2.0])
        np.testing.assert_allclose(plan.predicted_deviation_kwh, [1.0, -3.0])

    def test_ledger_uses_eur_per_mwh_to_eur_per_kwh_conversion(self) -> None:
        ledger = evaluate_actual_day(
            day_ahead_position_kwh=[10.0, 0.0],
            day_ahead_price_eur_per_mwh=[50.0, 100.0],
            actual_load_kwh=[10.0, 2.0],
            actual_pv_kwh=[0.0, 0.0],
            actual_discharge_kwh=[0.0, 1.0],
            actual_intraday_price_eur_per_mwh=[60.0, 120.0],
            battery=self.battery,
        )
        self.assertAlmostEqual(ledger.day_ahead_cost_eur, 0.5)
        self.assertAlmostEqual(ledger.intraday_deviation_cost_eur, 0.12)
        self.assertAlmostEqual(ledger.battery_cost_eur, 0.02)
        self.assertAlmostEqual(ledger.total_cost_eur, 0.64)
        np.testing.assert_allclose(ledger.actual_deviation_kwh, [0.0, 1.0])

    def test_oracle_uses_one_day_ahead_schedule_without_intraday_arbitrage(self) -> None:
        oracle = solve_oracle_perfect_information(
            actual_pv_kwh=[0.0, 0.0],
            actual_day_ahead_price_eur_per_mwh=[100.0, 10.0],
            load_min_kwh=[0.0, 0.0],
            load_max_kwh=[10.0, 10.0],
            required_load_kwh=10.0,
            battery=BatteryParameters(
                available_energy_kwh=0.0,
                max_discharge_kwh_per_hour=0.0,
                night_reference_price_eur_per_mwh=10.0,
                charge_efficiency=1.0,
                discharge_efficiency=1.0,
                degradation_eur_per_kwh=0.0,
            ),
            discharge_allowed=[False, False],
        )
        np.testing.assert_allclose(oracle.nominated_load_kwh, [0.0, 10.0])
        np.testing.assert_allclose(oracle.actual_load_kwh, [0.0, 10.0])
        np.testing.assert_allclose(oracle.day_ahead_position_kwh, [0.0, 10.0])



if __name__ == "__main__":
    unittest.main()
