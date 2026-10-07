from __future__ import annotations

import unittest

import numpy as np

from energy.optimization import BatteryParameters, solve_stochastic_day_ahead


class StochasticDayAheadOptimizationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.no_battery = BatteryParameters(
            available_energy_kwh=0.0,
            max_discharge_kwh_per_hour=0.0,
            night_reference_price_eur_per_mwh=0.0,
            charge_efficiency=1.0,
            discharge_efficiency=1.0,
            degradation_eur_per_kwh=0.0,
        )

    def test_returns_one_feasible_schedule_and_point_based_position(self) -> None:
        plan = solve_stochastic_day_ahead(
            pv_scenarios_kwh=np.array([[1.0, 2.0], [0.0, 4.0]]),
            price_scenarios_eur_per_mwh=np.array([[20.0, 100.0], [40.0, 80.0]]),
            point_pv_forecast_kwh=np.array([0.5, 3.0]),
            load_min_kwh=np.zeros(2),
            load_max_kwh=np.full(2, 10.0),
            required_load_kwh=10.0,
            battery=self.no_battery,
            discharge_allowed=np.zeros(2, dtype=bool),
        )
        self.assertAlmostEqual(float(plan.load_kwh.sum()), 10.0)
        np.testing.assert_allclose(plan.discharge_kwh, 0.0)
        np.testing.assert_allclose(
            plan.day_ahead_position_kwh,
            plan.load_kwh - np.array([0.5, 3.0]),
        )
        self.assertEqual(plan.scenario_cost_eur.shape, (2,))

    def test_identical_scenarios_reduce_to_the_point_schedule(self) -> None:
        price = np.tile(np.array([20.0, 100.0, 50.0]), (8, 1))
        plan = solve_stochastic_day_ahead(
            pv_scenarios_kwh=np.zeros((8, 3)),
            price_scenarios_eur_per_mwh=price,
            point_pv_forecast_kwh=np.zeros(3),
            load_min_kwh=np.zeros(3),
            load_max_kwh=np.full(3, 10.0),
            required_load_kwh=10.0,
            battery=self.no_battery,
            discharge_allowed=np.zeros(3, dtype=bool),
        )
        np.testing.assert_allclose(plan.load_kwh, [10.0, 0.0, 0.0])
        self.assertAlmostEqual(plan.expected_cost_eur, 0.2)

    def test_cvar_moves_load_away_from_a_rare_expensive_hour(self) -> None:
        prices = np.tile(np.array([10.0, 40.0]), (20, 1))
        prices[-1, 0] = 500.0
        common = dict(
            pv_scenarios_kwh=np.zeros((20, 2)),
            price_scenarios_eur_per_mwh=prices,
            point_pv_forecast_kwh=np.zeros(2),
            load_min_kwh=np.zeros(2),
            load_max_kwh=np.full(2, 10.0),
            required_load_kwh=10.0,
            battery=self.no_battery,
            discharge_allowed=np.zeros(2, dtype=bool),
            cvar_alpha=0.95,
        )
        risk_neutral = solve_stochastic_day_ahead(**common, risk_weight=0.0)
        risk_aware = solve_stochastic_day_ahead(**common, risk_weight=1.0)
        np.testing.assert_allclose(risk_neutral.load_kwh, [10.0, 0.0])
        np.testing.assert_allclose(risk_aware.load_kwh, [0.0, 10.0])
        self.assertLess(risk_aware.cvar_cost_eur, risk_neutral.cvar_cost_eur)


if __name__ == "__main__":
    unittest.main()
