from __future__ import annotations

from datetime import date
import unittest

import numpy as np
import pandas as pd

from energy.optimization import (
    ReferenceScenario,
    build_rule_based_plan,
    fit_rule_based_price_shape,
)


class ReferenceScenarioAndRuleBasedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scenario = ReferenceScenario()
        hours = list(cls.scenario.working_hours)
        # A frozen winter workday curve: 06:00 is cheapest and 21:00 costliest.
        cls.history = pd.DataFrame(
            {
                "delivery_date_local": [date(2024, 1, 2)] * len(hours),
                "hour_local": hours,
                "day_ahead_price_eur_per_mwh": [100.0 + 10.0 * rank for rank in range(len(hours))],
            }
        )

    def test_reference_load_bounds_and_pv_scaling(self) -> None:
        lower, upper = self.scenario.load_bounds()
        self.assertEqual(lower.sum(), 80.0)
        self.assertEqual(upper.sum(), 320.0)
        self.assertEqual(self.scenario.daily_load_kwh, 200.0)
        np.testing.assert_allclose(self.scenario.scale_pv_energy([0.0, 0.5]), [0.0, 10.0])

    def test_rule_based_plan_is_feasible_and_inverse_to_price_rank(self) -> None:
        shape = fit_rule_based_price_shape(self.history)
        plan = build_rule_based_plan(
            delivery_day=date(2025, 1, 6),  # winter workday
            price_shape=shape,
            scenario=self.scenario,
            night_reference_price_eur_per_mwh=10.0,
        )
        lower, upper = self.scenario.load_bounds()
        self.assertAlmostEqual(plan.load_kwh.sum(), 200.0)
        self.assertTrue(np.all(plan.load_kwh >= lower))
        self.assertTrue(np.all(plan.load_kwh <= upper))
        self.assertEqual(plan.load_kwh[6], 20.0)
        self.assertEqual(plan.load_kwh[21], 5.0)
        self.assertTrue(np.all(np.diff(plan.load_kwh[6:22]) <= 0.0))
        np.testing.assert_allclose(plan.day_ahead_position_kwh, plan.load_kwh)
        self.assertLessEqual(plan.discharge_kwh.sum(), 44.16)
        self.assertTrue(np.all(plan.discharge_kwh <= 5.0))
        self.assertEqual(plan.discharge_kwh[21], 5.0)

    def test_rule_based_battery_does_not_cycle_when_typical_value_is_too_low(self) -> None:
        low_history = self.history.assign(day_ahead_price_eur_per_mwh=10.0)
        shape = fit_rule_based_price_shape(low_history)
        plan = build_rule_based_plan(
            delivery_day=date(2025, 1, 6),
            price_shape=shape,
            scenario=self.scenario,
            night_reference_price_eur_per_mwh=100.0,
        )
        np.testing.assert_allclose(plan.discharge_kwh, np.zeros(24))


if __name__ == "__main__":
    unittest.main()
