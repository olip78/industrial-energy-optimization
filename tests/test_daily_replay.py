from __future__ import annotations

from datetime import date
import unittest

import numpy as np
import pandas as pd

from energy.optimization import (
    DayReplayInputs,
    ReferenceScenario,
    fit_rule_based_price_shape,
    replay_deterministic_day,
    replay_day_ahead_only_day,
    replay_oracle_day,
    replay_rule_based_day,
)


class DailyReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scenario = ReferenceScenario(
            daily_load_kwh=10.0,
            working_start_hour=0,
            working_end_hour=2,
            load_min_kwh_per_hour=0.0,
            load_max_kwh_per_hour=10.0,
            pv_profile_scale=1.0,
            battery_available_energy_kwh=5.0,
            battery_max_discharge_kwh_per_hour=5.0,
            charge_efficiency=1.0,
            discharge_efficiency=1.0,
            degradation_eur_per_kwh=0.01,
        )
        zeros = np.zeros(24)
        da_price = np.full(24, 30.0)
        id_price = np.full(24, 30.0)
        # In the first MPC decision the new ID forecast makes hour 0 valuable.
        id_forecast = np.tile(da_price, (24, 1))
        id_forecast[0, 0] = 100.0
        cls.inputs = DayReplayInputs(
            delivery_day=date(2025, 1, 6),
            day_ahead_pv_forecast_kwh=zeros,
            day_ahead_price_forecast_eur_per_mwh=da_price,
            actual_pv_kwh=zeros,
            actual_day_ahead_price_eur_per_mwh=da_price,
            actual_intraday_price_eur_per_mwh=id_price,
            mpc_pv_forecast_kwh=np.tile(zeros, (24, 1)),
            mpc_intraday_price_forecast_eur_per_mwh=id_forecast,
            night_reference_price_eur_per_mwh=10.0,
        )

    def test_deterministic_replay_executes_only_first_action_of_each_mpc_plan(self) -> None:
        replay = replay_deterministic_day(inputs=self.inputs, scenario=self.scenario)
        self.assertEqual(len(replay.mpc_plans), 24)
        self.assertAlmostEqual(replay.executed_load_kwh.sum(), 10.0)
        self.assertAlmostEqual(replay.executed_discharge_kwh.sum(), 5.0)
        self.assertAlmostEqual(replay.executed_load_kwh[0], 0.0)
        self.assertAlmostEqual(replay.executed_load_kwh[1], 10.0)
        self.assertAlmostEqual(replay.executed_discharge_kwh[0], 5.0)
        self.assertEqual(replay.ledger.actual_deviation_kwh.shape, (24,))

    def test_day_ahead_only_replay_executes_its_initial_plan(self) -> None:
        replay = replay_day_ahead_only_day(inputs=self.inputs, scenario=self.scenario)
        mpc_replay = replay_deterministic_day(inputs=self.inputs, scenario=self.scenario)
        np.testing.assert_allclose(
            replay.day_ahead_plan.day_ahead_position_kwh,
            mpc_replay.day_ahead_plan.day_ahead_position_kwh,
        )
        np.testing.assert_allclose(
            replay.ledger.actual_deviation_kwh,
            -self.inputs.actual_pv_kwh,
        )
        self.assertAlmostEqual(replay.day_ahead_plan.load_kwh.sum(), 10.0)

    def test_rule_based_replay_uses_the_same_ledger(self) -> None:
        history = pd.DataFrame(
            {
                "delivery_date_local": [date(2024, 1, 2)] * 24,
                "hour_local": list(range(24)),
                "day_ahead_price_eur_per_mwh": list(range(24)),
            }
        )
        price_shape = fit_rule_based_price_shape(history)
        replay = replay_rule_based_day(
            inputs=self.inputs,
            scenario=self.scenario,
            price_shape=price_shape,
        )
        self.assertAlmostEqual(replay.rule_plan.load_kwh.sum(), 10.0)
        np.testing.assert_allclose(
            replay.rule_plan.day_ahead_position_kwh,
            replay.rule_plan.load_kwh,
        )
        self.assertTrue(np.isfinite(replay.ledger.total_cost_eur))

    def test_oracle_replay_exposes_a_feasible_perfect_information_plan(self) -> None:
        replay = replay_oracle_day(inputs=self.inputs, scenario=self.scenario)
        self.assertAlmostEqual(replay.oracle_plan.nominated_load_kwh.sum(), 10.0)
        self.assertAlmostEqual(replay.oracle_plan.actual_load_kwh.sum(), 10.0)
        self.assertLessEqual(replay.oracle_plan.actual_discharge_kwh.sum(), 5.0)
        np.testing.assert_allclose(
            replay.oracle_plan.nominated_load_kwh,
            replay.oracle_plan.actual_load_kwh,
        )
        np.testing.assert_allclose(replay.ledger.actual_deviation_kwh, 0.0)
        self.assertAlmostEqual(replay.ledger.intraday_deviation_cost_eur, 0.0)
        self.assertTrue(np.isfinite(replay.ledger.total_cost_eur))


if __name__ == "__main__":
    unittest.main()
