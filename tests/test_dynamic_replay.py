from __future__ import annotations

from datetime import date
import unittest

import numpy as np
import pandas as pd

from energy.optimization.dynamic_replay import (
    replay_dynamic_deterministic_day,
    replay_dynamic_rule_based_day,
)
from energy.optimization.replay import DayReplayInputs
from energy.optimization.rule_based import fit_rule_based_price_shape
from energy.optimization.scenario import ReferenceScenario


def _inputs(
    *,
    day_ahead_price: np.ndarray,
    actual_pv: np.ndarray,
    mpc_price: np.ndarray,
) -> DayReplayInputs:
    zeros = np.zeros(24, dtype=float)
    return DayReplayInputs(
        delivery_day=date(2025, 1, 6),
        day_ahead_pv_forecast_kwh=zeros.copy(),
        day_ahead_price_forecast_eur_per_mwh=day_ahead_price,
        actual_pv_kwh=actual_pv,
        actual_day_ahead_price_eur_per_mwh=day_ahead_price,
        actual_intraday_price_eur_per_mwh=day_ahead_price,
        mpc_pv_forecast_kwh=np.tile(actual_pv, (24, 1)),
        mpc_intraday_price_forecast_eur_per_mwh=mpc_price,
        night_reference_price_eur_per_mwh=0.0,
    )


class DynamicReplayTest(unittest.TestCase):
    def test_mpc_reoptimizes_battery_while_day_ahead_position_stays_fixed(self) -> None:
        scenario = ReferenceScenario(
            daily_load_kwh=10.0,
            working_start_hour=0,
            working_end_hour=2,
            load_min_kwh_per_hour=5.0,
            load_max_kwh_per_hour=5.0,
            pv_profile_scale=1.0,
            battery_available_energy_kwh=5.0,
            battery_max_discharge_kwh_per_hour=5.0,
            charge_efficiency=1.0,
            discharge_efficiency=1.0,
            degradation_eur_per_kwh=0.0,
        )
        day_ahead_price = np.zeros(24, dtype=float)
        day_ahead_price[:2] = [100.0, 0.0]
        mpc_price = np.zeros((24, 24), dtype=float)
        mpc_price[0, :2] = [0.0, 100.0]
        mpc_price[1, 1] = 100.0
        replay = replay_dynamic_deterministic_day(
            inputs=_inputs(
                day_ahead_price=day_ahead_price,
                actual_pv=np.zeros(24, dtype=float),
                mpc_price=mpc_price,
            ),
            scenario=scenario,
        )

        np.testing.assert_allclose(
            replay.day_ahead_plan.discharge_kwh,
            [5.0, 0.0],
            atol=1e-8,
        )
        np.testing.assert_allclose(
            replay.executed_discharge_kwh,
            [0.0, 5.0],
            atol=1e-8,
        )
        np.testing.assert_allclose(
            replay.day_ahead_plan.net_position_kwh,
            [0.0, 5.0],
            atol=1e-8,
        )

    def test_rule_charges_only_from_pv_and_curtails_negative_price_surplus(self) -> None:
        scenario = ReferenceScenario(
            daily_load_kwh=10.0,
            working_start_hour=0,
            working_end_hour=2,
            load_min_kwh_per_hour=5.0,
            load_max_kwh_per_hour=5.0,
            pv_profile_scale=1.0,
            battery_available_energy_kwh=0.0,
            battery_max_discharge_kwh_per_hour=0.0,
            charge_efficiency=1.0,
            discharge_efficiency=1.0,
            degradation_eur_per_kwh=0.0,
        )
        day_ahead_price = np.zeros(24, dtype=float)
        day_ahead_price[0] = -50.0
        actual_pv = np.zeros(24, dtype=float)
        actual_pv[0] = 5.0
        history = pd.DataFrame(
            {
                "delivery_date_local": [date(2024, 1, 2)] * 24,
                "hour_local": np.arange(24),
                "day_ahead_price_eur_per_mwh": np.arange(24),
            }
        )
        replay = replay_dynamic_rule_based_day(
            inputs=_inputs(
                day_ahead_price=day_ahead_price,
                actual_pv=actual_pv,
                mpc_price=np.tile(day_ahead_price, (24, 1)),
            ),
            scenario=scenario,
            price_shape=fit_rule_based_price_shape(history),
        )

        np.testing.assert_allclose(replay.rule_plan.charge_kwh, [0.0, 0.0])
        np.testing.assert_allclose(replay.rule_plan.curtailment_kwh, [5.0, 0.0])
        np.testing.assert_allclose(replay.ledger.actual_deviation_kwh, [0.0, 0.0])

    def test_rule_refills_discharge_headroom_from_later_pv(self) -> None:
        scenario = ReferenceScenario(
            daily_load_kwh=10.0,
            working_start_hour=0,
            working_end_hour=2,
            load_min_kwh_per_hour=5.0,
            load_max_kwh_per_hour=5.0,
            pv_profile_scale=1.0,
            battery_available_energy_kwh=5.0,
            battery_max_discharge_kwh_per_hour=5.0,
            charge_efficiency=1.0,
            discharge_efficiency=1.0,
            degradation_eur_per_kwh=0.0,
        )
        day_ahead_price = np.zeros(24, dtype=float)
        day_ahead_price[:2] = [100.0, 20.0]
        actual_pv = np.zeros(24, dtype=float)
        actual_pv[1] = 5.0
        history_price = np.zeros(24, dtype=float)
        history_price[:2] = [100.0, 20.0]
        history = pd.DataFrame(
            {
                "delivery_date_local": [date(2024, 1, 2)] * 24,
                "hour_local": np.arange(24),
                "day_ahead_price_eur_per_mwh": history_price,
            }
        )
        replay = replay_dynamic_rule_based_day(
            inputs=_inputs(
                day_ahead_price=day_ahead_price,
                actual_pv=actual_pv,
                mpc_price=np.tile(day_ahead_price, (24, 1)),
            ),
            scenario=scenario,
            price_shape=fit_rule_based_price_shape(history),
        )

        np.testing.assert_allclose(replay.rule_plan.discharge_kwh, [5.0, 0.0])
        np.testing.assert_allclose(replay.rule_plan.charge_kwh, [0.0, 5.0])
        np.testing.assert_allclose(replay.rule_plan.soc_kwh, [5.0, 0.0, 5.0])


if __name__ == "__main__":
    unittest.main()
