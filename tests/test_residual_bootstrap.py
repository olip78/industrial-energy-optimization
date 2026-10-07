from __future__ import annotations

import unittest
from datetime import date

import numpy as np
import pandas as pd

from energy.uncertainty import JointResidualBootstrap


class JointResidualBootstrapTest(unittest.TestCase):
    def setUp(self) -> None:
        self.hours = (6, 7, 8)
        rows = []
        for day_index, day in enumerate(("2024-01-01", "2024-01-02", "2024-01-03")):
            for hour_index, hour in enumerate(self.hours):
                rows.append(
                    {
                        "delivery_date_local": day,
                        "hour_local": hour,
                        "pv_residual_kwh": float(10 * day_index + hour_index),
                        "price_residual_eur_per_mwh": float(-100 * day_index - hour_index),
                    }
                )
        self.library = pd.DataFrame(rows)

    def test_samples_pv_and_price_from_the_same_complete_day(self) -> None:
        generator = JointResidualBootstrap(self.library, hours_local=self.hours)
        batch = generator.sample(
            point_pv_kwh=np.full(3, 100.0),
            point_price_eur_per_mwh=np.full(3, 1_000.0),
            n_scenarios=50,
            random_state=7,
            as_of_date=date(2024, 1, 4),
        )
        for index, source_day in enumerate(batch.source_residual_days):
            source = self.library.loc[
                pd.to_datetime(self.library["delivery_date_local"]).dt.date.eq(source_day)
            ].sort_values("hour_local")
            np.testing.assert_allclose(
                batch.pv_kwh[index] - 100.0,
                source["pv_residual_kwh"],
            )
            np.testing.assert_allclose(
                batch.day_ahead_price_eur_per_mwh[index] - 1_000.0,
                source["price_residual_eur_per_mwh"],
            )

    def test_excludes_residual_days_not_strictly_before_as_of(self) -> None:
        generator = JointResidualBootstrap(self.library, hours_local=self.hours)
        batch = generator.sample(
            point_pv_kwh=np.ones(3),
            point_price_eur_per_mwh=np.ones(3),
            n_scenarios=20,
            random_state=3,
            as_of_date=date(2024, 1, 2),
        )
        self.assertEqual(set(batch.source_residual_days), {date(2024, 1, 1)})

    def test_applies_pv_physical_bounds(self) -> None:
        generator = JointResidualBootstrap(self.library, hours_local=self.hours)
        batch = generator.sample(
            point_pv_kwh=np.zeros(3),
            point_price_eur_per_mwh=np.zeros(3),
            n_scenarios=10,
            random_state=1,
            pv_capacity_kwh_per_hour=5.0,
        )
        self.assertTrue((batch.pv_kwh >= 0.0).all())
        self.assertTrue((batch.pv_kwh <= 5.0).all())


if __name__ == "__main__":
    unittest.main()
