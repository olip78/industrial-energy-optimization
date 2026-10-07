from __future__ import annotations

import unittest
from datetime import date

import numpy as np
import pandas as pd

from energy.uncertainty import QuantileEmpiricalCopula


class QuantileEmpiricalCopulaTest(unittest.TestCase):
    def setUp(self) -> None:
        self.hours = (6, 7)
        self.levels = (0.1, 0.5, 0.9)
        self.library = pd.DataFrame(
            {
                "delivery_date_local": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "pv_u_h06": [0.2, 0.5, 0.8],
                "pv_u_h07": [0.3, 0.6, 0.9],
                "price_u_h06": [0.8, 0.5, 0.2],
                "price_u_h07": [0.9, 0.6, 0.3],
            }
        )

    def test_preserves_each_sampled_daily_rank_vector(self) -> None:
        generator = QuantileEmpiricalCopula(
            self.library, quantile_levels=self.levels, hours_local=self.hours
        )
        batch = generator.sample(
            pv_quantiles_kwh=np.array([[0.0, 5.0, 10.0], [0.0, 10.0, 20.0]]),
            price_quantiles_eur_per_mwh=np.array(
                [[0.0, 50.0, 100.0], [100.0, 150.0, 200.0]]
            ),
            n_scenarios=20,
            random_state=7,
            as_of_date=date(2024, 1, 4),
        )
        indexed = self.library.assign(
            delivery_date_local=pd.to_datetime(
                self.library["delivery_date_local"]
            ).dt.date
        ).set_index("delivery_date_local")
        for index, source_day in enumerate(batch.source_copula_days):
            np.testing.assert_allclose(
                batch.pv_uniform_ranks[index],
                indexed.loc[source_day, ["pv_u_h06", "pv_u_h07"]],
            )
            np.testing.assert_allclose(
                batch.price_uniform_ranks[index],
                indexed.loc[source_day, ["price_u_h06", "price_u_h07"]],
            )

    def test_inverts_quantiles_and_applies_pv_cap(self) -> None:
        generator = QuantileEmpiricalCopula(
            self.library.iloc[[1]], quantile_levels=self.levels, hours_local=self.hours
        )
        batch = generator.sample(
            pv_quantiles_kwh=np.array([[0.0, 5.0, 10.0], [0.0, 10.0, 20.0]]),
            price_quantiles_eur_per_mwh=np.array(
                [[0.0, 50.0, 100.0], [100.0, 150.0, 200.0]]
            ),
            n_scenarios=1,
            random_state=1,
            pv_capacity_kwh_per_hour=8.0,
        )
        np.testing.assert_allclose(batch.pv_kwh, [[5.0, 8.0]])
        np.testing.assert_allclose(batch.day_ahead_price_eur_per_mwh, [[50.0, 162.5]])

    def test_rearranges_crossed_quantiles(self) -> None:
        generator = QuantileEmpiricalCopula(
            self.library.iloc[[1]], quantile_levels=self.levels, hours_local=self.hours
        )
        batch = generator.sample(
            pv_quantiles_kwh=np.array([[10.0, 0.0, 5.0], [20.0, 0.0, 10.0]]),
            price_quantiles_eur_per_mwh=np.array(
                [[100.0, 0.0, 50.0], [200.0, 100.0, 150.0]]
            ),
            n_scenarios=1,
            random_state=1,
        )
        np.testing.assert_allclose(batch.pv_kwh, [[5.0, 12.5]])
        np.testing.assert_allclose(batch.day_ahead_price_eur_per_mwh, [[50.0, 162.5]])


if __name__ == "__main__":
    unittest.main()
