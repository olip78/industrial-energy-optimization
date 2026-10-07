from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from energy.data.intraday_price import (
    DAY_AHEAD_COLUMN,
    TARGET_COLUMN,
    _feature_names,
    _make_mpc_rows,
    _validate_mpc_rows,
)
from energy.training.intraday_price import calibrated_intraday_price


class IntradayPriceFeaturesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        valid_time = pd.date_range("2023-12-01T00:00:00Z", periods=60 * 24, freq="h")
        hour = np.arange(len(valid_time), dtype=float)
        day_ahead = 50.0 + 0.2 * hour
        intraday = day_ahead + np.sin(hour / 3.0)
        cls.source = pd.DataFrame(
            {
                "valid_time_utc": valid_time,
                DAY_AHEAD_COLUMN: day_ahead,
                TARGET_COLUMN: intraday,
                "intraday_continuous_id1_price_eur_per_mwh": intraday + 0.1,
                "intraday_continuous_id3_price_eur_per_mwh": intraday - 0.1,
            }
        )
        cls.rows = _make_mpc_rows(cls.source)

    def test_features_are_available_and_target_is_future(self) -> None:
        _validate_mpc_rows(self.rows, tuple(_feature_names()))
        self.assertTrue(
            (self.rows["intraday_history_available_at_utc"] <= self.rows["as_of_utc"]).all()
        )
        self.assertTrue(
            (self.rows["day_ahead_price_available_at_utc"] <= self.rows["as_of_utc"]).all()
        )
        self.assertTrue(
            (self.rows["target_available_at_utc"] > self.rows["as_of_utc"]).all()
        )
        self.assertTrue(
            (self.rows["target_valid_time_utc"] > self.rows["as_of_utc"]).all()
        )

    def test_lag_one_uses_the_last_completed_hour(self) -> None:
        row = self.rows.iloc[len(self.rows) // 2]
        expected_time = row["as_of_utc"] - pd.Timedelta(hours=1)
        expected = self.source.loc[
            self.source["valid_time_utc"] == expected_time, TARGET_COLUMN
        ].iloc[0]
        self.assertAlmostEqual(row["intraday_price_lag_1_eur_per_mwh"], expected)

    def test_target_hour_is_not_part_of_the_observation_window(self) -> None:
        row = self.rows.iloc[len(self.rows) // 2]
        self.assertLess(row["as_of_utc"], row["target_valid_time_utc"])
        self.assertLessEqual(row["lead_hours"], 23)
        self.assertGreaterEqual(row["lead_hours"], 1)

    def test_enriched_features_keep_the_same_availability_contract(self) -> None:
        enriched = _make_mpc_rows(self.source, feature_version="v2")
        feature_names = _feature_names("v2")
        _validate_mpc_rows(enriched, tuple(feature_names))
        self.assertTrue(
            {
                "intraday_last6_mean_d7_eur_per_mwh",
                "intraday_last6_mean_d14_eur_per_mwh",
                "intraday_last6_mean_ratio_d7",
                "intraday_ar1_forecast_eur_per_mwh",
            }.issubset(enriched.columns)
        )
        self.assertFalse(enriched[feature_names].isna().any().any())

    def test_calibrated_spread_is_applied_only_to_next_hour(self) -> None:
        prediction = calibrated_intraday_price(
            np.array([100.0, 110.0, 120.0]),
            np.array([10.0, -20.0, 30.0]),
            np.array([1, 2, 1]),
            correction_weight=0.7,
        )
        np.testing.assert_allclose(prediction, np.array([107.0, 110.0, 141.0]))


if __name__ == "__main__":
    unittest.main()
