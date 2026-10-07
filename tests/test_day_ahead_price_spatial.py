from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

import pandas as pd


DATA_ROOT = Path(os.environ.get("ENERGY_TEST_DATA_ROOT", "data"))
DATASET = (
    DATA_ROOT
    / "features"
    / "day_ahead_price_spatial"
    / "day_ahead_price_spatial_2024-02-18_2025-09-30.parquet"
)
MANIFEST = DATA_ROOT / "metadata" / "day_ahead_price_spatial_v1_manifest.json"


class DayAheadPriceSpatialDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = pd.read_parquet(DATASET)
        cls.manifest = json.loads(MANIFEST.read_text())

    def test_selected_features_are_complete(self) -> None:
        features = sum(self.manifest["feature_groups"].values(), [])
        self.assertFalse(self.frame[features].isna().any().any())

    def test_every_temporal_feature_is_available_at_as_of(self) -> None:
        as_of = pd.to_datetime(self.frame["as_of_utc"], utc=True)
        history_available = pd.to_datetime(self.frame["price_history_available_at_utc"], utc=True)
        self.assertTrue((history_available <= as_of).all())
        for column in self.manifest["weather_availability_audit_columns"]:
            available = pd.to_datetime(self.frame[column], utc=True)
            self.assertTrue((available <= as_of).all(), msg=column)
        target_available = pd.to_datetime(self.frame["target_available_at_utc"], utc=True)
        self.assertTrue((target_available > as_of).all())

    def test_rows_represent_regular_24_hour_delivery_days(self) -> None:
        counts = self.frame.groupby("delivery_date_local").size()
        self.assertTrue((counts == 24).all())
        hour_sets = self.frame.groupby("delivery_date_local")["hour_local"].agg(set)
        self.assertTrue(hour_sets.map(lambda hours: hours == set(range(24))).all())
        self.assertFalse(self.frame.duplicated("valid_time_utc").any())


if __name__ == "__main__":
    unittest.main()
