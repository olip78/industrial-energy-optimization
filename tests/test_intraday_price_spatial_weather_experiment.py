from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from energy.training.intraday_price_spatial_weather_experiment import (
    attach_spatial_ifs_features,
)


VARIABLES = (
    "temperature_2m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "wind_speed_120m",
    "cloud_cover",
)


class IntradayPriceSpatialWeatherExperimentTest(unittest.TestCase):
    def test_selects_only_forecasts_published_before_each_cutoff(self) -> None:
        target = pd.Timestamp("2024-06-01T12:00:00Z")
        frame = pd.DataFrame(
            {
                "as_of_utc": [
                    pd.Timestamp("2024-06-01T05:00:00Z"),
                    pd.Timestamp("2024-06-01T07:00:00Z"),
                ],
                "target_valid_time_utc": [target, target],
                "delivery_date_local": ["2024-06-01", "2024-06-01"],
            }
        )
        rows: list[dict[str, object]] = []
        for location_id in ("north", "south"):
            for run_init, value in (
                (pd.Timestamp("2024-05-31T00:00:00Z"), 10.0),
                (pd.Timestamp("2024-06-01T00:00:00Z"), 20.0),
            ):
                rows.append(
                    {
                        "location_id": location_id,
                        "run_init_utc": run_init,
                        "available_at_utc": run_init + pd.Timedelta(hours=6),
                        "valid_time_utc": target,
                        **{
                            f"weather_forecast_{variable}": value
                            for variable in VARIABLES
                        },
                    }
                )
        weather = pd.DataFrame(rows)

        enriched, _, audit = attach_spatial_ifs_features(frame, weather)

        # At 05:00 the current-day run has not yet been published.  At 07:00 it
        # has; the day-ahead view remains the May-31 run for both rows.
        np.testing.assert_allclose(
            enriched["ifs_day_ahead_temperature_2m_mean"], [10.0, 10.0]
        )
        np.testing.assert_allclose(
            enriched["ifs_latest_temperature_2m_mean"], [10.0, 20.0]
        )
        np.testing.assert_allclose(
            enriched["ifs_revision_temperature_2m_mean"], [0.0, 10.0]
        )
        self.assertTrue(audit["all_day_ahead_available_by_cutoff"])
        self.assertTrue(audit["all_latest_available_by_decision"])


if __name__ == "__main__":
    unittest.main()
