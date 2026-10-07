from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from energy.data.intraday_weather import (
    _response_to_frame,
    build_mpc_weather_snapshots,
)


WEATHER_VALUES = {
    "temperature_2m": [10.0, 11.0, 12.0],
    "shortwave_radiation": [0.0, 100.0, 200.0],
    "direct_radiation": [0.0, 50.0, 120.0],
    "diffuse_radiation": [0.0, 50.0, 80.0],
    "wind_speed_10m": [5.0, 6.0, 7.0],
    "cloud_cover": [90.0, 80.0, 70.0],
}


class IntradayWeatherTest(unittest.TestCase):
    def test_response_is_normalised_with_a_separate_publication_time(self) -> None:
        frame = _response_to_frame(
            {
                "run_init_utc": "2024-06-01T00:00:00+00:00",
                "available_at_utc": "2024-06-01T06:00:00+00:00",
                "response": {
                    "hourly": {
                        "time": ["2024-06-01T00:00", "2024-06-01T01:00", "2024-06-01T02:00"],
                        **WEATHER_VALUES,
                    }
                },
            }
        )
        self.assertEqual(len(frame), 3)
        self.assertTrue((frame["available_at_utc"] > frame["run_init_utc"]).all())
        self.assertIn("weather_forecast_shortwave_radiation", frame)

    def test_snapshot_builder_selects_only_runs_published_before_the_decision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            feature_directory = root / "data" / "features" / "day_ahead_pv"
            feature_directory.mkdir(parents=True)
            delivery_day = "2024-06-01"
            valid_times = pd.date_range("2024-05-31T22:00:00Z", periods=24, freq="h")
            pd.DataFrame(
                {
                    "delivery_date_local": [delivery_day] * 24,
                    "valid_time_utc": valid_times,
                    "hour_local": list(range(24)),
                }
            ).to_parquet(feature_directory / "day_ahead_pv_2024.parquet", index=False)

            processed_directory = root / "data" / "processed"
            processed_directory.mkdir(parents=True)
            forecast_times = pd.date_range("2024-05-31T18:00:00Z", periods=48, freq="h")
            runs = []
            for run, available, value in (
                ("2024-05-31T18:00:00Z", "2024-06-01T00:00:00Z", 10.0),
                ("2024-06-01T00:00:00Z", "2024-06-01T06:00:00Z", 20.0),
            ):
                runs.append(
                    pd.DataFrame(
                        {
                            "run_init_utc": run,
                            "available_at_utc": available,
                            "valid_time_utc": forecast_times,
                            **{f"weather_forecast_{name}": value for name in WEATHER_VALUES},
                        }
                    )
                )
            source_path = processed_directory / "weather_forecast_hourly_ecmwf_ifs_single_runs.parquet"
            pd.concat(runs, ignore_index=True).to_parquet(source_path, index=False)

            result = build_mpc_weather_snapshots(root, year=2024, source_path=source_path)
            snapshots = pd.read_parquet(result.output_path)
            six_utc = snapshots.loc[
                (snapshots["decision_hour_local"] == 8)
                & (pd.to_datetime(snapshots["valid_time_utc"], utc=True) == pd.Timestamp("2024-06-01T07:00:00Z"))
            ]
            self.assertEqual(len(six_utc), 1)
            self.assertEqual(float(six_utc["weather_forecast_temperature_2m"].iloc[0]), 20.0)
            self.assertTrue(
                (
                    pd.to_datetime(snapshots["weather_snapshot_available_at_utc"], utc=True)
                    <= pd.to_datetime(snapshots["as_of_utc"], utc=True)
                ).all()
            )

if __name__ == "__main__":
    unittest.main()
