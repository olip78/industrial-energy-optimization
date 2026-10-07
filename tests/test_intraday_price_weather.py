from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from energy.data.intraday_price_weather import (
    SpatialEcmwfIfsArchiveConfig,
    _ModelRunUnavailableError,
    collect_spatial_ecmwf_ifs_single_runs,
    materialize_spatial_ecmwf_ifs_archive,
)


VARIABLES = (
    "temperature_2m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "wind_speed_120m",
    "cloud_cover",
)


class IntradayPriceWeatherTest(unittest.TestCase):
    def test_collection_keeps_each_requested_location_and_run_availability(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "locations.json"
            locations = [
                {"location_id": "north", "latitude": 54.4, "longitude": 7.7, "role": "wind"},
                {"location_id": "south", "latitude": 48.8, "longitude": 11.6, "role": "solar"},
            ]
            config_path.write_text(json.dumps({"locations": locations}))
            requested_urls: list[str] = []

            def fetcher(url: str) -> list[dict[str, object]]:
                requested_urls.append(url)
                return [
                    {"hourly": {"time": ["2024-03-14T00:00", "2024-03-14T01:00"], **{name: [index, index + 1] for index, name in enumerate(VARIABLES)}}}
                    for _ in locations
                ]

            result = collect_spatial_ecmwf_ifs_single_runs(
                SpatialEcmwfIfsArchiveConfig(
                    project_root=root,
                    locations_path=config_path,
                    start=date(2024, 3, 14),
                    end=date(2024, 3, 14),
                    forecast_hours=48,
                    run_hours_utc=(0,),
                ),
                fetcher=fetcher,
            )
            self.assertEqual(result.locations, 2)
            self.assertEqual(result.hourly_rows, 4)
            self.assertEqual(len(requested_urls), 1)
            query = parse_qs(urlparse(requested_urls[0]).query)
            self.assertEqual(query["latitude"], ["54.4,48.8"])
            table = pd.read_parquet(result.processed_path)
            self.assertEqual(set(table["location_id"]), {"north", "south"})
            self.assertTrue((pd.to_datetime(table["available_at_utc"], utc=True) > pd.to_datetime(table["run_init_utc"], utc=True)).all())

            materialized = materialize_spatial_ecmwf_ifs_archive(root, locations_path=config_path)
            self.assertEqual(materialized.locations, 2)
            self.assertEqual(materialized.hourly_rows, 4)

    def test_collection_caches_plain_text_model_run_gap_as_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "locations.json"
            config_path.write_text(
                json.dumps(
                    {
                        "locations": [
                            {
                                "location_id": "north",
                                "latitude": 54.4,
                                "longitude": 7.7,
                                "role": "wind",
                            }
                        ]
                    }
                )
            )

            def unavailable(_: str) -> list[dict[str, object]]:
                raise _ModelRunUnavailableError("modelRunUnavailable")

            with self.assertRaisesRegex(RuntimeError, "no successful runs"):
                collect_spatial_ecmwf_ifs_single_runs(
                    SpatialEcmwfIfsArchiveConfig(
                        project_root=root,
                        locations_path=config_path,
                        start=date(2024, 3, 14),
                        end=date(2024, 3, 14),
                        forecast_hours=48,
                        run_hours_utc=(0,),
                    ),
                    fetcher=unavailable,
                )
            records = list(
                (root / "data/raw/weather_forecasts_ecmwf_ifs_spatial_single_runs/runs").glob("*.json")
            )
            self.assertEqual(len(records), 1)
            payload = json.loads(records[0].read_text())
            self.assertEqual(payload["status"], "unavailable")
            self.assertEqual(payload["error"], "modelRunUnavailable")


if __name__ == "__main__":
    unittest.main()
