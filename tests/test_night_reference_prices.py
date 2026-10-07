from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from energy.training.economic_backtest import _night_reference_prices


class NightReferencePricesTest(unittest.TestCase):
    def test_reference_uses_fourteen_completed_nine_label_nights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            processed = root / "data" / "processed"
            processed.mkdir(parents=True)
            parts = []
            for night_index in range(14):
                night_start = pd.Timestamp(
                    "2025-01-01 22:00", tz="Europe/Berlin"
                ) + pd.Timedelta(days=night_index)
                local_time = pd.date_range(
                    night_start,
                    night_start + pd.Timedelta(hours=8),
                    freq="h",
                )
                price = -100.0 if night_index == 0 else float(night_index + 1)
                parts.append(
                    pd.DataFrame(
                        {
                            "valid_time_utc": local_time.tz_convert("UTC"),
                            "day_ahead_price_eur_per_mwh": price,
                        }
                    )
                )
            frame = pd.concat(parts, ignore_index=True)
            frame.to_csv(
                processed / "prices_day_ahead_de_lu.csv.gz",
                index=False,
                compression="gzip",
            )

            result = _night_reference_prices(root)

            # The negative first-night mean is floored before the rolling mean.
            self.assertEqual(list(result), [date(2025, 1, 16)])
            self.assertAlmostEqual(result[date(2025, 1, 16)], 104.0 / 14.0)


if __name__ == "__main__":
    unittest.main()
