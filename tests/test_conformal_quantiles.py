from __future__ import annotations

import unittest

import numpy as np

from energy.uncertainty import CentralIntervalCQR


class CentralIntervalCQRTest(unittest.TestCase):
    def test_expands_interval_by_finite_sample_score(self) -> None:
        levels = (0.1, 0.5, 0.9)
        prediction = np.tile(np.array([4.0, 5.0, 6.0]), (10, 1))
        actual = np.array([0.0] * 5 + [10.0] * 5)
        calibrator = CentralIntervalCQR.fit(
            actual=actual,
            quantile_predictions=prediction,
            groups=np.zeros(10, dtype=int),
            quantile_levels=levels,
        )
        calibrated = calibrator.transform(
            prediction[:1], np.zeros(1, dtype=int)
        )
        np.testing.assert_allclose(calibrated, [[0.0, 5.0, 10.0]])

    def test_never_shrinks_an_overcovering_interval(self) -> None:
        levels = (0.1, 0.5, 0.9)
        prediction = np.tile(np.array([0.0, 5.0, 10.0]), (20, 1))
        calibrator = CentralIntervalCQR.fit(
            actual=np.full(20, 5.0),
            quantile_predictions=prediction,
            groups=np.zeros(20, dtype=int),
            quantile_levels=levels,
        )
        self.assertEqual(float(calibrator.corrections["correction"].iloc[0]), 0.0)

    def test_uses_separate_group_corrections_and_physical_bounds(self) -> None:
        levels = (0.1, 0.5, 0.9)
        prediction = np.tile(np.array([2.0, 5.0, 8.0]), (20, 1))
        actual = np.concatenate((np.full(10, 5.0), np.full(10, 12.0)))
        groups = np.concatenate((np.zeros(10, dtype=int), np.ones(10, dtype=int)))
        calibrator = CentralIntervalCQR.fit(
            actual=actual,
            quantile_predictions=prediction,
            groups=groups,
            quantile_levels=levels,
        )
        calibrated = calibrator.transform(
            np.tile(np.array([2.0, 5.0, 8.0]), (2, 1)),
            np.array([0, 1]),
            lower_bound=0.0,
            upper_bound=10.0,
        )
        np.testing.assert_allclose(calibrated[0], [2.0, 5.0, 8.0])
        np.testing.assert_allclose(calibrated[1], [0.0, 5.0, 10.0])


if __name__ == "__main__":
    unittest.main()
